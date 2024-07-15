import logging
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from mltools.mltools.torch_utils import train_valid_split

log = logging.getLogger(__name__)


class Geant4H5Dataset(Dataset):
    """Dataset class for neutrino regression stored as H5 files."""

    def __init__(
        self,
        *,
        file_name: str,
        file_dir: str,
        group_name: str = "even",
        num_events: int | None = None,
    ) -> None:
        """Parameters
        ----------
        file_name: str
            The name of the file to load.
        file_dir: str
            The directory containing the file.
        group_name: str
            The name of the group in the file to load.
        num_events: int, optional
            The number of events to load. If None, all events are loaded.
        """
        super().__init__()

        file_path = Path(file_dir, file_name)
        with h5py.File(file_path, "r") as f:
            self.jet = f[group_name]["jet"][:num_events]
            self.lep = f[group_name]["lep"][:num_events]
            self.met = f[group_name]["met"][:num_events]
            self.misc = f[group_name]["misc"][:num_events]
            self.nu = f[group_name]["neutrinos"][:num_events]
            self.evt_info = f[group_name]["evt_info"][:num_events]
        log.info(f"{len(self.met)} events loaded")

        # Jets need to be padded so create a mask
        self.jet_mask = ~np.all(self.jet == 0, axis=-1)

        # Get the weight of the event
        ws = [
            "weight_mc_NOSYS",
            "weight_pileup_NOSYS",
            "weight_beamspot",
            "weight_btagSF_DL1dv01_Continuous_NOSYS",
        ]

        # Combine the weights by reducing the product
        self.weight = np.prod([self.evt_info[w] for w in ws], axis=0).astype(np.float32)
        self.weight /= np.mean(self.weight)  # Ensure average weight is 1

    def __len__(self) -> int:
        return len(self.met)

    def __getitem__(self, idx: int) -> list:
        """Return dictionaries for the inputs and the targets."""
        inputs = {
            "jets": (self.jet[idx], self.jet_mask[idx]),
            "leptons": self.lep[idx],
            "met": self.met[idx],
            "misc": self.misc[idx],
        }
        targets = {"neutrinos": self.nu[idx][0], "antineutrino": self.nu[idx][1]}
        return inputs, targets, self.weight[idx]

    def get_input_dims(self) -> tuple:
        """Return the typical dimensions of a data sample."""
        return {
            k: v[0].shape[-1] if isinstance(v, tuple) else v.shape[-1]
            for k, v in self[0][0].items()
        }

    def get_target_dims(self) -> tuple:
        """Return the typical dimensions of a data sample."""
        return {
            k: v[0].shape[-1] if isinstance(v, tuple) else v.shape[-1]
            for k, v in self[0][1].items()
        }


class Geant4H5DataModule(LightningDataModule):
    def __init__(
        self,
        *,
        train_conf: Mapping,
        test_conf: Mapping,
        loader_conf: Mapping,
        val_frac: float = 0.1,
    ) -> None:
        """The datamodule for providing dilepton information.

        Parameters
        ----------
        train_conf:
            Config for the training dataset class.
        test_conf:
            Config for the testing dataset class.
        loader_conf:
            Config for the pytorch dataloader.
        val_frac:
            Fraction of dataset held out for validaiton. Defaults to 0.1.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Initialise a mini test set for now to infer the dimensions of the samples
        mini_conf = deepcopy(test_conf)
        mini_conf["num_events"] = 2
        self.miniset = Geant4H5Dataset(**mini_conf)

    def setup(self, stage: str) -> None:
        if stage in {"fit", "validate"}:
            self.dataset = Geant4H5Dataset(**self.hparams.train_conf)
            self.train_set, self.valid_set = train_valid_split(self.dataset, self.hparams.val_frac)
            self.n_train = len(self.train_set)
            self.n_valid = len(self.valid_set)
            log.info(f"Split data into {self.n_train} train and {self.n_valid} valid")

        if stage in {"test", "predict"}:
            self.test_set = Geant4H5Dataset(**self.hparams.test_conf)
            self.n_test_samples = len(self.test_set)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, **self.hparams.loader_conf, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_set, **self.hparams.loader_conf, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        self.hparams.loader_conf["drop_last"] = False
        return DataLoader(self.test_set, **self.hparams.loader_conf, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    @property
    def model_kwargs(self) -> dict:
        """A dict used to instantiate the model."""
        return {
            "input_dimensions": self.miniset.get_input_dims(),
            "target_dimensions": self.miniset.get_target_dims(),
        }
