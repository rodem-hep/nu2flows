import logging
from copy import deepcopy
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from mltools.mltools.torch_utils import train_valid_split

log = logging.getLogger(__name__)


class Geant4H5Dataset(Dataset):
    """Dataset class for neutrino regression stored as H5 files."""

    def __init__(
        self,
        *,
        file_list: list,
        data_dir: str,
        n_per_file: int = None,
        table_name: str = "atlas_even",
    ) -> None:
        """
        Parameters
        ----------
        file_list:
            List of datasets to load
        data_dir:
            The location of the datafiles
        n_per_file:
            Maximum number of events to load from each file
        table_name:
            The name of the table in the hdf5 file
        """
        super().__init__()

        # Get the list of files to use for the dataset
        self.file_list = []
        for f in file_list:
            file = Path(data_dir, f)
            if not file.exists():
                raise ValueError(f"Can't find requested file: {file}")
            self.file_list.append(file)

        # All data loaded from the files will be stored in these tensors
        npf = n_per_file
        log.info(f"loading {npf} events from each files...")
        self.misc = []
        self.met = []
        self.lep = []
        self.jet = []
        self.nu = []
        for file in self.file_list:
            log.info(file.name)
            with h5py.File(file, "r") as f:
                table = f[table_name]

                # TODO Change this!
                # For now there is a bug causing some of the neutrinos to be inf
                # Or really small. We need to filter these out
                nu = table["neutrinos"][:npf]
                mask = ~np.isinf(nu).any(axis=(-1, -2))

                # Misc for now is just the number of jets
                self.misc.append(table["nJets"][:npf][mask][..., None])
                self.misc_vars = ["nJets"]

                # The MET is stored in two parts
                met_x = table["met_x"][:npf][mask][..., None]
                met_y = table["met_y"][:npf][mask][..., None]
                self.met.append(np.hstack([met_x, met_y]))

                self.lep.append(table["leptons"][:npf][mask])
                self.jet.append(table["jets"][:npf][mask])
                self.nu.append(table["neutrinos"][:npf][mask])

        self.misc = np.vstack(self.misc).astype(np.float32)
        self.met = np.vstack(self.met).astype(np.float32)
        self.lep = np.vstack(self.lep).astype(np.float32)
        self.jet = np.vstack(self.jet).astype(np.float32)
        self.nu = np.vstack(self.nu).astype(np.float32) / 1000  # Nus are in MeV?
        log.info(f"{len(self.met)} events loaded")

        # Clamp the neutrinos because some of them have inf values
        self.nuy

        np.max(self.misc)
        np.max(self.met)
        np.max(self.lep)
        np.max(self.jet)
        np.max(self.nu)
        a = self.nu
        np.unravel_index(a.argmax(), a.shape)
        (self.nu[self.nu != np.inf]).max()

        # Jets need to be padded so create a mask
        self.jet_mask = ~np.all(self.jet == 0, axis=-1)

    def __len__(self) -> int:
        return len(self.met)

    def __getitem__(self, idx: int) -> list:
        """Return dictionaries for the inputs and the targets."""
        inputs = {
            "misc": self.misc[idx],
            "met": self.met[idx],
            "leptons": self.lep[idx],
            "jets": (self.jet[idx], self.jet_mask[idx]),
        }
        targets = {"neutrino": self.nu[idx][0], "antineutrino": self.nu[idx][1]}
        return inputs, targets

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


class Geant4H5DataModule(pl.LightningDataModule):
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
        mini_conf["n_per_file"] = 2
        self.miniset = Geant4H5Dataset(**mini_conf)

    def setup(self, stage: str) -> None:
        if stage in ["fit", "validate"]:
            self.dataset = Geant4H5Dataset(**self.hparams.train_conf)
            # self.dataset.plot_variables("plots")
            self.train_set, self.valid_set = train_valid_split(
                self.dataset, self.hparams.val_frac
            )
            self.n_train_samples = len(self.train_set)
            self.n_valid_samples = len(self.valid_set)

        if stage in ["test", "predict"]:
            self.test_set = Geant4H5Dataset(**self.hparams.test_conf)
            self.n_test_samples = len(self.test_set)

    def input_dimensions(self) -> tuple:
        """Return the typical dimensions of a input sample."""
        return self.miniset.get_input_dims()

    def target_dimensions(self) -> tuple:
        """Return the typical dimensions of the target sample."""
        return self.miniset.get_target_dims()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, **self.hparams.loader_conf, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_set, **self.hparams.loader_conf, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        self.hparams.loader_conf["drop_last"] = False
        return DataLoader(self.test_set, **self.hparams.loader_conf, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
