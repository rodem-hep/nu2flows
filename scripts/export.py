from pathlib import Path

import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging

import h5py
import hydra
import torch as T
from omegaconf import DictConfig

from mltools.mltools.hydra_utils import reload_original_config
from mltools.mltools.torch_utils import to_np

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path=str(root / "configs"), config_name="export.yaml"
)
def main(cfg: DictConfig) -> None:
    log.info("Loading run information")
    orig_cfg = reload_original_config(ckpt_flag="*last*")  # TODO: Change this to best

    log.info("Loading best checkpoint")
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    model_class = hydra.utils.get_class(orig_cfg.model._target_)
    model = model_class.load_from_checkpoint(orig_cfg.ckpt_path, map_location=device)

    # Use the export batch size
    orig_cfg.datamodule.loader_conf.batch_size = cfg.batch_size

    # Cycle through the datasets and create the dataloader
    for dataset in cfg.datasets:
        log.info(f"Instantiating the data module for {dataset}")
        orig_cfg.datamodule.test_conf.file_list = [dataset]
        datamodule = hydra.utils.instantiate(orig_cfg.datamodule)

        log.info("Instantiating the trainer")
        orig_cfg.trainer["enable_progress_bar"] = True
        orig_cfg.trainer["logger"] = False  # Prevents a lightning_logs folder
        trainer = hydra.utils.instantiate(orig_cfg.trainer)

        log.info("Running the prediction loop")
        outputs = trainer.predict(model=model, datamodule=datamodule)

        log.info("Combining predictions across dataset")
        keys = list(outputs[0].keys())
        comb_dict = {key: T.vstack([o[key] for o in outputs]) for key in keys}

        log.info("Saving Outputs")
        Path("outputs").mkdir(exist_ok=True, parents=True)
        with h5py.File(f"outputs/{dataset}", mode="w") as file:
            for key in keys:
                file.create_dataset(key, data=to_np(comb_dict[key]))


if __name__ == "__main__":
    main()
