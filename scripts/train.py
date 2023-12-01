import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging

import hydra
import pytorch_lightning as pl
import torch as T
from omegaconf import DictConfig

from mltools.mltools.hydra_utils import (
    instantiate_collection,
    log_hyperparameters,
    print_config,
    reload_original_config,
    save_config,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path=str(root / "configs"), config_name="train.yaml"
)
def main(cfg: DictConfig) -> None:
    log.info("Setting up full job config")
    if cfg.full_resume:
        cfg = reload_original_config(cfg)
    print_config(cfg)

    if cfg.seed:
        log.info(f"Setting seed to: {cfg.seed}")
        pl.seed_everything(cfg.seed, workers=True)

    if cfg.precision:
        log.info(f"Setting matrix precision to: {cfg.precision}")
        T.set_float32_matmul_precision(cfg.precision)

    log.info("Instantiating the data module")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    log.info("Instantiating the model")
    model = hydra.utils.instantiate(
        cfg.model,
        input_dimensions=datamodule.input_dimensions(),
        target_dimensions=datamodule.target_dimensions(),
    )

    log.info("Instantiating all callbacks")
    callbacks = instantiate_collection(cfg.callbacks)

    log.info("Instantiating the loggers")
    loggers = instantiate_collection(cfg.loggers)

    log.info("Instantiating the trainer")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    if loggers:
        log.info("Logging all hyperparameters")
        log_hyperparameters(cfg, model, trainer)

    log.info("Saving config so job can be resumed")
    save_config(cfg)
    log.info(model)

    log.info("Starting training!")
    trainer.fit(model, datamodule, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
