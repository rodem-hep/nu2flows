from functools import partial
from typing import Mapping

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import wandb

from utils.bayesian import contains_bayesian_layers
from utils.modules import DenseNetwork
from utils.torch_utils import get_sched
from utils.transformers import TransformerVectorEncoder
from src.models.nuflows import NuFlowBase


class TransFeedForward(NuFlowBase):
    """FeedForward only network that uses a transformer-vector encoder."""

    def __init__(
        self,
        *,
        data_dims: list,
        part_embed_conf: Mapping,
        tve_conf: Mapping,
        sched_conf: Mapping,
        optimizer: partial,
        prior_weight: float = 1e-5
    ) -> None:
        """
        Args:
            data_dims: List of the dimensions of the data used in this model
                [misc, met, leptons, jets, nu]
            part_embed_conf: Config for the particle embedders
            tve_conf: Configuration for the transformer vector encoder.
            embed_conf: Context embedding network configuraion.
            flow_conf: Invertible neural network configuration.
            prior_weight: Weight for the bayesian loss term
            sched_conf: Config for the scheduler
            optimizer: Partially initialised optimizer
            prior_weight: Weight for the bayesian prior, only if bayesian
        """
        super().__init__(data_dims)

        # Initialise the transformer vector encoder
        self.transformer = TransformerVectorEncoder(**tve_conf, ctxt_dim=self.misc_dim)

        # Initialise each of the embedding networks
        self.met_embedder = DenseNetwork(
            inpt_dim=self.met_dim,
            outp_dim=self.transformer.model_dim,
            **part_embed_conf,
        )
        self.lep_embedder = DenseNetwork(
            inpt_dim=self.lep_dim[-1],
            outp_dim=self.transformer.model_dim,
            **part_embed_conf,
        )
        self.jet_embedder = DenseNetwork(
            inpt_dim=self.jet_dim[-1],
            outp_dim=self.transformer.model_dim,
            **part_embed_conf,
        )

        # Here we just use a single output linear layer
        self.out_linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.transformer.model_dim, self.n_nu * self.nu_features),
        )

        # Check to see if the network has any bayesian layers
        self.do_bayesian = contains_bayesian_layers(self)

    def on_fit_start(self, *_args) -> None:
        """Function to run at the start of training."""

        # Define the metrics for wandb (otherwise the min wont be stored!)
        if wandb.run is not None:
            wandb.define_metric("train/total_loss", summary="min")
            wandb.define_metric("valid/total_loss", summary="min")

    def forward(
        self,
        misc: T.Tensor,
        met: T.Tensor,
        lep: T.Tensor,
        jet: T.Tensor,
    ) -> tuple:
        """Predict directly the neutrino momenta from the network."""

        # Make a mask for the input jets based on zero padding
        jet_mask = T.any(jet != 0, dim=-1)

        # Pass the inputs through the normalisation layers
        misc = self.misc_normaliser(misc)
        met = self.met_normaliser(met)
        lep = self.lep_normaliser(lep)
        jet = self.jet_normaliser(jet, jet_mask)

        # Pass each of the particles through the embedding networks
        met = self.met_embedder(met)
        lep = self.lep_embedder(lep)
        jet = self.jet_embedder(jet)

        # Combine them all into a single tensor
        combined = T.concat([met.unsqueeze(1), lep, jet], dim=1)

        # Get a mask for all the elements
        mask = T.concat(
            [
                T.full((len(met), 1 + self.n_lep), True, device=self.device),
                jet_mask,
            ],
            dim=-1,
        )

        # Pass the combined tensor through the transformer
        output = self.transformer(
            seq=combined,
            mask=mask,
            ctxt=misc,
        )

        # Pass through final output linear
        return self.out_linear(output)

    def _shared_step(self, batch: tuple) -> T.Tensor:
        misc, met, lep, jet, nu = batch
        gen_nu = self.forward(misc, met, lep, jet)
        nu_normed = self.nu_normaliser(nu).view(len(nu), -1)
        loss = F.mse_loss(gen_nu, nu_normed)
        return loss

    def training_step(self, batch: tuple, _batch_idx: int) -> T.Tensor:
        loss = self._shared_step(batch)
        self.log("train/total_loss", loss)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> T.Tensor:
        loss = self._shared_step(batch)
        self.log("valid/total_loss", loss)
        return loss

    def predict_step(self, batch: tuple, _batch_idx: int) -> None:
        """Single prediction step which returns the neutrinos, unnormalised."""
        misc, met, lep, jet, _ = batch
        gen_nu = self.forward(misc, met, lep, jet)
        gen_nu = gen_nu.view(len(gen_nu), 1, self.n_nu, self.nu_features)
        gen_nu = self.nu_normaliser.reverse(gen_nu)

        # We also need the log_probs for saving consistantly with other models
        log_probs = T.zeros((len(gen_nu), 1), device=gen_nu.device)

        return {"gen_nu": gen_nu, "log_prob": log_probs}

    def configure_optimizers(self) -> dict:
        """Configure the optimisers and learning rate sheduler for this
        model."""

        # Finish initialising the partialy created methods
        opt = self.hparams.optimizer(params=self.parameters())

        # Use utils to initialise the scheduler (cyclic-epoch sync)
        sched = get_sched(
            self.hparams.sched_conf.utils,
            opt,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            max_epochs=self.trainer.max_epochs,
        )

        # Return the dict for the lightning trainer
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, **self.hparams.sched_conf.lightning},
        }
