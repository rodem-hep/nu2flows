from functools import partial
from typing import Mapping

import pytorch_lightning as pl
import torch as T
import wandb
from nflows import distributions, flows

from utils.bayesian import contains_bayesian_layers, prior_loss
from utils.flows.transforms import stacked_norm_flow
from utils.modules import DeepSet, DenseNetwork, IterativeNormLayer
from utils.torch_utils import get_sched
from utils.transformers import (
    TransformerEncoder,
    TransformerVectorEncoder,
)


class NuFlowBase(pl.LightningModule):
    """Base class for the neutrino flows."""

    def __init__(self, data_dims: Mapping) -> None:
        """
        Args:
            dim: List of the dimensions of the data used in this model
                [misc, met, leptons, jets, nu]
            embed_conf: Context embedding network configuraion.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # if the model has bayesian layers
        self.do_bayesian = False

        # Split the input dimension into its component parts
        self.misc_dim, self.met_dim, self.lep_dim, self.jet_dim, self.nu_dim = data_dims

        # The misc and met dimensions need to be converted to intergers
        self.misc_dim = self.misc_dim[0]
        self.met_dim = self.met_dim[0]

        # The number of leptons, jets and neutrinos used in this model
        self.n_nu = self.nu_dim[0]
        self.n_lep = self.lep_dim[0]
        self.n_jets = self.jet_dim[0]
        self.nu_features = self.nu_dim[-1]
        self.lep_features = self.lep_dim[-1]
        self.jet_features = self.jet_dim[-1]

        # Initialise the individual normalisation layers
        self.misc_normaliser = IterativeNormLayer(self.misc_dim)
        self.met_normaliser = IterativeNormLayer(self.met_dim)
        self.lep_normaliser = IterativeNormLayer(self.lep_dim, extra_dims=0)
        self.jet_normaliser = IterativeNormLayer(self.jet_dim[-1])  # Will be masked
        self.nu_normaliser = IterativeNormLayer(self.nu_dim, extra_dims=0)

    def on_fit_start(self, *_args) -> None:
        """Function to run at the start of training."""

        # Define the metrics for wandb (otherwise the min wont be stored!)
        if wandb.run is not None:
            wandb.define_metric("train/log_lik", summary="max")
            wandb.define_metric("train/prior", summary="min")
            wandb.define_metric("train/total_loss", summary="min")
            wandb.define_metric("valid/log_lik", summary="max")
            wandb.define_metric("valid/prior", summary="min")
            wandb.define_metric("valid/total_loss", summary="min")

    def get_flow_ctxt(
        self, misc: T.Tensor, met: T.Tensor, lep: T.Tensor, jet: T.Tensor
    ) -> tuple:
        raise NotImplementedError

    def _shared_step(self, sample: tuple) -> T.Tensor:
        """Combination of the get_flow_tensors and log_prob methods to allow
        the forward hooks used by other packages to be activated during
        training."""
        misc, met, lep, jet, nu = sample
        ctxt = self.get_flow_ctxt(misc, met, lep, jet)
        nu = self.nu_normaliser(nu).view(len(nu), -1)
        log_lik = self.flow.log_prob(nu, context=ctxt).mean()
        prior = prior_loss(self) if self.do_bayesian else T.zeros_like(log_lik)
        total_loss = -log_lik + self.hparams.prior_weight * prior
        return log_lik, prior, total_loss

    def training_step(self, batch: tuple, _batch_idx: int) -> T.Tensor:
        log_lik, prior, total_loss = self._shared_step(batch)
        self.log("train/log_lik", log_lik)
        self.log("train/prior", prior)
        self.log("train/total_loss", total_loss)
        return total_loss

    def validation_step(self, batch: tuple, batch_idx: int) -> T.Tensor:
        log_lik, prior, total_loss = self._shared_step(batch)
        self.log("valid/log_lik", log_lik)
        self.log("valid/prior", prior)
        self.log("valid/total_loss", total_loss)
        return total_loss

    def predict_step(self, batch: tuple, _batch_idx: int) -> None:
        """Single prediction step which add generates samples to the buffer."""
        misc, met, lep, jet, _ = batch
        gen_nu, log_prob = self.sample_and_log_prob((misc, met, lep, jet))
        return {"gen_nu": gen_nu, "log_prob": log_prob}

    def generate(self, inputs: tuple, n_points: int = 256) -> T.Tensor:
        """Generate points in the X space by sampling from the latent."""
        ctxt = self.get_flow_ctxt(*inputs)
        gen_nu = self.flow.sample(n_points, ctxt)
        gen_nu = gen_nu.view(len(gen_nu), n_points, self.n_nu, self.nu_features)
        gen_nu = self.nu_normaliser.reverse(gen_nu)
        return gen_nu

    def sample_and_log_prob(self, inputs: tuple, n_points: int = 256) -> tuple:
        """Generate many points per sample and return all with their log
        likelihoods."""
        ctxt = self.get_flow_ctxt(*inputs)
        gen_nu, log_probs = self.flow.sample_and_log_prob(n_points, ctxt)
        gen_nu = gen_nu.view(len(gen_nu), n_points, self.n_nu, self.nu_features)
        gen_nu = self.nu_normaliser.reverse(gen_nu)
        return gen_nu, log_probs

    def forward(self, *args) -> tuple:
        return self.get_flow_ctxt(*args)
        # return self.sample_and_log_prob(inputs, n_points=1)

    def get_mode(self, inputs: tuple, n_points: int = 256) -> T.Tensor:
        """Generate points, then select the one with the most likely value in
        the reconstruction space."""
        gen_nu, log_probs = self.sample_and_log_prob(inputs, n_points=n_points)
        return gen_nu[T.arange(len(gen_nu)), log_probs.argmax(dim=-1)]

    def get_latents(self, batch: tuple) -> T.Tensor:
        """Get the latent space embeddings given neutrino and context
        information."""
        misc, met, lep, jet, nu = batch
        ctxt = self.get_flow_ctxt(misc, met, lep, jet)
        nu = self.nu_normaliser(nu).view(len(nu), -1)
        return self.flow.transform_to_noise(nu, context=ctxt)

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


class NeutrinoFlow(NuFlowBase):
    """Standard flow which uses a deepset for the jet embedding."""

    def __init__(
        self,
        *,
        data_dims: list,
        embed_conf: Mapping,
        deepset_conf: Mapping,
        flow_conf: Mapping,
        sched_conf: Mapping,
        optimizer: partial,
        prior_weight: float = 1e-5
    ) -> None:
        """
        Args:
            dim: List of the dimensions of the data used in this model
                [misc, met, leptons, jets, nu]
            embed_conf: Context embedding network configuraion.
            deepset_conf: DeepSet configuraion.
            flow_conf: Invertible neural network configuration.
            prior_weight: Weight for the bayesian loss term
            sched_conf: Config for the scheduler
            optimizer: Partially initialised optimizer
            prior_weight: Weight for the bayesian prior, only if bayesian
        """
        super().__init__(data_dims)

        # Initialise the deep set
        self.jet_deepset = DeepSet(
            inpt_dim=self.jet_dim[-1],
            ctxt_dim=self.misc_dim + self.met_dim + self.n_lep * self.lep_features,
            **deepset_conf,
        )

        # Initialise the context embedding network
        self.embed_net = DenseNetwork(
            inpt_dim=self.misc_dim
            + self.met_dim
            + self.n_lep * self.lep_features
            + self.jet_deepset.outp_dim,
            **embed_conf,
        )

        # Save the flow: a combination of the inn and a gaussian
        self.flow = flows.Flow(
            stacked_norm_flow(
                xz_dim=(self.n_nu * self.nu_features),
                ctxt_dim=self.embed_net.outp_dim,
                **flow_conf,
            ),
            distributions.StandardNormal([self.n_nu * self.nu_features]),
        )

        # Check to see if the network has any bayesian layers
        self.do_bayesian = contains_bayesian_layers(self)

    def get_flow_ctxt(
        self, misc: T.Tensor, met: T.Tensor, lep: T.Tensor, jet: T.Tensor
    ) -> tuple:
        """Gets the context tensor for the flow using the FF components of the
        net."""

        # Make a mask for the input jets based on zero padding
        jet_mask = T.any(jet != 0, dim=-1)

        # Pass the inputs through the normalisation layers
        misc = self.misc_normaliser(misc)
        met = self.met_normaliser(met)
        lep = self.lep_normaliser(lep)
        jet = self.jet_normaliser(jet, jet_mask)

        # Flatten the lep tensor
        lep = lep.view(len(lep), -1)

        # Pass the jet tensor through the deep set
        jet = self.jet_deepset(
            inpt=jet,
            mask=jet_mask,
            ctxt=T.cat([misc, met, lep], dim=-1),
        )

        # Combine all inputs and pass through the embedding network
        return self.embed_net(T.cat([misc, met, lep, jet], dim=-1))


class TransNeutrinoFlow(NuFlowBase):
    """Standard flow which uses a transformer for all embedding."""

    def __init__(
        self,
        *,
        data_dims: list,
        part_embed_conf: Mapping,
        tve_conf: Mapping,
        embed_conf: Mapping,
        flow_conf: Mapping,
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

        # The final context embedding network for the flow
        self.embed_net = DenseNetwork(
            inpt_dim=self.transformer.model_dim,
            **embed_conf,
        )

        # Save the flow: a combination of the inn and a gaussian
        self.flow = flows.Flow(
            stacked_norm_flow(
                xz_dim=(self.n_nu * self.nu_features),
                ctxt_dim=self.embed_net.outp_dim,
                **flow_conf,
            ),
            distributions.StandardNormal([self.n_nu * self.nu_features]),
        )

        # Check to see if the network has any bayesian layers
        self.do_bayesian = contains_bayesian_layers(self)

    def get_flow_ctxt(
        self, misc: T.Tensor, met: T.Tensor, lep: T.Tensor, jet: T.Tensor
    ) -> tuple:
        """Gets the context tensor for the flow using the FF components of the
        net."""

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

        # Pass through final embedding and return
        return self.embed_net(output)
