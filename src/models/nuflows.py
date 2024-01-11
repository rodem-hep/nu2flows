from functools import partial
from typing import Any

import pytorch_lightning as pl
import torch as T
import wandb

from mltools.mltools.flows import rqs_flow
from mltools.mltools.lightning_utils import standard_optim_sched
from mltools.mltools.mlp import MLP
from mltools.mltools.modules import IterativeNormLayer
from mltools.mltools.transformers import TransformerVectorEncoder


class NuFlows(pl.LightningModule):
    """Transformer based conditional normalising flow for neutrino unfolding."""

    def __init__(
        self,
        *,
        input_dimensions: dict,
        target_dimensions: dict,
        embed_config: dict,
        transformer_config: dict,
        flow_config: dict,
        sched_config: dict,
        optimizer: partial,
    ) -> None:
        """
        Parameters
        ----------
        input_dimensions : dict
            Dictionary containing the names and dimensions of each of the inputs
        target_dimensions : dict
            Dictionary containing the names and dimensions of each of the targets
        embed_config : dict
            Configuration dictionary for the embedding networks
        transformer_config : dict
            Configuration dictionary for the transformer
        flow_config : dict
            Configuration dictionary for the conditional normalising flow
        sched_config : dict
            Configuration dictionary for the learning rate scheduler
        optimizer : partial
            Partially initialised optimizer for the model
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.input_dimensions = input_dimensions
        self.target_dimensions = target_dimensions

        # Initialise the transformer vector encoder
        self.transformer = TransformerVectorEncoder(**transformer_config)
        dim = self.transformer.dim

        # Record the input dimensions and initialise an embedding network for each
        for key, inpt_dim in input_dimensions.items():
            setattr(self, key + "_norm", IterativeNormLayer(inpt_dim))
            setattr(self, key + "_mlp", MLP(inpt_dim, outp_dim=dim, **embed_config))

        # Create the normalisation layer for the targets
        target_dim = sum(target_dimensions.values())
        self.target_norm = IterativeNormLayer(target_dim)

        # Initialise the normalising flow
        self.flow = rqs_flow(xz_dim=target_dim, ctxt_dim=dim, **flow_config)

    def get_context(self, inputs: dict) -> T.Tensor:
        """Pass the inputs through the transformer context extractor.

        Parameters
        ----------
        inputs : dict
            Dictionary of inputs to the model. Each entry must contain a tuple.
            The first element of the tuple is the tensor of values of shape
            (batch x N x D) where N is the multiplicity of that type.
            The second element is a boolean mask of shape (batch x N) to allow
            for padding.
        """

        # Loop through each of the inputs
        embeddings = []
        all_mask = []
        for key, inpt in inputs.items():
            # Check if a mask was provided or create one
            if isinstance(inpt, tuple | list):
                inpt, mask = inpt
            else:
                mask = None

            # Check that the input has a multiplicity dimension
            if inpt.ndim == 2:
                inpt = inpt.unsqueeze(1)

            if mask is None:
                mask = T.ones(inpt.shape[:2], device=self.device, dtype=bool)

            # Skip if the mask is all zeros
            if mask.sum() == 0:
                continue

            # Pass through the normalisation and embedding layers
            normed = getattr(self, key + "_norm")(inpt, mask)
            embedded = getattr(self, key + "_mlp")(normed)

            # Stack the embeddings and masks together
            embeddings.append(embedded)
            all_mask.append(mask)

        # Concatenate all the embeddings together
        embeddings = T.cat(embeddings, dim=1)
        all_mask = T.cat(all_mask, dim=1)

        # Pass the combined tensor through the transformer and return
        return self.transformer(embeddings, kv_mask=all_mask)

    def get_targets(self, targets: dict) -> T.Tensor:
        """Unpack the target dictionary as a single tensor."""
        return self.target_norm(T.cat(tuple(targets.values()), dim=-1))

    def pack_outputs(self, outputs: T.Tensor) -> dict:
        """Pack the targets of the flow into a dictionary."""
        output_dict = {}
        for key, dim in self.target_dimensions.items():
            output_dict[key] = outputs[..., :dim]
            outputs = outputs[..., dim:]
        return output_dict

    def on_fit_start(self, *_args) -> None:
        """Function to run at the start of training."""

        # Define the metrics for wandb (otherwise the min wont be stored!)
        if wandb.run is not None:
            wandb.define_metric("train/total_loss", summary="min")
            wandb.define_metric("valid/total_loss", summary="min")

    def _shared_step(self, sample: tuple) -> T.Tensor:
        """Shared step for training and validation."""

        # Unpack the sample
        inputs, targets = sample

        # Get the context and the flattened targets
        ctxt = self.get_context(inputs)
        targ = self.get_targets(targets)

        # Pass through the flow and get the log likelihood loss
        flow_loss = self.flow.forward_kld(targ, context=ctxt)

        return flow_loss

    def sample(self, inputs: dict, samples_per_event: int = 256) -> dict:
        """Generate many points per sample."""

        # Get the context from the event feature extractor
        ctxt = self.get_context(inputs)

        # Repeat the context for how many samples per event
        ctxt = ctxt.repeat_interleave(samples_per_event, dim=0)

        # Sample from the flow, undo normalisation and reshape
        sampled, log_probs = self.flow.sample(ctxt.shape[0], ctxt)
        sampled = self.target_norm.reverse(sampled)
        sampled = sampled.view(-1, samples_per_event, sampled.shape[-1])
        log_probs = log_probs.view(-1, samples_per_event)

        # Pack the targets into a dict and return
        out_dict = self.pack_outputs(sampled)
        out_dict["log_probs"] = log_probs
        return out_dict

    def forward(self, *args) -> Any:
        """Alias for sample required for onnx export that assumes order."""
        input_dict = {k: v for k, v in zip(self.input_dimensions.keys(), args)}
        sample_dict = self.sample(input_dict)
        return tuple(sample_dict.values())

    def training_step(self, batch: tuple, _batch_idx: int) -> T.Tensor:
        flow_loss = self._shared_step(batch)
        self.log("train/total_loss", flow_loss)
        return flow_loss

    def validation_step(self, batch: tuple, batch_idx: int) -> T.Tensor:
        flow_loss = self._shared_step(batch)
        self.log("valid/total_loss", flow_loss)
        return flow_loss

    def predict_step(self, batch: tuple, _batch_idx: int) -> dict:
        """Single prediction step which add generates samples."""
        inputs, targets = batch
        return self.sample(inputs)

    def configure_optimizers(self) -> dict:
        return standard_optim_sched(self)
