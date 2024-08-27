from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch as T
import wandb
from lightning import LightningModule

from mltools.mltools.flows import rqs_flow
from mltools.mltools.lightning_utils import simple_optim_sched
from mltools.mltools.mlp import MLP
from mltools.mltools.modules import IterativeNormLayer
from mltools.mltools.plotting import plot_corr_heatmaps, plot_multi_hists, quantile_bins
from mltools.mltools.torch_utils import to_np
from mltools.mltools.transformers import TransformerVectorEncoder


class NuFlows(LightningModule):
    """Transformer based conditional normalising flow for neutrino unfolding."""

    def __init__(
        self,
        *,
        input_dimensions: dict,
        target_dimensions: dict,
        embed_config: dict,
        transformer_config: dict,
        flow_config: dict,
        scheduler: partial,
        optimizer: partial,
        gen_validation: int = 0,
    ) -> None:
        """Parameters
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
        gen_validation : int
            The number of validation batches to generate samples for.
            By default, no samples are generated.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.input_dimensions = input_dimensions
        self.target_dimensions = target_dimensions
        self.gen_validation = gen_validation
        self.valid_outs = []
        self.valid_targets = []

        # Disable all masking, this is useful when running an ONNX export
        self.disable_masking = False

        # Initialise the transformer vector encoder
        self.transformer = TransformerVectorEncoder(**transformer_config)
        dim = self.transformer.inpt_dim

        # Record the input dimensions and initialise an embedding network for each
        for key, inpt_dim in input_dimensions.items():
            setattr(self, key + "_norm", IterativeNormLayer(inpt_dim))
            setattr(self, key + "_mlp", MLP(inpt_dim, outp_dim=dim, **embed_config))

        # Create the normalisation layer for the targets
        target_dim = sum(target_dimensions.values())
        self.target_norm = IterativeNormLayer(target_dim)

        # Initialise the normalising flow
        self.flow = rqs_flow(xz_dim=target_dim, ctxt_dim=self.transformer.outp_dim, **flow_config)

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
        for key, sample in inputs.items():
            # Check if a mask was provided or create one
            inpt, mask = sample if isinstance(sample, tuple | list) else (sample, None)

            # Check that the input has a multiplicity dimension
            if inpt.ndim == 2:
                inpt = inpt.unsqueeze(1)

            # Build the mask
            if self.disable_masking:
                mask = None
            elif mask is None:
                mask = T.ones(inpt.shape[:2], device=self.device, dtype=bool)
            elif mask.sum() == 0:  # Dont process empty tensors
                continue

            # Pass through the normalisation and embedding layers
            normed = getattr(self, key + "_norm")(inpt, mask)
            embedded = getattr(self, key + "_mlp")(normed)

            # Stack the embeddings and masks together
            embeddings.append(embedded)
            all_mask.append(mask)

        # Concatenate all the embeddings together
        embeddings = T.hstack(embeddings)
        all_mask = None if self.disable_masking else T.hstack(all_mask)

        # Pass the combined tensor through the transformer and return
        return self.transformer(embeddings, mask=all_mask)

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

    @T.cuda.amp.custom_fwd(cast_inputs=T.float32)
    def _get_log_probs(
        self,
        targets: T.Tensor,
        context: T.Tensor,
        weights: T.Tensor | None = None,
    ) -> T.Tensor:
        """Seperate function to allow torch autocasing."""
        log_prob = self.flow.log_prob(targets, context)
        if weights is not None:
            log_prob *= weights
        # TODO: Why are NaNs appearing?
        log_prob = T.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)
        return log_prob.mean()

    def _shared_step(self, sample: tuple, flag: str) -> T.Tensor:
        """Shared step for training and validation."""
        # Unpack the sample
        inputs, targets, _weights = sample

        # Get the context and the flattened targets
        ctxt = self.get_context(inputs)
        targ = self.get_targets(targets)

        # Pass through the flow and get the log likelihood loss
        loss = -self._get_log_probs(targ, ctxt)  # TODO: Use weights

        # Log the loss and return
        self.log(f"{flag}/total_loss", loss)
        return loss

    @T.no_grad()
    def sample(self, inputs: dict) -> dict:
        """Generate many points per sample."""
        # Get the context from the event feature extractor
        ctxt = self.get_context(inputs)

        # Repeat the context for how many samples per event
        # Commenting out the behaviour for multiple samples per event
        # ctxt = ctxt.repeat_interleave(samples_per_event, dim=0)

        # Sample from the flow, undo normalisation and reshape
        sampled, log_probs = self.flow.sample(ctxt.shape[0], ctxt)
        sampled = self.target_norm.reverse(sampled)
        # sampled = sampled.view(-1, samples_per_event, sampled.shape[-1])
        # log_probs = log_probs.view(-1, samples_per_event)

        # Pack the targets into a dict and return
        out_dict = self.pack_outputs(sampled)
        out_dict["log_probs"] = log_probs

        return out_dict

    def forward(self, *args) -> Any:
        """Alias for self.sample required for ONNX export that assumes order."""
        # Check if the inputs are batched which is required for masking/transformers
        is_batched = args[0].ndim == 3
        if not is_batched:
            args = [a.unsqueeze(0) for a in args]

        input_dict = dict(zip(self.input_dimensions.keys(), args, strict=False))
        sample_dict = self.sample(input_dict)
        return tuple(sample_dict.values())

    def training_step(self, batch: tuple, _batch_idx: int) -> T.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> T.Tensor:
        if batch_idx < self.gen_validation:
            self.valid_outs.append(to_np(self.sample(batch[0])))
            self.valid_targets.append(to_np(batch[1]))
        return self._shared_step(batch, "valid")

    def predict_step(self, batch: tuple, _batch_idx: int) -> dict:
        """Single prediction step which add generates samples."""
        return self.sample(batch[0])

    def configure_optimizers(self) -> dict:
        return simple_optim_sched(self)

    def on_validation_epoch_end(self) -> None:
        keys = self.target_dimensions.keys()
        val_outs = {k: np.vstack([v[k] for v in self.valid_outs]) for k in keys}
        val_targets = {k: np.vstack([v[k] for v in self.valid_targets]) for k in keys}
        self.valid_outs.clear()
        self.valid_targets.clear()

        # Log the validation histograms
        for k in keys:
            Path("plots").mkdir(exist_ok=True)
            img = plot_multi_hists(
                data_list=[val_targets[k], val_outs[k]],
                data_labels=["Target", "Output"],
                col_labels=[f"{k}-{i}" for i in range(val_outs[k].shape[-1])],
                bins=30,
                return_img=True,
                path=f"plots/hist_{k}.png",
            )
            if wandb.run is not None:
                wandb.log({f"valid/{k}": [wandb.Image(img)]})

            bins = quantile_bins(val_targets[k][:, 0], bins=100)
            img = plot_corr_heatmaps(
                val_outs[k][:, 0],
                val_targets[k][:, 0],
                bins=[bins, bins],
                xlabel=f"Output: {k}0",
                ylabel=f"Target: {k}0",
                path=f"plots/corr_{k}.png",
                return_img=True,
            )
            if wandb.run is not None:
                wandb.log({f"valid/corr_{k}": [wandb.Image(img)]})
