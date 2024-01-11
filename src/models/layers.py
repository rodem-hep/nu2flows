from copy import deepcopy
from typing import Callable, Mapping

import torch as T
import torch.nn as nn

from mltools.mltools.diffusion import c_values
from mltools.mltools.modules import CosineEncodingLayer
from mltools.mltools.torch_utils import append_dims, ema_param_sync
from mltools.mltools.transformers import Transformer


class PointCloudDiffuser(nn.Module):
    """Karras diffusion model for generating point clouds.

    NO NORMALISATION! THIS IS A LAYER!
    """

    def __init__(
        self,
        *,
        inpt_dim: int,
        cosine_config: Mapping,
        transformer_config: Mapping,
        ctxt_dim: int = 0,
        ctxt_pc_dim: int = 0,
        min_sigma: float = 0,
        max_sigma: float = 80.0,
        ema_sync: float = 0.999,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sampler_function: Callable | None = None,
        sigma_function: Callable | None = None,
    ) -> None:
        super().__init__()

        # Class attributes
        self.inpt_dim = inpt_dim
        self.ctxt_dim = ctxt_dim
        self.ctxt_pc_dim = ctxt_pc_dim
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.ema_sync = ema_sync
        self.p_mean = p_mean
        self.p_std = p_std

        # If there is a context point cloud -> use the transformer as a decoder
        self.use_decoder = ctxt_pc_dim > 0
        if self.use_decoder:
            transformer_config["dim"] = ctxt_pc_dim

        # The cosine encoder for the sigma values
        self.sigma_encoder = CosineEncodingLayer(
            inpt_dim=1, min_value=min_sigma, max_value=max_sigma, **cosine_config
        )

        # The transformer itself
        # Is automatically a decoder if there is a context point cloud
        self.net = Transformer(
            inpt_dim=inpt_dim,
            outp_dim=inpt_dim,
            ctxt_dim=ctxt_dim + self.sigma_encoder.outp_dim,
            do_input_linear=True,
            do_output_linear=True,
            use_decoder=self.use_decoder,
            **transformer_config,
        )

        # A copy of the network to run during validation
        self.ema_net = deepcopy(self.net)
        self.ema_net.requires_grad_(False)

        # Sampler to run in the validation/testing loop
        self.sampler_function = sampler_function
        self.sigma_function = sigma_function

    def get_outputs(
        self,
        nodes: T.Tensor,
        sigmas: T.Tensor,
        mask: T.BoolTensor = None,
        ctxt: T.Tensor | None = None,
        ctxt_pc: T.Tensor | None = None,
        ctxt_pc_mask: T.BoolTensor | None = None,
    ) -> T.Tensor:
        """Pass through the model Corresponds to F_theta in the Karras paper."""

        # Use the appropriate network for training or validation
        if self.training:
            network = self.net
        else:
            network = self.ema_net

        # Encode the sigmas and combine with existing context info
        context = self.sigma_encoder(sigmas)
        if self.ctxt_dim:
            context = T.cat([context, ctxt], dim=-1)

        # Pass through the network, arguments change if there is a context point cloud
        kwargs = {"ctxt": context, "kv_mask": mask}
        if self.use_decoder:
            kwargs.update({"kv": ctxt_pc, "kv_mask": ctxt_pc_mask, "x_mask": mask})
        return network(nodes, **kwargs)

    def forward(
        self,
        noisy_nodes: T.Tensor,
        sigmas: T.Tensor,
        **kwargs,
    ) -> T.Tensor:
        """Get the denoised estimates by applying the prescaling."""

        # Get the c values for the data scaling
        c_in, c_out, c_skip = c_values(append_dims(sigmas, noisy_nodes.dim()))

        # Scale the inputs and pass through the network
        outputs = self.get_outputs(c_in * noisy_nodes, sigmas, **kwargs)

        # Scale the outputs and add to the noisy data via skip scaling
        return c_skip * noisy_nodes + c_out * outputs

    def get_loss(
        self,
        nodes: T.Tensor,
        mask: T.BoolTensor = None,
        ctxt: T.Tensor | None = None,
        ctxt_pc: T.Tensor | None = None,
        ctxt_pc_mask: T.BoolTensor | None = None,
        target_mask: T.BoolTensor | None = None,
    ) -> T.Tensor:
        """Calculate the denoising objective loss given a sample."""

        # Sync the ema network if in training mode
        if self.training:
            ema_param_sync(self.net, self.ema_net, self.ema_sync)

        # Sample sigmas using the Karras method of a log normal distribution
        sigmas = T.randn(size=(nodes.shape[0], 1), device=nodes.device)
        sigmas.mul_(self.p_std).add_(self.p_mean).exp_()
        sigmas.clamp_(self.min_sigma, self.max_sigma)

        # Get the c values for the data scaling
        c_in, c_out, c_skip = c_values(append_dims(sigmas, nodes.dim()))

        # Sample from N(0, sigma**2)
        noises = T.randn_like(nodes) * append_dims(sigmas, nodes.dim())

        # Make the noisy samples by mixing with the real data
        noisy_nodes = nodes + noises

        # Pass through the just the base network (manually scale with c values)
        output = self.get_outputs(
            c_in * noisy_nodes,
            sigmas,
            mask,
            ctxt,
            ctxt_pc,
            ctxt_pc_mask,
        )

        # Calculate the effective training target
        target = (nodes - c_skip * noisy_nodes) / c_out

        # Zero out the elements that we do not care about in the loss
        if target_mask is not None:
            output[~target_mask] = 0
            target[~target_mask] = 0

        # Return the denoising loss
        return (output[mask] - target[mask]).square().mean()

    @T.no_grad()
    def full_generation(
        self,
        noise: T.Tensor | None = None,
        mask: T.BoolTensor = None,
        ctxt: T.Tensor | None = None,
        ctxt_pc: T.Tensor | None = None,
        ctxt_pc_mask: T.BoolTensor | None = None,
    ) -> T.Tensor:
        """Fully generate a batch of data from noise."""

        # Work out how many samples with how many points to generate
        if mask is None and noise is None:
            raise ValueError("Please provide either a mask or noise to generate from")
        if mask is None:
            mask = T.full(noise.shape[:-1], True, device=noise.device)
        if noise is None:
            noise = T.randn((*mask.shape, self.inpt_dim), device=mask.device)

        # Generate the descending sigma values
        sigmas = self.sigma_function(self.min_sigma, self.max_sigma)

        # Run the sampler
        outputs = self.sampler_function(
            model=self,
            x=noise * self.max_sigma,
            sigmas=sigmas,
            extra_args={
                "mask": mask,
                "ctxt": ctxt,
                "ctxt_pc": ctxt_pc,
                "ctxt_pc_mask": ctxt_pc_mask,
            },
        )

        return outputs
