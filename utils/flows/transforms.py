"""Functions and classes used to define the learnable and invertible
transformations used."""

from copy import deepcopy
from typing import Literal

import torch as T
from nflows.transforms import (
    ActNorm,
    AffineCouplingTransform,
    BatchNorm,
    CompositeTransform,
    LULinear,
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    PiecewiseRationalQuadraticCouplingTransform,
)

from ..modules import DenseNetwork
from ..torch_utils import get_act
from ..utils import key_change


def change_kwargs_for_made(old_kwargs):
    """Converts a dictionary of keyword arguments for configuring a custom
    DenseNetwork to one that can initialise a MADE network for the nflows
    package with similar (not exactly the same) hyperparameters."""
    new_kwargs = deepcopy(old_kwargs)

    # Certain keys must be changed
    key_change(new_kwargs, "ctxt_dim", "context_features")
    key_change(new_kwargs, "drp", "dropout_probability")
    key_change(new_kwargs, "do_res", "use_residual_blocks")

    # Certain keys are changed and their values modified
    if "act_h" in new_kwargs:
        new_kwargs["activation"] = get_act(new_kwargs.pop("act_h"))
    if "nrm" in new_kwargs:  # MADE only supports batch norm!
        new_kwargs["use_batch_norm"] = new_kwargs.pop("nrm") is not None

    # Some options are missing
    missing = ["ctxt_in_inpt", "ctxt_in_hddn", "n_lyr_pbk", "act_o", "do_out"]
    for miss in missing:
        if miss in new_kwargs:
            del new_kwargs[miss]

    # The hidden dimension passed to MADE as an arg, not a kwarg
    if "hddn_dim" in new_kwargs:
        hddn_dim = new_kwargs.pop("hddn_dim")
    # Otherwise use the same default value for utils.modules.DenseNet
    else:
        hddn_dim = 32

    return new_kwargs, hddn_dim


def stacked_norm_flow(
    xz_dim: int,
    ctxt_dim: int = 0,
    nstacks: int = 3,
    param_func: Literal["made", "cplng"] = "cplng",
    invrt_func: Literal["rqs", "aff"] = "aff",
    do_lu: bool = True,
    nrm: str = "none",
    net_kwargs: dict = None,
    rqs_kwargs: dict = None,
) -> CompositeTransform:
    """Create a stacked flow using a either autoregressive or coupling layers
    to learn the paramters which are then applied to elementwise invertible
    transforms, which can either be a rational quadratic spline or an affine
    layer.

    After each of these transforms, there can be an extra invertible
    linear layer, followed by some normalisation.

    args:
        xz_dim: The number of input X (and output Z) features
    kwargs:
        ctxt_dim: The dimension of the context feature vector
        nstacks: The number of NSF+Perm layers to use in the overall transform
        param_func: To use either autoregressive or coupling layers
        invrt_func: To use either spline or affine transformations
        do_lu: Use an invertible linear layer inbetween splines to encourage mixing
        nrm: Do a scale shift normalisation inbetween splines (batch or act)
        net_kwargs: Kwargs for the network constructor (includes ctxt dim)
        rqs_kwargs: Keyword args for the invertible spline layers
    """

    # Dictionary default arguments (also protecting dict from chaning on save)
    net_kwargs = deepcopy(net_kwargs) or {}
    rqs_kwargs = deepcopy(rqs_kwargs) or {}

    # We add the context dimension to the list of network keyword arguments
    net_kwargs["ctxt_dim"] = ctxt_dim

    # For MADE netwoks change kwargs from my to nflows format
    if param_func == "made":
        made_kwargs, hddn_dim = change_kwargs_for_made(net_kwargs)

    # For coupling layers we need to define a custom network maker function
    elif param_func == "cplng":

        def net_mkr(inpt, outp):
            return DenseNetwork(inpt, outp, **net_kwargs)

    # Start the list of transforms out as an empty list
    trans_list = []

    # Start with a mixing layer
    if do_lu:
        trans_list.append(LULinear(xz_dim))

    # Cycle through each stack
    for i in range(nstacks):
        # For autoregressive funcions
        if param_func == "made":
            if invrt_func == "aff":
                trans_list.append(
                    MaskedAffineAutoregressiveTransform(xz_dim, hddn_dim, **made_kwargs)
                )

            elif invrt_func == "rqs":
                trans_list.append(
                    MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                        xz_dim, hddn_dim, **made_kwargs, **rqs_kwargs
                    )
                )

        # For coupling layers
        elif param_func == "cplng":
            # Alternate between masking first half and second half (rounded up)
            mask = T.abs(T.round(T.arange(xz_dim) / (xz_dim - 1)).int() - i % 2)

            if invrt_func == "aff":
                trans_list.append(AffineCouplingTransform(mask, net_mkr))

            elif param_func == "cplng" and invrt_func == "rqs":
                trans_list.append(
                    PiecewiseRationalQuadraticCouplingTransform(
                        mask, net_mkr, **rqs_kwargs
                    )
                )

        # Add the mixing layers
        if do_lu:
            trans_list.append(LULinear(xz_dim))

        # Normalising layers (never on last layer in stack)
        if i < nstacks - 1:
            if nrm == "batch":
                trans_list.append(BatchNorm(xz_dim))
            elif nrm == "act":
                trans_list.append(ActNorm(xz_dim))

    # Return the list of transforms combined
    return CompositeTransform(trans_list)
