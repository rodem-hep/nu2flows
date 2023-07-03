"""Collection of pytorch modules that make up the common networks used in my
projects."""

import math
from typing import Optional, Union

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from utils.bayesian import BayesianLinear
from utils.torch_utils import get_act, get_nrm, masked_pool, smart_cat


class MLPBlock(nn.Module):
    """A simple MLP block that makes up a dense network.

    Made up of several layers containing:
    - linear map
    - activation function [Optional]
    - layer normalisation [Optional]
    - dropout [Optional]

    Only the input of the block is concatentated with context information.
    For residual blocks, the input is added to the output of the final layer.
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        n_layers: int = 1,
        act: str = "lrlu",
        nrm: str = "none",
        drp: float = 0,
        do_res: bool = False,
        do_bayesian: bool = False,
        init_zeros: bool = False,
    ) -> None:
        """Init method for MLPBlock.

        Parameters
        ----------
        inpt_dim : int
            The number of features for the input layer
        outp_dim : int
            The number of output features
        ctxt_dim : int, optional
            The number of contextual features to concat to the inputs, by default 0
        n_layers : int, optional
            The number of transform layers in this block, by default 1
        act : str, optional
            A string indicating the name of the activation function, by default "lrlu"
        nrm : str, optional
            A string indicating the name of the normalisation, by default "none"
        drp : float, optional
            The dropout probability, 0 implies no dropout, by default 0
        do_res : bool, optional
            Add to previous output, only if dim does not change, by default 0
        do_bayesian : bool, optional
            If to fill the block with bayesian linear layers, by default False
        init_zeros : bool, optional,
            If the final layer weights and bias values are set to zero
            Does not apply to bayesian layers
        """
        super().__init__()

        # Save the input and output dimensions of the module
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim

        # If this layer includes an additive residual connection
        self.do_res = do_res and (inpt_dim == outp_dim)

        # Initialise the block layers as a module list
        self.block = nn.ModuleList()
        for n in range(n_layers):
            # Increase the input dimension of the first layer to include context
            lyr_in = inpt_dim + ctxt_dim if n == 0 else outp_dim

            # Linear transform, activation, normalisation, dropout
            self.block.append(
                BayesianLinear(lyr_in, outp_dim)
                if do_bayesian
                else nn.Linear(lyr_in, outp_dim)
            )

            # Initialise the final layer with zeros
            if init_zeros and n == n_layers - 1 and not do_bayesian:
                self.block[-1].weight.data.fill_(0)
                self.block[-1].bias.data.fill_(0)

            if act != "none":
                self.block.append(get_act(act))
            if nrm != "none":
                self.block.append(get_nrm(nrm, outp_dim))
            if drp > 0:
                self.block.append(nn.Dropout(drp))

    def forward(self, inpt: T.Tensor, ctxt: Optional[T.Tensor] = None) -> T.Tensor:
        """
        args:
            tensor: Pytorch tensor to pass through the network
            ctxt: The conditioning tensor, can be ignored
        """

        # Concatenate the context information to the input of the block
        if self.ctxt_dim and ctxt is None:
            raise ValueError(
                "Was expecting contextual information but none has been provided!"
            )
        temp = T.cat([inpt, ctxt], dim=-1) if self.ctxt_dim else inpt

        # Pass through each transform in the block
        for layer in self.block:
            temp = layer(temp)

        # Add the original inputs again for the residual connection
        if self.do_res:
            temp = temp + inpt

        return temp

    def __repr__(self) -> str:
        """Generate a one line string summing up the components of the
        block."""
        string = str(self.inpt_dim)
        if self.ctxt_dim:
            string += f"({self.ctxt_dim})"
        string += "->"
        string += "->".join([str(b).split("(", 1)[0] for b in self.block])
        string += "->" + str(self.outp_dim)
        if self.do_res:
            string += "(add)"
        return string


class DenseNetwork(nn.Module):
    """A dense neural network made from a series of consecutive MLP blocks and
    context injection layers."""

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int = 0,
        ctxt_dim: int = 0,
        hddn_dim: Union[int, list] = 32,
        num_blocks: int = 1,
        n_lyr_pbk: int = 1,
        act_h: str = "lrlu",
        act_o: str = "none",
        do_out: bool = True,
        nrm: str = "none",
        drp: float = 0,
        do_res: bool = False,
        ctxt_in_inpt: bool = True,
        ctxt_in_hddn: bool = False,
        do_bayesian: bool = False,
        output_init_zeros: bool = False,
    ) -> None:
        """Initialise the DenseNetwork.

        Parameters
        ----------
        inpt_dim : int
            The number of input neurons
        outp_dim : int, optional
            The number of output neurons. If none it will take from inpt or hddn,
            by default 0
        ctxt_dim : int, optional
            The number of context features. The context feature use is determined by
            ctxt_type, by default 0
        hddn_dim : Union[int, list], optional
            The width of each hidden block. If a list it overides depth, by default 32
        num_blocks : int, optional
            The number of hidden blocks, can be overwritten by hddn_dim, by default 1
        n_lyr_pbk : int, optional
            The number of transform layers per hidden block, by default 1
        act_h : str, optional
            The name of the activation function to apply in the hidden blocks,
            by default "lrlu"
        act_o : str, optional
            The name of the activation function to apply to the outputs,
            by default "none"
        do_out : bool, optional
            If the network has a dedicated output block, by default True
        nrm : str, optional
            Type of normalisation (layer or batch) in each hidden block, by default "none"
        drp : float, optional
            Dropout probability for hidden layers (0 means no dropout), by default 0
        do_res : bool, optional
            Use resisdual-connections between hidden blocks (only if same size),
            by default False
        ctxt_in_inpt : bool, optional
            Include the ctxt tensor in the input block, by default True
        ctxt_in_hddn : bool, optional
            Include the ctxt tensor in the hidden blocks, by default False
        do_bayesian : bool, optional
            Create the network with bayesian linear layers, by default False
        output_init_zeros : bool, optional
            Initialise the output layer weights as zeros

        Raises
        ------
        ValueError
            If the network was given a context input but both ctxt_in_inpt and
            ctxt_in_hddn were False
        """
        super().__init__()

        # Check that the context is used somewhere
        if ctxt_dim:
            if not ctxt_in_hddn and not ctxt_in_inpt:
                raise ValueError("Network has context inputs but nowhere to use them!")

        # We store the input, hddn (list), output, and ctxt dims to query them later
        self.inpt_dim = inpt_dim
        if not isinstance(hddn_dim, int):
            self.hddn_dim = hddn_dim
        else:
            self.hddn_dim = num_blocks * [hddn_dim]
        self.outp_dim = outp_dim or inpt_dim if do_out else self.hddn_dim[-1]
        self.num_blocks = len(self.hddn_dim)
        self.ctxt_dim = ctxt_dim
        self.do_out = do_out

        # Necc for this module to work with the nflows package
        self.hidden_features = self.hddn_dim[-1]

        # Input MLP block
        self.input_block = MLPBlock(
            inpt_dim=self.inpt_dim,
            outp_dim=self.hddn_dim[0],
            ctxt_dim=self.ctxt_dim if ctxt_in_inpt else 0,
            act=act_h,
            nrm=nrm,
            drp=drp,
            do_bayesian=do_bayesian,
        )

        # All hidden blocks as a single module list
        self.hidden_blocks = []
        if self.num_blocks > 1:
            self.hidden_blocks = nn.ModuleList()
            for h_1, h_2 in zip(self.hddn_dim[:-1], self.hddn_dim[1:]):
                self.hidden_blocks.append(
                    MLPBlock(
                        inpt_dim=h_1,
                        outp_dim=h_2,
                        ctxt_dim=self.ctxt_dim if ctxt_in_hddn else 0,
                        n_layers=n_lyr_pbk,
                        act=act_h,
                        nrm=nrm,
                        drp=drp,
                        do_res=do_res,
                        do_bayesian=do_bayesian,
                    )
                )

        # Output block (optional and there is no normalisation, dropout or context)
        if do_out:
            self.output_block = MLPBlock(
                inpt_dim=self.hddn_dim[-1],
                outp_dim=self.outp_dim,
                act=act_o,
                do_bayesian=do_bayesian,
                init_zeros=output_init_zeros,
            )

    def forward(self, inputs: T.Tensor, ctxt: Optional[T.Tensor] = None) -> T.Tensor:
        """Pass through all layers of the dense network."""

        # Reshape the context if it is available
        if ctxt is not None:
            dim_diff = inputs.dim() - ctxt.dim()
            if dim_diff > 0:
                ctxt = ctxt.view(ctxt.shape[0], *dim_diff * (1,), *ctxt.shape[1:])
                ctxt = ctxt.expand(*inputs.shape[:-1], -1)

        # Pass through the input block
        inputs = self.input_block(inputs, ctxt)

        # Pass through each hidden block
        for h_block in self.hidden_blocks:  # Context tensor will only be used if
            inputs = h_block(inputs, ctxt)  # block was initialised with a ctxt dim

        # Pass through the output block
        if self.do_out:
            inputs = self.output_block(inputs)

        return inputs

    def __repr__(self):
        string = ""
        string += "\n  (inp): " + repr(self.input_block) + "\n"
        for i, h_block in enumerate(self.hidden_blocks):
            string += f"  (h-{i+1}): " + repr(h_block) + "\n"
        if self.do_out:
            string += "  (out): " + repr(self.output_block)
        return string

    def one_line_string(self):
        """Return a one line string that sums up the network structure."""
        string = str(self.inpt_dim)
        if self.ctxt_dim:
            string += f"({self.ctxt_dim})"
        string += ">"
        string += str(self.input_block.outp_dim) + ">"
        if self.num_blocks > 1:
            string += ">".join(
                [
                    str(layer.out_features)
                    for hidden in self.hidden_blocks
                    for layer in hidden.block
                    if isinstance(layer, nn.Linear)
                ]
            )
            string += ">"
        if self.do_out:
            string += str(self.outp_dim)
        return string


class DeepSet(nn.Module):
    """A deep set network that can provide attention pooling."""

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        pool_type: str = "mean",
        attn_type: str = "mean",
        feat_net_kwargs=None,
        attn_net_kwargs=None,
        post_net_kwargs=None,
    ) -> None:
        """
        args:
            inpt_dim: The number of input features
            outp_dim: The number of desired output featues
        kwargs:
            ctxt_dim: Dimension of the context information for all networks
            pool_type: The type of set pooling applied; mean, sum, max or attn
            attn_type: The type of attention; mean, sum, raw
            feat_net_kwargs: Keyword arguments for the feature network
            attn_net_kwargs: Keyword arguments for the attention network
            post_net_kwargs: Keyword arguments for the post network
        """
        super().__init__()

        # Dict default arguments
        feat_net_kwargs = feat_net_kwargs or {}
        attn_net_kwargs = attn_net_kwargs or {}
        post_net_kwargs = post_net_kwargs or {}

        # For the attention network the default output must be set to 1
        # The dense network default output is the same as the input
        if "outp_dim" not in attn_net_kwargs:
            attn_net_kwargs["outp_dim"] = 1

        # Save the class attributes
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim
        self.pool_type = pool_type
        self.attn_type = attn_type

        # Create the feature extraction network
        self.feat_net = DenseNetwork(
            self.inpt_dim, ctxt_dim=self.ctxt_dim, **feat_net_kwargs
        )

        # For an attention deepset
        if self.pool_type == "attn":
            # Create the attention network
            self.attn_net = DenseNetwork(
                self.inpt_dim, ctxt_dim=self.ctxt_dim, **attn_net_kwargs
            )

            # Check that the dimension of each head makes internal sense
            self.n_heads = self.attn_net.outp_dim
            assert self.feat_net.outp_dim % self.n_heads == 0
            self.head_dim = self.feat_net.outp_dim // self.n_heads

        # Create the post network to update the pooled features of the set
        self.post_net = DenseNetwork(
            self.feat_net.outp_dim, outp_dim, ctxt_dim=self.ctxt_dim, **post_net_kwargs
        )

    def forward(
        self, inpt: T.tensor, mask: T.BoolTensor, ctxt: Union[T.Tensor, list] = None
    ):
        """The expected shapes of the inputs are.

        - tensor: batch x setsize x features
        - mask: batch x setsize
        - ctxt: batch x features
        """

        # Combine the context information if it is a list
        if isinstance(ctxt, list):
            ctxt = smart_cat(ctxt)

        # Pass the values through the feature network
        feat_outs = self.feat_net(inpt, ctxt)

        # For attention
        if self.pool_type == "attn":
            attn_outs = self.attn_net(inpt, ctxt)

            # Change the attention weights of the padded elements
            attn_outs[~mask] = 0 if self.attn_type == "raw" else -T.inf

            # Apply either a softmax for weighted mean or softplus for weighted sum
            if self.attn_type == "mean":
                attn_outs = F.softmax(attn_outs, dim=-2)
            elif self.attn_type == "sum":
                attn_outs = F.softplus(attn_outs)

            # Kill the nans introduced by the empty sets
            attn_outs = T.nan_to_num(attn_outs, 0)
            # attn_outs[~mask] = 0

            # Broadcast the attention to get the multiple poolings and sum
            attn_outs = (
                attn_outs.unsqueeze(-1).expand(-1, -1, -1, self.head_dim).flatten(2)
            )
            feat_outs = (feat_outs * attn_outs).sum(dim=-2)

        # For the other types of pooling use the masked pool method
        else:
            feat_outs = masked_pool(self.pool_type, feat_outs, mask)

        # Pass the pooled information through post network and return
        return self.post_net(feat_outs, ctxt)


class GRF(Function):
    """A gradient reversal function.

    - The forward pass is the identity function
    - The backward pass multiplies the upstream gradients by -1
    """

    @staticmethod
    def forward(ctx, inpt, alpha):
        """Pass inputs without chaning them."""
        ctx.alpha = alpha
        return inpt.clone()

    @staticmethod
    def backward(ctx, grads):
        """Inverse the gradients."""
        alpha = ctx.alpha
        neg_grads = -alpha * grads
        return neg_grads, None


class GRL(nn.Module):
    """A gradient reversal layer.

    This layer has no parameters, and simply reverses the gradient in
    the backward pass.
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = T.tensor(alpha, requires_grad=False)

    def forward(self, inpt):
        """Pass to the GRF."""
        return GRF.apply(inpt, self.alpha)


class IterativeNormLayer(nn.Module):
    """A basic normalisation layer so it can be part of the model.

    Tracks the runnning mean and variances of an input vector over the batch dimension.
    Must always be passed batched data!
    Any additional dimension to calculate the stats must be provided as extra_dims.

    For example: Providing an image with inpt_dim = channels, width, height
    Will result in an operation with independant stats per pixel, per channel
    If instead you want the mean and shift only for each channel you will have to give
    extra_dims = (1, 2) or (-2, -1)
    This will tell the layer to average out the width and height dimensions

    Note! If a mask is provided in the forward pass, then this must be
    the dimension to apply over the masked inputs! For example: Graph
    nodes are usually batch x n_nodes x features so to average out the n_nodes
    one would typically give extra_dims as (0,). But nodes
    are always passed with the mask which flattens it to batch x features.
    Batch dimension is done automatically, so we dont pass any extra_dims!!!
    """

    def __init__(
        self,
        inpt_dim: Union[T.Tensor, tuple, int],
        means: Optional[T.Tensor] = None,
        vars: Optional[T.Tensor] = None,
        n: int = 0,
        max_n: int = 5_00_000,
        extra_dims: Union[tuple, int] = (),
        track_grad_forward: bool = False,
        track_grad_reverse: bool = False,
    ) -> None:
        """Init method for Normalisatiion module.

        Args:
            inpt_dim: Shape of the input tensor (non batched), required for reloading
            means: Calculated means for the mapping. Defaults to None.
            vars: Calculated variances for the mapping. Defaults to None.
            n: Number of samples used to make the mapping. Defaults to None.
            max_n: Maximum number of iterations before the means and vars are frozen
            extra_dims: The extra dimension(s) over which to calculate the stats
                Will always calculate over the batch dimension
            track_grad_forward: If the gradients should be tracked for this operation
            track_grad_reverse: If the gradients should be tracked for this operation
        """
        super().__init__()

        # Fail if only one of means or vars is provided
        if (means is None) ^ (vars is None):  # XOR
            raise ValueError(
                """Only one of 'means' and 'vars' is defined. Either both or
                neither must be defined"""
            )

        # Allow interger inpt_dim and n arguments
        if isinstance(inpt_dim, int):
            inpt_dim = (inpt_dim,)
        if isinstance(n, int):
            n = T.tensor(n)

        # The dimensions over which to apply the normalisation, make positive!
        if isinstance(extra_dims, int):  # Ensure it is a list
            extra_dims = [extra_dims]
        else:
            extra_dims = list(extra_dims)
        if any([abs(e) > len(inpt_dim) for e in extra_dims]):  # Check size
            raise ValueError("extra_dims argument lists dimensions outside input range")
        for d in range(len(extra_dims)):
            if extra_dims[d] < 0:  # make positive
                extra_dims[d] = len(inpt_dim) + extra_dims[d]
            extra_dims[d] += 1  # Add one because we are inserting a batch dimension
        self.extra_dims = extra_dims

        # Calculate the input and output shapes
        self.max_n = max_n
        self.inpt_dim = list(inpt_dim)
        self.stat_dim = [1] + list(inpt_dim)  # Add batch dimension
        for d in range(len(self.stat_dim)):
            if d in self.extra_dims:
                self.stat_dim[d] = 1

        # Buffers are needed for saving/loading the layer
        self.register_buffer(
            "means",
            T.zeros(self.stat_dim, dtype=T.float32)
            if means is None
            else T.as_tensor(means, dtype=T.float32),
        )
        self.register_buffer(
            "vars",
            T.ones(self.stat_dim, dtype=T.float32)
            if vars is None
            else T.as_tensor(vars, dtype=T.float32),
        )
        self.register_buffer("n", n)

        # For the welford algorithm it is useful to have another variable m2
        self.register_buffer(
            "m2",
            T.ones(self.stat_dim, dtype=T.float32)
            if vars is None
            else T.as_tensor(vars, dtype=T.float32),
        )

        # If the means are set here then the model is "frozen" and never updated
        self.register_buffer(
            "frozen",
            T.as_tensor(
                (means is not None and vars is not None) or self.n > self.max_n,
            ),
        )

        # Gradient tracking options
        self.track_grad_forward = track_grad_forward
        self.track_grad_reverse = track_grad_reverse

    def __repr__(self):
        return f"IterativeNormLayer({list(self.means.shape)})"

    def __str__(self) -> str:
        return f"IterativeNormLayer(m={self.means.squeeze()}, v={self.vars.squeeze()})"

    def _mask(self, inpt: T.Tensor, mask: Optional[T.BoolTensor] = None) -> T.Tensor:
        if mask is None:
            return inpt
        return inpt[mask]

    def _check_attributes(self) -> None:
        if self.means is None or self.vars is None:
            raise ValueError(
                "Stats for have not been initialised or fit() has not been run!"
            )

    @T.no_grad()
    def fit(
        self, inpt: T.Tensor, mask: Optional[T.BoolTensor] = None, freeze: bool = True
    ) -> None:
        """Set the stats given a population of data."""
        inpt = self._mask(inpt, mask)
        self.vars, self.means = T.var_mean(
            inpt,
            dim=(0, *self.extra_dims),
            keepdim=True,
        )
        T.nan_to_num_(self.vars, 1)  # Will be nans if the first batch is size 1
        self.n = T.tensor(len(inpt), device=self.means.device)
        self.m2 = self.vars * self.n
        self.frozen.fill_(True)

    def forward(self, inpt: T.Tensor, mask: Optional[T.BoolTensor] = None) -> T.Tensor:
        """Applies the standardisation to a batch of inputs, also uses the
        inputs to update the running stats if in training mode."""

        # Save and check the gradient tracking options
        grad_setting = T.is_grad_enabled()
        T.set_grad_enabled(self.track_grad_forward)

        # Mask the inputs and update the stats
        sel_inpt = self._mask(inpt, mask)
        if not self.frozen and self.training:
            self.update(sel_inpt)

        # Apply the mapping
        normed_inpt = (sel_inpt - self.means) / (self.vars.sqrt() + 1e-8)

        # Undo the masking
        if mask is not None:
            inpt = inpt.clone()  # prevents inplace operation, bad for autograd
            inpt[mask] = normed_inpt.type(inpt.dtype)
            normed_inpt = inpt

        # Revert the gradient setting
        T.set_grad_enabled(grad_setting)

        return normed_inpt

    def reverse(self, inpt: T.Tensor, mask: Optional[T.BoolTensor] = None) -> T.Tensor:
        """Unnormalises the inputs given the recorded stats."""

        # Save and check the gradient tracking options
        grad_setting = T.is_grad_enabled()
        T.set_grad_enabled(self.track_grad_reverse)

        # Mask and revert the inputs
        sel_inpt = self._mask(inpt, mask)
        unnormed_inpt = sel_inpt * self.vars.sqrt() + self.means

        # Undo the masking
        if mask is not None:
            inpt = inpt.clone()  # prevents inplace operation, bad for autograd
            inpt[mask] = unnormed_inpt.type(inpt.dtype)
            unnormed_inpt = inpt

        # Revert the gradient setting
        T.set_grad_enabled(grad_setting)

        return unnormed_inpt

    def update(self, inpt: T.Tensor) -> None:
        """Update the running stats using a batch of data."""

        # Freeze the model if we already exceed the requested stats
        T.fill_(self.frozen, self.n >= self.max_n)
        if self.frozen:
            return

        # For first iteration, just run the fit on the batch
        if self.n == 0:
            self.fit(inpt, freeze=False)
            return

        # later iterations based on batched welford algorithm
        with T.no_grad():
            self.n += len(inpt)
            delta = inpt - self.means
            self.means += (delta / self.n).mean(
                dim=(0, *self.extra_dims), keepdim=True
            ) * len(inpt)
            delta2 = inpt - self.means
            self.m2 += (delta * delta2).mean(
                dim=(0, *self.extra_dims), keepdim=True
            ) * len(inpt)
            self.vars = self.m2 / self.n


class SineCosineEncoding:
    def __init__(
        self,
        outp_dim: int = 32,
        min_value: float = 0.0,
        max_value: float = 1.0,
        frequency_scaling: str = "exponential",
    ) -> None:
        assert outp_dim % 2 == 0
        self.outp_dim = outp_dim
        self.min_value = min_value
        self.max_value = max_value
        self.frequency_scaling = frequency_scaling

    def __call__(self, inpt: T.Tensor) -> T.Tensor:
        cosine = cosine_encoding(
            inpt,
            self.outp_dim // 2,
            self.min_value,
            self.max_value,
            self.frequency_scaling,
        )
        sine = cosine_encoding(
            inpt,
            self.outp_dim // 2,
            self.min_value,
            self.max_value,
            self.frequency_scaling,
            use_sin=True,
        )
        return T.cat([cosine, sine], dim=-1)


class CosineEncoding:
    def __init__(
        self,
        outp_dim: int = 32,
        min_value: float = 0.0,
        max_value: float = 1.0,
        frequency_scaling: str = "exponential",
    ) -> None:
        self.outp_dim = outp_dim
        self.min_value = min_value
        self.max_value = max_value
        self.frequency_scaling = frequency_scaling

    def __call__(self, inpt: T.Tensor) -> T.Tensor:
        return cosine_encoding(
            inpt, self.outp_dim, self.min_value, self.max_value, self.frequency_scaling
        )


def cosine_encoding(
    x: T.Tensor,
    outp_dim: int = 32,
    min_value: float = 0.0,
    max_value: float = 1.0,
    frequency_scaling: str = "exponential",
) -> T.Tensor:
    """Computes a positional cosine encodings with an increasing series of
    frequencies.

    The frequencies either increase linearly or exponentially (default).
    The latter is good for when max_value is large and extremely high sensitivity to the
    input is required.
    If inputs greater than the max value are provided, the outputs become degenerate.
    If inputs smaller than the min value are provided, the inputs the the cosine will
    be both positive and negative, which may lead degenerate outputs.

    Always make sure that the min and max bounds are not exceeded!

    Args:
        x: The input, the final dimension is encoded. If 1D then it will be unqueezed
        out_dim: The dimension of the output encoding
        min_value: Added to x (and max) as cosine embedding works with positive inputs
        max_value: The maximum expected value, sets the scale of the lowest frequency
        frequency_scaling: Either 'linear' or 'exponential'

    Returns:
        The cosine embeddings of the input using (out_dim) many frequencies
    """

    # Unsqueeze if final dimension is flat
    if x.shape[-1] != 1 or x.dim() == 1:
        x = x.unsqueeze(-1)

    # Check the the bounds are obeyed
    if T.any(x > max_value):
        print("Warning! Passing values to cosine_encoding encoding that exceed max!")
    if T.any(x < min_value):
        print("Warning! Passing values to cosine_encoding encoding below min!")

    # Calculate the various frequencies
    if frequency_scaling == "exponential":
        freqs = T.arange(outp_dim, device=x.device).exp()
    elif frequency_scaling == "linear":
        freqs = T.arange(1, outp_dim + 1, device=x.device)
    else:
        raise RuntimeError(f"Unrecognised frequency scaling: {frequency_scaling}")

    return T.cos((x + min_value) * freqs * math.pi / (max_value + min_value))


def sine_cosine_encoding(
    x: T.Tensor,
    outp_dim: int = 32,
    min_value: float = 0.0,
    max_value: float = 1.0,
    frequency_scaling: str = "exponential",
) -> T.Tensor:
    """Computes a positional sine and cosine encodings with an increasing
    series of frequencies.

    See above for more details

    Returns:
        The cosine embeddings of the input using (out_dim) many frequencies
    """

    # Unsqueeze if final dimension is flat
    if x.shape[-1] != 1 or x.dim() == 1:
        x = x.unsqueeze(-1)

    # Check the the bounds are obeyed
    if T.any(x > max_value):
        print("Warning! Passing values to cosine_encoding encoding that exceed max!")
    if T.any(x < min_value):
        print("Warning! Passing values to cosine_encoding encoding below min!")

    # Calculate the various frequencies
    if frequency_scaling == "exponential":
        freqs = T.arange(outp_dim // 2, device=x.device).exp()
    elif frequency_scaling == "linear":
        freqs = T.arange(1, outp_dim // 2 + 1, device=x.device)
    else:
        raise RuntimeError(f"Unrecognised frequency scaling: {frequency_scaling}")

    # Scale the frequencies to match the cyclic requirements (0 -> pi)
    freqs = (x + min_value) * freqs * math.pi / (max_value + min_value)

    # Place the frequencies into the encodings tensor
    encodings = T.zeros(*x.shape[:-1], outp_dim, device=x.device)
    encodings[..., outp_dim // 2 :] = T.sin(freqs)
    encodings[..., : outp_dim // 2] = T.cos(freqs)

    return encodings
