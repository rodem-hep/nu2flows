"""Script to define the lookahead optimiser copied from:

https://github.com/alphadl/lookahead.pytorch
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable, Iterable, Optional

import torch as T
from torch.optim import Optimizer


class Lookahead(Optimizer):
    def __init__(
        self,
        inner_optimizer: partial | Optimizer = None,
        params: Iterable = None,
        k=10,
        alpha=0.5,
        **opt_kwargs,
    ) -> None:
        # If we have a fully initialised optimiser
        if isinstance(inner_optimizer, Optimizer):
            self.optimizer = inner_optimizer
        # Otherwise we initialise using our parameters
        else:
            self.optimizer = inner_optimizer(params, **opt_kwargs)

        # Other class features
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    @property
    def defaults(self):
        return self.optimizer.defaults

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = T.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, T.Tensor) else k): v for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super().load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)


class Lion(Optimizer):
    """The lion algorithm https://arxiv.org/pdf/2302.06675.pdf.

    Implementation derived from:
    https://github.com/google/automl/blob/master/lion/lion_pytorch.py
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """
        Args:
          params: iterable of parameters to optimize or dicts defining groups
          lr: Learning rate, should be ~5 lower than Adam (default: 1e-4)
          betas: coefficients used for computing running averages of gradient and square
          weight_decay: weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @T.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Any:
        """Performs a single optimization step.

        Args:
          closure: A closure that reevaluates the model and returns the loss
        """

        # Define the loss using the closure function
        loss = None
        if closure is not None:
            with T.enable_grad():
                loss = closure()

        # Iterate through the parameter groups
        for group in self.param_groups:
            for p in group["params"]:
                # Skip if the gradients are empty
                if p.grad is None:
                    continue

                # Perform weight decay step initially
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Get the gradients
                grad = p.grad

                # State (default-dict for each group) initialization at start of training
                state = self.state[p]
                if len(state) == 0:
                    state["mt"] = T.zeros_like(p)

                # Load the state's averages and betas
                mt = state["mt"]
                beta1, beta2 = group["betas"]

                # Perform the weight update
                ct = mt * beta1 + grad * (1 - beta1)
                p.add_(T.sign(ct), alpha=-group["lr"])

                # Decay the momentum running average coefficient
                mt.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
