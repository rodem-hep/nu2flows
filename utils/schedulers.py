"""Custom pytorch learning rate schedulers."""

import warnings

from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR, _LRScheduler


class LinearWarmupRootDecay(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        dim_model: int = 256,
        warmup_steps: int = 10000,
        last_epoch: int = -1,
        verbose: bool = False,
        use_max_lr: bool = False,
    ) -> None:
        # For calculating the learning rate profile
        self.dim_model = dim_model
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        # For overwritting the max learning rate (instead of using dim model)
        self.use_max_lr = use_max_lr
        self.max_lr_coef = (self.dim_model * self.warmup_steps) ** (0.5)

        # Super init at end for some reason
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        lr = self.dim_model ** (-0.5) * min(
            self._step_count ** (-0.5), self._step_count * self.warmup_steps ** (-1.5)
        )
        if self.use_max_lr:
            lr *= self.base_lrs[0] * self.max_lr_coef
        return [lr] * self.num_param_groups


class WarmupToConstant(_LRScheduler):
    """Gradually warm-up learning rate in optimizer to a constant value."""

    def __init__(self, optimizer: Optimizer, num_steps: int = 100):
        """
        args:
            optimizer (Optimizer): Wrapped optimizer.
            num_steps: target learning rate is reached at num_steps.
        """
        self.num_steps = num_steps
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.num_steps:
            return [base_lr for base_lr in self.base_lrs]
        return [
            (base_lr / self.num_steps) * self.last_epoch for base_lr in self.base_lrs
        ]


class CyclicWithWarmup(OneCycleLR):
    """A cyclic scheduler with dedicated warmup periods based on the onecycle
    LR.

    The only difference is the get_lr method, which resets the scheduler
    after each cycle instead of throwing out an error
    """

    def get_lr(self):
        """Overloaded method for aquiring new learning rates Only line that is
        changed from the original method is the step number!

        Also removed the warning that step > length
        """
        if not self._get_lr_called_within_step:  # pylint: disable=no-member
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        lrs = []
        step_num = self.last_epoch % self.total_steps  # Only changed line!!!

        for group in self.optimizer.param_groups:
            start_step = 0
            for i, phase in enumerate(self._schedule_phases):
                end_step = phase["end_step"]
                if step_num <= end_step or i == len(self._schedule_phases) - 1:
                    pct = (step_num - start_step) / (end_step - start_step)
                    computed_lr = self.anneal_func(
                        group[phase["start_lr"]], group[phase["end_lr"]], pct
                    )
                    if self.cycle_momentum:
                        computed_momentum = self.anneal_func(
                            group[phase["start_momentum"]],
                            group[phase["end_momentum"]],
                            pct,
                        )
                    break
                start_step = phase["end_step"]

            lrs.append(computed_lr)
            if self.cycle_momentum:
                if self.use_beta1:
                    _, beta2 = group["betas"]
                    group["betas"] = (computed_momentum, beta2)
                else:
                    group["momentum"] = computed_momentum

        return lrs
