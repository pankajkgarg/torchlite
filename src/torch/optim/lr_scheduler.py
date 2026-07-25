"""Learning-rate schedules used by the browser exercise notebooks."""

from __future__ import annotations

import math


class CosineAnnealingLR:
    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        if T_max <= 0:
            raise ValueError("T_max must be positive")
        self.optimizer = optimizer
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)
        self.last_epoch = int(last_epoch)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self):
        self.last_epoch += 1
        position = min(self.last_epoch, self.T_max)
        factor = (1.0 + math.cos(math.pi * position / self.T_max)) / 2.0
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self.eta_min + (base_lr - self.eta_min) * factor
