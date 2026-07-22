"""Optimizers supported by TorchLite's ``llm-core-v1`` profile."""

from __future__ import annotations

import numpy as np


class Optimizer:
    def __init__(self, params, defaults):
        self.defaults = dict(defaults)
        parameters = list(params)
        if parameters and isinstance(parameters[0], dict):
            self.param_groups = []
            for specification in parameters:
                group = {**self.defaults, **specification}
                group["params"] = list(specification["params"])
                self.param_groups.append(group)
        else:
            self.param_groups = [{"params": parameters, **self.defaults}]
        self.state = {}

    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for parameter in group["params"]:
                if set_to_none:
                    parameter.grad = None
                elif parameter.grad is not None:
                    parameter.grad._array.fill(0)


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        **kwargs,
    ):
        unsupported = {name: value for name, value in kwargs.items() if value not in (None, False)}
        if unsupported:
            raise NotImplementedError(f"AdamW options not in llm-core-v1: {tuple(unsupported)}")
        if lr < 0 or eps < 0 or not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError("invalid AdamW hyperparameters")
        super().__init__(params, {
            "lr": lr, "betas": tuple(betas), "eps": eps, "weight_decay": weight_decay,
        })

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state.setdefault(id(parameter), {
                    "step": 0,
                    "exp_avg": np.zeros_like(parameter._array),
                    "exp_avg_sq": np.zeros_like(parameter._array),
                })
                state["step"] += 1
                gradient = parameter.grad._array
                state["exp_avg"] *= beta1
                state["exp_avg"] += (1 - beta1) * gradient
                state["exp_avg_sq"] *= beta2
                state["exp_avg_sq"] += (1 - beta2) * gradient * gradient
                correction1 = 1 - beta1 ** state["step"]
                correction2 = 1 - beta2 ** state["step"]
                denominator = np.sqrt(state["exp_avg_sq"] / correction2) + group["eps"]
                if group["weight_decay"]:
                    parameter._array *= 1 - group["lr"] * group["weight_decay"]
                parameter._array -= group["lr"] * (state["exp_avg"] / correction1) / denominator
        return loss
