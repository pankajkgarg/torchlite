"""Neural-network building blocks for TorchLite."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .. import Tensor, randn, zeros


Parameter = Tensor


class Module:
    def __init__(self):
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if isinstance(value, Tensor) and value.requires_grad:
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        for parameter in self._parameters.values():
            yield parameter
        for module in self._modules.values():
            yield from module.parameters()

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = None

    def train(self, mode=True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def register_buffer(self, name, tensor, persistent=True):
        object.__setattr__(self, name, tensor)
        self._buffers[name] = tensor


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        scale = in_features ** -0.5
        self.weight = randn(out_features, in_features, requires_grad=True) * scale
        self.weight.requires_grad = True
        self.bias = zeros(out_features, requires_grad=True) if bias else None

    def forward(self, input):
        output = input @ self.weight.T
        return output if self.bias is None else output + self.bias


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.tanh()


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self._sequence = list(modules)
        for index, module in enumerate(modules):
            setattr(self, str(index), module)

    def forward(self, input):
        for module in self._sequence:
            input = module(input)
        return input


from . import functional  # noqa: E402
