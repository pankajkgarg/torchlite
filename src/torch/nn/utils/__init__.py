"""Gradient and sequence utilities used by educational models."""

from __future__ import annotations

import math

from ... import Tensor


def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0):
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    if norm_type != 2.0:
        raise NotImplementedError("TorchLite currently supports only the L2 gradient norm")
    total = math.sqrt(sum(float((parameter.grad._array ** 2).sum()) for parameter in parameters))
    coefficient = min(1.0, max_norm / (total + 1e-6))
    for parameter in parameters:
        parameter.grad._array *= coefficient
    return Tensor(total)


from .rnn import pad_sequence

__all__ = ["clip_grad_norm_", "pad_sequence"]
