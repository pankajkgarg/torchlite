"""Small in-place initialization subset used by educational GPTs."""

from __future__ import annotations

import math
import numpy as np

from .. import Tensor, _rng


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0, generator=None) -> Tensor:
    tensor._array[...] = _rng(generator).normal(mean, std, size=tensor.shape).astype(tensor.dtype)
    return tensor


def zeros_(tensor: Tensor) -> Tensor:
    tensor._array.fill(0)
    return tensor


def ones_(tensor: Tensor) -> Tensor:
    tensor._array.fill(1)
    return tensor


def constant_(tensor: Tensor, value: float) -> Tensor:
    tensor._array.fill(value)
    return tensor


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    if tensor.ndim < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    receptive_field = int(np.prod(tensor.shape[2:])) if tensor.ndim > 2 else 1
    return tensor.shape[1] * receptive_field, tensor.shape[0] * receptive_field


def calculate_gain(nonlinearity: str, param=None) -> float:
    if nonlinearity in {"linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d",
                        "conv_transpose2d", "conv_transpose3d", "sigmoid"}:
        return 1.0
    if nonlinearity == "tanh":
        return 5.0 / 3
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity == "leaky_relu":
        slope = 0.01 if param is None else float(param)
        return math.sqrt(2.0 / (1 + slope**2))
    if nonlinearity == "selu":
        return 3.0 / 4
    raise ValueError(f"Unsupported nonlinearity {nonlinearity!r}")


def xavier_uniform_(
    tensor: Tensor,
    gain: float = 1.0,
    generator=None,
) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return tensor.uniform_(-bound, bound, generator=generator)


def xavier_normal_(
    tensor: Tensor,
    gain: float = 1.0,
    generator=None,
) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal_(tensor, 0.0, std, generator=generator)


def _calculate_correct_fan(tensor: Tensor, mode: str) -> int:
    if mode not in {"fan_in", "fan_out"}:
        raise ValueError("Mode must be either 'fan_in' or 'fan_out'")
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator=None,
) -> Tensor:
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    bound = math.sqrt(3.0) * gain / math.sqrt(fan)
    return tensor.uniform_(-bound, bound, generator=generator)


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator=None,
) -> Tensor:
    fan = _calculate_correct_fan(tensor, mode)
    std = calculate_gain(nonlinearity, a) / math.sqrt(fan)
    return normal_(tensor, 0.0, std, generator=generator)
