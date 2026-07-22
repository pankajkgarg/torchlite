"""Small in-place initialization subset used by educational GPTs."""

from __future__ import annotations

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
