"""Functional neural-network operations."""

from __future__ import annotations

import numpy as np

from .. import Tensor, arange


def one_hot(input: Tensor, num_classes: int = -1) -> Tensor:
    indices = np.asarray(input._array, dtype=np.int64)
    if indices.size and np.any(indices < 0):
        raise RuntimeError("Class values must be non-negative")
    if num_classes == -1:
        num_classes = int(indices.max()) + 1 if indices.size else 0
    if indices.size and np.any(indices >= num_classes):
        raise RuntimeError("Class values must be smaller than num_classes")
    result = np.zeros(indices.shape + (num_classes,), dtype=np.int64)
    if indices.size:
        result.reshape(-1, num_classes)[np.arange(indices.size), indices.reshape(-1)] = 1
    return Tensor(result, dtype=np.int64)


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    shifted = input - Tensor(np.max(input._array, axis=dim, keepdims=True), dtype=input.dtype)
    exponentials = shifted.exp()
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


def log_softmax(input: Tensor, dim: int = -1) -> Tensor:
    shifted = input - Tensor(np.max(input._array, axis=dim, keepdims=True), dtype=input.dtype)
    return shifted - shifted.exp().sum(dim=dim, keepdim=True).log()


def cross_entropy(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    log_probs = log_softmax(input, dim=-1)
    flat = log_probs.view(-1, log_probs.shape[-1])
    targets = target.view(-1)
    losses = -flat[arange(targets.numel()), targets]
    if reduction == "none":
        return losses.view(target.shape)
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError(f"invalid reduction: {reduction}")
