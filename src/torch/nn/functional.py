"""Functional neural-network operations for the ``llm-core-v1`` profile."""

from __future__ import annotations

import math
import numpy as np

from .. import Tensor, arange, rand


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


def gelu(input: Tensor, approximate: str = "none") -> Tensor:
    if approximate not in {"none", "tanh"}:
        raise RuntimeError("approximate must be 'none' or 'tanh'")
    if approximate == "tanh":
        coefficient = math.sqrt(2.0 / math.pi)
        return 0.5 * input * (1.0 + (coefficient * (input + 0.044715 * input**3)).tanh())

    values = input._array
    erf_values = np.vectorize(math.erf)(values / math.sqrt(2.0))
    result = input._new(0.5 * values * (1.0 + erf_values), input, op="GeluBackward")
    if result.requires_grad:
        derivative = 0.5 * (1.0 + erf_values) + values * np.exp(-(values**2) / 2) / math.sqrt(2 * math.pi)
        result._backward = lambda: input._accumulate_grad(result.grad._array * derivative)
    return result


def relu(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        raise NotImplementedError("TorchLite relu does not support inplace=True")
    return input * (input > 0).to(dtype=input.dtype)


def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    if inplace:
        raise NotImplementedError("TorchLite dropout does not support inplace=True")
    if not training or p == 0:
        return input
    if p == 1:
        return input * 0
    mask = (rand(input.shape, device=input.device) >= p).to(dtype=input.dtype)
    return input * mask / (1.0 - p)


def layer_norm(
    input: Tensor,
    normalized_shape: int | tuple[int, ...],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    if tuple(input.shape[-len(shape):]) != shape:
        raise RuntimeError(f"expected input trailing shape {shape}, got {input.shape}")
    axes = tuple(range(input.ndim - len(shape), input.ndim))
    mean = input.mean(dim=axes, keepdim=True)
    variance = ((input - mean) ** 2).mean(dim=axes, keepdim=True)
    output = (input - mean) / (variance + eps).sqrt()
    if weight is not None:
        output = output * weight
    if bias is not None:
        output = output + bias
    return output


def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    output = input @ weight.T
    return output if bias is None else output + bias


def cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tensor:
    if weight is not None or label_smoothing:
        raise NotImplementedError("llm-core-v1 cross_entropy omits class weights and label smoothing")
    log_probs = log_softmax(input, dim=-1)
    flat = log_probs.view(-1, log_probs.shape[-1])
    targets = target.view(-1)
    valid = targets != ignore_index
    valid_count = int(np.count_nonzero(valid._array))
    safe_targets = Tensor(
        np.where(valid._array, targets._array, 0), dtype=np.int64, device=target.device
    )
    losses = -flat[arange(targets.numel()), safe_targets]
    losses = losses * valid.to(dtype=input.dtype)
    if reduction == "none":
        return losses.view(target.shape)
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.sum() / valid_count if valid_count else losses.sum() * float("nan")
    raise ValueError(f"invalid reduction: {reduction}")


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> Tensor:
    if is_causal and attn_mask is not None:
        raise RuntimeError("attn_mask and is_causal cannot both be set in llm-core-v1")
    scale_factor = (query.shape[-1] ** -0.5) if scale is None else scale
    scores = (query @ key.transpose(-2, -1)) * scale_factor
    if is_causal:
        query_length, key_length = query.shape[-2], key.shape[-2]
        causal = np.tril(np.ones((query_length, key_length), dtype=np.bool_))
        scores = scores.masked_fill(Tensor(~causal, dtype=np.bool_), -np.inf)
    elif attn_mask is not None:
        if attn_mask.dtype.kind == "b":
            scores = scores.masked_fill(attn_mask == False, -np.inf)  # noqa: E712
        else:
            scores = scores + attn_mask
    weights = softmax(scores, dim=-1)
    weights = dropout(weights, dropout_p, training=dropout_p > 0)
    return weights @ value
