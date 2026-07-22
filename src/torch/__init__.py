"""TorchLite's course-focused PyTorch compatibility package for Pyodide.

The implementation is deliberately small and explicit. NumPy is the correctness
backend; a WebGPU backend can implement the same primitive operation contract.
"""

from __future__ import annotations

from collections import namedtuple
from contextlib import ContextDecorator
from typing import Any, Iterable, Sequence

import numpy as np

__version__ = "0.2.0"

float32 = np.dtype("float32")
float64 = np.dtype("float64")
double = float64
float = float32
int32 = np.dtype("int32")
int64 = np.dtype("int64")
long = int64
int = int32
bool = np.dtype("bool")
uint8 = np.dtype("uint8")

_grad_enabled = True
_default_rng = np.random.RandomState()
MaxResult = namedtuple("max", ["values", "indices"])


def _resolve_dtype(data: Any, dtype: Any | None) -> np.dtype:
    if dtype is not None:
        return np.dtype(dtype)
    array = np.asarray(data)
    if array.dtype.kind in "iu":
        return int64
    if array.dtype.kind == "b":
        return bool
    return float32


def _as_array(value: Any) -> np.ndarray:
    return value._array if isinstance(value, Tensor) else np.asarray(value)


def _sum_to_shape(data: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    data = np.asarray(data)
    while data.ndim > len(shape):
        data = data.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and data.shape[axis] != 1:
            data = data.sum(axis=axis, keepdims=True)
    return data.reshape(shape)


class device:
    def __init__(self, device_type: str):
        self.type = "webgpu" if device_type == "cuda" else str(device_type)

    def __str__(self) -> str:
        return self.type

    def __repr__(self) -> str:
        return f"device(type='{self.type}')"


class Generator:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._seed = 0
        self._rng = np.random.RandomState(self._seed)

    def manual_seed(self, seed: int) -> "Generator":
        self._seed = builtins_int(seed)
        self._rng = np.random.RandomState(self._seed)
        return self

    def initial_seed(self) -> int:
        return self._seed


class Tensor:
    __array_priority__ = 1000

    def __init__(
        self,
        data: Any,
        *,
        dtype: Any | None = None,
        device: str | device = "cpu",
        requires_grad: bool = False,
        _children: Iterable["Tensor"] = (),
        _op: str = "",
    ):
        if isinstance(data, Tensor):
            data = data._array
        resolved_dtype = _resolve_dtype(data, dtype)
        self._array = np.array(data, dtype=resolved_dtype, copy=True)
        self.device = globals()["device"](device) if isinstance(device, str) else device
        self.requires_grad = builtins_bool(requires_grad and _grad_enabled)
        if self.requires_grad and self._array.dtype.kind not in "fc":
            raise RuntimeError("Only floating point tensors can require gradients")
        self.grad: Tensor | None = None
        self.grad_fn = _op or None
        self._prev = tuple(_children)
        self._backward = lambda: None

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def ndim(self) -> int:
        return self._array.ndim

    @property
    def dtype(self) -> np.dtype:
        return self._array.dtype

    @property
    def data(self) -> "Tensor":
        return self

    @data.setter
    def data(self, value: Any) -> None:
        if value is self:
            return
        self._array = np.asarray(_as_array(value), dtype=self.dtype).copy()

    @property
    def T(self) -> "Tensor":
        if self.ndim > 2:
            raise RuntimeError("Tensor.T is only supported for tensors with at most 2 dimensions")
        return self.transpose(0, 1) if self.ndim == 2 else self

    def __repr__(self) -> str:
        suffix = ", requires_grad=True" if self.requires_grad else ""
        return f"tensor({self._array!r}{suffix})"

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        array = np.asarray(self._array, dtype=dtype)
        return array.copy() if copy else array

    def __len__(self) -> int:
        return len(self._array)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __format__(self, spec: str) -> str:
        return format(self.item(), spec) if self.numel() == 1 else format(str(self), spec)

    def __float__(self) -> builtins_float:
        return builtins_float(self.item())

    def __int__(self) -> builtins_int:
        return builtins_int(self.item())

    def _new(self, data: Any, *parents: "Tensor", op: str = "") -> "Tensor":
        requires_grad = _grad_enabled and any(parent.requires_grad for parent in parents)
        return Tensor(
            data,
            dtype=np.asarray(data).dtype,
            device=self.device,
            requires_grad=requires_grad,
            _children=parents if requires_grad else (),
            _op=op,
        )

    def _accumulate_grad(self, value: Any) -> None:
        value = np.asarray(value, dtype=self._array.dtype)
        if self.grad is None:
            self.grad = Tensor(np.zeros_like(self._array), device=self.device)
        self.grad._array += value

    def backward(self, gradient: Any | None = None, retain_graph: bool = False) -> None:
        if not self.requires_grad:
            raise RuntimeError("element 0 of tensors does not require grad")
        if gradient is None:
            if self.numel() != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            gradient = np.ones_like(self._array)
        else:
            gradient = _as_array(gradient)

        topo: list[Tensor] = []
        visited: set[int] = set()

        def visit(node: Tensor) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._prev:
                visit(parent)
            topo.append(node)

        visit(self)
        self.grad = Tensor(gradient, dtype=self.dtype, device=self.device)
        for node in reversed(topo):
            node._backward()
        if not retain_graph:
            for node in topo:
                node._prev = ()

    def item(self) -> Any:
        if self.numel() != 1:
            raise ValueError("only one element tensors can be converted to Python scalars")
        return self._array.item()

    def tolist(self) -> list[Any]:
        return self._array.tolist()

    def numpy(self) -> np.ndarray:
        return self._array

    def cpu(self) -> "Tensor":
        return self

    def to(self, target: Any = None, *, dtype: Any | None = None) -> "Tensor":
        if dtype is None and target is not None and not isinstance(target, (str, device)):
            dtype = target
        target_device = self.device if target is None or not isinstance(target, (str, device)) else target
        return Tensor(self._array, dtype=dtype or self.dtype, device=target_device, requires_grad=self.requires_grad)

    def float(self) -> "Tensor":
        return self.to(dtype=float32)

    def double(self) -> "Tensor":
        return self.to(dtype=float64)

    def long(self) -> "Tensor":
        return self.to(dtype=int64)

    def int(self) -> "Tensor":
        return self.to(dtype=int32)

    def numel(self) -> int:
        return self._array.size

    def nelement(self) -> int:
        return self.numel()

    def size(self, dim: int | None = None):
        return self.shape if dim is None else self.shape[dim]

    def dim(self) -> int:
        return self.ndim

    def requires_grad_(self, requires_grad: bool = True) -> "Tensor":
        self.requires_grad = builtins_bool(requires_grad)
        return self

    def retain_grad(self) -> None:
        # This compact engine retains gradients for every differentiable node.
        return None

    def detach(self) -> "Tensor":
        return Tensor(self._array, dtype=self.dtype, device=self.device)

    def clone(self) -> "Tensor":
        result = self._new(self._array.copy(), self, op="CloneBackward")
        if result.requires_grad:
            result._backward = lambda: self._accumulate_grad(result.grad._array)
        return result

    def view(self, *shape: int | Sequence[int]) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        result = self._new(self._array.reshape(*shape), self, op="ViewBackward")
        if result.requires_grad:
            result._backward = lambda: self._accumulate_grad(result.grad._array.reshape(self.shape))
        return result

    reshape = view

    def squeeze(self, dim: int | None = None) -> "Tensor":
        values = np.squeeze(self._array, axis=dim)
        result = self._new(values, self, op="SqueezeBackward")
        if result.requires_grad:
            result._backward = lambda: self._accumulate_grad(result.grad._array.reshape(self.shape))
        return result

    def unsqueeze(self, dim: int) -> "Tensor":
        values = np.expand_dims(self._array, axis=dim)
        result = self._new(values, self, op="UnsqueezeBackward")
        if result.requires_grad:
            result._backward = lambda: self._accumulate_grad(result.grad._array.reshape(self.shape))
        return result

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        result = self._new(np.swapaxes(self._array, dim0, dim1), self, op="TransposeBackward")
        if result.requires_grad:
            result._backward = lambda: self._accumulate_grad(np.swapaxes(result.grad._array, dim0, dim1))
        return result

    def permute(self, *dims: int | Sequence[int]) -> "Tensor":
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        normalized = tuple(builtins_int(dim) % self.ndim for dim in dims)
        if sorted(normalized) != list(range(self.ndim)):
            raise RuntimeError("permute expects each dimension exactly once")
        result = self._new(np.transpose(self._array, normalized), self, op="PermuteBackward")
        if result.requires_grad:
            inverse = tuple(np.argsort(normalized))
            result._backward = lambda: self._accumulate_grad(
                np.transpose(result.grad._array, inverse)
            )
        return result

    def contiguous(self) -> "Tensor":
        return self.clone()

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        start_dim %= self.ndim
        end_dim %= self.ndim
        if start_dim > end_dim:
            raise RuntimeError("flatten start_dim cannot come after end_dim")
        collapsed = builtins_int(np.prod(self.shape[start_dim:end_dim + 1]))
        return self.view(self.shape[:start_dim] + (collapsed,) + self.shape[end_dim + 1:])

    def split(self, split_size_or_sections: int | Sequence[int], dim: int = 0) -> tuple["Tensor", ...]:
        axis = dim % self.ndim
        length = self.shape[axis]
        if isinstance(split_size_or_sections, builtins_int):
            size = builtins_int(split_size_or_sections)
            if size <= 0:
                raise RuntimeError("split_size must be greater than zero")
            sections = [min(size, length - start) for start in range(0, length, size)]
        else:
            sections = [builtins_int(section) for section in split_size_or_sections]
            if sum(sections) != length:
                raise RuntimeError("split_with_sizes expects sections to sum to the dimension length")
        output = []
        start = 0
        for section in sections:
            key = [slice(None)] * self.ndim
            key[axis] = slice(start, start + section)
            output.append(self[tuple(key)])
            start += section
        return tuple(output)

    def chunk(self, chunks: int, dim: int = 0) -> tuple["Tensor", ...]:
        chunks = builtins_int(chunks)
        if chunks <= 0:
            raise RuntimeError("chunks must be greater than zero")
        length = self.shape[dim % self.ndim]
        return self.split(max(1, (length + chunks - 1) // chunks), dim=dim)

    def t(self) -> "Tensor":
        if self.ndim != 2:
            raise RuntimeError("t() expects a 2D tensor")
        return self.transpose(0, 1)

    def __getitem__(self, key: Any) -> "Tensor":
        def convert(index: Any):
            if isinstance(index, Tensor):
                return index._array.astype(np.bool_ if index.dtype.kind == "b" else np.int64)
            return index

        converted = tuple(convert(part) for part in key) if isinstance(key, tuple) else convert(key)
        result = self._new(self._array[converted], self, op="IndexBackward")
        if result.requires_grad:
            def backward_index() -> None:
                grad = np.zeros_like(self._array)
                np.add.at(grad, converted, result.grad._array)
                self._accumulate_grad(grad)
            result._backward = backward_index
        return result

    def __eq__(self, other: Any) -> "Tensor":
        return Tensor(self._array == _as_array(other), dtype=bool, device=self.device)

    def __ne__(self, other: Any) -> "Tensor":
        return Tensor(self._array != _as_array(other), dtype=bool, device=self.device)

    def __lt__(self, other: Any) -> "Tensor":
        return Tensor(self._array < _as_array(other), dtype=bool, device=self.device)

    def __le__(self, other: Any) -> "Tensor":
        return Tensor(self._array <= _as_array(other), dtype=bool, device=self.device)

    def __gt__(self, other: Any) -> "Tensor":
        return Tensor(self._array > _as_array(other), dtype=bool, device=self.device)

    def __ge__(self, other: Any) -> "Tensor":
        return Tensor(self._array >= _as_array(other), dtype=bool, device=self.device)

    def __setitem__(self, key: Any, value: Any) -> None:
        def convert(index: Any):
            if isinstance(index, Tensor):
                return index._array.astype(np.bool_ if index.dtype.kind == "b" else np.int64)
            return index

        converted = tuple(convert(part) for part in key) if isinstance(key, tuple) else convert(key)
        self._array[converted] = _as_array(value)

    def masked_fill(self, mask: "Tensor", value: Any) -> "Tensor":
        mask_array = np.asarray(_as_array(mask), dtype=np.bool_)
        result = self._new(
            np.where(mask_array, _as_array(value), self._array),
            self,
            op="MaskedFillBackward",
        )
        if result.requires_grad:
            result._backward = lambda: self._accumulate_grad(
                _sum_to_shape(np.where(mask_array, 0, result.grad._array), self.shape)
            )
        return result

    def copy_(self, source: Any) -> "Tensor":
        self._array[...] = _as_array(source)
        return self

    def __add__(self, other: Any) -> "Tensor":
        other_array = _as_array(other)
        parents = (self, other) if isinstance(other, Tensor) else (self,)
        result = self._new(self._array + other_array, *parents, op="AddBackward")
        if result.requires_grad:
            def backward_add() -> None:
                if self.requires_grad:
                    self._accumulate_grad(_sum_to_shape(result.grad._array, self.shape))
                if isinstance(other, Tensor) and other.requires_grad:
                    other._accumulate_grad(_sum_to_shape(result.grad._array, other.shape))
            result._backward = backward_add
        return result

    __radd__ = __add__

    def __sub__(self, other: Any) -> "Tensor":
        return self + (-other if isinstance(other, Tensor) else -np.asarray(other))

    def __rsub__(self, other: Any) -> "Tensor":
        return (-self) + other

    def __neg__(self) -> "Tensor":
        result = self._new(-self._array, self, op="NegBackward")
        if result.requires_grad:
            result._backward = lambda: self._accumulate_grad(-result.grad._array)
        return result

    def __mul__(self, other: Any) -> "Tensor":
        other_array = _as_array(other)
        parents = (self, other) if isinstance(other, Tensor) else (self,)
        result = self._new(self._array * other_array, *parents, op="MulBackward")
        if result.requires_grad:
            def backward_mul() -> None:
                if self.requires_grad:
                    self._accumulate_grad(_sum_to_shape(result.grad._array * other_array, self.shape))
                if isinstance(other, Tensor) and other.requires_grad:
                    other._accumulate_grad(_sum_to_shape(result.grad._array * self._array, other.shape))
            result._backward = backward_mul
        return result

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "Tensor":
        other_array = _as_array(other)
        parents = (self, other) if isinstance(other, Tensor) else (self,)
        result = self._new(self._array / other_array, *parents, op="DivBackward")
        if result.requires_grad:
            def backward_div() -> None:
                if self.requires_grad:
                    self._accumulate_grad(_sum_to_shape(result.grad._array / other_array, self.shape))
                if isinstance(other, Tensor) and other.requires_grad:
                    grad = -result.grad._array * self._array / (other_array ** 2)
                    other._accumulate_grad(_sum_to_shape(grad, other.shape))
            result._backward = backward_div
        return result

    def __rtruediv__(self, other: Any) -> "Tensor":
        return tensor(other, dtype=self.dtype) / self

    def __pow__(self, exponent: Any) -> "Tensor":
        if isinstance(exponent, Tensor):
            raise NotImplementedError("tensor exponents are not implemented")
        result = self._new(self._array ** exponent, self, op="PowBackward")
        if result.requires_grad:
            result._backward = lambda: self._accumulate_grad(
                result.grad._array * exponent * (self._array ** (exponent - 1))
            )
        return result

    def __matmul__(self, other: Any) -> "Tensor":
        other_array = _as_array(other)
        parents = (self, other) if isinstance(other, Tensor) else (self,)
        result = self._new(np.matmul(self._array, other_array), *parents, op="MatmulBackward")
        if result.requires_grad:
            def backward_matmul() -> None:
                if self.requires_grad:
                    grad = np.matmul(result.grad._array, np.swapaxes(other_array, -1, -2))
                    self._accumulate_grad(_sum_to_shape(grad, self.shape))
                if isinstance(other, Tensor) and other.requires_grad:
                    grad = np.matmul(np.swapaxes(self._array, -1, -2), result.grad._array)
                    other._accumulate_grad(_sum_to_shape(grad, other.shape))
            result._backward = backward_matmul
        return result

    def __iadd__(self, other: Any) -> "Tensor":
        return self + other

    def __isub__(self, other: Any) -> "Tensor":
        return self - other

    def __imul__(self, other: Any) -> "Tensor":
        return self * other

    def __itruediv__(self, other: Any) -> "Tensor":
        return self / other

    def sum(self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False, **kwargs) -> "Tensor":
        keepdim = kwargs.pop("keepdims", keepdim)
        if kwargs:
            raise TypeError(f"unexpected keyword arguments: {tuple(kwargs)}")
        result = self._new(self._array.sum(axis=dim, keepdims=keepdim), self, op="SumBackward")
        if result.requires_grad:
            def backward_sum() -> None:
                grad = result.grad._array
                if dim is not None and not keepdim:
                    axes = (dim,) if isinstance(dim, builtins_int) else tuple(dim)
                    for axis in sorted((axis % self.ndim for axis in axes)):
                        grad = np.expand_dims(grad, axis=axis)
                self._accumulate_grad(np.broadcast_to(grad, self.shape))
            result._backward = backward_sum
        return result

    def mean(self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False) -> "Tensor":
        if dim is None:
            count = self.numel()
        else:
            axes = (dim,) if isinstance(dim, builtins_int) else tuple(dim)
            count = np.prod([self.shape[axis] for axis in axes])
        return self.sum(dim=dim, keepdim=keepdim) / count

    def var(
        self,
        dim: int | tuple[int, ...] | None = None,
        unbiased: bool = True,
        keepdim: bool = False,
        *,
        correction: int | None = None,
    ) -> "Tensor":
        axes = tuple(range(self.ndim)) if dim is None else (
            (dim,) if isinstance(dim, builtins_int) else tuple(dim)
        )
        normalized_axes = tuple(axis % self.ndim for axis in axes)
        count = builtins_int(np.prod([self.shape[axis] for axis in normalized_axes]))
        degrees = builtins_int(unbiased) if correction is None else builtins_int(correction)
        if count <= degrees:
            return Tensor(np.full(
                np.var(self._array, axis=dim, keepdims=keepdim).shape,
                np.nan,
                dtype=self.dtype,
            ))
        centered = self - self.mean(dim=dim, keepdim=True)
        return (centered * centered).sum(dim=dim, keepdim=keepdim) / (count - degrees)

    def std(
        self,
        dim: int | tuple[int, ...] | None = None,
        unbiased: bool = True,
        keepdim: bool = False,
        *,
        correction: int | None = None,
    ) -> "Tensor":
        return self.var(
            dim=dim,
            unbiased=unbiased,
            keepdim=keepdim,
            correction=correction,
        ).sqrt()

    def exp(self) -> "Tensor":
        values = np.exp(self._array)
        result = self._new(values, self, op="ExpBackward")
        if result.requires_grad:
            result._backward = lambda: self._accumulate_grad(result.grad._array * values)
        return result

    def log(self) -> "Tensor":
        result = self._new(np.log(self._array), self, op="LogBackward")
        if result.requires_grad:
            result._backward = lambda: self._accumulate_grad(result.grad._array / self._array)
        return result

    def log10(self) -> "Tensor":
        return self.log() / np.log(10.0)

    def sqrt(self) -> "Tensor":
        return self ** 0.5

    def abs(self) -> "Tensor":
        result = self._new(np.abs(self._array), self, op="AbsBackward")
        if result.requires_grad:
            result._backward = lambda: self._accumulate_grad(
                result.grad._array * np.sign(self._array)
            )
        return result

    def max(self, dim: int | None = None, keepdim: bool = False):
        if dim is None:
            flat_index = builtins_int(np.argmax(self._array))
            result = self._new(np.max(self._array), self, op="MaxBackward")
            if result.requires_grad:
                def backward_max_scalar() -> None:
                    grad = np.zeros_like(self._array)
                    grad.reshape(-1)[flat_index] = result.grad._array
                    self._accumulate_grad(grad)
                result._backward = backward_max_scalar
            return result

        axis = dim % self.ndim
        indices = np.argmax(self._array, axis=axis)
        values = np.take_along_axis(
            self._array,
            np.expand_dims(indices, axis=axis),
            axis=axis,
        )
        if not keepdim:
            values = np.squeeze(values, axis=axis)
        result = self._new(values, self, op="MaxBackward")
        if result.requires_grad:
            def backward_max_dim() -> None:
                grad = np.zeros_like(self._array)
                expanded_indices = np.expand_dims(indices, axis=axis)
                upstream = result.grad._array
                if not keepdim:
                    upstream = np.expand_dims(upstream, axis=axis)
                np.put_along_axis(grad, expanded_indices, upstream, axis=axis)
                self._accumulate_grad(grad)
            result._backward = backward_max_dim
        return MaxResult(result, Tensor(indices, dtype=int64, device=self.device))

    def tanh(self) -> "Tensor":
        values = np.tanh(self._array)
        result = self._new(values, self, op="TanhBackward")
        if result.requires_grad:
            result._backward = lambda: self._accumulate_grad(result.grad._array * (1 - values ** 2))
        return result

    def softmax(self, dim: int = -1) -> "Tensor":
        return nn.functional.softmax(self, dim=dim)

    def log_softmax(self, dim: int = -1) -> "Tensor":
        return nn.functional.log_softmax(self, dim=dim)

    def argmax(self, dim: int | None = None, keepdim: bool = False) -> "Tensor":
        values = np.argmax(self._array, axis=dim)
        if dim is not None and keepdim:
            values = np.expand_dims(values, axis=dim)
        return Tensor(values, dtype=int64, device=self.device)


class _GradContext(ContextDecorator):
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __enter__(self):
        global _grad_enabled
        self.previous = _grad_enabled
        _grad_enabled = self.enabled
        return self

    def __exit__(self, *exc):
        global _grad_enabled
        _grad_enabled = self.previous
        return False


def no_grad() -> _GradContext:
    return _GradContext(False)


def enable_grad() -> _GradContext:
    return _GradContext(True)


def is_grad_enabled() -> bool:
    return _grad_enabled


def set_grad_enabled(mode: bool) -> None:
    global _grad_enabled
    _grad_enabled = builtins_bool(mode)


def tensor(data: Any, *, dtype: Any | None = None, device: Any = "cpu", requires_grad: bool = False) -> Tensor:
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)


as_tensor = tensor


def _shape_args(shape: tuple[Any, ...]) -> tuple[int, ...]:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return tuple(builtins_int(value) for value in shape)


def zeros(*shape: Any, dtype: Any = float32, device: Any = "cpu", requires_grad: bool = False) -> Tensor:
    return Tensor(np.zeros(_shape_args(shape), dtype=dtype), device=device, requires_grad=requires_grad)


def ones(*shape: Any, dtype: Any = float32, device: Any = "cpu", requires_grad: bool = False) -> Tensor:
    return Tensor(np.ones(_shape_args(shape), dtype=dtype), device=device, requires_grad=requires_grad)


def empty(*shape: Any, dtype: Any = float32, device: Any = "cpu", requires_grad: bool = False) -> Tensor:
    return Tensor(np.empty(_shape_args(shape), dtype=dtype), device=device, requires_grad=requires_grad)


def full(
    shape: Sequence[int],
    fill_value: Any,
    *,
    dtype: Any | None = None,
    device: Any = "cpu",
    requires_grad: bool = False,
) -> Tensor:
    return Tensor(
        np.full(tuple(shape), fill_value, dtype=dtype or _resolve_dtype(fill_value, None)),
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(input: Tensor, **kwargs) -> Tensor:
    return Tensor(np.zeros_like(input._array), dtype=kwargs.get("dtype", input.dtype), device=kwargs.get("device", input.device))


def ones_like(input: Tensor, **kwargs) -> Tensor:
    return Tensor(np.ones_like(input._array), dtype=kwargs.get("dtype", input.dtype), device=kwargs.get("device", input.device))


def empty_like(input: Tensor, **kwargs) -> Tensor:
    return Tensor(np.empty_like(input._array), dtype=kwargs.get("dtype", input.dtype), device=kwargs.get("device", input.device))


def full_like(input: Tensor, fill_value: Any, **kwargs) -> Tensor:
    dtype = kwargs.get("dtype", input.dtype)
    return Tensor(
        np.full(input.shape, fill_value, dtype=dtype),
        dtype=dtype,
        device=kwargs.get("device", input.device),
    )


def _rng(generator: Generator | None):
    return _default_rng if generator is None else generator._rng


def randn(*shape: Any, generator: Generator | None = None, dtype: Any = float32, device: Any = "cpu", requires_grad: bool = False) -> Tensor:
    return Tensor(_rng(generator).randn(*_shape_args(shape)), dtype=dtype, device=device, requires_grad=requires_grad)


def rand(*shape: Any, generator: Generator | None = None, dtype: Any = float32, device: Any = "cpu", requires_grad: bool = False) -> Tensor:
    return Tensor(_rng(generator).rand(*_shape_args(shape)), dtype=dtype, device=device, requires_grad=requires_grad)


def randint(low: int, high: int | Sequence[int] | None = None, size: Sequence[int] | None = None, *, generator: Generator | None = None, device: Any = "cpu") -> Tensor:
    if isinstance(high, (tuple, list)) and size is None:
        size, high, low = high, low, 0
    elif high is None:
        high, low = low, 0
    return Tensor(_rng(generator).randint(low, high, size=None if size is None else tuple(size)), dtype=int64, device=device)


def arange(start: int, end: int | None = None, step: int = 1, *, dtype: Any | None = None, device: Any = "cpu") -> Tensor:
    if end is None:
        start, end = 0, start
    return Tensor(np.arange(start, end, step, dtype=dtype or int64), device=device)


def linspace(
    start: float,
    end: float,
    steps: int,
    *,
    dtype: Any = float32,
    device: Any = "cpu",
) -> Tensor:
    return Tensor(np.linspace(start, end, builtins_int(steps), dtype=dtype), device=device)


def manual_seed(seed: int) -> Generator:
    global _default_rng
    _default_rng = np.random.RandomState(builtins_int(seed))
    return Generator().manual_seed(seed)


def multinomial(input: Tensor, num_samples: int, replacement: bool = False, *, generator: Generator | None = None) -> Tensor:
    probabilities = np.asarray(input._array, dtype=np.float64)
    if probabilities.ndim not in (1, 2):
        raise RuntimeError("prob_dist must be 1 or 2 dimensional")
    rows = probabilities.reshape(1, -1) if probabilities.ndim == 1 else probabilities
    output = []
    for row in rows:
        if np.any(row < 0) or not np.all(np.isfinite(row)) or row.sum() <= 0:
            raise RuntimeError("invalid multinomial distribution")
        if not replacement and num_samples > np.count_nonzero(row):
            raise RuntimeError("cannot sample more values than non-zero probabilities")
        output.append(_rng(generator).choice(row.size, size=num_samples, replace=replacement, p=row / row.sum()))
    values = np.stack(output).astype(np.int64)
    return Tensor(values[0] if probabilities.ndim == 1 else values, dtype=int64)


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    tensors = tuple(tensors)
    requires_grad = _grad_enabled and any(value.requires_grad for value in tensors)
    result = Tensor(
        np.stack([value._array for value in tensors], axis=dim),
        dtype=tensors[0].dtype,
        device=tensors[0].device,
        requires_grad=requires_grad,
        _children=tensors if requires_grad else (),
        _op="StackBackward",
    )
    if result.requires_grad:
        def backward_stack() -> None:
            for index, value in enumerate(tensors):
                if value.requires_grad:
                    value._accumulate_grad(np.take(result.grad._array, index, axis=dim))
        result._backward = backward_stack
    return result


def cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    tensors = tuple(tensors)
    axis = dim % tensors[0].ndim
    requires_grad = _grad_enabled and any(value.requires_grad for value in tensors)
    result = Tensor(
        np.concatenate([value._array for value in tensors], axis=axis),
        dtype=tensors[0].dtype,
        device=tensors[0].device,
        requires_grad=requires_grad,
        _children=tensors if requires_grad else (),
        _op="CatBackward",
    )
    if result.requires_grad:
        boundaries = np.cumsum([value.shape[axis] for value in tensors[:-1]])
        def backward_cat() -> None:
            gradients = np.split(result.grad._array, boundaries, axis=axis)
            for value, gradient in zip(tensors, gradients):
                if value.requires_grad:
                    value._accumulate_grad(gradient)
        result._backward = backward_cat
    return result


def split(input: Tensor, split_size_or_sections: int | Sequence[int], dim: int = 0):
    return input.split(split_size_or_sections, dim=dim)


def chunk(input: Tensor, chunks: int, dim: int = 0):
    return input.chunk(chunks, dim=dim)


def tril(input: Tensor, diagonal: int = 0) -> Tensor:
    mask = np.tril(np.ones(input.shape, dtype=np.bool_), k=diagonal)
    return input.masked_fill(Tensor(~mask, dtype=bool), 0)


def log(input: Tensor) -> Tensor:
    return input.log()


def exp(input: Tensor) -> Tensor:
    return input.exp()


def tanh(input: Tensor) -> Tensor:
    return input.tanh()


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    return nn.functional.softmax(input, dim=dim)


def argmax(input: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    return input.argmax(dim=dim, keepdim=keepdim)


def sqrt(input: Tensor) -> Tensor:
    return input.sqrt()


def abs(input: Tensor) -> Tensor:
    return input.abs()


def all(input: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    return Tensor(np.all(input._array, axis=dim, keepdims=keepdim), dtype=bool)


def allclose(
    input: Tensor,
    other: Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> bool:
    return builtins_bool(np.allclose(
        input._array,
        other._array,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    ))


def histogram(
    input: Tensor,
    bins: int | Sequence[float] | Tensor = 100,
    *,
    range: tuple[float, float] | None = None,
    weight: Tensor | None = None,
    density: bool = False,
):
    bin_values = bins._array if isinstance(bins, Tensor) else bins
    weights = None if weight is None else weight._array
    values, edges = np.histogram(
        input._array,
        bins=bin_values,
        range=range,
        weights=weights,
        density=density,
    )
    return Tensor(values, dtype=float32), Tensor(edges, dtype=float32)


def matmul(left: Tensor, right: Tensor) -> Tensor:
    return left @ right


builtins_int = __builtins__["int"] if isinstance(__builtins__, dict) else __builtins__.int
builtins_float = __builtins__["float"] if isinstance(__builtins__, dict) else __builtins__.float
builtins_bool = __builtins__["bool"] if isinstance(__builtins__, dict) else __builtins__.bool

from . import nn  # noqa: E402  (registered after Tensor is defined)
from . import optim  # noqa: E402
