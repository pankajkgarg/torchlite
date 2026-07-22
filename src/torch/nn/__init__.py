"""Neural-network building blocks for TorchLite's ``llm-core-v1`` profile."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator

import numpy as np

from .. import Tensor, ones, randn, zeros


def Parameter(data, requires_grad: bool = True) -> Tensor:
    """Create a leaf tensor that ``Module`` will register as a parameter."""
    if isinstance(data, Tensor):
        return Tensor(data._array, dtype=data.dtype, device=data.device, requires_grad=requires_grad)
    return Tensor(data, requires_grad=requires_grad)


class Module:
    def __init__(self):
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        if name not in {"_parameters", "_modules", "_buffers"} and hasattr(self, "_parameters"):
            self._parameters.pop(name, None)
            self._modules.pop(name, None)
            self._buffers.pop(name, None)
            if isinstance(value, Tensor) and value.requires_grad:
                self._parameters[name] = value
            elif isinstance(value, Module):
                self._modules[name] = value
        object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        seen: set[int] = set()
        for name, parameter in self._parameters.items():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                yield prefix + name, parameter
        if recurse:
            for module_name, module in self._modules.items():
                child_prefix = prefix + module_name + "."
                for name, parameter in module.named_parameters(child_prefix, recurse=True):
                    if id(parameter) not in seen:
                        seen.add(id(parameter))
                        yield name, parameter

    def parameters(self, recurse: bool = True):
        for _, parameter in self.named_parameters(recurse=recurse):
            yield parameter

    def named_buffers(self, prefix: str = "", recurse: bool = True):
        for name, value in self._buffers.items():
            yield prefix + name, value
        if recurse:
            for module_name, module in self._modules.items():
                yield from module.named_buffers(prefix + module_name + ".", recurse=True)

    def buffers(self, recurse: bool = True):
        for _, value in self.named_buffers(recurse=recurse):
            yield value

    def children(self):
        return self._modules.values()

    def named_modules(self, memo=None, prefix: str = ""):
        memo = set() if memo is None else memo
        if id(self) in memo:
            return
        memo.add(id(self))
        yield prefix, self
        for name, module in self._modules.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_modules(memo, child_prefix)

    def modules(self):
        for _, module in self.named_modules():
            yield module

    def apply(self, function):
        for child in self.children():
            child.apply(function)
        function(self)
        return self

    def zero_grad(self, set_to_none: bool = True):
        for parameter in self.parameters():
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad._array.fill(0)

    def train(self, mode: bool = True):
        self.training = bool(mode)
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def register_buffer(self, name, tensor, persistent: bool = True):
        del persistent  # all TorchLite buffers are persistent in llm-core-v1
        object.__setattr__(self, name, tensor)
        self._buffers[name] = tensor

    def state_dict(self):
        state = OrderedDict()
        for name, parameter in self.named_parameters():
            state[name] = parameter.detach().clone()
        for name, value in self.named_buffers():
            state[name] = value.detach().clone()
        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        local = dict(self.named_parameters()) | dict(self.named_buffers())
        missing = [name for name in local if name not in state_dict]
        unexpected = [name for name in state_dict if name not in local]
        for name, target in local.items():
            if name not in state_dict:
                continue
            source = state_dict[name]
            if target.shape != source.shape:
                raise RuntimeError(f"size mismatch for {name}: {source.shape} vs {target.shape}")
            target.copy_(source)
        if strict and (missing or unexpected):
            raise RuntimeError(f"missing keys: {missing}; unexpected keys: {unexpected}")
        return missing, unexpected


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(randn(out_features, in_features) * (in_features ** -0.5))
        self.bias = Parameter(zeros(out_features)) if bias else None

    def forward(self, input):
        output = input @ self.weight.T
        return output if self.bias is None else output + self.bias


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(randn(num_embeddings, embedding_dim))

    def forward(self, input):
        return self.weight[input]


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.normalized_shape = ((normalized_shape,) if isinstance(normalized_shape, int)
                                 else tuple(normalized_shape))
        self.eps = eps
        self.weight = Parameter(ones(self.normalized_shape)) if elementwise_affine else None
        self.bias = Parameter(zeros(self.normalized_shape)) if elementwise_affine and bias else None

    def forward(self, input):
        return functional.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )


class RMSNorm(Module):
    def __init__(self, normalized_shape: int | tuple[int, ...], eps: float = 1e-6):
        super().__init__()
        self.normalized_shape = ((normalized_shape,) if isinstance(normalized_shape, int)
                                 else tuple(normalized_shape))
        self.eps = eps
        self.weight = Parameter(ones(self.normalized_shape))

    def forward(self, input):
        axes = tuple(range(input.ndim - len(self.normalized_shape), input.ndim))
        scale = (input * input).mean(dim=axes, keepdim=True)
        return input * (scale + self.eps) ** -0.5 * self.weight


class Dropout(Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError("dropout probability must be between 0 and 1")
        if inplace:
            raise NotImplementedError("TorchLite Dropout does not support inplace=True")
        self.p = float(p)

    def forward(self, input):
        return functional.dropout(input, self.p, self.training)


class GELU(Module):
    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, input):
        return functional.gelu(input, approximate=self.approximate)


class Tanh(Module):
    def forward(self, input):
        return input.tanh()


class Identity(Module):
    def forward(self, input):
        return input


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        if len(modules) == 1 and isinstance(modules[0], OrderedDict):
            items = list(modules[0].items())
        else:
            items = [(str(index), module) for index, module in enumerate(modules)]
        self._sequence = []
        for name, module in items:
            setattr(self, name, module)
            self._sequence.append(module)

    def forward(self, input):
        for module in self._sequence:
            input = module(input)
        return input


class ModuleList(Module):
    def __init__(self, modules: Iterable[Module] | None = None):
        super().__init__()
        self._sequence: list[Module] = []
        for module in modules or ():
            self.append(module)

    def append(self, module: Module):
        setattr(self, str(len(self._sequence)), module)
        self._sequence.append(module)
        return self

    def __iter__(self) -> Iterator[Module]:
        return iter(self._sequence)

    def __len__(self) -> int:
        return len(self._sequence)

    def __getitem__(self, index):
        return self._sequence[index]


class ModuleDict(Module):
    def __init__(self, modules=None):
        super().__init__()
        self._keys = []
        for name, module in dict(modules or {}).items():
            self[name] = module

    def __setitem__(self, name, module):
        if name not in self._keys:
            self._keys.append(name)
        setattr(self, name, module)

    def __getitem__(self, name):
        return getattr(self, name)

    def __iter__(self):
        return iter(self._keys)

    def items(self):
        return ((name, self[name]) for name in self._keys)


class CrossEntropyLoss(Module):
    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        return functional.cross_entropy(
            input, target, ignore_index=self.ignore_index, reduction=self.reduction
        )


from . import functional  # noqa: E402
from . import init  # noqa: E402
from . import utils  # noqa: E402
