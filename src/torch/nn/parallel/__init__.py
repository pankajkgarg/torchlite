"""Single-process DataParallel compatibility."""

from __future__ import annotations

from .. import Module


class DataParallel(Module):
    """Forward to one module because a browser worker exposes no CUDA devices."""

    def __init__(self, module, device_ids=None, output_device=None, dim: int = 0):
        super().__init__()
        del device_ids, output_device, dim
        self.module = module

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(self.module, name)
