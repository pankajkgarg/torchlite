"""CPU-only automatic-mixed-precision compatibility for browser notebooks."""

from __future__ import annotations

from contextlib import nullcontext


def autocast(device_type: str = "cpu", dtype=None, enabled: bool = True, **kwargs):
    """Return a no-op context because TorchLite currently computes on NumPy CPU."""
    del device_type, dtype, enabled, kwargs
    return nullcontext()


class GradScaler:
    """API-compatible scaler for a backend that does not use low-precision grads."""

    def __init__(self, device: str = "cpu", enabled: bool = True, **kwargs):
        del device, kwargs
        self._enabled = bool(enabled)

    def scale(self, outputs):
        return outputs

    def step(self, optimizer, *args, **kwargs):
        return optimizer.step(*args, **kwargs)

    def update(self, new_scale=None):
        del new_scale

    def unscale_(self, optimizer):
        return optimizer

    def is_enabled(self) -> bool:
        return self._enabled
