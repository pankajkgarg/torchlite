"""CUDA capability probes with honest CPU-only browser behavior."""

from __future__ import annotations

from .. import manual_seed as _manual_seed


def is_available() -> bool:
    return False


def device_count() -> int:
    return 0


def get_device_name(device=None) -> str:
    del device
    raise RuntimeError("TorchLite runs on browser CPU; CUDA is not available")


def manual_seed(seed: int):
    return _manual_seed(seed)


def manual_seed_all(seed: int):
    return _manual_seed(seed)


def synchronize(device=None) -> None:
    del device


def memory_summary(*args, **kwargs) -> str:
    del args, kwargs
    return "TorchLite browser CPU backend: CUDA memory is unavailable."
