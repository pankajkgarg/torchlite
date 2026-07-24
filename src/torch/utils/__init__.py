"""Small utility subset used by Zero-to-Hero notebooks."""

from __future__ import annotations

from types import SimpleNamespace

from . import data

backcompat = SimpleNamespace(
    broadcast_warning=SimpleNamespace(enabled=False),
)

__all__ = ["backcompat", "data"]
