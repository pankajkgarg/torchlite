"""Sequence padding helper."""

from __future__ import annotations

import numpy as np

from ... import Tensor, stack


def pad_sequence(
    sequences,
    batch_first: bool = False,
    padding_value: float = 0.0,
    padding_side: str = "right",
):
    sequences = list(sequences)
    if not sequences:
        raise RuntimeError("received an empty list of sequences")
    if padding_side not in {"left", "right"}:
        raise RuntimeError("padding_side must be 'left' or 'right'")
    max_length = max(sequence.shape[0] for sequence in sequences)
    padded = []
    for sequence in sequences:
        missing = max_length - sequence.shape[0]
        widths = [(0, 0)] * sequence.ndim
        widths[0] = (missing, 0) if padding_side == "left" else (0, missing)
        values = np.pad(sequence._array, widths, constant_values=padding_value)
        padded.append(Tensor(values, dtype=sequence.dtype, device=sequence.device))
    output = stack(padded, dim=0)
    return output if batch_first else output.transpose(0, 1)
