"""Minimal in-process dataset helpers for educational browser notebooks."""

from __future__ import annotations

import math

import numpy as np

from .. import Generator, Tensor, randperm, stack


class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


def _collate(batch):
    first = batch[0]
    if isinstance(first, Tensor):
        return stack(batch)
    if isinstance(first, tuple):
        return tuple(_collate(list(items)) for items in zip(*batch))
    if isinstance(first, list):
        return [_collate(list(items)) for items in zip(*batch)]
    if isinstance(first, (int, float, bool, np.number)):
        return Tensor(batch)
    return batch


class DataLoader:
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        generator: Generator | None = None,
        collate_fn=None,
        **kwargs,
    ):
        # Browser execution is single-process. Accept common desktop-only knobs
        # so course code does not need branching.
        del kwargs
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.generator = generator
        self.collate_fn = collate_fn or _collate

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            indices = randperm(len(indices), generator=self.generator).tolist()
        for start in range(0, len(indices), self.batch_size):
            selected = indices[start:start + self.batch_size]
            if self.drop_last and len(selected) < self.batch_size:
                break
            yield self.collate_fn([self.dataset[index] for index in selected])


def random_split(dataset, lengths, generator: Generator | None = None):
    total = len(dataset)
    if lengths and all(isinstance(length, float) for length in lengths):
        counts = [math.floor(total * length) for length in lengths]
        for index in range(total - sum(counts)):
            counts[index % len(counts)] += 1
    else:
        counts = [int(length) for length in lengths]
    if sum(counts) != total:
        raise ValueError("Sum of input lengths does not equal the length of the input dataset")
    indices = randperm(total, generator=generator).tolist()
    output = []
    start = 0
    for count in counts:
        output.append(Subset(dataset, indices[start:start + count]))
        start += count
    return output
