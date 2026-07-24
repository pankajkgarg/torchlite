"""Regression coverage for APIs used by the source Zero-to-Hero notebooks."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split


class ToyDataset(Dataset):
    def __init__(self):
        self.values = [torch.tensor([index, index + 1]) for index in range(6)]

    def __getitem__(self, index):
        return self.values[index], index

    def __len__(self):
        return len(self.values)


def run_zero_to_hero_compatibility():
    # Lectures 2–6: broadcasting, unbinding, diagnostics, and initialization.
    broadcast = torch.add(torch.ones(4, 1), torch.ones(4))
    assert broadcast.shape == (4, 4)

    embedding = torch.arange(24).view(2, 3, 4).float().requires_grad_()
    pieces = torch.unbind(embedding, 1)
    joined = torch.cat(pieces, 1)
    assert joined.shape == (2, 12)
    joined.sum().backward()
    assert torch.allclose(embedding.grad, torch.ones_like(embedding))

    assert math.isclose(nn.init.calculate_gain("tanh"), 5 / 3)
    weight = torch.empty(6, 4)
    nn.init.xavier_uniform_(weight, generator=torch.Generator().manual_seed(7))
    assert float(weight.abs().max()) <= math.sqrt(6 / 10)

    probabilities = torch.full((32,), 0.25)
    samples = torch.bernoulli(probabilities, generator=torch.Generator().manual_seed(1))
    assert samples.shape == probabilities.shape
    assert not torch.isnan(samples).any()
    assert torch.isinf(torch.tensor([0.0, float("inf")])).tolist() == [False, True]

    # GPT lessons: top-k sampling helpers, gather, modules, CPU probes, and I/O.
    scores = torch.tensor([[0.1, 0.7, -0.2, 0.4]])
    top = torch.topk(scores, 2)
    assert top.indices.tolist() == [[1, 3]]
    assert torch.gather(scores, 1, top.indices).tolist() == top.values.tolist()
    assert torch.cuda.is_available() is False
    assert torch.backends.mps.is_available() is False
    assert torch.compile(lambda value: value + 1)(2) == 3
    with torch.autocast(device_type="cpu"):
        assert torch.mean(torch.pow(scores, 2)).item() > 0

    model = DataParallel(nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max=4)
    before = [parameter.clone() for parameter in model.parameters()]
    loss = F.cross_entropy(model(torch.randn(3, 4)), torch.tensor([0, 1, 0]))
    loss.backward()
    optimizer.step()
    scheduler.step()
    assert any(
        not torch.allclose(previous, current)
        for previous, current in zip(before, model.parameters())
    )

    state_path = Path(tempfile.gettempdir()) / "torchlite-zero-to-hero-state.pkl"
    torch.save(model.module.state_dict(), state_path)
    restored = torch.load(state_path)
    assert set(restored) == set(model.module.state_dict())
    state_path.unlink()

    # Exercise notebooks: simple batching, deterministic splits, and padding.
    dataset = ToyDataset()
    splits = random_split(dataset, [4, 2], generator=torch.Generator().manual_seed(3))
    batch_values, batch_indices = next(iter(DataLoader(splits[0], batch_size=2)))
    assert batch_values.shape == (2, 2)
    assert batch_indices.shape == (2,)
    padded = pad_sequence(
        [torch.tensor([1, 2]), torch.tensor([3])],
        batch_first=True,
        padding_value=-1,
    )
    assert padded.tolist() == [[1, 2], [3, -1]]

    return {
        "broadcast_shape": broadcast.shape,
        "unbound_parts": len(pieces),
        "top_indices": top.indices.tolist(),
        "loss": loss.item(),
    }


if __name__ == "__main__":
    print(run_zero_to_hero_compatibility())
