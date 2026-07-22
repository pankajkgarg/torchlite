"""Executable contract for TorchLite's reduced decoder-only transformer profile."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_size: int, heads: int):
        super().__init__()
        assert embedding_size % heads == 0
        self.heads = heads
        self.projection = nn.Linear(embedding_size, 3 * embedding_size)
        self.output = nn.Linear(embedding_size, embedding_size)

    def forward(self, values):
        batch, time, channels = values.shape
        query, key, value = self.projection(values).split(channels, dim=2)
        query = query.view(batch, time, self.heads, channels // self.heads).transpose(1, 2)
        key = key.view(batch, time, self.heads, channels // self.heads).transpose(1, 2)
        value = value.view(batch, time, self.heads, channels // self.heads).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.output(attended)


class Block(nn.Module):
    def __init__(self, embedding_size: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_size)
        self.attention = CausalSelfAttention(embedding_size, heads)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * embedding_size, embedding_size),
        )

    def forward(self, values):
        values = values + self.attention(self.norm1(values))
        return values + self.mlp(self.norm2(values))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int = 11, context: int = 6, embedding_size: int = 8):
        super().__init__()
        self.transformer = nn.ModuleDict({
            "token_embedding": nn.Embedding(vocab_size, embedding_size),
            "position_embedding": nn.Embedding(context, embedding_size),
            "blocks": nn.ModuleList([Block(embedding_size, heads=2)]),
            "norm": nn.LayerNorm(embedding_size),
        })
        self.head = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, tokens, targets=None):
        _, time = tokens.shape
        positions = torch.arange(time, dtype=torch.long, device=tokens.device)
        values = self.transformer["token_embedding"](tokens)
        values = values + self.transformer["position_embedding"](positions)
        for block in self.transformer["blocks"]:
            values = block(values)
        logits = self.head(self.transformer["norm"](values))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss


def initialize_deterministically(model):
    """Avoid RNG differences so this file can also run against real PyTorch."""
    for index, parameter in enumerate(model.parameters()):
        values = torch.linspace(
            -0.04 + index * 0.001,
            0.04 + index * 0.001,
            parameter.numel(),
            dtype=parameter.dtype,
        ).view(parameter.shape)
        parameter.data.copy_(values)


def run_llm_core_regression(steps: int = 24):
    model = TinyGPT()
    visited = []
    model.apply(lambda module: visited.append(type(module).__name__))
    assert visited[-1] == "TinyGPT" and "CausalSelfAttention" in visited
    initialize_deterministically(model)
    tokens = torch.tensor([
        [0, 1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1, 0],
        [2, 3, 5, 7, 1, 4],
        [8, 6, 4, 2, 0, 9],
    ])
    targets = torch.tensor([
        [1, 2, 3, 4, 5, 6],
        [4, 3, 2, 1, 0, -1],
        [3, 5, 7, 1, 4, 6],
        [6, 4, 2, 0, 9, 10],
    ])
    decay = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=0.025)

    losses = []
    for _ in range(steps):
        _, loss = model(tokens, targets)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert float(total_norm) > 0
        optimizer.step()

    logits, final_loss = model(tokens, targets)
    assert logits.shape == (4, 6, 11)
    assert losses[-1] < losses[0] * 0.72, losses
    assert all(parameter.grad is not None for parameter in model.parameters())

    state = model.state_dict()
    restored = TinyGPT()
    restored.load_state_dict(state)
    restored_logits, _ = restored(tokens)
    assert torch.allclose(logits, restored_logits, atol=1e-6)

    rms = nn.RMSNorm(8)
    normalized = rms(torch.linspace(-1, 1, 16).view(2, 8))
    assert normalized.shape == (2, 8)
    return {
        "initial_loss": losses[0],
        "last_step_loss": losses[-1],
        "final_loss": final_loss.item(),
        "logit_checksum": logits.sum().item(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def assert_pytorch_reference(metrics):
    fixture_path = Path(__file__).parent / "fixtures" / "llm_core_pytorch_2_13.json"
    fixture = json.loads(fixture_path.read_text())
    reference = fixture["metrics"]
    tolerances = fixture["tolerances"]
    assert metrics["parameter_count"] == reference["parameter_count"]
    for name in ("initial_loss", "last_step_loss", "final_loss"):
        assert abs(metrics[name] - reference[name]) <= tolerances["loss_atol"], (name, metrics)
    assert abs(metrics["logit_checksum"] - reference["logit_checksum"]) <= tolerances["logit_checksum_atol"]


if __name__ == "__main__":
    result = run_llm_core_regression()
    assert_pytorch_reference(result)
    print(json.dumps(result, sort_keys=True))
