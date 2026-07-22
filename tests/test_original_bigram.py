from pathlib import Path

import torch
import torch.nn.functional as F


def run_original_corpus(steps=50):
    repository_root = Path(__file__).resolve().parents[1]
    words = (repository_root / "jupyterlite/content/data/names.txt").read_text().splitlines()
    chars = sorted(set("".join(words)))
    stoi = {char: index + 1 for index, char in enumerate(chars)}
    stoi["."] = 0

    xs, ys = [], []
    for word in words:
        sequence = [".", *word, "."]
        for first, second in zip(sequence, sequence[1:]):
            xs.append(stoi[first])
            ys.append(stoi[second])

    xs, ys = torch.tensor(xs), torch.tensor(ys)
    generator = torch.Generator().manual_seed(2147483647)
    weights = torch.randn((len(stoi), len(stoi)), generator=generator, requires_grad=True)
    losses = []

    encoded = F.one_hot(xs, num_classes=len(stoi)).float()
    for _ in range(steps):
        logits = encoded @ weights
        counts = logits.exp()
        probabilities = counts / counts.sum(1, keepdim=True)
        loss = -probabilities[torch.arange(xs.nelement()), ys].log().mean()
        loss = loss + 0.01 * (weights**2).mean()
        losses.append(loss.item())
        weights.grad = None
        loss.backward()
        weights.data += -20 * weights.grad

    assert xs.nelement() == 228146
    assert len(stoi) == 27
    assert losses[-1] < losses[0]
    return losses[0], losses[-1]


if __name__ == "__main__":
    before, after = run_original_corpus()
    print({"initial_loss": before, "final_loss": after})
