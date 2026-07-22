import torch
import torch.nn.functional as F


def run_bigram_regression():
    words = ["emma", "olivia", "ava", "isabella", "sophia", "charlotte", "mia", "amelia"]
    chars = sorted(set("".join(words)))
    stoi = {char: index + 1 for index, char in enumerate(chars)}
    stoi["."] = 0

    xs, ys = [], []
    for word in words:
        sequence = [".", *word, "."]
        for first, second in zip(sequence, sequence[1:]):
            xs.append(stoi[first])
            ys.append(stoi[second])

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    count = xs.nelement()
    classes = len(stoi)
    generator = torch.Generator().manual_seed(2147483647)
    weights = torch.randn((classes, classes), generator=generator, requires_grad=True)

    initial_loss = None
    for _ in range(50):
        encoded = F.one_hot(xs, num_classes=classes).float()
        logits = encoded @ weights
        counts = logits.exp()
        probabilities = counts / counts.sum(1, keepdims=True)
        loss = -probabilities[torch.arange(count), ys].log().mean() + 0.01 * (weights**2).mean()
        if initial_loss is None:
            initial_loss = loss.item()
        weights.grad = None
        loss.backward()
        weights.data += -20 * weights.grad

    assert loss.item() < initial_loss
    assert weights.grad is not None
    return initial_loss, loss.item()


if __name__ == "__main__":
    before, after = run_bigram_regression()
    print({"initial_loss": before, "final_loss": after})
