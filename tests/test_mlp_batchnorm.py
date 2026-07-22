import math

import torch
import torch.nn.functional as F


def make_dataset():
    words = ["emma", "olivia", "ava", "isabella", "sophia", "charlotte", "mia", "amelia"]
    chars = sorted(set("".join(words)))
    stoi = {char: index + 1 for index, char in enumerate(chars)}
    stoi["."] = 0

    block_size = 3
    inputs, labels = [], []
    for word in words:
        context = [0] * block_size
        for char in word + ".":
            index = stoi[char]
            inputs.append(context)
            labels.append(index)
            context = context[1:] + [index]
    return torch.tensor(inputs), torch.tensor(labels), len(stoi)


def run_mlp_regression(steps=120):
    inputs, labels, classes = make_dataset()
    generator = torch.Generator().manual_seed(2147483647)
    embedding_size, hidden_size = 8, 48

    embeddings = torch.randn((classes, embedding_size), generator=generator)
    hidden_weights = torch.randn((3 * embedding_size, hidden_size), generator=generator)
    hidden_weights.data *= (5 / 3) / math.sqrt(3 * embedding_size)
    hidden_bias = torch.zeros(hidden_size)
    output_weights = torch.randn((hidden_size, classes), generator=generator)
    output_weights.data *= 0.01
    output_bias = torch.zeros(classes)
    parameters = [embeddings, hidden_weights, hidden_bias, output_weights, output_bias]
    for parameter in parameters:
        parameter.requires_grad = True

    losses = []
    for _ in range(steps):
        embedded = embeddings[inputs]
        hidden = torch.tanh(embedded.view(-1, 3 * embedding_size) @ hidden_weights + hidden_bias)
        logits = hidden @ output_weights + output_bias
        loss = F.cross_entropy(logits, labels)
        losses.append(loss.item())
        for parameter in parameters:
            parameter.grad = None
        loss.backward()
        for parameter in parameters:
            parameter.data += -0.2 * parameter.grad

    assert losses[-1] < losses[0] * 0.8, losses
    return losses[0], losses[-1]


def test_batchnorm_primitives():
    generator = torch.Generator().manual_seed(42)
    values = torch.randn((16, 8), generator=generator, requires_grad=True)
    gain = torch.ones((1, 8), requires_grad=True)
    bias = torch.zeros((1, 8), requires_grad=True)

    normalized = (values - values.mean(0, keepdim=True)) / torch.sqrt(
        values.var(0, keepdim=True, unbiased=True) + 1e-5
    )
    output = gain * normalized + bias
    loss = (output**2).mean()
    output.retain_grad()
    loss.backward()

    assert output.grad is not None
    assert values.grad is not None
    assert gain.grad is not None
    assert bias.grad is not None
    assert torch.allclose(values.mean(0), torch.tensor(values.numpy().mean(axis=0)), atol=1e-6)
    assert abs(values.std().item() - values.numpy().std(ddof=1)) < 1e-6


def test_diagnostic_primitives():
    values = torch.tensor([[1.0, 4.0, 2.0], [9.0, 3.0, 5.0]], requires_grad=True)
    maximum = values.max(1)
    assert maximum.values.tolist() == [4.0, 9.0]
    assert maximum.indices.tolist() == [1, 0]
    maximum.values.sum().backward()
    assert values.grad.tolist() == [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    assert torch.linspace(-3, 0, 4).tolist() == [-3.0, -2.0, -1.0, 0.0]
    hist, edges = torch.histogram(torch.tensor([0.0, 0.5, 1.0]), bins=2)
    assert hist.tolist() == [1.0, 2.0]
    assert len(edges) == 3


if __name__ == "__main__":
    before, after = run_mlp_regression()
    test_batchnorm_primitives()
    test_diagnostic_primitives()
    print({"initial_loss": before, "final_loss": after})
