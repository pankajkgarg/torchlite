import torch
import torch.nn.functional as F


class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, values):
        self.out = values @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dimension, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dimension)
        self.beta = torch.zeros(dimension)
        self.running_mean = torch.zeros(dimension)
        self.running_var = torch.ones(dimension)

    def __call__(self, values):
        if self.training:
            dimensions = 0 if values.ndim == 2 else (0, 1)
            mean = values.mean(dimensions, keepdim=True)
            variance = values.var(dimensions, keepdim=True)
        else:
            mean, variance = self.running_mean, self.running_var
        normalized = (values - mean) / torch.sqrt(variance + self.eps)
        self.out = self.gamma * normalized + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * variance
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, values):
        self.out = torch.tanh(values)
        return self.out

    def parameters(self):
        return []


class Embedding:
    def __init__(self, count, dimension):
        self.weight = torch.randn((count, dimension))

    def __call__(self, indices):
        self.out = self.weight[indices]
        return self.out

    def parameters(self):
        return [self.weight]


class FlattenConsecutive:
    def __init__(self, count):
        self.count = count

    def __call__(self, values):
        batch, time, channels = values.shape
        values = values.view(batch, time // self.count, channels * self.count)
        if values.shape[1] == 1:
            values = values.squeeze(1)
        self.out = values
        return self.out

    def parameters(self):
        return []


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, values):
        for layer in self.layers:
            values = layer(values)
        self.out = values
        return self.out

    def parameters(self):
        return [parameter for layer in self.layers for parameter in layer.parameters()]


def run_wavenet_regression(steps=100):
    torch.manual_seed(42)
    count, block_size, vocab_size = 128, 8, 15
    inputs = torch.randint(0, vocab_size, (count, block_size))
    labels = torch.randint(0, vocab_size, (count,))
    embedding_size, hidden_size = 8, 32

    model = Sequential([
        Embedding(vocab_size, embedding_size),
        FlattenConsecutive(2), Linear(embedding_size * 2, hidden_size, bias=False), BatchNorm1d(hidden_size), Tanh(),
        FlattenConsecutive(2), Linear(hidden_size * 2, hidden_size, bias=False), BatchNorm1d(hidden_size), Tanh(),
        FlattenConsecutive(2), Linear(hidden_size * 2, hidden_size, bias=False), BatchNorm1d(hidden_size), Tanh(),
        Linear(hidden_size, vocab_size),
    ])
    with torch.no_grad():
        model.layers[-1].weight *= 0.1
    parameters = model.parameters()
    for parameter in parameters:
        parameter.requires_grad = True

    losses = []
    for _ in range(steps):
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels)
        losses.append(loss.item())
        for parameter in parameters:
            parameter.grad = None
        loss.backward()
        assert all(parameter.grad is not None for parameter in parameters)
        for parameter in parameters:
            parameter.data += -0.1 * parameter.grad

    for layer in model.layers:
        layer.training = False
    assert model(inputs[:8]).shape == (8, vocab_size)
    assert losses[-1] < losses[0] * 0.8, losses
    return losses[0], losses[-1], sum(parameter.nelement() for parameter in parameters)


if __name__ == "__main__":
    before, after, parameter_count = run_wavenet_regression()
    print({"initial_loss": before, "final_loss": after, "parameters": parameter_count})
