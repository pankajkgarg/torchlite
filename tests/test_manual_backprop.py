import torch
import torch.nn.functional as F


def run_manual_backprop_fixture():
    generator = torch.Generator().manual_seed(2147483647)
    batch_size, block_size = 32, 3
    vocab_size, embedding_size, hidden_size = 15, 6, 24
    inputs = torch.randint(0, vocab_size, (batch_size, block_size), generator=generator)
    labels = torch.randint(0, vocab_size, (batch_size,), generator=generator)

    embeddings_table = torch.randn((vocab_size, embedding_size), generator=generator)
    hidden_weights = torch.randn((embedding_size * block_size, hidden_size), generator=generator)
    hidden_bias = torch.randn(hidden_size, generator=generator) * 0.1
    output_weights = torch.randn((hidden_size, vocab_size), generator=generator) * 0.1
    output_bias = torch.randn(vocab_size, generator=generator) * 0.1
    batchnorm_gain = torch.randn((1, hidden_size), generator=generator) * 0.1 + 1.0
    batchnorm_bias = torch.randn((1, hidden_size), generator=generator) * 0.1
    parameters = [
        embeddings_table,
        hidden_weights,
        hidden_bias,
        output_weights,
        output_bias,
        batchnorm_gain,
        batchnorm_bias,
    ]
    for parameter in parameters:
        parameter.requires_grad = True

    embedded = embeddings_table[inputs]
    embedded_flat = embedded.view(batch_size, -1)
    hidden_pre_bn = embedded_flat @ hidden_weights + hidden_bias
    bn_mean = hidden_pre_bn.sum(0, keepdim=True) / batch_size
    bn_diff = hidden_pre_bn - bn_mean
    bn_diff_squared = bn_diff**2
    bn_variance = bn_diff_squared.sum(0, keepdim=True) / (batch_size - 1)
    bn_inverse_std = (bn_variance + 1e-5) ** -0.5
    bn_raw = bn_diff * bn_inverse_std
    hidden_pre_activation = batchnorm_gain * bn_raw + batchnorm_bias
    hidden = torch.tanh(hidden_pre_activation)
    logits = hidden @ output_weights + output_bias
    logit_maxes = logits.max(1, keepdim=True).values
    normalized_logits = logits - logit_maxes
    counts = normalized_logits.exp()
    counts_sum = counts.sum(1, keepdim=True)
    counts_sum_inverse = counts_sum**-1
    probabilities = counts * counts_sum_inverse
    log_probabilities = probabilities.log()
    loss = -log_probabilities[range(batch_size), labels].mean()
    loss.backward()

    d_log_probabilities = torch.zeros_like(log_probabilities)
    d_log_probabilities[range(batch_size), labels] = -1.0 / batch_size
    d_probabilities = (1.0 / probabilities) * d_log_probabilities
    d_counts_sum_inverse = (counts * d_probabilities).sum(1, keepdim=True)
    d_counts = counts_sum_inverse * d_probabilities
    d_counts_sum = (-counts_sum**-2) * d_counts_sum_inverse
    d_counts += torch.ones_like(counts) * d_counts_sum
    d_normalized_logits = counts * d_counts
    d_logits = d_normalized_logits.clone()
    d_logit_maxes = (-d_normalized_logits).sum(1, keepdim=True)
    d_logits += F.one_hot(logits.max(1).indices, num_classes=vocab_size) * d_logit_maxes
    d_hidden = d_logits @ output_weights.T
    d_output_weights = hidden.T @ d_logits
    d_output_bias = d_logits.sum(0)
    d_hidden_pre_activation = (1.0 - hidden**2) * d_hidden
    d_batchnorm_gain = (bn_raw * d_hidden_pre_activation).sum(0, keepdim=True)
    d_bn_raw = batchnorm_gain * d_hidden_pre_activation
    d_batchnorm_bias = d_hidden_pre_activation.sum(0, keepdim=True)
    d_bn_diff = bn_inverse_std * d_bn_raw
    d_bn_inverse_std = (bn_diff * d_bn_raw).sum(0, keepdim=True)
    d_bn_variance = (-0.5 * (bn_variance + 1e-5) ** -1.5) * d_bn_inverse_std
    d_bn_diff_squared = torch.ones_like(bn_diff_squared) * d_bn_variance / (batch_size - 1)
    d_bn_diff += (2 * bn_diff) * d_bn_diff_squared
    d_hidden_pre_bn = d_bn_diff.clone()
    d_bn_mean = (-d_bn_diff).sum(0)
    d_hidden_pre_bn += torch.ones_like(hidden_pre_bn) * d_bn_mean / batch_size
    d_embedded_flat = d_hidden_pre_bn @ hidden_weights.T
    d_hidden_weights = embedded_flat.T @ d_hidden_pre_bn
    d_hidden_bias = d_hidden_pre_bn.sum(0)
    d_embedded = d_embedded_flat.view(embedded.shape)
    d_embeddings_table = torch.zeros_like(embeddings_table)
    for row in range(inputs.shape[0]):
        for column in range(inputs.shape[1]):
            index = inputs[row, column]
            d_embeddings_table[index] += d_embedded[row, column]

    comparisons = {
        "log_probabilities": (d_log_probabilities, log_probabilities.grad),
        "probabilities": (d_probabilities, probabilities.grad),
        "logits": (d_logits, logits.grad),
        "hidden": (d_hidden, hidden.grad),
        "output_weights": (d_output_weights, output_weights.grad),
        "output_bias": (d_output_bias, output_bias.grad),
        "batchnorm_gain": (d_batchnorm_gain, batchnorm_gain.grad),
        "batchnorm_bias": (d_batchnorm_bias, batchnorm_bias.grad),
        "hidden_weights": (d_hidden_weights, hidden_weights.grad),
        "hidden_bias": (d_hidden_bias, hidden_bias.grad),
        "embeddings": (d_embeddings_table, embeddings_table.grad),
    }
    failures = {
        name: (manual - automatic).abs().max().item()
        for name, (manual, automatic) in comparisons.items()
        if not torch.allclose(manual, automatic, rtol=1e-4, atol=1e-6)
    }
    assert not failures, failures
    return loss.item(), len(comparisons)


if __name__ == "__main__":
    loss, comparisons = run_manual_backprop_fixture()
    print({"loss": loss, "gradient_comparisons": comparisons})
