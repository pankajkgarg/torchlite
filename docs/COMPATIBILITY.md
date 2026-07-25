# Zero-to-Hero compatibility

Date: 2026-07-25

TorchLite supports the PyTorch surface used by the core lesson path from
micrograd comparisons through makemore and a reduced GPT. The browser site
contains executable teaching fixtures rather than byte-for-byte copies of
every source notebook: several source notebooks contain intentional error
cells, desktop shell commands, large downloads, and training schedules that
are not suitable for JupyterLite's Run All action.

## Executed browser notebooks

| Notebook | Current result |
|---|---|
| Original 32,033-name bigram, 50 steps | Loss 3.7557 → 2.5812 |
| MLP + BatchNorm slice | Loss 2.7188 → 0.4631; gradients present |
| Hierarchical WaveNet-shaped slice | Training and all parameter gradients pass |
| Tiny GPT | One-block causal transformer trains and checkpoints |
| Source-lesson API check | Makemore, initialization, top-k, device probes, SGD, and file I/O pass |

The combined Chrome suite completes in about one minute on the development
machine, including a fresh Pyodide kernel for each notebook. These are
compatibility fixtures, not claims that every original 200,000-step schedule
has completed in the browser.

## Local numerical fixtures

- Miniature and original-corpus bigram training.
- MLP and BatchNorm autograd.
- Manual backprop compared across 11 intermediate and parameter gradients.
- Three-dimensional batched matrix multiplication and hierarchical flattening.
- Random generators, multinomial sampling, advanced indexing, variance/std,
  histogram diagnostics, and `max(dim).indices`.
- Source-notebook helpers including `add`, `unbind`, `bernoulli`, `randperm`,
  `topk`, `gather`, save/load, CPU-safe CUDA/MPS probes, ReLU, SGD, common
  initializers, simple data loaders, sequence padding, and cosine scheduling.

## Course outlook

| Section | Status | Remaining work |
|---|---|---|
| Micrograd | Pure-Python engine works in Pyodide; PyTorch comparison surface supported | Bundle optional Graphviz rendering |
| Bigram | Browser regression passing | Broader real-PyTorch reference fixtures |
| MLP + BatchNorm | Browser slice passing | Original plots and long schedules |
| Manual backprop | Local parity fixture passing | Editable browser notebook |
| WaveNet | Browser slice passing | Original long schedule as opt-in benchmark |
| GPT | Reduced one-block decoder trains and checkpoints in browser | Full-size schedules need acceleration |
| Tokenizer | Core BPE lesson is pure Python | Optional `regex`, tiktoken, and SentencePiece cells need compatible wheels/data |
| GPT-2 reproduction | Core model APIs are present | 124M+ training, Hugging Face downloads, and CUDA profiling are not interactive browser workloads |

## Performance contract

- Early-course CPU/Wasm cells should remain interactive.
- Long makemore schedules are opt-in benchmarks until measured on the browser
  and hardware matrix.
- GPT defaults to a reduced educational preset. Original settings are an
  optional desktop benchmark after WebGPU is correct.
- No operation may claim WebGPU execution unless GPU dispatch and CPU/GPU
  numerical parity are both verified.

## Verify

```sh
PYTHONPATH=src temp/jupyterlite-venv/bin/python tests/test_original_bigram.py
PYTHONPATH=src temp/jupyterlite-venv/bin/python tests/test_manual_backprop.py
PYTHONPATH=src temp/jupyterlite-venv/bin/python tests/test_wavenet.py
PYTHONPATH=src temp/jupyterlite-venv/bin/python tests/test_llm_core.py
PYTHONPATH=src temp/jupyterlite-venv/bin/python tests/test_zero_to_hero_compat.py
npm run test:browser
```
