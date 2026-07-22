# Zero-to-Hero compatibility

Date: 2026-07-22

TorchLite does not yet run the entire Neural Networks: Zero to Hero course
unchanged. It does demonstrate that a fully local JupyterLite/Pyodide route is
practical for the makemore workloads.

## Executed browser notebooks

| Notebook | Current result |
|---|---|
| Original 32,033-name bigram, 50 steps | Loss 3.7557 → 2.5812 |
| MLP + BatchNorm slice | Loss 2.7188 → 0.4631; gradients present |
| Hierarchical WaveNet-shaped slice | Training and all parameter gradients pass |

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

## Course outlook

| Section | Status | Remaining work |
|---|---|---|
| Micrograd | Expected to work with Pyodide | Bundle plotting/Graphviz dependencies and notebook fixture |
| Bigram | Browser regression passing | Broader real-PyTorch reference fixtures |
| MLP + BatchNorm | Browser slice passing | Original plots and long schedules |
| Manual backprop | Local parity fixture passing | Editable browser notebook |
| WaveNet | Browser slice passing | Original long schedule as opt-in benchmark |
| GPT | Not yet supported | LayerNorm, masks/`tril`, initialization, buffers, optimizer/state APIs, WebGPU |
| Tokenizer | Mostly pure Python | Package `regex`/tiktoken-compatible paths and browser files |

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
npm run test:browser
```
