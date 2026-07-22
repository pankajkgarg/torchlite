# TorchLite

TorchLite runs PyTorch-style learning notebooks locally in the browser with
JupyterLite and Pyodide. It installs a compact, course-focused top-level
`torch` package backed by NumPy/Wasm today, with WebGPU acceleration planned as
a separate backend.

The first compatibility target is Andrej Karpathy's *Neural Networks: Zero to
Hero*. The repository currently includes browser-tested notebooks for:

- the full 32,033-name makemore bigram dataset;
- MLP training and BatchNorm gradients;
- a hierarchical, WaveNet-shaped model.
- a one-block decoder-only transformer with causal attention and AdamW.

This is not a complete PyTorch implementation. It is an intentionally small
compatibility layer for educational workloads, with executable tests defining
what is supported.

## Why this architecture

JupyterLite supplies the notebook interface, files, persistence, and rich
outputs. Pyodide supplies Python and scientific packages inside a Web Worker.
TorchLite supplies the missing PyTorch-style tensor and autograd surface. That
keeps computation local without building another notebook application.

## Build and run

All commands run from the repository root:

```sh
python3 -m venv temp/jupyterlite-venv
temp/jupyterlite-venv/bin/pip install \
  build hatchling jupyter-server jupyterlite-core jupyterlite-pyodide-kernel numpy
npm install

temp/jupyterlite-venv/bin/python -m build --wheel --no-isolation
temp/jupyterlite-venv/bin/jupyter lite build \
  --lite-dir jupyterlite \
  --contents content \
  --piplite-wheels ../dist/torchlite-0.2.0-py3-none-any.whl
temp/jupyterlite-venv/bin/jupyter lite serve --lite-dir jupyterlite
```

Then open `http://127.0.0.1:8000/lab/index.html`.

## Verify

```sh
PYTHONPATH=src temp/jupyterlite-venv/bin/python tests/test_bigram.py
PYTHONPATH=src temp/jupyterlite-venv/bin/python tests/test_original_bigram.py
PYTHONPATH=src temp/jupyterlite-venv/bin/python tests/test_mlp_batchnorm.py
PYTHONPATH=src temp/jupyterlite-venv/bin/python tests/test_manual_backprop.py
PYTHONPATH=src temp/jupyterlite-venv/bin/python tests/test_wavenet.py
PYTHONPATH=src temp/jupyterlite-venv/bin/python tests/test_llm_core.py
temp/jupyterlite-venv/bin/jupyter lite check --lite-dir jupyterlite
npm run test:browser
```

See the executable [`llm-core-v1` profile](docs/LLM_CORE_V1.md) and [roadmap](docs/IMPLEMENTATION_ROADMAP.md) for broader GPT and WebGPU work.

## Attribution

TorchLite is an independent project. Its architecture was informed by an
evaluation of [Deep-ML-codebase/greed](https://github.com/Deep-ML-codebase/greed),
JupyterLite, Pyodide, and WgPy. The bundled `names.txt` comes from Andrej
Karpathy's MIT-licensed `makemore` repository; its license is included beside
the data.
