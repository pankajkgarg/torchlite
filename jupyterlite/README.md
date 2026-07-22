# TorchLite JupyterLite site

This directory contains the static notebook application and bundled course
content. Notebooks execute in a Pyodide worker, and `%pip install torchlite`
resolves to the wheel copied into the site during the build.

Included notebooks:

1. `01-makemore-bigram.ipynb` — original 32,033-name course corpus.
2. `02-makemore-mlp-batchnorm.ipynb` — editable MLP and BatchNorm slice.
3. `03-makemore-wavenet.ipynb` — hierarchical WaveNet-shaped network.

Build from the repository root:

```sh
temp/jupyterlite-venv/bin/python -m build --wheel --no-isolation
temp/jupyterlite-venv/bin/jupyter lite build \
  --lite-dir jupyterlite \
  --contents content \
  --piplite-wheels ../dist/torchlite-0.1.0-py3-none-any.whl
```

Validate and test:

```sh
temp/jupyterlite-venv/bin/jupyter lite check --lite-dir jupyterlite
npm run test:browser
```

The browser test server supplies cross-origin isolation headers for worker
filesystem synchronization and the planned shared-memory WebGPU fallback. The
current Pyodide runtime loads from its configured CDN; vendoring it for fully
offline use is a release milestone.
