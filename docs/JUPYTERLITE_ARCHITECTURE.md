# JupyterLite + Pyodide architecture

Date: 2026-07-22

## Decision

JupyterLite is the notebook application and persistence layer. Pyodide is the Python kernel. TorchLite consists of two reusable components:

1. A pure-Python distribution named `torchlite` that installs the top-level `torch` package into Pyodide.
2. A JupyterLite extension that provides a WebGPU broker and connects it to the Pyodide kernel.

TorchLite does not build a second notebook UI, filesystem, output renderer, or session manager. Those remain JupyterLite responsibilities.

## Why this boundary

- JupyterLite already supplies notebooks, kernel workers, rich output, IndexedDB-backed files, package integration, and static deployment.
- Pyodide supplies CPython, NumPy, matplotlib, pure-Python wheels, and the JavaScript/Python FFI.
- Pyodide 0.27.7 and newer expose JSPI-backed `pyodide.ffi.run_sync()`, allowing synchronous Python APIs to consume asynchronous JavaScript operations when the browser supports JSPI.
- WgPy demonstrates the fallback design: a Pyodide worker issues GPU commands and uses `SharedArrayBuffer` plus `Atomics` for synchronous readback.
- TensorFlow.js and Burn prove that browser training and WebGPU autodiff are viable, but neither exposes the Python/PyTorch semantics required by the course. They are kernel and optimization references, not the public API.

Primary references:

- https://jupyterlite.readthedocs.io/en/latest/
- https://blog.pyodide.org/posts/jspi/
- https://github.com/mil-tokyo/wgpy
- https://github.com/tensorflow/tfjs
- https://github.com/tracel-ai/burn

## Package boundaries

| Component | Responsibility |
|---|---|
| `torchlite` wheel | PyTorch-compatible tensors, modules, autograd, random generators, CPU backend, backend-neutral tensor metadata |
| JupyterLite kernel integration | Install/load the wheel, register the JavaScript GPU module, surface backend capabilities |
| WebGPU broker | GPU device, buffer ownership, command queue, shader/pipeline cache, memory pool, readback |
| Course distribution | Original notebooks, data files, dependency lock, browser presets, expected outputs |
| Browser conformance suite | CPython-PyTorch reference fixtures, Pyodide CPU tests, WebGPU numerical and gradient parity |

## Execution model

### CPU correctness path

The canonical first implementation stores tensors as NumPy arrays inside the Pyodide worker. It must run on every JupyterLite-supported browser and is the reference for debugging the GPU backend.

### WebGPU path

Python tensors hold shape, dtype, autograd metadata, and an opaque GPU buffer handle. GPU data is not shadowed on CPU.

Operations that do not need host data enqueue commands and return a tensor handle immediately. Readback operations such as `.item()`, `.tolist()`, plotting, and Python data-dependent control flow synchronize explicitly.

Synchronization has two implementations:

1. JSPI through `pyodide.ffi.run_sync()` on supported browsers. This is the preferred path because it avoids blocking a worker and does not require cross-origin isolation.
2. A GPU broker worker plus `SharedArrayBuffer`/`Atomics` on browsers without JSPI. This requires COOP/COEP and is the compatibility path.

No operation may claim `webgpu` unless its output buffer was produced by a GPU command. CPU fallback is explicit and observable.

## Autograd

Autograd semantics remain in Python so course code can inspect gradients normally. Forward operations record backend-neutral backward recipes. Backward execution dispatches the same primitive operations to CPU or WebGPU.

The initial GPU implementation uses hand-written, cached primitives for the course operation set. Kernel fusion and graph compilation are later optimizations and may not change eager semantics.

## Scope

The compatibility contract is the complete Neural Networks: Zero to Hero course, not every public PyTorch API. APIs outside the course are supported only when required by shared infrastructure or when they are inexpensive consequences of implemented primitives.

Every supported API needs:

- a real-PyTorch reference fixture;
- a Pyodide CPU result/gradient parity test;
- a WebGPU result/gradient parity test when GPU-backed;
- an end-to-end course cell or notebook regression.

## Rejected foundations

- Compiling all of upstream PyTorch to Wasm: large native dependency surface, no browser WebGPU training backend, and far more scope than the course.
- ONNX Runtime Web: inference-oriented and cannot execute the course's eager autograd/training code.
- TensorFlow.js as the public engine: mature browser training, but PyTorch eager semantics, synchronous scalar reads, state dicts, and notebook-visible gradients still require a substantial adapter.
- Burn as the immediate core: strong WebGPU/autodiff technology, but the Rust/Wasm/Python boundary adds packaging and semantic work before it helps the first notebooks. It remains a candidate for later kernel reuse.

## Deployment requirements

- Static JupyterLite site with a pinned Pyodide release and pinned wheel set.
- Chromium with JSPI for the primary fast path.
- COOP/COEP-enabled deployment for the SharedArrayBuffer fallback.
- CPU mode for unsupported GPU/browser combinations.
- Capability report in the notebook UI showing Python, Pyodide, JSPI, WebGPU, limits, and active backend.
