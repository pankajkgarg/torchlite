# Notes

## Anchor
- current default/champion: independent `pankajkgarg/torchlite` repository; JupyterLite + Pyodide + NumPy-backed `torch` wheel
- alias map: TorchLite = this repo; Z2H = `karpathy/nn-zero-to-hero`
- shortlist / active threads: manual-backprop browser notebook; GPT API surface; PyTorch parity fixtures; JSPI/WebGPU backend
- next: add the manual-backprop browser notebook; add reference PyTorch fixtures; implement GPT tensor/nn APIs; prototype JSPI WebGPU; add CI/static preview
- standing gates: no compatibility or GPU claim without an executable browser or numerical parity test

## Overview
Run PyTorch-style educational notebooks locally in the browser using JupyterLite, Pyodide, and a compact compatibility package.

## Current status
2026-07-22: TorchLite is validated as an independent build: the `torchlite-0.1.0` wheel, five numerical regressions, JupyterLite integrity checks, and three Chrome/Pyodide notebooks pass from a clean ColdStore-local environment in 1.2m; the repository is ready for its fresh GitHub history, while GPT APIs and true WebGPU execution remain future milestones.

## Architecture
- Decision: JupyterLite owns the application, notebook, persistence, and output layers; TorchLite owns the top-level `torch` compatibility package and future WebGPU extension.

## Invariants & gotchas
- Keep environments, caches, generated sites, and browser artifacts on the ColdStore external disk.
- A `webgpu` label is never evidence of GPU execution; verify numerical parity and actual dispatch.
