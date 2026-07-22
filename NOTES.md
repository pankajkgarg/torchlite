# Notes

## Anchor
- current default/champion: TorchLite 0.2 `llm-core-v1`; JupyterLite + Pyodide + NumPy-backed `torch` wheel
- alias map: TorchLite = this repo; Z2H = `karpathy/nn-zero-to-hero`
- shortlist / active threads: trace-driven nanoGPT coverage; tokenizer/checkpoints; JSPI/WebGPU backend; CI/static preview
- next: trace nanoGPT/Zero-to-Hero APIs under PyTorch; add an operation manifest/generated wrappers; add browser checkpoint/tokenizer paths; prototype JSPI WebGPU with parity fixtures; add CI/static preview
- standing gates: no compatibility or GPU claim without an executable browser or numerical parity test

## Overview
Run PyTorch-style educational notebooks locally in the browser using JupyterLite, Pyodide, and a compact compatibility package.

## Current status
2026-07-22: TorchLite 0.2's `llm-core-v1` milestone is published on `agent/llm-core-v1` in draft PR #1 against `main`; the one-block, 1,112-parameter transformer finishes at loss 1.291235 versus PyTorch 2.13's 1.291306, six local regressions, the wheel, JupyterLite checks, and all four Chrome/Pyodide notebooks pass (1.2m), while broader nanoGPT, checkpoints/tokenizers, CI/static preview, and true WebGPU execution remain open.

## Architecture
- Decision: JupyterLite owns the application, notebook, persistence, and output layers; TorchLite owns the top-level `torch` compatibility package and future WebGPU extension.

## Invariants & gotchas
- Keep environments, caches, generated sites, and browser artifacts on the ColdStore external disk.
- A `webgpu` label is never evidence of GPU execution; verify numerical parity and actual dispatch.
