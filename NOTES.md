# Notes

## Anchor
- current default/champion: TorchLite 0.2 `llm-core-v1`; JupyterLite + Pyodide + NumPy-backed `torch` wheel
- alias map: TorchLite = this repo; Z2H = `karpathy/nn-zero-to-hero`
- shortlist / active threads: trace-driven nanoGPT coverage; tokenizer/checkpoints; JSPI/WebGPU backend; CI/static preview
- next: publish the tested 0.2 feature branch and draft PR; trace nanoGPT/Zero-to-Hero APIs under PyTorch; add an operation manifest/generated wrappers; add browser checkpoint/tokenizer paths; prototype JSPI WebGPU with parity fixtures
- standing gates: no compatibility or GPU claim without an executable browser or numerical parity test

## Overview
Run PyTorch-style educational notebooks locally in the browser using JupyterLite, Pyodide, and a compact compatibility package.

## Current status
2026-07-22: TorchLite 0.2 implements the explicit `llm-core-v1` profile and trains a one-block, 1,112-parameter causal transformer end to end; its nanoGPT-shaped fixture finishes at loss 1.291235 versus PyTorch 2.13's 1.291306, six local regressions, the built wheel, JupyterLite checks, and all four Chrome/Pyodide notebooks pass (1.2m final suite), GitHub authentication is restored and the tested feature branch is ready to publish, while broader nanoGPT, checkpoints/tokenizers, and WebGPU remain open.

## Architecture
- Decision: JupyterLite owns the application, notebook, persistence, and output layers; TorchLite owns the top-level `torch` compatibility package and future WebGPU extension.

## Invariants & gotchas
- Keep environments, caches, generated sites, and browser artifacts on the ColdStore external disk.
- A `webgpu` label is never evidence of GPU execution; verify numerical parity and actual dispatch.
