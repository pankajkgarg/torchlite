# Implementation roadmap

Date: 2026-07-22

| Milestone | Outcome | Acceptance gate | Status |
|---|---|---|---|
| M0 Architecture | JupyterLite/Pyodide package boundaries and GPU synchronization strategy | Architecture recorded; reference stacks evaluated; unsupported claims removed | Complete |
| M1 Bigram | Installable `torchlite` wheel and lecture-2 bigram in JupyterLite | Original bigram training/sampling cells run with course data; loss and seeded samples match reference tolerances | Complete |
| M2 Makemore | MLP, activation/gradient, BatchNorm, manual-backprop, and WaveNet notebooks | Every cell runs; losses, shapes, selected gradients, plots, and samples pass fixtures | Bigram, MLP/BatchNorm, and WaveNet browser slices passing; manual fixture local |
| M3 WebGPU | Correct GPU broker and backend-neutral autograd | CPU/GPU forward and gradient parity; no silent fallback; memory lifecycle test; course benchmark gates | Planned |
| M4 GPT + tokenizer | GPT and minBPE plus complete course distribution | Reduced GPT preset completes; original settings launch and checkpoint; tokenizer variants and files work | `llm-core-v1` Tiny GPT complete; broader model/tokenizer work remains |
| M5 Release | Reproducible static site and documented package | Clean build, browser matrix, offline cache, security review, independent repository and preview | In progress |

## Performance gates

| Workload | Required browser result |
|---|---|
| Micrograd and bigram | Interactive; ordinary cells complete in seconds |
| Makemore MLP through WaveNet | Original iteration counts complete in a practical desktop session |
| GPT reduced profile | Completes in a normal learning session without exceeding configured memory |
| GPT original profile | Optional desktop benchmark with checkpointing and honest hardware warning |

M1 has an executable JupyterLite/Chrome regression over the original `names.txt` corpus: the bundled wheel supports tuple factories, tensor indexing with backward scatter, `one_hot`, `Generator`, `multinomial`, broadcast-gradient reduction, `exp`/`log` autograd, and 50-step bigram training/sampling. M2's MLP/BatchNorm slice adds `linspace`, `log10`, `sqrt`, variance/standard deviation autograd, reductions, max indices, histograms, and retained diagnostics. M4's first slice adds a one-block causal transformer, a PyTorch 2.13 differential fixture, AdamW, state handling, and an executable browser notebook; see [`LLM_CORE_V1.md`](LLM_CORE_V1.md).
