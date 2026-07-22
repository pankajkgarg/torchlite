## [2026-07-22 12:40] decision | Independent TorchLite repository
Extracted the JupyterLite/Pyodide implementation from the Greed evaluation into a clean project identity; retained the new tensor package, notebooks, tests, architecture, and upstream attribution without fork history.

## [2026-07-22 13:02] experiment | Independent build gate
Built `torchlite-0.1.0`; five Python regressions and JupyterLite checks passed, then all three Pyodide notebooks passed in Chrome in 1.2m from the new repository.

## [2026-07-22 13:07] status | Independent repository published
Created public `pankajkgarg/torchlite` with fresh `main` history; GitHub reports `isFork: false`. The original Greed fork was not modified or deleted during extraction.

## [2026-07-22 15:05] experiment | LLM-core differential fixture
Implemented a 1,112-parameter causal transformer and ran the same 24-step program under TorchLite 0.2 and PyTorch 2.13; final loss was 1.290711 vs 1.290762 and logit checksum differed by 0.000463.

## [2026-07-22 15:18] status | Previous 0.1 browser baseline superseded
Before the LLM-core slice, the `torchlite-0.1.0` wheel, five numerical regressions, JupyterLite checks, and three Chrome/Pyodide notebooks passed in 1.2m; GPT APIs and true WebGPU execution were still future milestones.

## [2026-07-22 15:20] experiment | Tiny GPT browser gate
Built `torchlite-0.2.0`, passed six local regressions plus the PyTorch 2.13 reference and JupyterLite checks, then all four Pyodide notebooks—including causal Tiny GPT training—passed in Chrome in 1.3m.

## [2026-07-22 15:27] refactor | nanoGPT-shaped module and optimizer groups
Added `ModuleDict`, recursive `apply`, and per-group AdamW decay; the revised fixture ended at loss 1.291235 vs PyTorch 2.13's 1.291306, a 0.000071 difference.

## [2026-07-22 15:36] bug | Pyodide kernel startup stalled
The first final browser rerun passed bigram, then its second fresh kernel remained `Unknown` with all cells busy until the 240s output timeout; no code exception appeared and later serial tests did not run. The prior full four-notebook build had passed, so retry the changed Tiny GPT fixture in isolation.

## [2026-07-22 15:42] fix | Browser gate recovered
Tiny GPT passed alone in 8.5s, MLP and WaveNet passed together in 15.3s, and a final clean serial run passed all four notebooks in 1.2m; the earlier `Unknown` kernel timeout was transient.

## [2026-07-22 15:45] status | Pre-publish LLM-core milestone
TorchLite 0.2's final wheel, six local regressions, PyTorch 2.13 differential fixture, JupyterLite integrity checks, and four-notebook Chrome suite pass; only GitHub publication remained.

## [2026-07-22 15:46] bug | GitHub CLI authentication invalid
The publish workflow stopped before branch, staging, commit, or push because `gh auth status` reports the active `pankajkgarg` token is invalid; run `gh auth login -h github.com` before retrying publication.

## [2026-07-22 16:08] fix | GitHub authentication restored
Completed a fresh GitHub device flow while keeping the CLI session alive; `gh` confirmed authentication as `pankajkgarg`, superseding the earlier invalid-token blocker.
