## [2026-07-22 12:40] decision | Independent TorchLite repository
Extracted the JupyterLite/Pyodide implementation from the Greed evaluation into a clean project identity; retained the new tensor package, notebooks, tests, architecture, and upstream attribution without fork history.

## [2026-07-22 13:02] experiment | Independent build gate
Built `torchlite-0.1.0`; five Python regressions and JupyterLite checks passed, then all three Pyodide notebooks passed in Chrome in 1.2m from the new repository.
