# 📂 SARSSA `src/`

> **What this is:** a 30-second map of the Python source tree and
> where each part is documented. This is an index, not a deep doc —
> follow the links for the detail.

`src/` is the **Python import root**: the backend is launched from
here (`cd src && uvicorn app.main:app …`, i.e. `just run`), so modules
import as `app.*`, `plugins.*`, and `utils.*`. Keep that in mind when
reading import paths anywhere in the codebase.

## Layout

| Folder | What it is | Docs |
|--------|------------|------|
| `app/` | The **FastAPI backend** — HTTP API, the pipeline engine, plugin discovery, MLflow integration. | [`app/README.md`](app/README.md) |
| `plugins/` | The **plugin system** and the seven pipeline categories (dataset loading → CFM/SAE training → neuron labeling → evaluation / inspection / steering). | [`plugins/README.md`](plugins/README.md) · dataset loading tutorial: [`plugins/dataset_loading/README.md`](plugins/dataset_loading/README.md) |
| `utils/` | Shared, provider-agnostic helpers used by both `app/` and `plugins/` (MLflow access, embedders, LLM clients, torch model loaders, logging). | `utils/README.md` *(planned)* |
| `tests/` | Pytest suite; mirrors the tree (`tests/app`, `tests/plugins`, `tests/utils`). | — |

**Extending SARSSA?** Almost all extension happens in `plugins/`: you
add your own dataset loader, trainer, or labeling / inspection /
steering method as a new implementation **inside the relevant
`plugins/<category>/` subfolder** — no backend or frontend changes
needed. Start with the plugin contract in
[`plugins/README.md`](plugins/README.md), and for the most common case
— bringing your own dataset — the step-by-step tutorial in
[`plugins/dataset_loading/README.md`](plugins/dataset_loading/README.md).

`src/` may also contain **runtime-generated MLflow data** —
`mlflow.db`, `mlartifacts/`, `mlruns/` — created when pipelines run.
These are not source code (they are gitignored); ignore them when
reading the tree.

## More

- 🖥️ Frontend (the web UI): [`../frontend/README.md`](../frontend/README.md)
- 📘 Project overview, setup & Docker: [root `README.md`](../README.md)
