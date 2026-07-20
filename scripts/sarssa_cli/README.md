# SARSSA CLI

A thin command-line client over the SARSSA HTTP API — run pipelines,
follow their progress, and browse runs/experiments from the terminal.

Needs a **running SARSSA stack** — backend + MLflow, and in Docker
also the frontend container, which is what serves the API at
`:5173/api`. All compute happens there, so CLI tasks share the
compute queue, cancellation, and experiment selection with the web
UI. Stdlib-only; YAML definitions use PyYAML, already a project
dependency.

## Examples

```bash
# run a pipeline definition and follow it until it finishes
uv run python scripts/sarssa_cli/sarssa_cli.py run scripts/sarssa_cli/pipeline.example.yaml

# fire-and-forget into a chosen experiment, watch it later in the UI
uv run python scripts/sarssa_cli/sarssa_cli.py run pipeline.yaml \
    --experiment my_experiments --name overnight --no-wait

# one multi_run step on an existing pipeline run
uv run python scripts/sarssa_cli/sarssa_cli.py step <run_id> \
    inspection.single.sae_inspection.sae_inspection --param neuron_id=42

# browse state (add --params to see every plugin's parameters)
uv run python scripts/sarssa_cli/sarssa_cli.py runs list --experiment my_experiments
uv run python scripts/sarssa_cli/sarssa_cli.py tasks list
uv run python scripts/sarssa_cli/sarssa_cli.py plugins list training_sae --params
```

## Pipeline definitions

YAML/JSON files mirroring the `POST /pipelines/run-async` body — see
the commented [`pipeline.example.yaml`](pipeline.example.yaml)
(including how to inherit steps from a past run via `context`).

## Commands → endpoints

| Command | Endpoint(s) |
|---|---|
| `run` | `POST /pipelines/run-async` + task polling |
| `step` | `POST /pipelines/runs/{id}/execute-step-async` + polling |
| `runs list` / `runs context` | `GET /pipelines/runs` / `…/{id}/context` |
| `tasks list` / `tasks cancel` | `GET /pipelines/tasks` / `POST …/{id}/cancel` |
| `experiments list` / `create` | `GET` / `POST /pipelines/experiments` |
| `plugins list` | `GET /plugins/registry` |

## Notes

- Base URL defaults to `http://localhost:5173` (`--base-url` /
  `SARSSA_URL`); the `/api` prefix is auto-detected, so running
  locally you can point it straight at the backend
  (`--base-url http://localhost:8000`) and skip the frontend — only
  `just mlflow` + `just run` need to be up. There is no
  authentication support: the CLI targets local stacks, not
  password-protected deployments.
- Progress streams to **stderr**, machine-readable results (the final
  `context.json`) to **stdout** — `… run x.yaml > context.json` is
  scriptable.
- Exit codes: `0` completed, `1` error, `2` cancelled, `3`
  `--timeout` reached (the task keeps running).
- Ctrl-C while following offers to cancel the task (`--yes` skips the
  prompt); a second Ctrl-C detaches and leaves it running — it stays
  adoptable from the UI's *Running* menu.
- Endpoint details: [`src/app/README.md`](../../src/app/README.md) §4.
