#!/usr/bin/env python3
"""SARSSA command-line utility — a thin client over the HTTP API.

Requires a running SARSSA stack. Usage, the pipeline-file
format, and examples: ``README.md`` next to this script.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://localhost:5173"
DEFAULT_POLL_INTERVAL_S = 3.0

# task terminal states - process exit codes
EXIT_BY_STATUS = {"completed": 0, "error": 1, "cancelled": 2}
EXIT_USAGE = 1
EXIT_TIMEOUT = 3


class CliError(Exception):
    """A user-facing error: printed to stderr, exits with code 1."""


def eprint(*args: Any) -> None:
    """Print progress/diagnostics to stderr (stdout stays scriptable)."""
    print(*args, file=sys.stderr, flush=True)


# ── spec / argument parsing helpers ─────────


def load_spec(path: str | Path) -> dict:
    """Load a pipeline definition from a JSON or YAML file.

    Args:
        path: File path; ``.json`` is parsed with the stdlib,
            anything else is treated as YAML (requires PyYAML).

    Returns:
        dict: The parsed definition.

    Raises:
        CliError: If the file is missing, unparsable, not a mapping,
            has no non-empty ``steps`` list, or YAML is requested
            without PyYAML installed.
    """
    p = Path(path)
    if not p.is_file():
        raise CliError(f"pipeline file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        try:
            spec = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CliError(f"invalid JSON in {p}: {exc}") from exc
    else:
        try:
            import yaml
        except ImportError as exc:
            raise CliError(
                f"{p} looks like YAML but PyYAML is not installed; "
                "use a .json file or run inside the project venv (uv run)."
            ) from exc
        try:
            spec = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise CliError(f"invalid YAML in {p}: {exc}") from exc
    if not isinstance(spec, dict):
        raise CliError(f"{p} must contain a mapping at the top level")
    if not isinstance(spec.get("steps"), list) or not spec.get("steps"):
        raise CliError(f"{p} must define a non-empty 'steps' list")
    return spec


def parse_kv(pairs: list[str] | None) -> dict[str, str]:
    """Turn repeated ``key=value`` options into a dict.

    Raises:
        CliError: If an item has no ``=`` or an empty key.
    """
    out: dict[str, str] = {}
    for item in pairs or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise CliError(f"expected key=value, got: {item!r}")
        out[key] = value
    return out


def exit_code_for(status: str) -> int:
    """Map a terminal task status to a process exit code."""
    return EXIT_BY_STATUS.get(status, 1)


# ── HTTP client ──────────────────────────────


class Api:
    """Minimal JSON-over-HTTP client with ``/api`` prefix autodetection.

    The nginx single entry serves the backend under ``/api``;
    a bare local uvicorn serves it at the root. The first request
    probes ``/api``, falls back to the root, and
    remembers the answer.
    """

    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")
        self._prefix: str | None = None

    def _raw_request(self, method: str, url: str, body: dict | None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = resp.read()
        return json.loads(payload) if payload else None

    def _resolve_prefix(self) -> str:
        if self._prefix is not None:
            return self._prefix
        last_error: Exception | None = None
        for prefix in ("/api", ""):
            try:
                self._raw_request("GET", f"{self.base}{prefix}/plugins/registry", None)
                self._prefix = prefix
                return prefix
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code != 404:
                    # 401/500 etc. still tell us the prefix is right
                    self._prefix = prefix
                    return prefix
            except urllib.error.URLError as exc:
                raise CliError(
                    f"cannot reach SARSSA at {self.base}: {exc.reason}. "
                    "Is the stack running? (--base-url / SARSSA_URL)"
                ) from exc
        raise CliError(f"no SARSSA API found at {self.base} (tried /api and /): {last_error}")

    def request(self, method: str, path: str, body: dict | None = None) -> Any:
        url = f"{self.base}{self._resolve_prefix()}{path}"
        try:
            return self._raw_request(method, url, body)
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("detail", exc.reason)
            except Exception:
                detail = exc.reason
            raise CliError(f"{method} {path} → {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise CliError(f"cannot reach SARSSA at {self.base}: {exc.reason}") from exc

    def get(self, path: str) -> Any:
        return self.request("GET", path)

    def post(self, path: str, body: dict | None = None) -> Any:
        return self.request("POST", path, body or {})


# ── task following ───────────────────────


def _print_new_messages(messages: list[dict], seen: int) -> int:
    """Print plugin notifications not shown yet; return the new count."""
    for msg in messages[seen:]:
        eprint(f"  [{msg.get('level', 'info')}] {msg.get('text', '')}")
    return len(messages)


def poll_task(
    api: Api,
    task_id: str,
    interval: float,
    timeout: float | None,
    assume_yes: bool,
) -> int:
    """Follow a task until it reaches a terminal state.

    Streams plugin messages and step transitions to stderr; on
    completion prints the final pipeline context to stdout (so
    ``... > context.json`` is scriptable). Ctrl-C offers to cancel
    the task (``--yes`` skips the prompt); a second Ctrl-C detaches
    and leaves the task running on the backend.

    Returns:
        int: Process exit code derived from the terminal status.
    """
    seen_messages = 0
    last_step: tuple[Any, Any] | None = None
    started = time.monotonic()

    while True:
        try:
            status = api.get(f"/pipelines/tasks/{task_id}")
            seen_messages = _print_new_messages(status.get("messages") or [], seen_messages)

            step = (status.get("current_step_index"), status.get("current_step"))
            if step != last_step and status.get("current_step"):
                eprint(
                    f"step {status['current_step_index'] + 1}/{status['total_steps']}: "
                    f"{status['current_step']}"
                )
                last_step = step

            state = status["status"]
            if state in EXIT_BY_STATUS:
                eprint(f"task {task_id}: {state}")
                if state == "error" and status.get("error"):
                    eprint(f"error: {status['error']}")
                if status.get("run_id"):
                    eprint(f"run_id: {status['run_id']}")
                if status.get("context"):
                    print(json.dumps(status["context"], indent=2))
                return exit_code_for(state)

            if timeout is not None and time.monotonic() - started > timeout:
                eprint(f"timeout after {timeout:.0f}s; task {task_id} keeps running")
                return EXIT_TIMEOUT

            time.sleep(interval)
        except KeyboardInterrupt:
            try:
                if assume_yes:
                    answer = "y"
                else:
                    eprint("\nCancel the running task? [y/N] (Ctrl-C again to detach)")
                    answer = input().strip().lower()
            except (KeyboardInterrupt, EOFError):
                eprint(f"detached; task {task_id} keeps running (see the Running menu)")
                return EXIT_BY_STATUS["cancelled"]
            if answer == "y":
                api.post(f"/pipelines/tasks/{task_id}/cancel?mode=now")
                eprint("cancellation requested; waiting for the task to stop...")
            else:
                eprint("continuing to follow the task")


# ── commands ────────────────────────


def cmd_run(api: Api, args: argparse.Namespace) -> int:
    spec = load_spec(args.file)
    body = {
        "steps": spec["steps"],
        "context": spec.get("context") or {},
        "tags": {**(spec.get("tags") or {}), **parse_kv(args.tag)},
        "description": args.description or spec.get("description", ""),
        "pipeline_name": args.name or spec.get("name", ""),
        "experiment_name": args.experiment or spec.get("experiment", ""),
    }
    task = api.post("/pipelines/run-async", body)
    eprint(f"task {task['task_id']} queued ({len(body['steps'])} steps)")
    if args.no_wait:
        print(json.dumps(task))
        return 0
    return poll_task(api, task["task_id"], args.poll_interval, args.timeout, args.yes)


def cmd_step(api: Api, args: argparse.Namespace) -> int:
    params: dict[str, Any] = parse_kv(args.param)
    if args.params_json:
        try:
            params.update(json.loads(args.params_json))
        except json.JSONDecodeError as exc:
            raise CliError(f"invalid --params-json: {exc}") from exc
    task = api.post(
        f"/pipelines/runs/{args.run_id}/execute-step-async",
        {"plugin": args.plugin, "params": params},
    )
    eprint(f"task {task['task_id']} queued (step {args.plugin})")
    if args.no_wait:
        print(json.dumps(task))
        return 0
    return poll_task(api, task["task_id"], args.poll_interval, args.timeout, args.yes)


def cmd_runs_list(api: Api, args: argparse.Namespace) -> int:
    parts = [
        f"required_steps={urllib.parse.quote(step.strip())}"
        for step in (args.required_steps or "").split(",")
        if step.strip()
    ]
    if args.experiment:
        parts.append(f"experiment={urllib.parse.quote(args.experiment)}")
    query = "?" + "&".join(parts) if parts else ""
    runs = api.get(f"/pipelines/runs{query}")
    for run in runs:
        shared = " (shared)" if args.experiment and run.get("shared") else ""
        print(f"{run['run_id']}  {run['status']:<9}  {run['run_name']}{shared}")
    eprint(f"{len(runs)} run(s)")
    return 0


def cmd_runs_context(api: Api, args: argparse.Namespace) -> int:
    print(json.dumps(api.get(f"/pipelines/runs/{args.run_id}/context"), indent=2))
    return 0


def cmd_tasks_list(api: Api, args: argparse.Namespace) -> int:  # noqa: ARG001
    tasks = api.get("/pipelines/tasks")
    for task in tasks:
        step = task.get("current_step") or "-"
        experiment = task.get("experiment_name") or "(shared)"
        print(
            f"{task['task_id']}  {task['status']:<8}  "
            f"step {task['current_step_index'] + 1}/{task['total_steps']} {step}  "
            f"[{experiment}]  {task.get('pipeline_name') or ''}"
        )
    eprint(f"{len(tasks)} active task(s)")
    return 0


def cmd_tasks_cancel(api: Api, args: argparse.Namespace) -> int:
    mode = "now" if args.now else "graceful"
    result = api.post(f"/pipelines/tasks/{args.task_id}/cancel?mode={mode}")
    eprint((result or {}).get("message", "cancellation requested"))
    return 0


def cmd_experiments_list(api: Api, args: argparse.Namespace) -> int:  # noqa: ARG001
    for experiment in api.get("/pipelines/experiments"):
        marker = " (shared)" if experiment.get("shared") else ""
        print(f"{experiment['experiment_id']:>4}  {experiment['name']}{marker}")
    return 0


def cmd_experiments_create(api: Api, args: argparse.Namespace) -> int:
    experiment = api.post("/pipelines/experiments", {"name": args.name})
    eprint(f"experiment {experiment['name']} (id {experiment['experiment_id']})")
    return 0


def cmd_plugins_list(api: Api, args: argparse.Namespace) -> int:
    registry = api.get("/plugins/registry")
    for category, entry in registry.items():
        if args.category and category != args.category:
            continue
        info = entry.get("category_info") or {}
        print(f"{category}  ({info.get('type', '?')})")
        for impl in entry.get("implementations") or []:
            print(f"  {impl['plugin_name']}")
            if args.params:
                for param in impl.get("params") or []:
                    default = param.get("default")
                    default_str = "required" if default is None else f"default {default!r}"
                    print(f"    {param['name']}: {param.get('type', '?')} ({default_str})")
    return 0


# ── entry point ─────────────────────────


def _add_follow_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-wait", action="store_true", help="queue and exit immediately")
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_S, metavar="S")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="S",
        help="stop following after S seconds (task keeps running); exit code 3",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="cancel without asking on Ctrl-C")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sarssa_cli",
        description="SARSSA command-line utility (thin client over the HTTP API).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SARSSA_URL", DEFAULT_BASE_URL),
        help=f"SARSSA base URL (env SARSSA_URL; default {DEFAULT_BASE_URL})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run a pipeline from a YAML/JSON definition")
    p.add_argument(
        "file", help="pipeline definition (see scripts/sarssa_cli/pipeline.example.yaml)"
    )
    p.add_argument("--name", help="pipeline name (overrides the file)")
    p.add_argument("--experiment", help="target MLflow experiment (overrides the file)")
    p.add_argument("--description", help="run description (overrides the file)")
    p.add_argument("--tag", action="append", metavar="K=V", help="extra run tag (repeatable)")
    _add_follow_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("step", help="execute one multi_run step on an existing run")
    p.add_argument("run_id")
    p.add_argument("plugin", help="dotted plugin path (see: plugins list)")
    p.add_argument("--param", action="append", metavar="K=V", help="plugin param (repeatable)")
    p.add_argument("--params-json", help="plugin params as a JSON object")
    _add_follow_args(p)
    p.set_defaults(func=cmd_step)

    runs = sub.add_parser("runs", help="browse past pipeline runs").add_subparsers(
        dest="subcommand", required=True
    )
    p = runs.add_parser("list")
    p.add_argument("--experiment", help="also search this experiment (besides shared)")
    p.add_argument("--required-steps", metavar="A,B", help="only runs containing these steps")
    p.set_defaults(func=cmd_runs_list)
    p = runs.add_parser("context")
    p.add_argument("run_id")
    p.set_defaults(func=cmd_runs_context)

    tasks = sub.add_parser("tasks", help="active (queued/running) tasks").add_subparsers(
        dest="subcommand", required=True
    )
    tasks.add_parser("list").set_defaults(func=cmd_tasks_list)
    p = tasks.add_parser("cancel")
    p.add_argument("task_id")
    p.add_argument("--now", action="store_true", help="also abort the current step")
    p.set_defaults(func=cmd_tasks_cancel)

    experiments = sub.add_parser("experiments", help="MLflow experiments").add_subparsers(
        dest="subcommand", required=True
    )
    experiments.add_parser("list").set_defaults(func=cmd_experiments_list)
    p = experiments.add_parser("create")
    p.add_argument("name")
    p.set_defaults(func=cmd_experiments_create)

    plugins = sub.add_parser("plugins", help="browse the plugin registry").add_subparsers(
        dest="subcommand", required=True
    )
    p = plugins.add_parser("list")
    p.add_argument("category", nargs="?", help="limit to one category")
    p.add_argument("--params", action="store_true", help="also list run() parameters")
    p.set_defaults(func=cmd_plugins_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api = Api(args.base_url)
    try:
        return args.func(api, args)
    except CliError as exc:
        eprint(f"error: {exc}")
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
