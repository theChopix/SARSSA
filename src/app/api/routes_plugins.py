"""API routes for plugin discovery and registry."""

import threading
from collections import OrderedDict
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.core.param_choices import resolve_dependent_choices
from app.core.pipeline_runs import get_run_context
from app.core.plugin_discovery.plugin_manager import PluginManager
from app.core.plugin_discovery.plugin_registry import get_plugin_registry
from plugins.plugin_interface import (
    DependentDropdownHint,
    DropdownArtifactSpec,
    DynamicDropdownHint,
)
from utils.mlflow_manager import MLflowRunLoader

router = APIRouter()

# formatted-options cache for server_search dropdowns, keyed by
# (plugin_name, param_name, run_id)
_OPTIONS_CACHE: OrderedDict[tuple[str, str, str], list[dict[str, Any]]] = OrderedDict()
_OPTIONS_CACHE_MAX = 4
_OPTIONS_CACHE_LOCK = threading.Lock()


@router.get("/registry")
def get_registry() -> dict[str, Any]:
    """Return the full plugin registry.

    Returns:
        dict[str, Any]: A mapping of category keys to
            ``CategoryRegistryEntry`` data (serialised via Pydantic).
    """
    registry = get_plugin_registry()
    return {key: entry.model_dump() for key, entry in registry.items()}


@router.get(
    "/param-choices/{category}/{plugin_name:path}/{param_name}",
)
def get_param_choices(
    category: str,  # noqa: ARG001
    plugin_name: str,
    param_name: str,
    run_id: str | None = Query(
        None,
        description="MLflow run ID of the step that produced the source artifact.",
    ),
    value: str | None = Query(
        None,
        description="Current value of the controlling param (dependent dropdowns).",
    ),
    search: str | None = Query(
        None,
        description="Substring filter on option labels (server_search dropdowns).",
    ),
    limit: int = Query(
        200,
        ge=1,
        le=1000,
        description="Maximum options returned by a server_search dropdown.",
    ),
    sort_by_tint: bool = Query(
        False,
        description="Sort options by tint descending before limiting (server_search dropdowns).",
    ),
) -> list[dict[str, Any]] | dict[str, Any]:
    """Return dropdown options for a plugin parameter.

    For a ``DependentDropdownHint`` the options are computed from the
    controlling param's *value*. Otherwise the parameter's
    ``DynamicDropdownHint`` artifact(s) are loaded from *run_id* and
    passed through the plugin's formatter method.

    Args:
        category: Plugin category key (e.g. ``"steering"``).
        plugin_name: Dotted plugin module path.
        param_name: Name of the ``run()`` parameter.
        run_id: MLflow run ID for the upstream step that
            produced the source artifact (dynamic dropdowns).
        value: Current value of the controlling param
            (dependent dropdowns).
        search: Case-insensitive substring filter on option labels
            (only applied for ``server_search`` hints).
        limit: Page size for ``server_search`` hints.
        sort_by_tint: Sort the (filtered) options by tint descending
            before applying *limit*, so the page holds the global top.

    Returns:
        list[dict[str, Any]] | dict[str, Any]: For plain dropdowns a
            list of option dicts (``"label"``/``"value"`` keys, plus
            optional ``"tint"``/``"emphasis"``). For ``server_search``
            dropdowns an envelope ``{"options": [...], "total": int}``
            where *total* counts all options matching *search*.

    Raises:
        HTTPException: 404 if the plugin, parameter hint, resolver, or
            artifact cannot be found; 422 if ``run_id`` is missing for
            a dynamic dropdown.
    """
    try:
        plugin_instance = PluginManager.load(plugin_name)
    except (ImportError, AttributeError) as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Plugin '{plugin_name}' not found: {exc}",
        ) from exc

    dependent_hint = _find_dependent_hint(plugin_instance, param_name)
    if dependent_hint is not None:
        try:
            return resolve_dependent_choices(dependent_hint.resolver, value)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No resolver '{dependent_hint.resolver}' for parameter "
                    f"'{param_name}' on plugin '{plugin_name}'."
                ),
            ) from exc

    hint = _find_dropdown_hint(plugin_instance, param_name)
    if hint is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No DynamicDropdownHint for parameter '{param_name}' on plugin '{plugin_name}'."
            ),
        )

    if run_id is None:
        raise HTTPException(
            status_code=422,
            detail=f"Query param 'run_id' is required for parameter '{param_name}'.",
        )

    resolved_run_id = _resolve_artifact_run_id(hint, run_id)
    formatter = getattr(plugin_instance.__class__, hint.formatter, None)
    if formatter is None:
        raise HTTPException(
            status_code=404,
            detail=(f"Formatter '{hint.formatter}' not found on plugin '{plugin_name}'."),
        )

    if not hint.server_search:
        return formatter(_load_hint_artifact(hint, resolved_run_id))

    options = _cached_options(plugin_name, param_name, resolved_run_id, hint, formatter)
    if search:
        needle = search.lower()
        options = [o for o in options if needle in str(o.get("label", "")).lower()]
    if sort_by_tint:
        options = sorted(
            options,
            key=lambda o: -(o["tint"] if o.get("tint") is not None else float("-inf")),
        )
    return {"options": options[:limit], "total": len(options)}


def _cached_options(
    plugin_name: str,
    param_name: str,
    run_id: str,
    hint: DynamicDropdownHint,
    formatter: Any,
) -> list[dict[str, Any]]:
    """Return the full formatted option list, cached per run.

    Args:
        plugin_name: Dotted plugin module path.
        param_name: Name of the ``run()`` parameter.
        run_id: Resolved run id holding the source artifact(s).
        hint: The dropdown hint.
        formatter: The plugin's formatter static method.

    Returns:
        list[dict[str, Any]]: All formatted options.
    """
    key = (plugin_name, param_name, run_id)
    with _OPTIONS_CACHE_LOCK:
        cached = _OPTIONS_CACHE.get(key)
        if cached is not None:
            _OPTIONS_CACHE.move_to_end(key)
            return cached

    options = formatter(_load_hint_artifact(hint, run_id))
    with _OPTIONS_CACHE_LOCK:
        _OPTIONS_CACHE[key] = options
        if len(_OPTIONS_CACHE) > _OPTIONS_CACHE_MAX:
            _OPTIONS_CACHE.popitem(last=False)
    return options


def _find_dropdown_hint(
    plugin_instance: Any,
    param_name: str,
) -> DynamicDropdownHint | None:
    """Find a DynamicDropdownHint for a parameter on a plugin.

    Args:
        plugin_instance: An instantiated plugin object.
        param_name: Name of the ``run()`` parameter.

    Returns:
        DynamicDropdownHint | None: The matching hint, or
            ``None`` if no dropdown hint exists for this param.
    """
    for hint in plugin_instance.io_spec.param_ui_hints:
        if isinstance(hint, DynamicDropdownHint) and hint.param_name == param_name:
            return hint
    return None


def _find_dependent_hint(
    plugin_instance: Any,
    param_name: str,
) -> DependentDropdownHint | None:
    """Find a DependentDropdownHint for a parameter on a plugin.

    Args:
        plugin_instance: An instantiated plugin object.
        param_name: Name of the ``run()`` parameter.

    Returns:
        DependentDropdownHint | None: The matching hint, or
            ``None`` if no dependent dropdown hint exists for this param.
    """
    for hint in plugin_instance.io_spec.param_ui_hints:
        if isinstance(hint, DependentDropdownHint) and hint.param_name == param_name:
            return hint
    return None


def _resolve_artifact_run_id(
    hint: DynamicDropdownHint,
    run_id: str,
) -> str:
    """Resolve the run id that holds the dropdown's source artifact.

    For non-cascading hints (``hint.source_run_param is None``) the
    caller already supplied the per-step run id, so it is returned
    unchanged.

    For cascading hints, *run_id* is the parent pipeline run id
    selected via a :class:`PastRunsDropdownHint`.  This function
    loads that run's ``context.json`` and returns the per-step
    run id recorded for ``hint.artifact_step``.

    Args:
        hint: The dropdown hint declaring how to resolve the source.
        run_id: The run id supplied by the frontend.

    Returns:
        str: The run id that directly holds the source artifact.

    Raises:
        HTTPException: 404 if the parent run's context cannot be
            loaded or does not contain the requested step.
    """
    if hint.source_run_param is None:
        return run_id

    try:
        context = get_run_context(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Parent run '{run_id}' has no context.json: {exc}",
        ) from exc

    step_entry = context.get(hint.artifact_step)
    if not isinstance(step_entry, dict) or "run_id" not in step_entry:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Parent run '{run_id}' context is missing step "
                f"'{hint.artifact_step}' or its run_id."
            ),
        )

    return str(step_entry["run_id"])


def _load_hint_artifact(
    hint: DynamicDropdownHint,
    run_id: str,
) -> Any:
    """Load the artifact(s) referenced by a DynamicDropdownHint.

    Args:
        hint: The dropdown hint specifying the artifact(s).
        run_id: MLflow run ID to load the artifact(s) from.

    Returns:
        Any: The loaded artifact data — a single object for the
            single-file form, a tuple (in ``artifact_files`` order)
            for the multi-file form.

    Raises:
        HTTPException: 404 if an artifact cannot be loaded.
    """
    loader = MLflowRunLoader(run_id)
    specs = hint.artifact_files or [
        DropdownArtifactSpec(file=hint.artifact_file, loader=hint.artifact_loader)
    ]
    loaded = []
    for spec in specs:
        try:
            match spec.loader:
                case "json":
                    loaded.append(loader.get_json_artifact(spec.file, **spec.kwargs))
                case "npy":
                    loaded.append(loader.get_npy_artifact(spec.file, **spec.kwargs))
                case "npz":
                    loaded.append(loader.get_npz_artifact(spec.file, **spec.kwargs))
                case _:
                    raise ValueError(f"Unsupported artifact loader: '{spec.loader}'")
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail=(f"Failed to load artifact '{spec.file}' from run '{run_id}': {exc}"),
            ) from exc
    return tuple(loaded) if hint.artifact_files else loaded[0]
