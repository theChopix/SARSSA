"""API routes for plugin discovery and registry."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.core.plugin_discovery.plugin_manager import PluginManager
from app.core.plugin_discovery.plugin_registry import get_plugin_registry
from plugins.plugin_interface import DynamicDropdownHint
from utils.mlflow_manager import MLflowRunLoader

router = APIRouter()


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
    run_id: str = Query(
        ...,
        description="MLflow run ID of the step that produced the source artifact.",
    ),
) -> list[dict[str, str]]:
    """Return dynamic dropdown options for a plugin parameter.

    Loads the artifact declared in the plugin's
    ``DynamicDropdownHint``, passes it through the plugin's
    formatter method, and returns the result.

    Args:
        category: Plugin category key (e.g. ``"steering"``).
        plugin_name: Dotted plugin module path.
        param_name: Name of the ``run()`` parameter.
        run_id: MLflow run ID for the upstream step that
            produced the source artifact.

    Returns:
        list[dict[str, str]]: Each dict has ``"label"`` and
            ``"value"`` keys.

    Raises:
        HTTPException: 404 if the plugin, parameter hint, or
            artifact cannot be found.
    """
    try:
        plugin_instance = PluginManager.load(plugin_name)
    except (ImportError, AttributeError) as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Plugin '{plugin_name}' not found: {exc}",
        ) from exc

    hint = _find_dropdown_hint(plugin_instance, param_name)
    if hint is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No DynamicDropdownHint for parameter '{param_name}' on plugin '{plugin_name}'."
            ),
        )

    artifact_data = _load_hint_artifact(hint, run_id)
    formatter = getattr(plugin_instance.__class__, hint.formatter, None)
    if formatter is None:
        raise HTTPException(
            status_code=404,
            detail=(f"Formatter '{hint.formatter}' not found on plugin '{plugin_name}'."),
        )

    return formatter(artifact_data)


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


def _load_hint_artifact(
    hint: DynamicDropdownHint,
    run_id: str,
) -> Any:
    """Load the artifact referenced by a DynamicDropdownHint.

    Args:
        hint: The dropdown hint specifying the artifact.
        run_id: MLflow run ID to load the artifact from.

    Returns:
        Any: The loaded artifact data.

    Raises:
        HTTPException: 404 if the artifact cannot be loaded.
    """
    loader = MLflowRunLoader(run_id)
    try:
        match hint.artifact_loader:
            case "json":
                return loader.get_json_artifact(hint.artifact_file)
            case "npy":
                return loader.get_npy_artifact(hint.artifact_file)
            case "npz":
                return loader.get_npz_artifact(hint.artifact_file)
            case _:
                raise ValueError(f"Unsupported artifact loader: '{hint.artifact_loader}'")
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=(f"Failed to load artifact '{hint.artifact_file}' from run '{run_id}': {exc}"),
        ) from exc
