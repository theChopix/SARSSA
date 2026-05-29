"""Human-readable naming helpers for plugins and pipeline steps.

Centralises the two ways a dotted plugin module path is turned into a
label so the plugin registry and the pipeline engine share one
definition.
"""

from app.config.config import PLUGIN_CATEGORIES


def make_plugin_display_name(plugin_module_path: str) -> str:
    """Derive a human-readable display name from a dotted module path.

    Takes the second-to-last segment (the implementation directory name)
    and converts it from snake_case to Title Case.

    Args:
        plugin_module_path: Dotted module path
            (e.g. ``dataset_loading.movieLens_loader.movieLens_loader``).

    Returns:
        str: Human-readable name (e.g. ``Movielens Loader``).
    """
    parts = plugin_module_path.split(".")
    impl_name = parts[-2]
    return impl_name.replace("_", " ").title()


def format_step_run_name(
    plugin_name: str,
    plugin_display_name: str,
    execution_order: int,
) -> str:
    """Build the display name for a nested (plugin execution) MLflow run.

    Produces ``[<execution_order>] <category> / <plugin>``, e.g.
    ``[5] Labeling Evaluation / Embedding Map``.  The bracket number is
    the execution sequence of the step within its parent pipeline run
    (not the category's static config order).

    Args:
        plugin_name: Dotted plugin module path
            (e.g. ``dataset_loading.movieLens_loader.movieLens_loader``).
        plugin_display_name: Human-readable plugin name.
        execution_order: 1-based position of this step within the
            parent pipeline run.

    Returns:
        str: The formatted nested-run name.  Falls back to the raw
            category key when the category is unknown.
    """
    category = plugin_name.split(".")[0]
    info = PLUGIN_CATEGORIES.get(category)
    category_label = info.display_name if info is not None else category
    return f"[{execution_order}] {category_label} / {plugin_display_name}"
