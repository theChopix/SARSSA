"""Human-readable naming helpers for plugins and pipeline steps.

Centralises the ways a dotted plugin module path (and a parent pipeline
run) is turned into a label so the plugin registry and the pipeline
engine share one definition.
"""

import datetime

from app.config.config import PLUGIN_CATEGORIES

_PIPELINE_RUN_TS_FORMAT = "%d/%m/%Y | %H:%M"


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


_DATASET_LOADER_SUFFIX = "_loader"


def make_dataset_label(plugin_module_path: str) -> str:
    """Derive a short dataset label from a dataset-loader module path.

    Takes the implementation directory segment and strips a trailing
    ``_loader`` suffix, preserving the original casing.  So
    ``dataset_loading.movieLens_loader.movieLens_loader`` becomes
    ``movieLens``.  Used to populate the MLflow UI ``Dataset`` column
    for a pipeline run.

    Args:
        plugin_module_path: Dotted plugin module path
            (e.g. ``dataset_loading.movieLens_loader.movieLens_loader``).

    Returns:
        str: Short dataset label (e.g. ``movieLens``).
    """
    impl_name = plugin_module_path.split(".")[-2]
    if impl_name.endswith(_DATASET_LOADER_SUFFIX):
        impl_name = impl_name[: -len(_DATASET_LOADER_SUFFIX)]
    return impl_name


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


def format_pipeline_run_name(
    user_pipeline_name: str,
    now: datetime.datetime,
) -> str:
    """Build the display name for a parent pipeline run.

    Produces a timestamped label, optionally including a user-provided
    pipeline name:

    - With a name: ``Pipeline Run | <name> [ 31/05/2026 | 14:07 ]``
    - Without:     ``Pipeline Run [ 31/05/2026 | 14:07 ]``

    A blank or whitespace-only ``user_pipeline_name`` yields the no-name
    form.  ``now`` is taken as an argument (rather than read from the
    clock) so the function stays pure and trivially testable.

    Args:
        user_pipeline_name: Optional user-supplied label for the run.
        now: Timestamp to render into the name.

    Returns:
        str: The formatted parent-run name.
    """
    stamp = now.strftime(_PIPELINE_RUN_TS_FORMAT)
    name = user_pipeline_name.strip()
    label = f" | {name}" if name else ""
    return f"Pipeline Run{label} [ {stamp} ]"
