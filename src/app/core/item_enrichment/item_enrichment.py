"""Item enrichment logic.

Loads ``item_metadata.json`` from a dataset-loading MLflow run and
joins requested item IDs with the stored metadata.  Results are
cached per ``run_id`` so repeated calls avoid redundant I/O.
"""

import functools
from typing import Any

from app.utils.logger import logger
from utils.mlflow_manager import MLflowRunLoader


@functools.lru_cache(maxsize=32)
def load_item_metadata(run_id: str) -> dict[str, dict[str, Any]]:
    """Download and cache ``item_metadata.json`` from an MLflow run.

    Args:
        run_id: MLflow run ID that produced the metadata
            artifact.

    Returns:
        dict[str, dict[str, Any]]: Mapping of item ID to
            metadata fields.  Empty dict if the artifact does
            not exist or cannot be loaded.
    """
    try:
        loader = MLflowRunLoader(run_id)
        if not loader.artifact_exists("item_metadata.json"):
            logger.warning(
                "item_metadata.json not found for run %s",
                run_id,
            )
            return {}
        return loader.get_json_artifact("item_metadata.json")
    except Exception:
        logger.exception(
            "Failed to load item_metadata.json for run %s",
            run_id,
        )
        return {}


def load_step_artifact(
    run_id: str,
    filename: str,
) -> Any:
    """Download a JSON artifact from an MLflow run.

    This is a thin proxy so the frontend never needs direct
    MLflow access.  No caching is applied here because the
    set of possible filenames is unbounded.

    Args:
        run_id: MLflow run ID of the step.
        filename: Artifact filename (e.g. ``"steered_recommendations.json"``).

    Returns:
        Any: Parsed JSON content of the artifact.

    Raises:
        FileNotFoundError: If the artifact does not exist.
    """
    loader = MLflowRunLoader(run_id)
    if not loader.artifact_exists(filename):
        raise FileNotFoundError(f"Artifact '{filename}' not found for run {run_id}")
    return loader.get_json_artifact(filename)


def enrich_items(
    run_id: str,
    item_ids: list[str],
) -> tuple[list[dict[str, Any]], bool]:
    """Join item IDs with metadata from the given MLflow run.

    For each requested ID the corresponding metadata entry is
    returned when available.  When metadata is missing for an item
    (or the entire artifact is absent), a minimal fallback record
    ``{"id": <id>, "title": <id>}`` is produced instead.

    Args:
        run_id: MLflow run ID of the dataset-loading step.
        item_ids: Item IDs to enrich.

    Returns:
        tuple[list[dict[str, Any]], bool]: A two-element tuple:
            - Enriched item dicts (one per requested ID, order
              preserved).
            - ``True`` if the metadata artifact was found,
              ``False`` otherwise.
    """
    metadata = load_item_metadata(run_id)
    metadata_available = bool(metadata)

    enriched: list[dict[str, Any]] = []
    for item_id in item_ids:
        entry = metadata.get(item_id)
        if entry is not None:
            enriched.append({"id": item_id, **entry})
        else:
            enriched.append({"id": item_id, "title": item_id})

    return enriched, metadata_available
