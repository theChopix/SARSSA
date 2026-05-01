"""API routes for item enrichment and artifact access."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.core.item_enrichment.item_enrichment import enrich_items, load_step_artifact
from app.utils.logger import logger

router = APIRouter()


@router.get("/artifact")
def get_step_artifact(
    run_id: str = Query(..., description="MLflow run ID of the step"),
    filename: str = Query(..., description="Artifact filename to download"),
) -> Any:
    """Download a JSON artifact from any MLflow run.

    Acts as a proxy so the frontend does not need direct MLflow
    access.

    Args:
        run_id: MLflow run ID of the plugin step.
        filename: Name of the JSON artifact file.

    Returns:
        Any: Parsed JSON content of the artifact.

    Raises:
        HTTPException: 404 if the artifact does not exist.
    """
    try:
        return load_step_artifact(run_id, filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error fetching artifact %s for run %s", filename, run_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/enrich")
def get_enriched_items(
    run_id: str = Query(..., description="MLflow run ID"),
    ids: str = Query("", description="Comma-separated item IDs"),
) -> dict[str, Any]:
    """Enrich item IDs with metadata from a dataset-loading run.

    Args:
        run_id: MLflow run ID that produced item_metadata.json.
        ids: Comma-separated list of item IDs to enrich.

    Returns:
        dict[str, Any]: ``{"items": [...], "metadata_available": bool}``.
    """
    item_ids = [i.strip() for i in ids.split(",") if i.strip()] if ids else []
    try:
        items, metadata_available = enrich_items(run_id, item_ids)
    except Exception as exc:
        logger.exception("Unexpected error enriching items for run %s", run_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"items": items, "metadata_available": metadata_available}
