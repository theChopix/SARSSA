"""API routes for item enrichment."""

from typing import Any

from fastapi import APIRouter, Query

from app.core.item_enrichment.item_enrichment import enrich_items

router = APIRouter()


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
    items, metadata_available = enrich_items(run_id, item_ids)
    return {"items": items, "metadata_available": metadata_available}
