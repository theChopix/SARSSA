"""Pydantic models for item enrichment endpoints."""

from pydantic import BaseModel, Field


class EnrichItemsRequest(BaseModel):
    """Request body for ``POST /items/enrich``.

    The item ids travel in the body because an interaction history
    can hold thousands of ids — far past proxy URL-length limits.
    """

    run_id: str = Field(description="MLflow run ID that produced item_metadata.json.")
    ids: list[str] = Field(default_factory=list, description="Item IDs to enrich.")
