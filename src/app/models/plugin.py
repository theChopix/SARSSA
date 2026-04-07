"""Pydantic models for plugin-related data."""

from enum import StrEnum

from pydantic import BaseModel


class CategoryType(StrEnum):
    """Allowed execution types for a plugin category."""

    ONE_TIME = "one_time"
    MULTI_RUN = "multi_run"


class CategoryInfo(BaseModel):
    """Metadata for a single plugin category.

    Attributes:
        order: Execution order within the pipeline (0-indexed).
        type: Execution type of the category.
        display_name: Human-readable label for the UI.
    """

    order: int
    type: CategoryType
    display_name: str
