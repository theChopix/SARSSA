"""Pydantic models for plugin-related data."""

from enum import StrEnum
from typing import Any

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


class ParameterInfo(BaseModel):
    """Schema for a single plugin parameter.

    Attributes:
        name: Parameter name as it appears in the run() signature.
        type: Python type name (e.g. ``int``, ``float``, ``str``).
        default: Default value, or None if the parameter is required.
        required: Whether the parameter must be provided by the user.
    """

    name: str
    type: str
    default: Any | None = None
    required: bool = True


class ImplementationInfo(BaseModel):
    """Schema for a discovered plugin implementation.

    Attributes:
        plugin_name: Dotted module path used by PluginManager.load()
            (e.g. ``dataset_loading.movieLens_loader.movieLens_loader``).
        display_name: Human-readable name derived from the module name.
        params: List of configurable parameters from the run() signature.
    """

    plugin_name: str
    display_name: str
    params: list[ParameterInfo]


class CategoryRegistryEntry(BaseModel):
    """Full registry entry for a single plugin category.

    Attributes:
        category_info: Static metadata for the category.
        implementations: Discovered plugin implementations.
    """

    category_info: CategoryInfo
    implementations: list[ImplementationInfo]
