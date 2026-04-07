"""API routes for plugin discovery and registry."""

from typing import Any

from fastapi import APIRouter

from app.core.plugin_discovery.plugin_registry import get_plugin_registry

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
