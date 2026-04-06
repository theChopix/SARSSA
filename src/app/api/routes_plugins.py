from fastapi import APIRouter

from app.core.plugin_registry import get_plugin_registry

router = APIRouter()


@router.get("/registry")
def plugin_registry():
    return get_plugin_registry()
