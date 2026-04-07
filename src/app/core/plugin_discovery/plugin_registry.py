"""Plugin discovery and registry logic.

Walks the ``src/plugins/`` directory tree, imports each plugin module,
introspects the ``Plugin.run()`` signature, and builds a structured
registry consumable by the frontend.
"""

import inspect
from pathlib import Path

from app.config.config import PLUGIN_CATEGORIES
from app.core.plugin_discovery.plugin_manager import PluginManager
from app.models.plugin import (
    CategoryRegistryEntry,
    ImplementationInfo,
    ParameterInfo,
)

PLUGINS_DIR = Path(__file__).resolve().parents[2].parent / "plugins"

SKIP_PARAMS = {"self", "context"}


def _extract_parameters(plugin_module_path: str) -> list[ParameterInfo]:
    """Extract configurable parameters from a plugin's run() method.

    Loads the plugin via ``PluginManager``, inspects the ``run()``
    signature, and returns metadata for every parameter except
    ``self`` and ``context``.

    Args:
        plugin_module_path: Dotted module path relative to ``plugins/``
            (e.g. ``dataset_loading.movieLens_loader.movieLens_loader``).

    Returns:
        list[ParameterInfo]: Ordered list of parameter descriptors.
    """
    plugin_instance = PluginManager.load(plugin_module_path)
    sig = inspect.signature(plugin_instance.run)

    params: list[ParameterInfo] = []
    for name, param in sig.parameters.items():
        if name in SKIP_PARAMS:
            continue

        has_default = param.default is not inspect.Parameter.empty
        type_name = (
            param.annotation.__name__ if param.annotation is not inspect.Parameter.empty else "str"
        )

        params.append(
            ParameterInfo(
                name=name,
                type=type_name,
                default=param.default if has_default else None,
                required=not has_default,
            )
        )

    return params


def _find_plugin_modules(category_dir: Path, prefix: str) -> list[str]:
    """Recursively find plugin module paths under a category directory.

    A valid plugin module is a ``.py`` file whose stem matches its
    parent directory name (e.g. ``elsa_trainer/elsa_trainer.py``).
    Intermediate directories like ``single/`` or ``compare/`` are
    traversed transparently.

    Args:
        category_dir: Absolute path to the category folder
            (e.g. ``src/plugins/dataset_loading/``).
        prefix: Dotted path prefix accumulated so far
            (e.g. ``dataset_loading``).

    Returns:
        list[str]: Dotted module paths suitable for ``PluginManager.load()``.
    """
    modules: list[str] = []

    for child in sorted(category_dir.iterdir()):
        if not child.is_dir() or child.name.startswith(("_", ".")):
            continue

        plugin_file = child / f"{child.name}.py"
        if plugin_file.is_file():
            modules.append(f"{prefix}.{child.name}.{child.name}")
        else:
            modules.extend(_find_plugin_modules(child, f"{prefix}.{child.name}"))

    return modules


def _make_display_name(plugin_module_path: str) -> str:
    """Derive a human-readable display name from a dotted module path.

    Takes the second-to-last segment (the implementation directory name)
    and converts it from snake_case to Title Case.

    Args:
        plugin_module_path: Dotted module path
            (e.g. ``dataset_loading.movieLens_loader.movieLens_loader``).

    Returns:
        str: Human-readable name (e.g. ``Movieles Loader``).
    """
    parts = plugin_module_path.split(".")
    impl_name = parts[-2]
    return impl_name.replace("_", " ").title()


def _discover_implementations(
    category_name: str,
) -> list[ImplementationInfo]:
    """Discover all plugin implementations for a given category.

    Args:
        category_name: Key from ``PLUGIN_CATEGORIES``
            (e.g. ``dataset_loading``).

    Returns:
        list[ImplementationInfo]: All discovered implementations
            with their parameters.
    """
    category_dir = PLUGINS_DIR / category_name
    module_paths = _find_plugin_modules(category_dir, category_name)

    implementations: list[ImplementationInfo] = []
    for module_path in module_paths:
        params = _extract_parameters(module_path)
        implementations.append(
            ImplementationInfo(
                plugin_name=module_path,
                display_name=_make_display_name(module_path),
                params=params,
            )
        )

    return implementations


def get_plugin_registry() -> dict[str, CategoryRegistryEntry]:
    """Build the complete plugin registry for all categories.

    Iterates over ``PLUGIN_CATEGORIES``, discovers implementations
    for each, and assembles a registry keyed by category name.

    Returns:
        dict[str, CategoryRegistryEntry]: Full registry mapping
            category names to their metadata and implementations.
    """
    registry: dict[str, CategoryRegistryEntry] = {}

    for category_name, category_info in PLUGIN_CATEGORIES.items():
        implementations = _discover_implementations(category_name)
        registry[category_name] = CategoryRegistryEntry(
            category_info=category_info,
            implementations=implementations,
        )

    return registry
