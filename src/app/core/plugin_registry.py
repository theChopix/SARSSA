import importlib
import inspect
from pathlib import Path
from typing import Any

from app.core.plugin_categories import PLUGIN_CATEGORIES


def _python_type_to_str(annotation) -> str:
    """Convert a Python type annotation to a simple string for the frontend."""
    type_map = {
        int: "int",
        float: "float",
        str: "str",
        bool: "bool",
    }
    return type_map.get(annotation, "str")


def _introspect_run_params(plugin_cls) -> list[dict[str, Any]]:
    """Extract parameter metadata from a Plugin.run() method signature.

    Skips 'self' and 'context' parameters since those are internal.
    Returns a list of dicts with keys: name, type, default.
    """
    sig = inspect.signature(plugin_cls.run)
    params = []
    for name, param in sig.parameters.items():
        if name in ("self", "context"):
            continue

        param_type = "str"
        if param.annotation != inspect.Parameter.empty:
            param_type = _python_type_to_str(param.annotation)

        default = None
        if param.default != inspect.Parameter.empty:
            default = param.default

        params.append(
            {
                "name": name,
                "type": param_type,
                "default": default,
            }
        )

    return params


def _discover_implementations(category: str) -> list[dict[str, Any]]:
    """Walk plugins/<category>/ to find all Plugin classes and introspect them.

    The convention is: plugins/<category>/<impl_name>/<impl_name>.py
    contains a class named Plugin.
    """
    plugins_root = Path(__file__).resolve().parent.parent.parent / "plugins"
    category_dir = plugins_root / category

    if not category_dir.is_dir():
        return []

    implementations = []

    for entry in sorted(category_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue

        # Look for <impl_name>/<impl_name>.py or recurse one more level
        # for nested structures like labeling_evaluation/single/dendrogram/dendrogram.py
        impl_modules = _find_plugin_modules(category, entry, category_dir)
        implementations.extend(impl_modules)

    return implementations


def _find_plugin_modules(
    category: str, directory: Path, category_dir: Path
) -> list[dict[str, Any]]:
    """Recursively find plugin modules in a directory.

    Handles both flat (training_cfm/elsa_trainer/elsa_trainer.py) and
    nested (labeling_evaluation/single/dendrogram/dendrogram.py) structures.
    """
    results = []

    # Check if this directory contains a module with the same name (leaf plugin)
    module_file = directory / f"{directory.name}.py"
    if module_file.exists():
        # Build the dotted plugin path relative to plugins/<category>/
        rel_path = module_file.relative_to(category_dir)
        # e.g. elsa_trainer/elsa_trainer.py -> elsa_trainer.elsa_trainer
        parts = list(rel_path.with_suffix("").parts)
        plugin_path = f"{category}.{'.'.join(parts)}"

        display_name = _make_display_name(parts[-1])

        # Derive group from intermediate directory for nested plugins
        # e.g. single/dendrogram/dendrogram.py -> group = "single"
        # e.g. elsa_trainer/elsa_trainer.py -> group = None
        group = parts[0] if len(parts) > 2 else None

        try:
            module = importlib.import_module(f"plugins.{plugin_path}")
            plugin_cls = getattr(module, "Plugin", None)
            if plugin_cls is not None:
                params = _introspect_run_params(plugin_cls)
                results.append(
                    {
                        "plugin_path": plugin_path,
                        "display_name": display_name,
                        "group": group,
                        "params": params,
                    }
                )
        except Exception:
            pass

        return results

    # Otherwise recurse into subdirectories (for nested structures)
    for sub_entry in sorted(directory.iterdir()):
        if sub_entry.is_dir() and not sub_entry.name.startswith("_"):
            results.extend(_find_plugin_modules(category, sub_entry, category_dir))

    return results


def _make_display_name(module_name: str) -> str:
    """Convert a snake_case module name to a human-readable display name.

    Examples:
        elsa_trainer     -> ELSA Trainer
        movieLens_loader -> MovieLens Loader
        sae_steering     -> SAE Steering
        tf_idf           -> TF-IDF
    """
    special_cases = {
        "elsa_trainer": "ELSA Trainer",
        "movieLens_loader": "MovieLens Loader",
        "lastFm1k_loader": "LastFm1k Loader",
        "sae_trainer": "SAE Trainer",
        "sae_steering": "SAE Steering",
        "sae_inspection": "SAE Inspection",
        "tf_idf": "TF-IDF",
    }

    if module_name in special_cases:
        return special_cases[module_name]

    return module_name.replace("_", " ").title()


def get_plugin_registry() -> dict[str, Any]:
    """Build the full plugin registry for the frontend.

    Returns a dict with a 'categories' key containing the ordered list of
    plugin categories, each with their implementations and parameter metadata.
    """
    categories = []

    for category_name, meta in sorted(PLUGIN_CATEGORIES.items(), key=lambda x: x[1]["order"]):
        implementations = _discover_implementations(category_name)
        categories.append(
            {
                "name": category_name,
                "display_name": meta["display_name"],
                "type": meta["type"],
                "order": meta["order"],
                "implementations": implementations,
            }
        )

    return {"categories": categories}
