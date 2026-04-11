"""Application configuration loaded from config.yaml."""

from pathlib import Path
from typing import Any

import yaml

from app.models.plugin import CategoryInfo

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _load_config() -> dict[str, Any]:
    """Load and parse the application config file.

    Returns:
        dict[str, Any]: Parsed YAML configuration.
    """
    with open(_CONFIG_PATH) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def _load_plugin_categories(
    raw: dict[str, Any],
) -> dict[str, CategoryInfo]:
    """Parse the plugin_categories section into typed models.

    Args:
        raw: Raw dict from the YAML ``plugin_categories`` key.

    Returns:
        dict[str, CategoryInfo]: Validated category metadata.
    """
    return {key: CategoryInfo(**value) for key, value in raw.items()}


_config = _load_config()

EXPERIMENT_NAME: str = _config["mlflow"]["experiment_name"]
TRACKING_URI: str = _config["mlflow"]["tracking_uri"]
ARTIFACT_ROOT: str = _config["mlflow"]["artifact_root"]
PLUGIN_CATEGORIES: dict[str, CategoryInfo] = _load_plugin_categories(_config["plugin_categories"])
