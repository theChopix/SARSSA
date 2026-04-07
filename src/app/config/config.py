"""Application configuration loaded from config.yaml."""

from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _load_config() -> dict[str, Any]:
    """Load and parse the application config file.

    Returns:
        dict[str, Any]: Parsed YAML configuration.
    """
    with open(_CONFIG_PATH) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


_config = _load_config()

EXPERIMENT_NAME: str = _config["mlflow"]["experiment_name"]
