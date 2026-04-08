"""Centralised logger for the SARSSA backend.

All modules under ``src/app`` should import ``logger`` from here.
"""

import logging
from pathlib import Path

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
_LOG_FILE = Path(__file__).resolve().parents[3] / "sarssa.log"

logger = logging.getLogger("sarssa")
logger.setLevel(logging.DEBUG)

# Console handler
_console = logging.StreamHandler()
_console.setLevel(logging.DEBUG)
_console.setFormatter(logging.Formatter(_LOG_FORMAT))
logger.addHandler(_console)

# File handler
_file = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_file.setLevel(logging.DEBUG)
_file.setFormatter(logging.Formatter(_LOG_FORMAT))
logger.addHandler(_file)
