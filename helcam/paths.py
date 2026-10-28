"""Carpetas de usuario (XDG): configuración propia y caché de modelos."""

from __future__ import annotations

import os
from pathlib import Path


def _xdg(variable: str, fallback: str) -> Path:
    return Path(os.environ.get(variable) or Path.home() / fallback) / "helcam"


CONFIG_DIR = _xdg("XDG_CONFIG_HOME", ".config")
CACHE_DIR = _xdg("XDG_CACHE_HOME", ".cache")
MODEL_DIR = CACHE_DIR / "models"
