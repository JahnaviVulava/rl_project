"""Configuration loading with recursive mode-specific overrides."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(mode: str = "default") -> dict[str, Any]:
    """Return default settings optionally overlaid by ``review`` or ``live``."""
    with (PROJECT_ROOT / "config" / "default.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if mode != "default":
        path = PROJECT_ROOT / "config" / f"{mode}.yaml"
        if not path.exists():
            raise ValueError(f"Unknown configuration mode: {mode}")
        with path.open(encoding="utf-8") as stream:
            config = _merge(config, yaml.safe_load(stream))
    return config


def project_path(relative_path: str) -> Path:
    """Resolve a configuration path relative to the repository root."""
    return PROJECT_ROOT / relative_path
