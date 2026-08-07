"""YAML configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PACKAGE_ROOT / "configs" / "so0"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping at the top level."""

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def load_default_config() -> dict[str, Any]:
    """Load the frozen SO-0 MVP configuration."""

    return load_yaml(DEFAULT_CONFIG_DIR / "forward_model_mvp.yaml")


def load_thresholds() -> dict[str, Any]:
    """Load acceptance thresholds."""

    return load_yaml(DEFAULT_CONFIG_DIR / "acceptance_thresholds.yaml")


def load_formula_registry() -> dict[str, Any]:
    """Load the formula registry and ensure no empty fields are present."""

    registry = load_yaml(DEFAULT_CONFIG_DIR / "forward_formula_registry.yaml")

    def walk(value: Any, path: str) -> None:
        if value is None or value == "":
            raise ValueError(f"Empty formula registry field: {path}")
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                walk(child, f"{path}[{idx}]")

    walk(registry, "registry")
    return registry
