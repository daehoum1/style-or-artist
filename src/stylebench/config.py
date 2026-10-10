"""YAML config loading. Paths are resolved relative to the project root."""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "artifacts"


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path) as f:
        return yaml.safe_load(f)


def dataset_config(name: str) -> dict:
    return load_yaml(f"configs/datasets/{name}.yaml")


def split_config(name: str = "default") -> dict:
    return load_yaml(f"configs/splits/{name}.yaml")
