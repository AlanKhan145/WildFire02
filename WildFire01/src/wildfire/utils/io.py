"""I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def read_yaml(path: Path) -> Dict[str, Any]:
    """Read YAML file into a dict."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write dict to JSON with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def read_json(path: Path) -> Dict[str, Any]:
    """Read JSON into dict."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
