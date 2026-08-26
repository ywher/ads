from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config with recursive ``base`` inheritance."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}

    bases = current.pop("base", [])
    if isinstance(bases, str):
        bases = [bases]

    merged: Dict[str, Any] = {}
    for base in bases:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = _merge(merged, load_config(base_path))

    merged = _merge(merged, current)
    merged = _expand(merged)
    merged["_config_path"] = str(path)
    merged["_repo_root"] = str(REPO_ROOT)
    return merged


def resolve_repo_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def dump_config(config: Dict[str, Any], path: str | Path) -> None:
    serializable = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False)

