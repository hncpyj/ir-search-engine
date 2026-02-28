"""
Config loading utility.
Loads default.yaml, then deep-merges an optional override file (full.yaml).
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str, override_path: str | None = None) -> dict:
    """
    Load a YAML config file and optionally merge an override file on top.

    Args:
        config_path:  Path to the base config (e.g. configs/default.yaml).
        override_path: Optional path to an override config (e.g. configs/full.yaml).

    Returns:
        Merged config dict.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if override_path and Path(override_path).exists():
        with open(override_path) as f:
            override = yaml.safe_load(f)
        if override:
            cfg = _deep_merge(cfg, override)

    return cfg
