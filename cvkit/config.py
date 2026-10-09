"""YAML configs with one small superpower: `base:` inheritance.

    # tasks/classification/configs/quick.yaml
    base: default.yaml          # start from default.yaml ...
    train:
      epochs: 2                 # ... and override only what differs

Command-line overrides use dotted keys:  --set train.epochs=3 data.root=data/mine
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Return `base` updated with `override`, merging nested dicts key by key."""
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Read a YAML config (following `base:` chains) and apply `key.sub=value` overrides.

    Pure function: it never touches the disk beyond reading YAML. Output
    folders are created later, by whatever actually writes into them.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    for item in overrides or []:
        apply_override(cfg, item)
    return cfg


def apply_override(cfg: dict, item: str) -> None:
    """Apply one `a.b.c=value` override in place. Values are parsed as YAML (so 3 -> int)."""
    if "=" not in item:
        raise ValueError(f"override must look like key.sub=value, got '{item}'")
    dotted, raw = item.split("=", 1)
    keys = dotted.strip().split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = yaml.safe_load(raw)


def output_dir(cfg: dict) -> Path:
    """Where this run writes checkpoints, history and reports (not created here)."""
    return Path(cfg.get("output", {}).get("dir", "outputs"))
