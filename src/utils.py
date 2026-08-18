"""Shared helpers: config loading, reproducibility, device selection.

Every entry point (train / evaluate / inference / app) goes through
`load_config`, so the YAML file stays the single source of truth.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    """Read the YAML config and resolve output paths."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = Path(cfg.get("output", {}).get("dir", "outputs"))
    cfg.setdefault("output", {})["dir"] = str(out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    return cfg


def seed_everything(seed: int = 42) -> None:
    """Make runs repeatable (as far as cuDNN allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def checkpoint_path(cfg: dict, name: str = "best.pt") -> Path:
    return Path(cfg["output"]["dir"]) / "checkpoints" / name
