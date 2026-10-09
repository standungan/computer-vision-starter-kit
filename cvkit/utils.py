"""Small helpers shared by every task: seeding, device choice, parameter counts."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 42) -> None:
    """Make runs repeatable (as far as the hardware allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(preference: str = "auto") -> torch.device:
    """'auto' picks CUDA, then Apple MPS, then CPU. Or force 'cpu' / 'cuda' / 'mps'."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable,
            "trainable_pct": 100 * trainable / max(total, 1)}
