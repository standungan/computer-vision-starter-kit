"""Save / load model checkpoints.

A checkpoint is self-describing: it stores the weights, the config the model
was trained with, and task metadata (class names, target statistics, ...).
Evaluation, inference and the apps rebuild the model from *the checkpoint's*
config, so editing the YAML after training can't silently break them.

Loading uses `weights_only=True`: a checkpoint file can't run code on load.
Keep `meta` to plain Python types (str / int / float / list / dict).
"""
from __future__ import annotations

from pathlib import Path

import torch


def checkpoint_path(cfg: dict, name: str = "best.pt") -> Path:
    return Path(cfg.get("output", {}).get("dir", "outputs")) / "checkpoints" / name


def save_checkpoint(path: Path, model, cfg: dict, meta: dict, epoch: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "meta": meta,
                "epoch": epoch, "metrics": metrics}, path)


def load_checkpoint(path: str | Path, device="cpu") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {path}\n"
            "Train a model first:  python -m cvkit <task> train --config <config.yaml>"
        )
    return torch.load(path, map_location=device, weights_only=True)


def restore_model(ckpt: dict, build_model, device) -> torch.nn.Module:
    """Rebuild the model from the checkpoint's own config (no pretrained download)."""
    cfg = ckpt["config"]
    cfg = {**cfg, "model": {**cfg["model"], "pretrained": False}}
    model = build_model(cfg, ckpt["meta"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()
