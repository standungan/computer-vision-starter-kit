"""Train a classifier.

    python -m cvkit classification train --config tasks/classification/configs/default.yaml

Writes outputs/<run>/checkpoints/best.pt (highest val accuracy), last.pt, history.csv.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from cvkit.engine import TrainSpec, fit
from cvkit.utils import get_device, seed_everything

from .data import build_loaders
from .model import build_model


@torch.inference_mode()
def evaluate_loader(model, loader, device) -> dict:
    """Loss and accuracy over a loader (used for validation every epoch)."""
    model.eval()
    ce = nn.CrossEntropyLoss()
    loss_sum, correct, seen = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss_sum += ce(logits, targets).item() * targets.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        seen += targets.size(0)
    return {"loss": loss_sum / max(seen, 1), "acc": correct / max(seen, 1)}


def train(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    loaders = build_loaders(cfg)
    classes = loaders["classes"]
    print(f"arch: {cfg['model']['arch']} | classes ({len(classes)}): {classes}")
    model = build_model(cfg, {"classes": classes}).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"].get("label_smoothing", 0.0))

    def train_step(model, batch, device):
        images, targets = (t.to(device) for t in batch)
        logits = model(images)
        loss = criterion(logits, targets)
        acc = (logits.argmax(1) == targets).float().mean().item()
        return loss, targets.size(0), {"acc": acc}

    spec = TrainSpec(train_step=train_step, evaluate=evaluate_loader, monitor="acc",
                     mode="max", meta={"classes": list(classes)})
    return fit(cfg, model, loaders["train"], loaders["val"], spec, device)
