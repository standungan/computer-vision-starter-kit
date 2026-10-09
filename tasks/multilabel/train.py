"""Train a multi-label tagger.

    python -m cvkit multilabel train --config tasks/multilabel/configs/default.yaml

Loss is BCEWithLogits: every label gets its own independent sigmoid, instead of the
softmax that makes classes compete in single-label classification. The checkpoint
chosen as best.pt is the one with the highest mAP (threshold-free) on the val split.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from cvkit.engine import TrainSpec, fit
from cvkit.utils import get_device, seed_everything

from .data import build_loaders
from .metrics import multilabel_metrics
from .model import build_model


@torch.inference_mode()
def predict_loader(model, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    """Sigmoid probabilities, targets and mean BCE loss over a loader."""
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    probs, targets, loss_sum = [], [], 0.0
    for images, y in loader:
        logits = model(images.to(device))
        loss_sum += bce(logits, y.to(device)).item()
        probs.append(torch.sigmoid(logits).cpu())
        targets.append(y)
    p, t = torch.cat(probs).numpy(), torch.cat(targets).numpy()
    return p, t, loss_sum / max(t.size, 1)


def make_evaluate(threshold: float):
    def evaluate_loader(model, loader, device) -> dict:
        probs, targets, loss = predict_loader(model, loader, device)
        return {"loss": loss, **multilabel_metrics(probs, targets, threshold)}
    return evaluate_loader


def train(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    loaders = build_loaders(cfg)
    classes = loaders["classes"]
    print(f"arch: {cfg['model']['arch']} | labels ({len(classes)}): {classes}")
    model = build_model(cfg, {"classes": classes}).to(device)

    pos_weight = None
    if cfg["train"].get("pos_weight", False):
        pos = torch.tensor(loaders["train_positive_counts"]).clamp(min=1)
        pos_weight = ((loaders["n_train"] - pos) / pos).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    threshold = float(cfg.get("eval", {}).get("threshold", 0.5))

    def train_step(model, batch, device):
        images, y = (t.to(device) for t in batch)
        logits = model(images)
        loss = criterion(logits, y)
        acc = ((logits.sigmoid() >= threshold) == (y > 0.5)).float().mean().item()
        return loss, y.size(0), {"label_acc": acc}

    spec = TrainSpec(train_step=train_step, evaluate=make_evaluate(threshold), monitor="map",
                     mode="max", meta={"classes": list(classes)})
    return fit(cfg, model, loaders["train"], loaders["val"], spec, device)
