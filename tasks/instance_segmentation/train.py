"""Train an instance-segmentation model.

    python -m cvkit instance_segmentation train --config tasks/instance_segmentation/configs/default.yaml

Training loss = detection losses + `loss_mask` (per-pixel cross-entropy inside each box).
best.pt = highest validation *mask* mAP — the overlap of predicted and true masks, which is
stricter than box overlap because it checks the shape, not just the bounding rectangle.
"""
from __future__ import annotations

from cvkit.detection import box_map, detection_train_step, mask_map, predict_loader
from cvkit.engine import TrainSpec, fit
from cvkit.utils import get_device, seed_everything

from .data import build_loaders
from .model import build_model


def validation_metrics(model, loader, device, num_classes: int, mask_threshold: float = 0.5) -> dict:
    preds, gts = predict_loader(model, loader, device, with_masks=True, mask_threshold=mask_threshold)
    b, k = box_map(preds, gts, num_classes), mask_map(preds, gts, num_classes)
    return {"mask_map": k["map"], "mask_map50": k["map50"], "box_map": b["map"], "box_map50": b["map50"]}


def train(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    loaders = build_loaders(cfg)
    classes = loaders["classes"]
    print(f"arch: {cfg['model']['arch']} | classes ({len(classes)}): {classes}")
    model = build_model(cfg, {"classes": classes}).to(device)
    mthr = float(cfg.get("eval", {}).get("mask_threshold", 0.5))

    spec = TrainSpec(train_step=detection_train_step,
                     evaluate=lambda m, l, d: validation_metrics(m, l, d, len(classes), mthr),
                     monitor="mask_map", mode="max", meta={"classes": list(classes)})
    return fit(cfg, model, loaders["train"], loaders["val"], spec, device)
