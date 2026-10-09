"""Train an object detector.

    python -m cvkit detection train --config tasks/detection/configs/default.yaml

The training loss is the sum of four parts, all logged per epoch in history.csv:
  loss_objectness / loss_rpn_box_reg   did the proposal stage find object-like regions?
  loss_classifier / loss_box_reg       did the second stage classify and tighten them?
best.pt = the epoch with the highest validation mAP (averaged over IoU 0.50..0.95).
"""
from __future__ import annotations

from cvkit.detection import box_map, build_coco_loaders, detection_train_step, predict_loader
from cvkit.engine import TrainSpec, fit
from cvkit.utils import get_device, seed_everything

from .model import build_model


def train(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    loaders = build_coco_loaders(cfg)
    classes = loaders["classes"]
    print(f"arch: {cfg['model']['arch']} | classes ({len(classes)}): {classes}")
    model = build_model(cfg, {"classes": classes}).to(device)

    def evaluate_loader(model, loader, device):
        preds, gts = predict_loader(model, loader, device)
        m = box_map(preds, gts, len(classes))
        return {k: m[k] for k in ("map", "map50", "map75")}

    spec = TrainSpec(train_step=detection_train_step, evaluate=evaluate_loader, monitor="map",
                     mode="max", meta={"classes": list(classes)})
    return fit(cfg, model, loaders["train"], loaders["val"], spec, device)
