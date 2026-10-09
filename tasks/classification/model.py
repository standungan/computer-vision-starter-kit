"""Classification model: a pretrained backbone with a fresh `num_classes` head."""
from __future__ import annotations

from cvkit.backbones import build_image_model


def build_model(cfg: dict, meta: dict):
    m = cfg["model"]
    return build_image_model(m["arch"], num_outputs=len(meta["classes"]),
                             pretrained=m.get("pretrained", True),
                             freeze_backbone=m.get("freeze_backbone", False),
                             dropout=m.get("dropout", 0.0))
