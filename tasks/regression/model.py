"""Regression model: a pretrained backbone with a single-output head."""
from __future__ import annotations

from cvkit.backbones import build_image_model


def build_model(cfg: dict, meta: dict):
    m = cfg["model"]
    return build_image_model(m["arch"], num_outputs=1, pretrained=m.get("pretrained", True),
                             freeze_backbone=m.get("freeze_backbone", False),
                             dropout=m.get("dropout", 0.0))
