"""Model factory — swap architectures from YAML, no code edits.

Supported `model.arch` values map to torchvision classifiers with their
final layer replaced to match the number of classes discovered from the
dataset. `freeze_backbone: true` turns the model into a feature extractor
(only the new head trains) — the right starting point for small datasets.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

# arch name -> (constructor, weights enum, name of the classifier attribute)
_ARCHS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, "fc"),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, "fc"),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, "fc"),
    "mobilenet_v3_small": (
        models.mobilenet_v3_small,
        models.MobileNet_V3_Small_Weights.DEFAULT,
        "classifier",
    ),
    "efficientnet_b0": (
        models.efficientnet_b0,
        models.EfficientNet_B0_Weights.DEFAULT,
        "classifier",
    ),
}


def build_model(cfg: dict, num_classes: int) -> nn.Module:
    arch = cfg["model"]["arch"]
    if arch not in _ARCHS:
        raise ValueError(f"unknown arch '{arch}' — choose from {sorted(_ARCHS)}")
    ctor, weights, head_attr = _ARCHS[arch]
    model = ctor(weights=weights if cfg["model"].get("pretrained", True) else None)

    if cfg["model"].get("freeze_backbone", False):
        for p in model.parameters():
            p.requires_grad = False

    dropout = cfg["model"].get("dropout", 0.0)
    head = getattr(model, head_attr)
    if isinstance(head, nn.Linear):  # resnets
        in_f = head.in_features
        new_head = _make_head(in_f, num_classes, dropout)
    else:  # mobilenet / efficientnet: classifier is a Sequential ending in Linear
        in_f = head[-1].in_features
        head = list(head.children())[:-1]
        head.append(_make_head(in_f, num_classes, dropout))
        new_head = nn.Sequential(*head)
    setattr(model, head_attr, new_head)
    return model


def _make_head(in_features: int, num_classes: int, dropout: float) -> nn.Module:
    if dropout > 0:
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return nn.Linear(in_features, num_classes)


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_pct": 100 * trainable / total}
