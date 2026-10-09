"""Pretrained backbones, shared by every task that looks at a whole image.

`build_image_model` is the classic transfer-learning recipe: take a network
pretrained on ImageNet, throw away its final layer, and put a fresh one with
`num_outputs` units in its place:

    classification  -> num_outputs = number of classes
    multilabel      -> num_outputs = number of labels
    regression      -> num_outputs = 1

`resnet_stages` exposes a ResNet as a list of stages (stem, layer1..layer4),
which the segmentation U-Net and the keypoint heatmap model build on.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

ARCHS = ("resnet18", "resnet34", "resnet50",
         "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0")


def _make(arch: str, pretrained: bool) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"unknown arch '{arch}' — choose from {list(ARCHS)}")
    return models.get_model(arch, weights="DEFAULT" if pretrained else None)


def build_image_model(arch: str, num_outputs: int, pretrained: bool = True,
                      freeze_backbone: bool = False, dropout: float = 0.0) -> nn.Module:
    """Backbone + new head. With freeze_backbone=True only the head trains."""
    model = _make(arch, pretrained)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    head_attr = "fc" if hasattr(model, "fc") else "classifier"
    head = getattr(model, head_attr)
    if isinstance(head, nn.Linear):                      # resnets
        new_head = _head(head.in_features, num_outputs, dropout)
    else:                                                # mobilenet / efficientnet:
        layers = list(head.children())                   # keep the hidden layers,
        layers[-1] = _head(layers[-1].in_features, num_outputs, dropout)  # swap the last Linear
        new_head = nn.Sequential(*layers)
    setattr(model, head_attr, new_head)
    for p in getattr(model, head_attr).parameters():     # the whole head is always trainable
        p.requires_grad = True
    return model


def _head(in_features: int, num_outputs: int, dropout: float) -> nn.Module:
    linear = nn.Linear(in_features, num_outputs)
    return nn.Sequential(nn.Dropout(dropout), linear) if dropout > 0 else linear


def resnet_stages(arch: str = "resnet18", pretrained: bool = True) -> tuple[nn.ModuleList, list[int]]:
    """A ResNet cut into stages, and the channel count after each.

    Strides: stem /2 (after conv+bn+relu), then maxpool+layer1 /4, layer2 /8,
    layer3 /16, layer4 /32.
    """
    if not arch.startswith("resnet"):
        raise ValueError("resnet_stages needs a resnet arch (resnet18/34/50)")
    net = _make(arch, pretrained)
    stem = nn.Sequential(net.conv1, net.bn1, net.relu)
    stages = nn.ModuleList([stem, nn.Sequential(net.maxpool, net.layer1),
                            net.layer2, net.layer3, net.layer4])
    widths = [64] + [_out_channels(l) for l in (net.layer1, net.layer2, net.layer3, net.layer4)]
    return stages, widths


def _out_channels(layer: nn.Module) -> int:
    last = layer[-1]
    return last.conv3.out_channels if hasattr(last, "conv3") else last.conv2.out_channels
