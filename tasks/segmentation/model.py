"""Segmentation models: every one maps (B,3,H,W) images to (B,C,H,W) per-pixel class scores.

unet_resnet18  a U-Net written out in full below: a ResNet18 encoder squeezes the image into
               smaller and smaller feature maps; the decoder grows them back, and at every
               step concatenates the matching encoder map ("skip connection") so fine detail
               that the squeezing lost can be recovered. Read it — it's ~40 lines.
lraspp_*, deeplabv3_*   torchvision's ready-made models, wrapped to return a plain tensor.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from cvkit.backbones import resnet_stages

ARCHS = ["unet_resnet18", "lraspp_mobilenet_v3_large", "deeplabv3_mobilenet_v3_large"]


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Up(nn.Module):
    """Upsample the deeper map to the skip's size, concatenate, then refine with two convs."""

    def __init__(self, cin: int, skip: int, cout: int):
        super().__init__()
        self.block = conv_block(cin + skip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    def __init__(self, num_classes: int, arch: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.encoder, w = resnet_stages(arch, pretrained)          # strides /2 /4 /8 /16 /32
        self.up4 = Up(w[4], w[3], 256)
        self.up3 = Up(256, w[2], 128)
        self.up2 = Up(128, w[1], 64)
        self.up1 = Up(64, w[0], 32)
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        size, skips = x.shape[-2:], []
        for stage in self.encoder:
            x = stage(x)
            skips.append(x)
        x = self.up4(skips[4], skips[3])
        x = self.up3(x, skips[2])
        x = self.up2(x, skips[1])
        x = self.up1(x, skips[0])
        return F.interpolate(self.head(x), size=size, mode="bilinear", align_corners=False)


class DictOutput(nn.Module):
    """torchvision segmentation models return {"out": logits}; unwrap to a tensor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


def build_model(cfg: dict, meta: dict) -> nn.Module:
    m = cfg["model"]
    arch, pretrained, n = m["arch"], m.get("pretrained", True), len(meta["classes"])
    backbone = "DEFAULT" if pretrained else None
    if arch == "unet_resnet18":
        return UNet(n, "resnet18", pretrained)
    if arch == "lraspp_mobilenet_v3_large":
        return DictOutput(models.segmentation.lraspp_mobilenet_v3_large(
            weights=None, weights_backbone=backbone, num_classes=n))
    if arch == "deeplabv3_mobilenet_v3_large":
        return DictOutput(models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=None, weights_backbone=backbone, num_classes=n, aux_loss=False))
    raise ValueError(f"unknown arch '{arch}' — choose from {ARCHS}")
