"""Instance-segmentation models: Mask R-CNN.

Mask R-CNN is Faster R-CNN plus a third head: for every detected box it predicts a small
mask of which pixels inside the box belong to the object. So it returns, per object, a box,
a class, a score and a mask. Class 0 is background, hence K + 1 outputs.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d

from cvkit.detection import detector_input_size

ARCHS = ["maskrcnn_resnet18_fpn", "maskrcnn_resnet50_fpn"]
SMALL_MODEL_KWARGS = dict(rpn_pre_nms_top_n_train=600, rpn_post_nms_top_n_train=300,
                          rpn_pre_nms_top_n_test=300, rpn_post_nms_top_n_test=150,
                          box_batch_size_per_image=64)


def build_model(cfg: dict, meta: dict) -> nn.Module:
    m = cfg["model"]
    arch, pretrained = m["arch"], m.get("pretrained", True)
    num_classes = len(meta["classes"]) + 1
    size = detector_input_size(cfg)
    if "roi_samples_per_image" in m:
        size = {**size, "box_batch_size_per_image": int(m["roi_samples_per_image"])}
    if arch not in ARCHS:
        raise ValueError(f"unknown arch '{arch}' — choose from {ARCHS}")

    if arch == "maskrcnn_resnet18_fpn":
        backbone = resnet_fpn_backbone(
            backbone_name="resnet18", weights=models.ResNet18_Weights.DEFAULT if pretrained else None,
            norm_layer=FrozenBatchNorm2d if pretrained else nn.BatchNorm2d,
            trainable_layers=3 if pretrained else 5)
        return MaskRCNN(backbone, num_classes=num_classes, **{**SMALL_MODEL_KWARGS, **size})

    if m.get("pretrained_detector", False):                   # COCO-trained, then swap both heads
        model = models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT", **size)
        in_box = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_box, num_classes)
        in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, num_classes)
        return model
    return models.detection.maskrcnn_resnet50_fpn(
        weights=None, weights_backbone="DEFAULT" if pretrained else None,
        num_classes=num_classes, **size)
