"""Detection models: torchvision Faster R-CNN, with the final classifier swapped for your classes.

Faster R-CNN works in two stages: a Region Proposal Network suggests "something might be here"
boxes, then a second head classifies each proposal and refines its box. Class 0 is always
"background", so a dataset with K classes gives the model K + 1 outputs.

`fasterrcnn_resnet18_fpn` is a small custom variant (ResNet18 + feature pyramid) that trains
on a CPU; the others are torchvision's stock models.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d

from cvkit.detection import detector_input_size

STOCK = {"fasterrcnn_resnet50_fpn": models.detection.fasterrcnn_resnet50_fpn,
         "fasterrcnn_mobilenet_v3_large_fpn": models.detection.fasterrcnn_mobilenet_v3_large_fpn,
         "fasterrcnn_mobilenet_v3_large_320_fpn": models.detection.fasterrcnn_mobilenet_v3_large_320_fpn}
ARCHS = ["fasterrcnn_resnet18_fpn", *STOCK]

# fewer proposals = much faster on CPU, plenty for images with a handful of objects
SMALL_MODEL_KWARGS = dict(rpn_pre_nms_top_n_train=600, rpn_post_nms_top_n_train=300,
                          rpn_pre_nms_top_n_test=300, rpn_post_nms_top_n_test=150,
                          box_batch_size_per_image=64)


def build_model(cfg: dict, meta: dict) -> nn.Module:
    m = cfg["model"]
    arch, pretrained = m["arch"], m.get("pretrained", True)
    num_classes = len(meta["classes"]) + 1                    # + background
    size = detector_input_size(cfg)
    if "roi_samples_per_image" in m:                          # speed knob: ROIs sampled per image
        size = {**size, "box_batch_size_per_image": int(m["roi_samples_per_image"])}
    if arch not in ARCHS:
        raise ValueError(f"unknown arch '{arch}' — choose from {ARCHS}")

    if arch == "fasterrcnn_resnet18_fpn":
        backbone = resnet_fpn_backbone(
            backbone_name="resnet18", weights=models.ResNet18_Weights.DEFAULT if pretrained else None,
            norm_layer=FrozenBatchNorm2d if pretrained else nn.BatchNorm2d,
            trainable_layers=3 if pretrained else 5)
        return FasterRCNN(backbone, num_classes=num_classes, **{**SMALL_MODEL_KWARGS, **size})

    if m.get("pretrained_detector", False):                   # COCO-trained, then swap the head
        model = STOCK[arch](weights="DEFAULT", **size)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    return STOCK[arch](weights=None, weights_backbone="DEFAULT" if pretrained else None,
                       num_classes=num_classes, **size)
