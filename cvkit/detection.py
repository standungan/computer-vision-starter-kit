"""Shared plumbing for the detection family: object detection and instance segmentation.

Both read COCO annotations, feed torchvision's R-CNN models, and are scored with mAP, so
they share the dataset, the augmentation (which keeps boxes and masks in sync with the
image), the prediction loop and the evaluation function. What differs per task — the model
and which mAP is the headline — stays in the task folders.

torchvision R-CNN models take a *list* of images and a *list* of target dicts, and resize
internally, so batches here are tuples of variable-size items rather than stacked tensors.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.ops import box_iou
from torchvision.transforms import v2

from .coco import load_coco, polygons_to_mask
from .data import make_loader, require_dir, split_indices
from .metrics import mask_iou, mean_average_precision


def collate(batch):
    return tuple(zip(*batch))


def detection_transforms(augment: dict | None, train: bool) -> v2.Compose:
    """Photometric + flip augmentation (boxes and masks follow the flips), then to float tensor."""
    aug = augment or {}
    ops: list = []
    if train:
        if aug.get("hflip", False):
            ops.append(v2.RandomHorizontalFlip())
        if aug.get("vflip", False):
            ops.append(v2.RandomVerticalFlip())
        if aug.get("color_jitter", 0):
            j = aug["color_jitter"]
            ops.append(v2.ColorJitter(brightness=j, contrast=j, saturation=j))
    ops += [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.SanitizeBoundingBoxes()]
    return v2.Compose(ops)


class CocoDataset(Dataset):
    """Returns (image float tensor [0,1], target dict) per torchvision's detection contract."""

    def __init__(self, images: list[dict], transform, with_masks: bool = False):
        self.images, self.transform, self.with_masks = images, transform, with_masks

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        info = self.images[i]
        with Image.open(info["path"]) as im:
            image = im.convert("RGB")
        h, w = info["height"], info["width"]
        anns = info["annotations"]
        boxes = torch.tensor([a["bbox"] for a in anns], dtype=torch.float32).reshape(-1, 4)
        target = {"boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h, w)),
                  "labels": torch.tensor([a["label"] for a in anns], dtype=torch.int64)}
        if self.with_masks:
            if any("segmentation" not in a for a in anns):
                raise ValueError(f"{info['path']}: instance segmentation needs polygon "
                                 "'segmentation' for every annotation")
            masks = np.stack([polygons_to_mask(a["segmentation"], h, w) for a in anns]) if anns \
                else np.zeros((0, h, w), dtype=np.uint8)
            target["masks"] = tv_tensors.Mask(torch.from_numpy(masks))
        return self.transform(image, target)


def build_coco_datasets(cfg: dict, with_masks: bool = False) -> dict:
    d = cfg["data"]
    coco = load_coco(require_dir(d["root"]), d.get("annotations", "annotations.json"),
                     d.get("images_dir", "images"))
    images = coco["images"]
    splits = split_indices(len(images), d["split"], cfg["seed"])
    aug = d.get("augment", {})

    def subset(name, train):
        return CocoDataset([images[i] for i in splits[name]], detection_transforms(aug, train), with_masks)

    return {"train": subset("train", True), "val": subset("val", False), "test": subset("test", False),
            "classes": coco["classes"], "images": images, "split_indices": splits}


def build_coco_loaders(cfg: dict, with_masks: bool = False) -> dict:
    ds = build_coco_datasets(cfg, with_masks)
    return {"train": make_loader(ds["train"], cfg, True, collate),
            "val": make_loader(ds["val"], cfg, False, collate),
            "test": make_loader(ds["test"], cfg, False, collate), "classes": ds["classes"]}


def detector_input_size(cfg: dict) -> dict:
    """data.image_size -> torchvision's min_size / max_size (the model resizes internally)."""
    s = int(cfg["data"]["image_size"])
    return {"min_size": s, "max_size": int(s * 1333 / 800)}


def _plain(t: torch.Tensor) -> torch.Tensor:
    return t.as_subclass(torch.Tensor) if isinstance(t, tv_tensors.TVTensor) else t


def detection_train_step(model, batch, device):
    """R-CNN models return a dict of losses in train mode; the total is their sum."""
    images, targets = batch
    images = [im.to(device) for im in images]
    targets = [{k: _plain(v).to(device) for k, v in t.items()} for t in targets]
    losses = model(images, targets)
    loss = sum(losses.values())
    return loss, len(images), {k: float(v.detach()) for k, v in losses.items()}


@torch.inference_mode()
def predict_loader(model, loader, device, with_masks: bool = False, mask_threshold: float = 0.5):
    """Run a detector over a loader -> (predictions, ground_truths), lists of CPU dicts."""
    model.eval()
    preds, gts = [], []
    for images, targets in loader:
        outputs = model([im.to(device) for im in images])
        for out, tgt in zip(outputs, targets):
            p = {"boxes": out["boxes"].cpu(), "labels": out["labels"].cpu(), "scores": out["scores"].cpu()}
            g = {"boxes": _plain(tgt["boxes"]), "labels": tgt["labels"]}
            if with_masks:
                p["masks"] = (out["masks"].squeeze(1) > mask_threshold).cpu()
                g["masks"] = _plain(tgt["masks"]).bool()
            preds.append(p)
            gts.append(g)
    return preds, gts


def box_map(preds, gts, num_classes: int) -> dict:
    return mean_average_precision(preds, gts, num_classes, lambda p, g: box_iou(p["boxes"], g["boxes"]))


def mask_map(preds, gts, num_classes: int) -> dict:
    return mean_average_precision(preds, gts, num_classes, lambda p, g: mask_iou(p["masks"], g["masks"]))


def match_at_threshold(preds, gts, score_threshold: float = 0.5, iou_threshold: float = 0.5) -> dict:
    """Plain counts at one operating point: TP / FP / FN, precision, recall, per-image error counts."""
    tp = fp = fn = 0
    per_image = []
    for p, g in zip(preds, gts):
        keep = p["scores"] >= score_threshold
        pb, pl = p["boxes"][keep], p["labels"][keep]
        used = torch.zeros(len(g["boxes"]), dtype=torch.bool)
        itp = 0
        if len(pb) and len(g["boxes"]):
            iou = box_iou(pb, g["boxes"])
            for k in torch.argsort(p["scores"][keep], descending=True).tolist():
                cand = iou[k].clone()
                cand[(g["labels"] != pl[k]) | used] = -1
                best = int(cand.argmax())
                if cand[best] >= iou_threshold:
                    used[best] = True
                    itp += 1
        ifp, ifn = len(pb) - itp, len(g["boxes"]) - itp
        tp, fp, fn = tp + itp, fp + ifp, fn + ifn
        per_image.append(ifp + ifn)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "errors_per_image": per_image}
