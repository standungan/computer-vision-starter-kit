"""Metrics that more than one task needs.

Written from scratch (rather than imported) on purpose: each is ~30 lines
and reading them is the best way to learn what the number actually means.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch


# ------------------------------------------------- segmentation: confusion -> IoU
def update_confusion(cm: torch.Tensor, pred: torch.Tensor, target: torch.Tensor,
                     num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    """Add one batch of (B,H,W) class-id maps to a (C,C) confusion matrix [true, pred]."""
    keep = target != ignore_index
    t, p = target[keep].long().cpu(), pred[keep].long().cpu()
    cm += torch.bincount(t * num_classes + p, minlength=num_classes ** 2).view(num_classes, num_classes)
    return cm


def iou_from_confusion(cm: torch.Tensor) -> dict:
    """Per-class IoU / Dice and their means, plus pixel accuracy.

    IoU = TP / (TP + FP + FN): the overlap between predicted and true region
    divided by their union. Classes that never appear (no TP, FP or FN) are
    skipped in the mean instead of counting as 0 or 1.
    """
    cm = cm.double()
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    union = tp + fp + fn
    present = union > 0
    iou = torch.where(present, tp / union.clamp(min=1), torch.zeros_like(tp))
    dice = torch.where(present, 2 * tp / (2 * tp + fp + fn).clamp(min=1), torch.zeros_like(tp))
    return {"iou": iou.tolist(), "dice": dice.tolist(),
            "miou": float(iou[present].mean()) if present.any() else 0.0,
            "mdice": float(dice[present].mean()) if present.any() else 0.0,
            "pixel_acc": float(tp.sum() / cm.sum().clamp(min=1))}


# --------------------------------------------- detection: average precision (AP)
def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the precision-recall curve, COCO style (101-point interpolation)."""
    if len(recall) == 0:
        return 0.0
    # make precision monotonically decreasing, then sample at 101 recall points
    prec = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, points, side="left")
    sampled = np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0.0)
    return float(sampled.mean())


def mean_average_precision(
    preds: list[dict], gts: list[dict], num_classes: int, iou_fn: Callable,
    iou_thresholds: tuple[float, ...] = tuple(np.arange(0.5, 0.96, 0.05)),
) -> dict:
    """COCO-style mAP for detection AND instance segmentation.

    preds[i] / gts[i] describe image i: dicts with a "labels" tensor (1..num_classes),
    preds also have "scores", plus whatever `iou_fn(pred_subset, gt_subset)` needs
    ("boxes" for box IoU, "masks" for mask IoU). iou_fn returns a (P, G) IoU matrix.

    For one class and one IoU threshold: sort all predictions by score; walk down
    the list; a prediction is a true positive if it overlaps a not-yet-matched
    ground truth by at least the threshold, otherwise a false positive. That gives a
    precision/recall curve; its area is AP. mAP averages AP over classes
    (and over thresholds 0.50:0.05:0.95 for the headline "map").
    Classes absent from the ground truth are skipped.
    """
    thresholds = list(iou_thresholds)
    ap = np.full((num_classes, len(thresholds)), np.nan)
    for c in range(1, num_classes + 1):
        n_gt = sum(int((g["labels"] == c).sum()) for g in gts)
        if n_gt == 0:
            continue
        records = []                  # (score, image index, iou row of this prediction)
        ious = {}
        for i, (p, g) in enumerate(zip(preds, gts)):
            pm, gm = p["labels"] == c, g["labels"] == c
            if pm.sum() == 0:
                continue
            sub_p = {k: v[pm] for k, v in p.items()}
            sub_g = {k: v[gm] for k, v in g.items()}
            iou = iou_fn(sub_p, sub_g).cpu().numpy() if gm.sum() else np.zeros((int(pm.sum()), 0))
            ious[i] = iou
            for j, s in enumerate(sub_p["scores"].tolist()):
                records.append((s, i, j))
        records.sort(key=lambda r: -r[0])
        for t_idx, thr in enumerate(thresholds):
            matched = {i: np.zeros(m.shape[1], dtype=bool) for i, m in ious.items()}
            tp = np.zeros(len(records))
            for r, (_, i, j) in enumerate(records):
                row = ious[i][j]
                if row.size == 0:
                    continue
                cand = np.where(~matched[i], row, -1.0)
                best = int(cand.argmax())
                if cand[best] >= thr:
                    matched[i][best] = True
                    tp[r] = 1
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(1 - tp)
            recall = cum_tp / n_gt
            precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
            ap[c - 1, t_idx] = average_precision(recall, precision)
    valid = ~np.isnan(ap[:, 0])
    if not valid.any():
        return {"map": 0.0, "map50": 0.0, "map75": 0.0, "ap50_per_class": [0.0] * num_classes}
    def at(thr):
        k = int(np.argmin([abs(x - thr) for x in thresholds]))
        return float(np.nanmean(ap[valid, k]))
    return {"map": float(np.nanmean(ap[valid])), "map50": at(0.5), "map75": at(0.75),
            "ap50_per_class": [float(x) if v else 0.0
                               for x, v in zip(ap[:, int(np.argmin([abs(x - 0.5) for x in thresholds]))], valid)]}


def mask_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """(P,H,W) x (G,H,W) boolean masks -> (P,G) IoU."""
    p = pred_masks.flatten(1).float()
    g = gt_masks.flatten(1).float()
    inter = p @ g.T
    union = p.sum(1)[:, None] + g.sum(1)[None, :] - inter
    return inter / union.clamp(min=1)
