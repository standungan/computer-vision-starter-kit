"""Keypoint metrics. Distances are divided by the image side, so they mean the same at any resolution.

    NME   normalized mean error: average distance to the true point, as a fraction of the image side (lower is better)
    PCK   percentage of correct keypoints: share of points within `threshold` x image side of the truth
"""
from __future__ import annotations

import torch


def keypoint_metrics(pred: torch.Tensor, true: torch.Tensor, visible: torch.Tensor,
                     size: int, threshold: float = 0.05) -> dict:
    """pred/true (N,K,2) in pixels, visible (N,K) bool -> aggregate numbers (+ per-keypoint lists)."""
    dist = (pred - true).norm(dim=-1) / size                       # (N,K)
    vis = visible.float()
    per_kp_n = vis.sum(0).clamp(min=1)
    nme_k = (dist * vis).sum(0) / per_kp_n
    pck_k = ((dist <= threshold).float() * vis).sum(0) / per_kp_n
    total = vis.sum().clamp(min=1)
    return {"nme": float((dist * vis).sum() / total), "pck": float(((dist <= threshold).float() * vis).sum() / total),
            "mean_err_px": float((dist * vis).sum() / total * size),
            "nme_per_keypoint": nme_k.tolist(), "pck_per_keypoint": pck_k.tolist()}


def pck_curve(pred: torch.Tensor, true: torch.Tensor, visible: torch.Tensor, size: int,
              thresholds: list[float]) -> list[float]:
    dist = ((pred - true).norm(dim=-1) / size)[visible]
    return [float((dist <= t).float().mean()) if dist.numel() else 0.0 for t in thresholds]
