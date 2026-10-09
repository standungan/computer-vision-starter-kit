"""Train a keypoint model.

    python -m cvkit keypoints train --config tasks/keypoints/configs/default.yaml

Each keypoint's score map is turned into a probability distribution over pixels (a softmax over
the whole map) and trained with cross-entropy against a Gaussian blob at the true position
(visible keypoints only). best.pt = the epoch with the lowest validation NME.
"""
from __future__ import annotations

import torch

from cvkit.engine import TrainSpec, fit
from cvkit.utils import get_device, seed_everything

from .data import build_loaders
from .metrics import keypoint_metrics
from .model import STRIDE, build_model, decode, make_heatmaps, to_probs


@torch.inference_mode()
def predict_loader(model, loader, device):
    """Decoded keypoints for every sample: (pred (N,K,2), true (N,K,2), visible (N,K)) on CPU."""
    model.eval()
    preds, trues, vis = [], [], []
    for images, pts, v in loader:
        p, _ = decode(to_probs(model(images.to(device))))
        preds.append(p.cpu())
        trues.append(pts)
        vis.append(v)
    return torch.cat(preds), torch.cat(trues), torch.cat(vis)


def train(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    loaders = build_loaders(cfg)
    names = loaders["keypoint_names"]
    size = int(cfg["data"]["image_size"])
    if size % STRIDE:
        raise ValueError(f"data.image_size must be a multiple of {STRIDE}, got {size}")
    sigma = float(cfg["model"].get("heatmap_sigma", 2.0))
    thr = float(cfg.get("eval", {}).get("pck_threshold", 0.05))
    print(f"arch: {cfg['model']['arch']} | keypoints ({len(names)}): {names}")
    model = build_model(cfg, {"keypoint_names": names}).to(device)

    def train_step(model, batch, device):
        images, pts, vis = (t.to(device) for t in batch)
        target = make_heatmaps(pts, vis.float(), size, sigma).flatten(2)
        target = target / target.sum(-1, keepdim=True).clamp(min=1e-8)      # a distribution over pixels
        log_p = model(images).float().flatten(2).log_softmax(-1)
        per_keypoint = -(target * log_p).sum(-1)                            # cross-entropy, (B,K)
        loss = (per_keypoint * vis).sum() / vis.sum().clamp(min=1)
        return loss, images.size(0), {}

    def evaluate_loader(model, loader, device):
        p, t, v = predict_loader(model, loader, device)
        m = keypoint_metrics(p, t, v, size, thr)
        return {"nme": m["nme"], "pck": m["pck"], "mean_err_px": m["mean_err_px"]}

    spec = TrainSpec(train_step=train_step, evaluate=evaluate_loader, monitor="nme", mode="min",
                     meta={"keypoint_names": list(names)})
    return fit(cfg, model, loaders["train"], loaders["val"], spec, device)
