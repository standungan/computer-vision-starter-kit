"""Train an image regressor.

    python -m cvkit regression train --config tasks/regression/configs/default.yaml

best.pt = the epoch with the lowest validation MAE (in real units).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from cvkit.engine import TrainSpec, fit
from cvkit.utils import get_device, seed_everything

from .data import build_loaders
from .metrics import regression_metrics
from .model import build_model

LOSSES = {"huber": F.smooth_l1_loss, "mse": F.mse_loss, "l1": F.l1_loss}


@torch.inference_mode()
def predict_loader(model, loader, device, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    """(predictions, targets) in original units."""
    model.eval()
    preds, targets = [], []
    for images, y in loader:
        out = model(images.to(device)).squeeze(1).float().cpu()
        preds.append(out * stats["std"] + stats["mean"])
        targets.append(y)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def train(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    loaders = build_loaders(cfg)
    stats = loaders["stats"]
    print(f"arch: {cfg['model']['arch']} | target '{stats['target_name']}': "
          f"train mean {stats['mean']:.3f}, std {stats['std']:.3f}")
    model = build_model(cfg, {}).to(device)
    loss_name = cfg["train"].get("loss", "huber")
    if loss_name not in LOSSES:
        raise ValueError(f"train.loss must be one of {sorted(LOSSES)}, got '{loss_name}'")
    loss_fn = LOSSES[loss_name]

    def train_step(model, batch, device):
        images, y = (t.to(device) for t in batch)
        out = model(images).squeeze(1)
        y_norm = (y - stats["mean"]) / stats["std"]
        loss = loss_fn(out.float(), y_norm)
        mae = ((out.detach().float() - y_norm).abs().mean() * stats["std"]).item()
        return loss, y.size(0), {"mae": mae}

    def evaluate_loader(model, loader, device):
        pred, true = predict_loader(model, loader, device, stats)
        return regression_metrics(pred, true)

    spec = TrainSpec(train_step=train_step, evaluate=evaluate_loader, monitor="mae", mode="min",
                     meta={"stats": stats})
    return fit(cfg, model, loaders["train"], loaders["val"], spec, device)
