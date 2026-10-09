"""Train a semantic-segmentation model.

    python -m cvkit segmentation train --config tasks/segmentation/configs/default.yaml

Loss = cross-entropy (every pixel is a classification problem) + dice_weight * Dice loss.
Cross-entropy alone lets a huge background drown small objects; Dice measures overlap per
class, so a tiny object missed entirely costs as much as a big one.
best.pt = the epoch with the highest validation mean IoU.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from cvkit.engine import TrainSpec, fit
from cvkit.metrics import iou_from_confusion, update_confusion
from cvkit.utils import get_device, seed_everything

from .data import build_loaders
from .model import build_model


def dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """1 - mean per-class Dice over the classes present in the batch; ignored pixels don't count."""
    n = logits.shape[1]
    valid = (target != ignore_index).unsqueeze(1)
    probs = logits.float().softmax(1) * valid
    onehot = F.one_hot(target.clamp(max=n - 1), n).permute(0, 3, 1, 2).float() * valid
    inter = (probs * onehot).sum((0, 2, 3))
    denom = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    present = onehot.sum((0, 2, 3)) > 0
    dice = (2 * inter + 1e-6) / (denom + 1e-6)
    return 1 - dice[present].mean() if present.any() else logits.sum() * 0


@torch.inference_mode()
def confusion_over_loader(model, loader, device, num_classes: int, ignore_index: int) -> torch.Tensor:
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for images, masks in loader:
        pred = model(images.to(device)).argmax(1)
        update_confusion(cm, pred.cpu(), masks, num_classes, ignore_index)
    return cm


def train(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    loaders = build_loaders(cfg)
    classes = loaders["classes"]
    n, ignore = len(classes), int(cfg["data"].get("ignore_index", 255))
    print(f"arch: {cfg['model']['arch']} | classes ({n}): {classes}")
    model = build_model(cfg, {"classes": classes}).to(device)
    dice_w = float(cfg["train"].get("dice_weight", 0.0))

    def train_step(model, batch, device):
        images, masks = (t.to(device) for t in batch)
        logits = model(images)
        loss = F.cross_entropy(logits, masks, ignore_index=ignore)
        if dice_w > 0:
            loss = loss + dice_w * dice_loss(logits, masks, ignore)
        keep = masks != ignore
        acc = ((logits.argmax(1) == masks) & keep).sum().item() / max(keep.sum().item(), 1)
        return loss, masks.size(0), {"pixel_acc": acc}

    def evaluate_loader(model, loader, device):
        res = iou_from_confusion(confusion_over_loader(model, loader, device, n, ignore))
        return {"miou": res["miou"], "mdice": res["mdice"], "pixel_acc": res["pixel_acc"]}

    spec = TrainSpec(train_step=train_step, evaluate=evaluate_loader, monitor="miou",
                     mode="max", meta={"classes": list(classes)})
    return fit(cfg, model, loaders["train"], loaders["val"], spec, device)
