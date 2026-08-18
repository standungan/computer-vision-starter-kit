"""Training loop: AMP, cosine schedule, early stopping, best-checkpoint saving.

Run from the repo root:

    python -m src.train --config configs/config.yaml

Everything worth tweaking lives in the YAML. The loop writes:
    outputs/checkpoints/best.pt   (highest val accuracy — used everywhere downstream)
    outputs/checkpoints/last.pt
    outputs/history.csv           (per-epoch metrics, plotted in notebook 03)
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from .dataset import build_loaders
from .model import build_model, count_parameters
from .utils import checkpoint_path, get_device, load_config, seed_everything


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
    """One pass over a loader. With an optimizer it trains, without it evaluates."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.inference_mode()
    with ctx:
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            with autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            seen += targets.size(0)
    return total_loss / seen, correct / seen


def train(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    device = get_device()
    loaders = build_loaders(cfg)
    classes = loaders["classes"]

    model = build_model(cfg, num_classes=len(classes)).to(device)
    params = count_parameters(model)
    print(f"device: {device} | arch: {cfg['model']['arch']} | classes: {len(classes)}")
    print(f"parameters: {params['total']:,} total, "
          f"{params['trainable']:,} trainable ({params['trainable_pct']:.1f}%)")

    tcfg = cfg["train"]
    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg.get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg["epochs"])
    amp = tcfg.get("amp", True) and device.type == "cuda"
    scaler = GradScaler(device.type) if amp else None

    best_acc, best_epoch = 0.0, -1
    patience = tcfg.get("early_stopping_patience", 0)
    history_path = Path(cfg["output"]["dir"]) / "history.csv"
    history: list[dict] = []

    for epoch in range(tcfg["epochs"]):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, device,
                                    optimizer=optimizer, scaler=scaler, amp=amp)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion, device, amp=amp)
        scheduler.step()

        row = {"epoch": epoch + 1, "train_loss": tr_loss, "train_acc": tr_acc,
               "val_loss": va_loss, "val_acc": va_acc,
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0}
        history.append(row)
        with open(history_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(history)

        marker = ""
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch
            torch.save({"model": model.state_dict(), "classes": classes,
                        "config": cfg, "epoch": epoch + 1, "val_acc": va_acc},
                       checkpoint_path(cfg, "best.pt"))
            marker = "  <- best"
        print(f"epoch {epoch + 1:>3}/{tcfg['epochs']} | "
              f"train {tr_loss:.4f}/{tr_acc:.3f} | val {va_loss:.4f}/{va_acc:.3f} | "
              f"{row['seconds']:.1f}s{marker}")

        torch.save({"model": model.state_dict(), "classes": classes,
                    "config": cfg, "epoch": epoch + 1, "val_acc": va_acc},
                   checkpoint_path(cfg, "last.pt"))

        if patience and epoch - best_epoch >= patience:
            print(f"early stopping: no val improvement in {patience} epochs")
            break

    print(f"best val acc {best_acc:.3f} (epoch {best_epoch + 1}) -> "
          f"{checkpoint_path(cfg, 'best.pt')}")
    return {"best_val_acc": best_acc, "best_epoch": best_epoch + 1, "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
