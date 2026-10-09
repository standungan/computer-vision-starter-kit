"""The generic training loop every task shares.

A task plugs in two small functions and the engine does the rest
(AdamW, cosine LR schedule, mixed precision, early stopping, checkpoints,
history.csv):

    train_step(model, batch, device) -> (loss, batch_size, stats)
        forward pass + loss for ONE training batch. `stats` is a dict of
        floats to average over the epoch (e.g. {"acc": 0.9}).

    evaluate(model, loader, device) -> {"acc": 0.93, "loss": 0.21, ...}
        metrics on a validation loader. One key is the `monitor`: the best
        value of it decides which epoch becomes best.pt.

Because the loop never looks inside a batch, the same engine trains
classifiers, detectors, segmenters, and keypoint models.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from torch.amp import GradScaler, autocast

from .checkpoint import checkpoint_path, save_checkpoint
from .config import output_dir
from .utils import count_parameters


@dataclass
class TrainSpec:
    train_step: Callable
    evaluate: Callable
    monitor: str                     # key returned by `evaluate`, e.g. "acc" or "map50"
    mode: str = "max"                # "max" (accuracy, mAP, IoU) or "min" (loss, MAE)
    meta: dict = field(default_factory=dict)   # saved into the checkpoint


def fit(cfg: dict, model: torch.nn.Module, train_loader, val_loader,
        spec: TrainSpec, device: torch.device) -> dict:
    tcfg = cfg["train"]
    params = count_parameters(model)
    print(f"device: {device} | parameters: {params['total']:,} total, "
          f"{params['trainable']:,} trainable ({params['trainable_pct']:.1f}%)")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(tcfg["lr"]), weight_decay=float(tcfg.get("weight_decay", 0.0)))
    epochs = int(tcfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(device.type) if amp else None
    grad_clip = float(tcfg.get("grad_clip", 0.0))
    limit = int(tcfg.get("limit_batches", 0))      # >0: stop each epoch early (debug runs)
    patience = int(tcfg.get("early_stopping_patience", 0))

    sign = 1.0 if spec.mode == "max" else -1.0
    best_score, best_epoch, best_metrics = -float("inf"), -1, {}
    out = output_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        loss_sum, seen, stat_sums = 0.0, 0, {}
        for i, batch in enumerate(train_loader):
            if limit and i >= limit:
                break
            with autocast(device_type=device.type, enabled=amp):
                loss, n, stats = spec.train_step(model, batch, device)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            loss_sum += loss.item() * n
            seen += n
            for k, v in stats.items():
                stat_sums[k] = stat_sums.get(k, 0.0) + float(v) * n
        scheduler.step()

        val = spec.evaluate(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_loss": loss_sum / max(seen, 1),
               **{f"train_{k}": v / max(seen, 1) for k, v in stat_sums.items()},
               **{f"val_{k}": float(v) for k, v in val.items()},
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0}
        history.append(row)
        _write_history(out / "history.csv", history)

        score = sign * float(val[spec.monitor])
        marker = ""
        if score > best_score:
            best_score, best_epoch, best_metrics = score, epoch, {k: float(v) for k, v in val.items()}
            save_checkpoint(checkpoint_path(cfg, "best.pt"), model, cfg, spec.meta,
                            epoch + 1, best_metrics)
            marker = "  <- best"
        save_checkpoint(checkpoint_path(cfg, "last.pt"), model, cfg, spec.meta,
                        epoch + 1, {k: float(v) for k, v in val.items()})
        shown = " ".join(f"{k}={v:.3f}" for k, v in val.items())
        print(f"epoch {epoch + 1:>3}/{epochs} | train_loss {row['train_loss']:.4f} | "
              f"val {shown} | {row['seconds']:.1f}s{marker}")

        if patience and epoch - best_epoch >= patience:
            print(f"early stopping: no '{spec.monitor}' improvement in {patience} epochs")
            break

    print(f"best val {spec.monitor} = {sign * best_score:.4f} (epoch {best_epoch + 1}) "
          f"-> {checkpoint_path(cfg, 'best.pt')}")
    return {"best_epoch": best_epoch + 1, "best_metrics": best_metrics, "history": history}


def _write_history(path: Path, history: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[-1]))
        w.writeheader()
        w.writerows(history)
