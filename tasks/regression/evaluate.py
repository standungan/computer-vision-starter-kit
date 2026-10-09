"""Evaluate a regressor on the test split.

    python -m cvkit regression evaluate --config <config.yaml>

Writes to <output.dir>/reports/:
    pred_vs_true.png    every test image as a dot: on the diagonal = perfect
    residuals.png       histogram of errors (pred - true): centred on 0? narrow? skewed?
    worst_errors.png    the images with the largest absolute error
    predictions.csv     path, true, pred, error
    metrics.json        MAE / RMSE / R2
"""
from __future__ import annotations

import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cvkit.checkpoint import checkpoint_path, load_checkpoint, restore_model
from cvkit.config import output_dir
from cvkit.data import make_loader
from cvkit.utils import get_device, seed_everything
from cvkit.viz import BLUE, INK_2, ORANGE, image_grid, save_fig

from .data import build_datasets
from .metrics import regression_metrics
from .model import build_model
from .train import predict_loader


def evaluate(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    ckpt = load_checkpoint(checkpoint_path(cfg, "best.pt"), device)
    model = restore_model(ckpt, build_model, device)
    stats = ckpt["meta"]["stats"]
    test_ds = build_datasets(cfg)["test"]
    pred, true = predict_loader(model, make_loader(test_ds, cfg, shuffle=False), device, stats)

    reports = output_dir(cfg) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    metrics = {**regression_metrics(pred, true), "n_test": len(pred),
               "baseline_mae": float(np.abs(true - stats["mean"]).mean())}
    name = stats["target_name"]
    _plot_pred_vs_true(pred, true, name, reports / "pred_vs_true.png")
    _plot_residuals(pred - true, name, reports / "residuals.png")
    worst = np.argsort(-np.abs(pred - true))[:8]
    image_grid([Image.open(test_ds.samples[i][0]).convert("RGB") for i in worst],
               [f"true {true[i]:.1f}  pred {pred[i]:.1f}" for i in worst], cols=4,
               out=reports / "worst_errors.png", suptitle="largest errors on the test split")
    with open(reports / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "error"])
        for (path, _), t, p in zip(test_ds.samples, true, pred):
            w.writerow([path, f"{t:.4f}", f"{p:.4f}", f"{p - t:.4f}"])
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"{name}: MAE {metrics['mae']:.3f} | RMSE {metrics['rmse']:.3f} | R2 {metrics['r2']:.3f} "
          f"({len(pred)} images)")
    print(f"for reference, always guessing the train mean would give MAE {metrics['baseline_mae']:.3f}")
    print(f"reports -> {reports}")
    return metrics


def _plot_pred_vs_true(pred, true, name, out) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    lo, hi = float(min(pred.min(), true.min())), float(max(pred.max(), true.max()))
    ax.plot([lo, hi], [lo, hi], color=INK_2, linestyle="--", linewidth=1.2)
    mid = lo + 0.8 * (hi - lo)
    ax.annotate("perfect: pred = true", (mid, mid), xytext=(8, -16), textcoords="offset points",
                color=INK_2, fontsize=9, ha="left")
    ax.scatter(true, pred, s=22, color=BLUE, edgecolor="#fcfcfb", linewidth=0.8, alpha=0.9)
    ax.set_xlabel(f"true {name}")
    ax.set_ylabel(f"predicted {name}")
    ax.set_title("predicted vs true")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    save_fig(fig, out)


def _plot_residuals(err, name, out) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.hist(err, bins=20, color=BLUE, edgecolor="#fcfcfb")
    ax.axvline(0, color=ORANGE, linestyle="--")
    ax.set_xlabel(f"error (predicted - true {name})")
    ax.set_ylabel("images")
    ax.set_title("residuals")
    fig.tight_layout()
    save_fig(fig, out)
