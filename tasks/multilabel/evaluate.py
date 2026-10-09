"""Evaluate a multi-label model on the test split.

    python -m cvkit multilabel evaluate --config <config.yaml>

Writes to <output.dir>/reports/:
    classification_report.txt   per-label precision / recall / F1 at the config threshold
    per_label_metrics.png       the same, as bars: which labels are weak?
    threshold_sweep.png         macro-F1 as the threshold moves: is 0.5 the right cut-off?
    worst_images.png            test images with the most wrong labels
    predictions.csv             probability for every label on every test image
    metrics.json                headline numbers
"""
from __future__ import annotations

import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, precision_recall_fscore_support

from cvkit.checkpoint import checkpoint_path, load_checkpoint, restore_model
from cvkit.config import output_dir
from cvkit.data import make_loader
from cvkit.utils import get_device, seed_everything
from cvkit.viz import AQUA, BLUE, INK_2, ORANGE, image_grid, save_fig

from .data import build_datasets
from .metrics import multilabel_metrics
from .model import build_model
from .train import predict_loader


def evaluate(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    ckpt = load_checkpoint(checkpoint_path(cfg, "best.pt"), device)
    model = restore_model(ckpt, build_model, device)
    classes = ckpt["meta"]["classes"]
    threshold = float(cfg.get("eval", {}).get("threshold", 0.5))
    test_ds = build_datasets(cfg)["test"]
    probs, targets, _ = predict_loader(model, make_loader(test_ds, cfg, shuffle=False), device)

    reports = output_dir(cfg) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    metrics = {**multilabel_metrics(probs, targets, threshold), "threshold": threshold, "n_test": len(probs)}
    pred, true = probs >= threshold, targets > 0.5

    report = classification_report(true, pred, target_names=classes, zero_division=0, digits=3)
    (reports / "classification_report.txt").write_text(
        f"threshold {threshold} | mAP {metrics['map']:.4f} | macro-F1 {metrics['f1_macro']:.4f} | "
        f"exact-match {metrics['exact']:.4f} ({len(probs)} images)\n\n{report}", encoding="utf-8")
    _plot_per_label(true, pred, classes, reports / "per_label_metrics.png")
    _plot_threshold_sweep(probs, targets, threshold, reports / "threshold_sweep.png")
    _plot_worst(test_ds.samples, probs, true, pred, classes, reports / "worst_images.png")
    with open(reports / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", *[f"p_{c}" for c in classes], *[f"true_{c}" for c in classes]])
        for (path, _), p, t in zip(test_ds.samples, probs, true):
            w.writerow([path, *[f"{v:.4f}" for v in p], *[int(v) for v in t]])
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"threshold {threshold}: mAP {metrics['map']:.4f} | macro-F1 {metrics['f1_macro']:.4f} | "
          f"micro-F1 {metrics['f1_micro']:.4f} | exact-match {metrics['exact']:.4f}")
    print(report)
    print(f"reports -> {reports}")
    return metrics


def _plot_per_label(true, pred, classes, out) -> None:
    p, r, f, _ = precision_recall_fscore_support(true, pred, zero_division=0)
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(classes) + 2), 3.8))
    for k, (vals, name, color) in enumerate(((p, "precision", BLUE), (r, "recall", ORANGE), (f, "F1", AQUA))):
        ax.bar(x + (k - 1) * 0.26, vals, 0.24, label=name, color=color)
    ax.set_xticks(x, classes, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("per-label metrics (test split)")
    ax.legend(frameon=False, ncol=3, loc="lower center")
    fig.tight_layout()
    save_fig(fig, out)


def _plot_threshold_sweep(probs, targets, chosen, out) -> None:
    grid = np.linspace(0.05, 0.95, 19)
    f1 = [multilabel_metrics(probs, targets, t)["f1_macro"] for t in grid]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(grid, f1, color=BLUE)
    ax.axvline(chosen, color=ORANGE, linestyle="--")
    ax.annotate(f"config threshold {chosen}", (chosen, min(f1)), xytext=(4, 4),
                textcoords="offset points", color=INK_2, fontsize=9)
    ax.set_xlabel("threshold")
    ax.set_ylabel("macro-F1")
    ax.set_title("does the threshold matter?")
    fig.tight_layout()
    save_fig(fig, out)


def _plot_worst(samples, probs, true, pred, classes, out, k: int = 8) -> None:
    wrong = (pred != true).sum(1)
    order = np.argsort(-wrong, kind="stable")[:k]
    images, titles = [], []
    for i in order:
        images.append(Image.open(samples[i][0]).convert("RGB"))
        t = ",".join(c for c, v in zip(classes, true[i]) if v) or "-"
        pr = ",".join(c for c, v in zip(classes, pred[i]) if v) or "-"
        titles.append(f"{'✓' if wrong[i] == 0 else '✗'} true: {t}\npred: {pr}")
    image_grid(images, titles, cols=4, out=out, suptitle="test images with the most wrong labels (✗ = wrong)")
