"""Evaluate on the held-out test split + error analysis.

    python -m cvkit classification evaluate --config <config.yaml>

Writes to <output.dir>/reports/:
    classification_report.txt   per-class precision / recall / F1
    confusion_matrix.png        which classes get mistaken for which
    top_losses.png              the images the model got most confidently wrong
    predictions.csv             every test image: path, true, pred, confidence, loss
    metrics.json                headline numbers

The top-losses grid is the error-analysis step. Sorted by loss, these images tell
you whether mistakes come from wrong labels, genuinely ambiguous images, or a real
model weakness.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from cvkit.checkpoint import checkpoint_path, load_checkpoint, restore_model
from cvkit.config import output_dir
from cvkit.data import make_loader
from cvkit.utils import get_device, seed_everything
from cvkit.viz import image_grid, plot_confusion_matrix

from .data import build_datasets
from .model import build_model


@torch.inference_mode()
def collect_predictions(cfg: dict, device) -> dict:
    """Run best.pt over the test split, keeping per-image confidence and loss."""
    ckpt = load_checkpoint(checkpoint_path(cfg, "best.pt"), device)
    model = restore_model(ckpt, build_model, device)
    test_ds = build_datasets(cfg)["test"]
    rows = []
    offset = 0
    for images, targets in make_loader(test_ds, cfg, shuffle=False):
        logits = model(images.to(device))
        probs = F.softmax(logits, dim=1).cpu()
        losses = F.cross_entropy(logits.cpu(), targets, reduction="none")
        for k in range(len(targets)):
            pred = int(probs[k].argmax())
            rows.append({"path": test_ds.samples[offset + k][0], "true": int(targets[k]),
                         "pred": pred, "confidence": float(probs[k][pred]), "loss": float(losses[k])})
        offset += len(targets)
    return {"rows": rows, "classes": ckpt["meta"]["classes"]}


def evaluate(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    result = collect_predictions(cfg, get_device(cfg.get("device", "auto")))
    rows, classes = result["rows"], result["classes"]
    reports = output_dir(cfg) / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    y_true, y_pred = [r["true"] for r in rows], [r["pred"] for r in rows]
    labels = list(range(len(classes)))
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(rows)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, target_names=classes,
                                   digits=3, zero_division=0)
    (reports / "classification_report.txt").write_text(
        f"test accuracy: {acc:.4f}  macro-F1: {macro_f1:.4f}  ({len(rows)} images)\n\n{report}",
        encoding="utf-8")
    plot_confusion_matrix(confusion_matrix(y_true, y_pred, labels=labels), classes,
                          reports / "confusion_matrix.png")
    _plot_top_losses(rows, classes, reports / "top_losses.png")
    with open(reports / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "true", "pred", "confidence", "loss"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "true": classes[r["true"]], "pred": classes[r["pred"]]})
    metrics = {"accuracy": acc, "macro_f1": macro_f1, "n_test": len(rows)}
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"test accuracy: {acc:.4f} | macro-F1: {macro_f1:.4f} ({len(rows)} images)")
    print(report)
    print(f"reports -> {reports}")
    return {**metrics, "rows": rows, "classes": classes}


def _plot_top_losses(rows: list[dict], classes: list[str], out: Path, k: int = 12) -> None:
    worst = sorted(rows, key=lambda r: r["loss"], reverse=True)[:k]
    images = [Image.open(r["path"]).convert("RGB") for r in worst]
    titles = [f"{'✓' if r['true'] == r['pred'] else '✗'} true: {classes[r['true']]}\n"
              f"pred: {classes[r['pred']]} ({r['confidence']:.2f})" for r in worst]
    image_grid(images, titles, cols=4, out=out, suptitle="highest-loss test images (✗ = wrong) — start error analysis here")

