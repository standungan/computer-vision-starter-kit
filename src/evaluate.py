"""Evaluation + error analysis on the held-out test split.

    python -m src.evaluate --config configs/config.yaml

Writes to outputs/reports/:
    classification_report.txt   per-class precision / recall / F1
    confusion_matrix.png
    top_losses.png              the images the model got most confidently wrong
    predictions.csv             every test image: path, true, pred, confidence

The top-losses grid is the error-analysis step of the pipeline: sorted by
loss, these are the images that tell you whether the model's mistakes are
label noise, class overlap, or a preprocessing bug.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from .dataset import build_datasets, build_transforms
from .model import build_model
from .utils import checkpoint_path, get_device, load_config, seed_everything


@torch.inference_mode()
def collect_predictions(cfg: dict, device) -> dict:
    """Run the best checkpoint over the test split, keeping per-sample loss."""
    ckpt = torch.load(checkpoint_path(cfg, "best.pt"), map_location=device, weights_only=False)
    classes = ckpt["classes"]
    model = build_model(cfg, num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = build_datasets(cfg)
    tf = build_transforms(cfg, train=False)
    test_indices = ds["split_indices"]["test"]
    samples = ds["samples"]

    rows = []
    batch, meta = [], []
    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        losses = F.cross_entropy(logits, torch.tensor([m[1] for m in meta], device=device),
                                 reduction="none")
        for (path, target), p, l in zip(meta, probs.cpu(), losses.cpu()):
            pred = int(p.argmax())
            rows.append({"path": path, "true": target, "pred": pred,
                         "confidence": float(p[pred]), "loss": float(l)})
        batch.clear(); meta.clear()

    for idx in test_indices:
        path, target = samples[idx]
        batch.append(tf(Image.open(path).convert("RGB")))
        meta.append((path, target))
        if len(batch) == cfg["data"]["batch_size"]:
            flush()
    flush()
    return {"rows": rows, "classes": classes, "val_acc": ckpt.get("val_acc"),
            "epoch": ckpt.get("epoch")}


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], out: Path):
    fig, ax = plt.subplots(figsize=(1.1 * len(classes) + 2, 1.1 * len(classes) + 1))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    thresh = cm.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_top_losses(rows: list[dict], classes: list[str], out: Path, k: int = 12):
    worst = sorted(rows, key=lambda r: r["loss"], reverse=True)[:k]
    cols = 4
    n_rows = (len(worst) + cols - 1) // cols
    fig, axes = plt.subplots(n_rows, cols, figsize=(3.2 * cols, 3.4 * n_rows))
    for ax, r in zip(np.atleast_1d(axes).ravel(), worst):
        ax.imshow(Image.open(r["path"]).convert("RGB"))
        ax.set_title(f"true: {classes[r['true']]}\npred: {classes[r['pred']]} "
                     f"({r['confidence']:.2f})", fontsize=9)
        ax.axis("off")
    for ax in np.atleast_1d(axes).ravel()[len(worst):]:
        ax.axis("off")
    fig.suptitle("highest-loss test images — start error analysis here", y=1.0)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def evaluate(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    device = get_device()
    result = collect_predictions(cfg, device)
    rows, classes = result["rows"], result["classes"]
    reports = Path(cfg["output"]["dir"]) / "reports"

    y_true = [r["true"] for r in rows]
    y_pred = [r["pred"] for r in rows]
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

    report = classification_report(y_true, y_pred, target_names=classes, digits=3)
    (reports / "classification_report.txt").write_text(
        f"test accuracy: {acc:.4f}  ({len(rows)} images)\n\n{report}", encoding="utf-8")
    plot_confusion_matrix(confusion_matrix(y_true, y_pred), classes,
                          reports / "confusion_matrix.png")
    plot_top_losses(rows, classes, reports / "top_losses.png")

    with open(reports / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "true", "pred", "confidence", "loss"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "true": classes[r["true"]], "pred": classes[r["pred"]]})

    print(f"test accuracy: {acc:.4f} ({len(rows)} images)")
    print(report)
    print(f"reports -> {reports}")
    return {"accuracy": acc, "rows": rows, "classes": classes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
