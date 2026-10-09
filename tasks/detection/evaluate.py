"""Evaluate a detector on the test split.

    python -m cvkit detection evaluate --config <config.yaml>

Writes to <output.dir>/reports/:
    metrics.json         mAP@[.5:.95], mAP@.5, mAP@.75, precision / recall at the score threshold
    ap_per_class.png     AP@.5 per class: which classes are hard?
    predictions.png      ground truth next to the model's predictions on test images
    worst_images.png     the same, for the images with the most missed + false detections

Reading mAP: a prediction counts as correct if its box overlaps a true box of the same class
by at least some IoU (intersection-over-union). mAP@.5 uses IoU >= 0.5 (generous);
mAP@[.5:.95] averages IoU thresholds from 0.5 to 0.95 (strict: boxes must be tight).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cvkit.checkpoint import checkpoint_path, load_checkpoint, restore_model
from cvkit.config import output_dir
from cvkit.data import make_loader
from cvkit.detection import box_map, build_coco_datasets, collate, match_at_threshold, predict_loader
from cvkit.utils import get_device, seed_everything
from cvkit.viz import BLUE, draw_boxes, image_grid, save_fig

from .model import build_model


def evaluate(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    ckpt = load_checkpoint(checkpoint_path(cfg, "best.pt"), device)
    model = restore_model(ckpt, build_model, device)
    classes = ckpt["meta"]["classes"]
    thr = float(cfg.get("eval", {}).get("score_threshold", 0.5))

    test_ds = build_coco_datasets(cfg)["test"]
    preds, gts = predict_loader(model, make_loader(test_ds, cfg, False, collate), device)
    m = box_map(preds, gts, len(classes))
    op = match_at_threshold(preds, gts, thr)
    metrics = {"map": m["map"], "map50": m["map50"], "map75": m["map75"],
               "precision": op["precision"], "recall": op["recall"], "score_threshold": thr,
               "n_test": len(gts), **{f"ap50_{c}": a for c, a in zip(classes, m["ap50_per_class"])}}

    reports = output_dir(cfg) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _plot_ap(classes, m["ap50_per_class"], reports / "ap_per_class.png")
    first = list(range(min(4, len(gts))))
    worst = list(np.argsort(-np.array(op["errors_per_image"]), kind="stable")[:4])
    _plot_pairs(test_ds, preds, first, classes, thr, reports / "predictions.png", "ground truth vs prediction")
    _plot_pairs(test_ds, preds, worst, classes, thr, reports / "worst_images.png",
                "images with the most missed + false detections")

    print(f"mAP@[.5:.95] {m['map']:.3f} | mAP@.5 {m['map50']:.3f} | mAP@.75 {m['map75']:.3f} "
          f"({len(gts)} test images)")
    print(f"at score >= {thr}: precision {op['precision']:.3f} | recall {op['recall']:.3f} "
          f"(TP {op['tp']}, FP {op['fp']}, FN {op['fn']})")
    for c, a in zip(classes, m["ap50_per_class"]):
        print(f"  AP@.5 {c:<16} {a:.3f}")
    print(f"reports -> {reports}")
    return metrics


def _plot_ap(classes, aps, out) -> None:
    fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * len(classes) + 2), 3.4))
    ax.bar(classes, aps, color=BLUE, width=0.55)
    for i, a in enumerate(aps):
        ax.text(i, a + 0.02, f"{a:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("AP@.5")
    ax.set_title("average precision per class")
    fig.tight_layout()
    save_fig(fig, out)


def _plot_pairs(ds, preds, indices, classes, thr, out, title) -> None:
    """Each chosen test image appears twice: ground truth, then predictions."""
    images, titles = [], []
    for i in indices:
        info = ds.images[i]
        im = Image.open(info["path"]).convert("RGB")
        gt = info["annotations"]
        images.append(draw_boxes(im, [a["bbox"] for a in gt], [a["label"] for a in gt], classes=classes))
        titles.append("ground truth")
        keep = preds[i]["scores"] >= thr
        images.append(draw_boxes(im, preds[i]["boxes"][keep].numpy(), preds[i]["labels"][keep].numpy(),
                                 preds[i]["scores"][keep].numpy(), classes))
        titles.append("prediction")
    image_grid(images, titles, cols=4, out=out, suptitle=title, cell=2.8)
