"""Evaluate an instance-segmentation model on the test split.

    python -m cvkit instance_segmentation evaluate --config <config.yaml>

Writes to <output.dir>/reports/:
    metrics.json          mask mAP and box mAP, precision / recall at the score threshold
    ap_per_class.png      mask AP@.5 per class
    predictions.png       ground-truth masks next to predicted masks
    worst_images.png      the same, for the images with the most missed + false detections

Mask mAP works like box mAP, but "overlap" is the IoU of the two *masks*: shared pixels divided
by the pixels covered by either. It rewards getting the outline right, not just the rectangle.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cvkit.checkpoint import checkpoint_path, load_checkpoint, restore_model
from cvkit.config import output_dir
from cvkit.data import make_loader
from cvkit.detection import box_map, collate, mask_map, match_at_threshold, predict_loader
from cvkit.utils import get_device, seed_everything
from cvkit.viz import BLUE, draw_boxes, image_grid, overlay_masks, save_fig

from .data import build_datasets
from .model import build_model


def evaluate(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    ckpt = load_checkpoint(checkpoint_path(cfg, "best.pt"), device)
    model = restore_model(ckpt, build_model, device)
    classes = ckpt["meta"]["classes"]
    ev = cfg.get("eval", {})
    thr, mthr = float(ev.get("score_threshold", 0.5)), float(ev.get("mask_threshold", 0.5))

    test_ds = build_datasets(cfg)["test"]
    preds, gts = predict_loader(model, make_loader(test_ds, cfg, False, collate), device,
                                with_masks=True, mask_threshold=mthr)
    k, b = mask_map(preds, gts, len(classes)), box_map(preds, gts, len(classes))
    op = match_at_threshold(preds, gts, thr)
    metrics = {"mask_map": k["map"], "mask_map50": k["map50"], "mask_map75": k["map75"],
               "box_map": b["map"], "box_map50": b["map50"],
               "precision": op["precision"], "recall": op["recall"], "score_threshold": thr,
               "n_test": len(gts), **{f"mask_ap50_{c}": a for c, a in zip(classes, k["ap50_per_class"])}}

    reports = output_dir(cfg) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _plot_ap(classes, k["ap50_per_class"], reports / "ap_per_class.png")
    first = list(range(min(4, len(gts))))
    worst = list(np.argsort(-np.array(op["errors_per_image"]), kind="stable")[:4])
    _plot_pairs(test_ds, preds, gts, first, classes, thr, reports / "predictions.png", "ground truth vs prediction")
    _plot_pairs(test_ds, preds, gts, worst, classes, thr, reports / "worst_images.png",
                "images with the most missed + false detections")

    print(f"mask mAP@[.5:.95] {k['map']:.3f} | mask mAP@.5 {k['map50']:.3f} | mask mAP@.75 {k['map75']:.3f}")
    print(f"box  mAP@[.5:.95] {b['map']:.3f} | box  mAP@.5 {b['map50']:.3f}  ({len(gts)} test images)")
    print(f"at score >= {thr}: precision {op['precision']:.3f} | recall {op['recall']:.3f} "
          f"(TP {op['tp']}, FP {op['fp']}, FN {op['fn']})")
    print(f"reports -> {reports}")
    return metrics


def _plot_ap(classes, aps, out) -> None:
    fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * len(classes) + 2), 3.4))
    ax.bar(classes, aps, color=BLUE, width=0.55)
    for i, a in enumerate(aps):
        ax.text(i, a + 0.02, f"{a:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("mask AP@.5")
    ax.set_title("mask average precision per class")
    fig.tight_layout()
    save_fig(fig, out)


def _plot_pairs(ds, preds, gts, indices, classes, thr, out, title) -> None:
    images, titles = [], []
    for i in indices:
        im = Image.open(ds.images[i]["path"]).convert("RGB")
        g = gts[i]
        images.append(draw_boxes(overlay_masks(im, g["masks"].numpy(), g["labels"].numpy()),
                                 g["boxes"].numpy(), g["labels"].numpy(), classes=classes, width=1))
        titles.append("ground truth")
        keep = preds[i]["scores"] >= thr
        p = {k: v[keep] for k, v in preds[i].items()}
        shaded = overlay_masks(im, p["masks"].numpy(), p["labels"].numpy()) if len(p["labels"]) else im
        images.append(draw_boxes(shaded, p["boxes"].numpy(), p["labels"].numpy(), p["scores"].numpy(),
                                 classes, width=1))
        titles.append("prediction")
    image_grid(images, titles, cols=4, out=out, suptitle=title, cell=2.8)
