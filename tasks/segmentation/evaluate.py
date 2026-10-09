"""Evaluate a segmentation model on the test split.

    python -m cvkit segmentation evaluate --config <config.yaml>

Writes to <output.dir>/reports/:
    metrics.json          mean IoU, mean Dice, pixel accuracy
    classification_report.txt   per-class IoU / Dice
    per_class_iou.png     IoU per class
    confusion_matrix.png  pixels of each true class (rows), by predicted class, in %
    predictions.png       image | ground truth | prediction, for a few test images
    worst_images.png      the same, for the test images with the lowest IoU

IoU = pixels both agree on / pixels either one marks. 1.0 is a perfect outline; accuracy alone
misleads when the background is 90% of the image (predicting "all background" scores 90%).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from cvkit.checkpoint import checkpoint_path, load_checkpoint, restore_model
from cvkit.config import output_dir
from cvkit.metrics import iou_from_confusion, update_confusion
from cvkit.utils import get_device, seed_everything
from cvkit.viz import BLUE, image_grid, overlay_label_map, plot_confusion_matrix, save_fig

from .data import build_datasets, read_mask
from .model import build_model


@torch.inference_mode()
def evaluate(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    ckpt = load_checkpoint(checkpoint_path(cfg, "best.pt"), device)
    model = restore_model(ckpt, build_model, device)
    classes = ckpt["meta"]["classes"]
    n, ignore = len(classes), int(cfg["data"].get("ignore_index", 255))
    test_ds = build_datasets(cfg)["test"]

    cm = torch.zeros(n, n, dtype=torch.long)
    per_image, preds = [], []
    for i in range(len(test_ds)):                         # one image at a time: need per-image scores
        image, mask = test_ds[i]
        pred = model(image.unsqueeze(0).to(device)).argmax(1)[0].cpu()
        update_confusion(cm, pred[None], mask[None], n, ignore)
        one = update_confusion(torch.zeros(n, n, dtype=torch.long), pred[None], mask[None], n, ignore)
        per_image.append(iou_from_confusion(one)["miou"])
        preds.append(pred.numpy().astype(np.uint8))
    res = iou_from_confusion(cm)

    reports = output_dir(cfg) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    metrics = {"miou": res["miou"], "mdice": res["mdice"], "pixel_acc": res["pixel_acc"], "n_test": len(test_ds),
               **{f"iou_{c}": v for c, v in zip(classes, res["iou"])}}
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = [f"{c:<20} IoU {iou:.3f}   Dice {d:.3f}" for c, iou, d in zip(classes, res["iou"], res["dice"])]
    (reports / "classification_report.txt").write_text(
        f"mean IoU {res['miou']:.4f} | mean Dice {res['mdice']:.4f} | pixel accuracy {res['pixel_acc']:.4f} "
        f"({len(test_ds)} images)\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    _plot_iou(classes, res["iou"], reports / "per_class_iou.png")
    pct = (cm.double() / cm.sum(1, keepdim=True).clamp(min=1) * 100).round().numpy().astype(int)
    plot_confusion_matrix(pct, classes, reports / "confusion_matrix.png", "pixels by true class (% of row)")
    order = np.argsort(per_image, kind="stable")
    _plot_triples(test_ds, preds, list(range(min(3, len(test_ds)))), reports / "predictions.png",
                  "image | ground truth | prediction")
    _plot_triples(test_ds, preds, [int(i) for i in order[:3]], reports / "worst_images.png",
                  "test images with the lowest IoU: image | ground truth | prediction")

    print(f"mean IoU {res['miou']:.3f} | mean Dice {res['mdice']:.3f} | pixel accuracy {res['pixel_acc']:.3f} "
          f"({len(test_ds)} test images)")
    print("\n".join(lines))
    print(f"reports -> {reports}")
    return metrics


def _plot_iou(classes, ious, out) -> None:
    fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * len(classes) + 2), 3.4))
    ax.bar(classes, ious, color=BLUE, width=0.55)
    for i, v in enumerate(ious):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("IoU")
    ax.set_title("IoU per class (test split)")
    fig.tight_layout()
    save_fig(fig, out)


def _plot_triples(ds, preds, indices, out, title) -> None:
    images, titles = [], []
    for i in indices:
        img_path, mask_path = ds.pairs[i]
        im = Image.open(img_path).convert("RGB").resize(preds[i].shape[::-1])
        gt = np.asarray(Image.fromarray(read_mask(mask_path).numpy()).resize(preds[i].shape[::-1], Image.NEAREST))
        images += [im, overlay_label_map(im, gt), overlay_label_map(im, preds[i])]
        titles += ["image", "ground truth", "prediction"]
    image_grid(images, titles, cols=3, out=out, suptitle=title, cell=2.6)
