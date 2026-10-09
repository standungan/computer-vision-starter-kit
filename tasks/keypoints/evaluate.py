"""Evaluate a keypoint model on the test split.

    python -m cvkit keypoints evaluate --config <config.yaml>

Writes to <output.dir>/reports/:
    metrics.json          NME, PCK at the config threshold, mean error in pixels, per-keypoint values
    pck_curve.png         share of keypoints within a given distance: the whole accuracy picture at once
    error_per_keypoint.png  which landmark is hardest?
    predictions.png       truth (white ring) vs prediction (coloured dot) on test images
    worst_images.png      the same for the test images with the largest error

Reading the PCK curve: at x = 0.05 the curve's height is the fraction of keypoints within 5% of the
image side. A curve that shoots up near 0 means precise localisation; a slow climb means sloppy.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from cvkit.checkpoint import checkpoint_path, load_checkpoint, restore_model
from cvkit.config import output_dir
from cvkit.data import make_loader
from cvkit.utils import get_device, seed_everything
from cvkit.viz import BLUE, draw_keypoints, image_grid, save_fig

from .data import build_datasets
from .metrics import keypoint_metrics, pck_curve
from .model import build_model
from .train import predict_loader


def evaluate(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    ckpt = load_checkpoint(checkpoint_path(cfg, "best.pt"), device)
    model = restore_model(ckpt, build_model, device)
    names = ckpt["meta"]["keypoint_names"]
    size = int(ckpt["config"]["data"]["image_size"])
    thr = float(cfg.get("eval", {}).get("pck_threshold", 0.05))

    test_ds = build_datasets(cfg)["test"]
    pred, true, vis = predict_loader(model, make_loader(test_ds, cfg, shuffle=False), device)
    m = keypoint_metrics(pred, true, vis, size, thr)

    reports = output_dir(cfg) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    metrics = {"nme": m["nme"], "pck": m["pck"], "pck_threshold": thr, "mean_err_px": m["mean_err_px"],
               "n_test": len(pred), **{f"pck_{n}": v for n, v in zip(names, m["pck_per_keypoint"])},
               **{f"nme_{n}": v for n, v in zip(names, m["nme_per_keypoint"])}}
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _plot_curve(pred, true, vis, size, thr, reports / "pck_curve.png")
    _plot_error(names, m["nme_per_keypoint"], reports / "error_per_keypoint.png")
    per_image = ((pred - true).norm(dim=-1) * vis).sum(1) / vis.sum(1).clamp(min=1)
    worst = [int(i) for i in torch.argsort(-per_image)[:4]]
    _plot_pairs(test_ds, pred, true, vis, names, size, list(range(min(4, len(pred)))),
                reports / "predictions.png", "truth (white ring) vs prediction (dot)")
    _plot_pairs(test_ds, pred, true, vis, names, size, worst, reports / "worst_images.png",
                "test images with the largest keypoint error")

    print(f"NME {m['nme']:.4f} (of image side) | mean error {m['mean_err_px']:.2f}px at {size}px | "
          f"PCK@{thr} {m['pck']:.3f} ({len(pred)} test images)")
    for n, p, e in zip(names, m["pck_per_keypoint"], m["nme_per_keypoint"]):
        print(f"  {n:<14} PCK {p:.3f}   NME {e:.4f}")
    print(f"reports -> {reports}")
    return metrics


def _plot_curve(pred, true, vis, size, thr, out) -> None:
    grid = np.linspace(0, 0.2, 41)
    ys = pck_curve(pred, true, vis, size, grid.tolist())
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(grid, ys, color=BLUE)
    ax.axvline(thr, color="#eb6834", linestyle="--")
    ax.annotate(f"threshold {thr}", (thr, 0.02), xytext=(4, 0), textcoords="offset points", color="#52514e", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("allowed error (fraction of image side)")
    ax.set_ylabel("share of keypoints correct")
    ax.set_title("PCK curve")
    fig.tight_layout()
    save_fig(fig, out)


def _plot_error(names, nme, out) -> None:
    fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * len(names) + 2), 3.4))
    ax.bar(names, nme, color=BLUE, width=0.55)
    for i, v in enumerate(nme):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("NME (fraction of image side)")
    ax.set_title("localisation error per keypoint")
    fig.tight_layout()
    save_fig(fig, out)


def _plot_pairs(ds, pred, true, vis, names, size, indices, out, title) -> None:
    images, titles = [], []
    for i in indices:
        im = Image.open(ds.items[i]["path"]).convert("RGB").resize((size, size))
        im = draw_keypoints(im, pred[i].numpy(), names, radius=3)
        d = ImageDraw.Draw(im)
        for (x, y), v in zip(true[i].tolist(), vis[i].tolist()):
            if v:
                d.ellipse([x - 7, y - 7, x + 7, y + 7], outline="white", width=2)
        images.append(im)
        titles.append(f"mean error {float((pred[i] - true[i]).norm(dim=-1).mean()):.1f}px")
    image_grid(images, titles, cols=4, out=out, suptitle=title, cell=2.8)
