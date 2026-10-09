"""Plotting + image-drawing helpers shared by every task.

Chart colors come from one validated palette (blue = train, orange = val, aqua = a
third series; a one-hue blue ramp for magnitudes such as confusion matrices) so
every figure in the kit looks like it belongs to the same family. Overlays on
photos (boxes, masks, keypoints) use an 8-colour categorical cycle assigned by
class index, so a class keeps its colour across images.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "kit_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "lines.linewidth": 2.0, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


apply_style()


def save_fig(fig, path: str | Path | None, dpi: int = 150) -> None:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------- learning curves
def read_history(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def col_is_train(col: str) -> bool:
    return col.startswith("train_")


def plot_history(history: list[dict] | str | Path, out: str | Path | None = None):
    """One panel per metric that has both a train_ and a val_ column (loss, acc, ...).

    Reading guide: a widening gap between train and val means overfitting;
    both curves flat and poor means underfitting; a jumpy val curve means the
    learning rate is too high or the val split is small.
    """
    if not isinstance(history, list):
        history = read_history(history)
    names = [k[len("train_"):] for k in history[0]
             if k.startswith("train_") and f"val_{k[len('train_'):]}" in history[0]]
    singles = [k for k in history[0] if k.startswith("val_") and f"train_{k[4:]}" not in history[0]]
    panels = [(n, [f"train_{n}", f"val_{n}"]) for n in names] + [(s[4:], [s]) for s in singles]
    if "train_loss" in history[0] and "val_loss" not in history[0]:   # e.g. detection: no val loss
        panels.insert(0, ("train loss", ["train_loss"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.6), squeeze=False)
    epochs = [int(h["epoch"]) for h in history]
    for ax, (name, cols) in zip(axes[0], panels):
        for col, color in zip(cols, (BLUE, ORANGE) if len(cols) == 2 else
                              ((BLUE,) if col_is_train(cols[0]) else (ORANGE,))):
            vals = [h[col] for h in history]
            ax.plot(epochs, vals, color=color, label=col.split("_")[0])
            ax.annotate(col.split("_")[0], (epochs[-1], vals[-1]), xytext=(4, 0),
                        textcoords="offset points", color=INK_2, fontsize=9, va="center")
        ax.set_title(name)
        ax.set_xlabel("epoch")
        if len(epochs) <= 12:
            ax.set_xticks(epochs)
        if len(cols) == 2:
            ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_fig(fig, out)
    return fig


# --------------------------------------------------------------- confusion matrix
def plot_confusion_matrix(cm: np.ndarray, classes: list[str], out: str | Path | None = None,
                          title: str = "confusion matrix"):
    """Rows = true class, columns = predicted. A perfect model is a bright diagonal."""
    n = len(classes)
    fig, ax = plt.subplots(figsize=(0.9 * n + 2.5, 0.9 * n + 1.8))
    ax.grid(False)
    ax.imshow(cm, cmap=SEQUENTIAL)
    ax.set_xticks(range(n), classes, rotation=45, ha="right")
    ax.set_yticks(range(n), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)
    top = cm.max() / 2 if cm.max() else 1
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > top else INK, fontsize=9)
    fig.tight_layout()
    save_fig(fig, out)
    return fig


# --------------------------------------------------------------------- image grids
def image_grid(images: list, titles: list[str] | None = None, cols: int = 4,
               out: str | Path | None = None, suptitle: str | None = None, cell: float = 3.0):
    """Show images (PIL / ndarray / HxWx3 tensor) in a grid with optional titles."""
    n = len(images)
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cell * cols, (cell + 0.4) * rows), squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if k < n:
            ax.imshow(images[k], cmap="gray" if getattr(images[k], "ndim", 3) == 2 else None)
            if titles:
                ax.set_title(titles[k], fontsize=9, loc="left")
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out)
    return fig


# -------------------------------------------------------- drawing on a single image
def _font(size: int = 12):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                       # very old Pillow
        return ImageFont.load_default()


def color_for(label: int) -> str:
    return CLASS_COLORS[label % len(CLASS_COLORS)]


def draw_boxes(image: Image.Image, boxes, labels, scores=None, classes: list[str] | None = None,
               width: int = 2) -> Image.Image:
    """Return a copy of `image` with labelled boxes. `labels` are 1-based (0 = background)."""
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    font = _font(12)
    for k, (box, lab) in enumerate(zip(np.asarray(boxes).reshape(-1, 4), np.asarray(labels).reshape(-1))):
        lab = int(lab)
        x0, y0, x1, y1 = [float(v) for v in box]
        color = color_for(lab - 1)
        d.rectangle([x0, y0, x1, y1], outline=color, width=width)
        name = classes[lab - 1] if classes and 0 < lab <= len(classes) else str(lab)
        text = f"{name} {float(scores[k]):.2f}" if scores is not None else name
        tw = d.textlength(text, font=font)
        tx = max(0.0, min(x0, img.width - tw - 6))                  # keep the label inside the image
        ty = max(y0 - 15, 0)
        d.rectangle([tx, ty, tx + tw + 6, ty + 15], fill=color)
        d.text((tx + 3, ty), text, fill="white", font=font)
    return img


def _outline(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Boundary pixels of a boolean mask (mask minus its erosion), `thickness` pixels wide."""
    m = np.pad(mask.astype(bool), 1)
    inner = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    edge = mask.astype(bool) & ~inner
    for _ in range(thickness - 1):
        e = np.pad(edge, 1)
        edge = e[1:-1, 1:-1] | e[:-2, 1:-1] | e[2:, 1:-1] | e[1:-1, :-2] | e[1:-1, 2:]
    return edge


def overlay_label_map(image: Image.Image, label_map: np.ndarray, alpha: float = 0.45,
                      background: int = 0) -> Image.Image:
    """Tint every pixel by its class id (semantic segmentation). Background stays untouched."""
    img = image.convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    lm = np.asarray(Image.fromarray(label_map.astype(np.uint8)).resize(img.size, Image.NEAREST))
    for lab in np.unique(lm):
        if lab == background or lab == 255:
            continue
        rgb = np.array(matplotlib.colors.to_rgb(color_for(int(lab) - 1))) * 255
        m = lm == lab
        arr[m] = (1 - alpha) * arr[m] + alpha * rgb
        arr[_outline(m, max(1, round(min(m.shape) / 160)))] = rgb     # solid edge: visible on any fill
    return Image.fromarray(arr.astype(np.uint8))


def overlay_masks(image: Image.Image, masks, labels, alpha: float = 0.5) -> Image.Image:
    """Tint each instance mask (N,H,W) with its class colour (instance segmentation)."""
    img = image.convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    for m, lab in zip(np.asarray(masks), np.asarray(labels).reshape(-1)):
        rgb = np.array(matplotlib.colors.to_rgb(color_for(int(lab) - 1))) * 255
        mm = np.asarray(m).astype(bool)
        arr[mm] = (1 - alpha) * arr[mm] + alpha * rgb
        arr[_outline(mm, max(1, round(min(mm.shape) / 160)))] = rgb   # solid edge: visible on any fill
    return Image.fromarray(arr.astype(np.uint8))


def draw_keypoints(image: Image.Image, points, names: list[str] | None = None,
                   radius: int = 4, visible=None) -> Image.Image:
    """Dot (and optional name) for every (x, y) keypoint."""
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    font = _font(11)
    for k, (x, y) in enumerate(np.asarray(points).reshape(-1, 2)):
        if visible is not None and not visible[k]:
            continue
        color = color_for(k)
        d.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline="white")
        if names:
            d.text((x + radius + 2, y - radius), names[k], fill=color, font=font)
    return img
