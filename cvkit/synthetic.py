"""Synthetic demo data: tiny "shapes" datasets, generated in seconds, offline.

Why: the fastest way to learn a pipeline is to run it end to end. Real datasets
need accounts, downloads and the right folder layout; these need nothing.
Every task has a generator that writes exactly the layout the task expects
(see docs/data-formats.md), so the demo doubles as a format example.

The scenes contain circles, squares and triangles with random colours,
positions, rotations and noise. Colour is independent of class on purpose, so
a model has to learn *shape*, not "the red one is a circle".

    python -m cvkit classification demo-data --config tasks/classification/configs/quick.yaml
"""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SHAPES = ["circle", "square", "triangle"]


# ----------------------------------------------------------------- scene drawing
def _polygon(shape: str, cx: float, cy: float, r: float, angle: float) -> list[float]:
    """Flat [x0, y0, x1, y1, ...] outline of a shape of 'radius' r, rotated by `angle` degrees."""
    if shape == "circle":
        pts = [(cx + r * math.cos(2 * math.pi * k / 24), cy + r * math.sin(2 * math.pi * k / 24))
               for k in range(24)]
    else:
        n = 4 if shape == "square" else 3
        a0 = math.radians(angle) + (math.pi / 4 if n == 4 else -math.pi / 2)
        pts = [(cx + r * math.cos(a0 + 2 * math.pi * k / n),
                cy + r * math.sin(a0 + 2 * math.pi * k / n)) for k in range(n)]
    return [float(v) for p in pts for v in p]


def _bbox(poly: list[float]) -> list[float]:
    xs, ys = poly[0::2], poly[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def _background(rng: random.Random, size: int) -> Image.Image:
    base = rng.randint(190, 240)
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, 7, (size, size, 3))
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _color(rng: random.Random) -> tuple[int, int, int]:
    return tuple(rng.randint(20, 170) for _ in range(3))


def _overlap(a: list[float], b: list[float], pad: float = 3.0) -> bool:
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def render_scene(rng: random.Random, size: int, shapes: list[str], radius=(0.10, 0.22),
                 separate: bool = True) -> tuple[Image.Image, list[dict]]:
    """Draw the given shapes on a noisy background. Returns the image and, per shape,
    {"label": 1-based index into SHAPES, "polygon": [...], "bbox": [x0,y0,x1,y1]}."""
    img = _background(rng, size)
    draw = ImageDraw.Draw(img)
    items: list[dict] = []
    for name in shapes:
        for _ in range(50):                                       # rejection-sample a free spot
            r = rng.uniform(*radius) * size
            cx, cy = rng.uniform(r + 2, size - r - 2), rng.uniform(r + 2, size - r - 2)
            poly = _polygon(name, cx, cy, r, rng.uniform(0, 360))
            box = _bbox(poly)
            if not separate or not any(_overlap(box, it["bbox"]) for it in items):
                break
        draw.polygon(list(zip(poly[0::2], poly[1::2])), fill=_color(rng))
        items.append({"label": SHAPES.index(name) + 1, "polygon": poly, "bbox": box})
    return img, items


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------ one per task
def make_classification(root, n: int = 120, size: int = 96, seed: int = 0) -> Path:
    """root/<class>/img_*.png — one shape per image; n images in total, balanced."""
    root, rng = Path(root), random.Random(seed)
    per_class = max(n // len(SHAPES), 3)
    for name in SHAPES:
        folder = _mkdir(root / name)
        for i in range(per_class):
            img, _ = render_scene(rng, size, [name], radius=(0.15, 0.35))
            img.save(folder / f"img_{i:04d}.png")
    return root


def make_multilabel(root, n: int = 200, size: int = 96, seed: int = 0) -> Path:
    """root/images/*.png + root/labels.csv (filename, circle, square, triangle as 0/1)."""
    root, rng = Path(root), random.Random(seed)
    images = _mkdir(root / "images")
    rows = [["filename", *SHAPES]]
    for i in range(n):
        present = [s for s in SHAPES if rng.random() < 0.5] or [rng.choice(SHAPES)]
        img, _ = render_scene(rng, size, present, radius=(0.12, 0.2))
        name = f"img_{i:04d}.png"
        img.save(images / name)
        rows.append([name, *[int(s in present) for s in SHAPES]])
    _write_csv(root / "labels.csv", rows)
    return root


def make_regression(root, n: int = 200, size: int = 96, seed: int = 0) -> Path:
    """root/images/*.png + root/labels.csv (filename, value): value = the circle's radius in pixels."""
    root, rng = Path(root), random.Random(seed)
    images = _mkdir(root / "images")
    rows = [["filename", "value"]]
    for i in range(n):
        frac = rng.uniform(0.08, 0.38)
        img, items = render_scene(rng, size, ["circle"], radius=(frac, frac))
        name = f"img_{i:04d}.png"
        img.save(images / name)
        rows.append([name, round(frac * size, 2)])
    _write_csv(root / "labels.csv", rows)
    return root


def make_coco_shapes(root, n: int = 120, size: int = 128, seed: int = 0) -> Path:
    """root/images/*.png + root/annotations.json (COCO: boxes AND polygons) with 1-3 shapes per image.
    Used by both object detection and instance segmentation."""
    root, rng = Path(root), random.Random(seed)
    images = _mkdir(root / "images")
    coco = {"images": [], "annotations": [],
            "categories": [{"id": i + 1, "name": s} for i, s in enumerate(SHAPES)]}
    ann_id = 1
    for i in range(n):
        shapes = [rng.choice(SHAPES) for _ in range(rng.randint(1, 3))]
        img, items = render_scene(rng, size, shapes)
        name = f"img_{i:04d}.png"
        img.save(images / name)
        coco["images"].append({"id": i + 1, "file_name": name, "width": size, "height": size})
        for it in items:
            x0, y0, x1, y1 = it["bbox"]
            coco["annotations"].append({
                "id": ann_id, "image_id": i + 1, "category_id": it["label"], "iscrowd": 0,
                "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                "area": round((x1 - x0) * (y1 - y0), 2),
                "segmentation": [[round(v, 2) for v in it["polygon"]]]})
            ann_id += 1
    (root / "annotations.json").write_text(json.dumps(coco), encoding="utf-8")
    return root


def make_segmentation(root, n: int = 120, size: int = 128, seed: int = 0) -> Path:
    """root/images/*.png, root/masks/*.png (pixel value = class id, 0 = background), root/classes.txt."""
    root, rng = Path(root), random.Random(seed)
    images, masks = _mkdir(root / "images"), _mkdir(root / "masks")
    (root / "classes.txt").write_text("\n".join(["background", *SHAPES]) + "\n", encoding="utf-8")
    for i in range(n):
        shapes = [rng.choice(SHAPES) for _ in range(rng.randint(1, 3))]
        img, items = render_scene(rng, size, shapes, separate=False)
        mask = Image.new("L", (size, size), 0)
        md = ImageDraw.Draw(mask)
        for it in items:                                          # same paint order as the image
            md.polygon(list(zip(it["polygon"][0::2], it["polygon"][1::2])), fill=it["label"])
        name = f"img_{i:04d}.png"
        img.save(images / name)
        mask.save(masks / name)
    return root


KEYPOINT_NAMES = ["apex", "left", "right"]


def make_keypoints(root, n: int = 160, size: int = 128, seed: int = 0) -> Path:
    """root/images/*.png + root/annotations.json (COCO keypoints): one isosceles triangle per
    image, keypoints = apex, base-left, base-right (visibility 2 = labelled and visible)."""
    root, rng = Path(root), random.Random(seed)
    images = _mkdir(root / "images")
    coco = {"images": [], "annotations": [],
            "categories": [{"id": 1, "name": "triangle", "keypoints": KEYPOINT_NAMES}]}
    for i in range(n):
        img = _background(rng, size)
        r = rng.uniform(0.18, 0.32) * size
        cx, cy = rng.uniform(r + 2, size - r - 2), rng.uniform(r + 2, size - r - 2)
        tilt = math.radians(rng.uniform(-35, 35))
        # apex points "up" (-90 deg) before tilt; base corners 140 / 40 deg off the apex direction
        pts = []
        for off in (0.0, math.radians(140), math.radians(-140)):
            a = -math.pi / 2 + tilt + off
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        ImageDraw.Draw(img).polygon(pts, fill=_color(rng))
        # image-left base corner first: sort the two base corners by x
        base = sorted(pts[1:], key=lambda p: p[0])
        kps = [pts[0], base[0], base[1]]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        name = f"img_{i:04d}.png"
        img.save(images / name)
        coco["images"].append({"id": i + 1, "file_name": name, "width": size, "height": size})
        coco["annotations"].append({
            "id": i + 1, "image_id": i + 1, "category_id": 1, "iscrowd": 0,
            "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
            "keypoints": [round(v, 2) for p in kps for v in (p[0], p[1], 2)],
            "num_keypoints": 3})
    (root / "annotations.json").write_text(json.dumps(coco), encoding="utf-8")
    return root


def _write_csv(path: Path, rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


MAKERS = {
    "classification": make_classification,
    "multilabel": make_multilabel,
    "regression": make_regression,
    "detection": make_coco_shapes,
    "instance_segmentation": make_coco_shapes,
    "segmentation": make_segmentation,
    "keypoints": make_keypoints,
}


def make_demo_data(task: str, cfg: dict) -> Path:
    """Generate the demo dataset for `task` at cfg['data']['root'] (sizes from cfg['demo'])."""
    demo = cfg.get("demo", {})
    kwargs = {k: demo[k] for k in ("n", "size", "seed") if k in demo}
    return MAKERS[task](cfg["data"]["root"], **kwargs)
