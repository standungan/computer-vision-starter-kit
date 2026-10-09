"""A small, dependency-free reader for COCO-format annotation files.

Used by object detection, instance segmentation and keypoints. We only need
a slice of the format, so we parse the JSON directly instead of pulling in
pycocotools:

    root/
      images/            *.jpg / *.png
      annotations.json   {"images": [...], "annotations": [...], "categories": [...]}

Supported per-annotation fields: `bbox` [x, y, w, h], `category_id`,
`segmentation` as POLYGONS (lists of x,y,...; not RLE), `keypoints`
[x, y, v, ...] and `iscrowd`. Labels are made contiguous: 0 is reserved for
background, so category k (sorted by id) becomes label k + 1 — the convention
torchvision's detection models expect.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_coco(root: str | Path, annotations: str = "annotations.json",
              images_dir: str = "images") -> dict:
    root = Path(root)
    ann_path = root / annotations
    if not ann_path.exists():
        raise FileNotFoundError(
            f"annotation file not found: {ann_path}\n"
            "Expected COCO layout: <root>/images/*.jpg and <root>/annotations.json "
            "(see docs/data-formats.md)")
    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)

    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_to_label = {c["id"]: i + 1 for i, c in enumerate(cats)}
    by_image: dict[int, list] = {}
    for a in coco.get("annotations", []):
        if a.get("iscrowd", 0):
            continue
        x, y, w, h = a["bbox"]
        if w < 1 or h < 1:                                  # degenerate box
            continue
        item = {"bbox": [x, y, x + w, y + h], "label": cat_to_label[a["category_id"]]}
        if a.get("segmentation"):
            if not isinstance(a["segmentation"], list):
                raise ValueError("segmentation must be polygons (list of [x,y,...]); RLE is not supported")
            item["segmentation"] = a["segmentation"]
        if a.get("keypoints"):
            item["keypoints"] = a["keypoints"]
        by_image.setdefault(a["image_id"], []).append(item)

    images = []
    for im in sorted(coco["images"], key=lambda i: i["id"]):
        images.append({"id": im["id"], "path": str(root / images_dir / im["file_name"]),
                       "width": im["width"], "height": im["height"],
                       "annotations": by_image.get(im["id"], [])})
    return {"images": images, "classes": [c["name"] for c in cats],
            "keypoint_names": cats[0].get("keypoints", []) if cats else []}


def polygons_to_mask(polygons: list[list[float]], height: int, width: int) -> np.ndarray:
    """Rasterize COCO polygons into a (H, W) uint8 mask of 0/1."""
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for poly in polygons:
        pts = list(zip(poly[0::2], poly[1::2]))
        if len(pts) >= 3:
            draw.polygon(pts, fill=1)
    return np.array(canvas, dtype=np.uint8)
