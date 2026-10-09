"""Detection data: COCO-format boxes. The loading code is shared with instance segmentation.

    root/
      images/            img_0001.jpg ...
      annotations.json   COCO: images[], annotations[] (bbox = [x, y, width, height]), categories[]

See docs/data-formats.md for a 15-line example and how to convert from YOLO or Pascal VOC.
"""
from __future__ import annotations

from cvkit.detection import build_coco_datasets, build_coco_loaders  # noqa: F401  (re-exported)


def test_image_paths(cfg: dict, n: int = 3) -> list[str]:
    ds = build_coco_datasets(cfg)
    return [im["path"] for im in ds["test"].images[:n]]
