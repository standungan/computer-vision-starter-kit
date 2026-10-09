"""Instance-segmentation data: COCO annotations with polygon masks.

    root/
      images/            img_0001.jpg ...
      annotations.json   COCO; every annotation has bbox AND segmentation (polygons)

Masks are rasterized from the polygons on the fly. RLE-encoded masks (iscrowd) are not supported.
"""
from __future__ import annotations

from cvkit.detection import build_coco_datasets, build_coco_loaders  # noqa: F401


def build_datasets(cfg: dict) -> dict:
    return build_coco_datasets(cfg, with_masks=True)


def build_loaders(cfg: dict) -> dict:
    return build_coco_loaders(cfg, with_masks=True)


def test_image_paths(cfg: dict, n: int = 3) -> list[str]:
    return [im["path"] for im in build_datasets(cfg)["test"].images[:n]]
