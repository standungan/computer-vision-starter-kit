"""Segmentation data: images plus same-named mask images where each pixel value is a class id.

    root/
      images/      img_0001.jpg ...
      masks/       img_0001.png ...     (single-channel; pixel value = class id; 255 = ignore)
      classes.txt  background\ncircle\nsquare\ntriangle      (line number = class id)

Image and mask go through the *same* random transform, so a flipped image always has a
flipped mask; masks are resized with nearest-neighbour so class ids are never blended.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from cvkit.data import IMAGENET_MEAN, IMAGENET_STD, list_images, make_loader, require_dir, split_indices


def seg_transforms(size: int, augment: dict | None, train: bool) -> v2.Compose:
    aug = augment or {}
    ops: list = [v2.Resize((size, size))]
    if train:
        if aug.get("hflip", False):
            ops.append(v2.RandomHorizontalFlip())
        if aug.get("vflip", False):
            ops.append(v2.RandomVerticalFlip())
        if aug.get("rotation", 0):
            ops.append(v2.RandomRotation(aug["rotation"]))
        if aug.get("color_jitter", 0):
            j = aug["color_jitter"]
            ops.append(v2.ColorJitter(brightness=j, contrast=j, saturation=j))
    ops += [v2.ToImage(),
            v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64, "others": None}, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return v2.Compose(ops)


class SegmentationDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], transform):
        self.pairs, self.transform = pairs, transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        img_path, mask_path = self.pairs[i]
        with Image.open(img_path) as im:
            image = im.convert("RGB")
        mask = read_mask(mask_path)
        image, mask = self.transform(image, tv_tensors.Mask(mask))
        return image, mask.as_subclass(torch.Tensor)


def read_mask(path: str | Path) -> torch.Tensor:
    arr = np.array(Image.open(path))                 # no .convert(): palette PNGs keep their class ids
    if arr.ndim != 2:
        raise ValueError(f"{path}: mask must be single-channel (pixel value = class id), "
                         f"got shape {arr.shape}. Colour masks need converting to class ids first.")
    return torch.from_numpy(arr.astype(np.uint8))


def read_classes(root: Path, classes_file: str, masks: list[str], ignore_index: int) -> list[str]:
    path = root / classes_file
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    top = max(int(read_mask(m)[read_mask(m) != ignore_index].max(initial=0)) for m in masks[:50])
    print(f"note: {path} not found — inferring {top + 1} classes named class_0..class_{top}")
    return [f"class_{i}" for i in range(top + 1)]


def build_datasets(cfg: dict) -> dict:
    d = cfg["data"]
    root = require_dir(d["root"])
    images = list_images(root / d.get("images_dir", "images"))
    if not images:
        raise FileNotFoundError(f"no images in {root / d.get('images_dir', 'images')}")
    mask_dir = root / d.get("masks_dir", "masks")
    by_stem = {p.stem: p for p in mask_dir.iterdir()} if mask_dir.exists() else {}
    missing = [p.name for p in images if p.stem not in by_stem]
    if missing:
        raise FileNotFoundError(f"{len(missing)} image(s) have no mask in {mask_dir} "
                                f"(masks must share the image's file name stem), e.g. {missing[:5]}")
    pairs = [(str(p), str(by_stem[p.stem])) for p in images]
    ignore = int(d.get("ignore_index", 255))
    classes = read_classes(root, d.get("classes_file", "classes.txt"), [m for _, m in pairs], ignore)
    for _, m in pairs[:50]:                           # catch bad class ids before epoch 1
        bad = read_mask(m)
        bad = bad[(bad != ignore) & (bad >= len(classes))]
        if bad.numel():
            raise ValueError(f"{m} contains class id {int(bad.max())} but classes.txt lists only "
                             f"{len(classes)} classes (ids 0..{len(classes) - 1}); use {ignore} for 'ignore'")
    splits = split_indices(len(pairs), d["split"], cfg["seed"])
    aug, size = d.get("augment", {}), d["image_size"]

    def subset(name, train):
        return SegmentationDataset([pairs[i] for i in splits[name]], seg_transforms(size, aug, train))

    return {"train": subset("train", True), "val": subset("val", False), "test": subset("test", False),
            "classes": classes, "pairs": pairs, "split_indices": splits}


def build_loaders(cfg: dict) -> dict:
    ds = build_datasets(cfg)
    return {"train": make_loader(ds["train"], cfg, shuffle=True),
            "val": make_loader(ds["val"], cfg, shuffle=False),
            "test": make_loader(ds["test"], cfg, shuffle=False), "classes": ds["classes"]}


def test_image_paths(cfg: dict, n: int = 3) -> list[str]:
    return [p for p, _ in build_datasets(cfg)["test"].pairs[:n]]
