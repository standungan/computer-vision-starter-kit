"""Dataset pipeline — dataset-agnostic by design.

Point `data.root` in configs/config.yaml at ANY folder laid out as

    root/
      class_a/ img1.jpg img2.jpg ...
      class_b/ ...

and everything downstream (training, evaluation, the app) adapts:
class names are discovered from folder names, splits are stratified
per class with a fixed seed, and normalization uses ImageNet stats
to match the pretrained backbones in `model.py`.

The example project uses the NEU surface-defect dataset (6 classes of
steel-surface defects, 300 images each) — see README.md for the download.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(cfg: dict, train: bool) -> transforms.Compose:
    """Eval: resize + normalize. Train: adds the augmentations enabled in YAML."""
    size = cfg["data"]["image_size"]
    aug = cfg["data"].get("augment", {})
    ops: list = [transforms.Resize((size, size))]
    if train:
        if aug.get("hflip", False):
            ops.append(transforms.RandomHorizontalFlip())
        if aug.get("vflip", False):
            ops.append(transforms.RandomVerticalFlip())
        if aug.get("rotation", 0):
            ops.append(transforms.RandomRotation(aug["rotation"]))
        if aug.get("color_jitter", 0):
            j = aug["color_jitter"]
            ops.append(transforms.ColorJitter(brightness=j, contrast=j, saturation=j))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if train and aug.get("random_erasing", 0):
        ops.append(transforms.RandomErasing(p=aug["random_erasing"]))
    return transforms.Compose(ops)


def stratified_split(targets: list[int], ratios: dict, seed: int) -> dict[str, list[int]]:
    """Split indices per class so every split keeps the class balance.

    ratios = {"train": 0.7, "val": 0.15, "test": 0.15} (must sum to 1.0).
    """
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    by_class: dict[int, list[int]] = {}
    for idx, t in enumerate(targets):
        by_class.setdefault(t, []).append(idx)
    g = torch.Generator().manual_seed(seed)
    splits: dict[str, list[int]] = {k: [] for k in ratios}
    for cls_indices in by_class.values():
        perm = torch.randperm(len(cls_indices), generator=g).tolist()
        shuffled = [cls_indices[i] for i in perm]
        n = len(shuffled)
        n_train = round(n * ratios["train"])
        n_val = round(n * ratios["val"])
        splits["train"] += shuffled[:n_train]
        splits["val"] += shuffled[n_train : n_train + n_val]
        splits["test"] += shuffled[n_train + n_val :]
    return splits


def build_datasets(cfg: dict) -> dict:
    """Discover classes, split once, return Subsets with the right transforms.

    Two ImageFolder instances share the same file list but carry different
    transforms — Subset then routes each split to the right one.
    """
    root = Path(cfg["data"]["root"])
    if not root.exists():
        raise FileNotFoundError(
            f"data root not found: {root}\n"
            "Download the example dataset (README.md → Dataset) or point "
            "data.root at your own ImageFolder-style directory."
        )
    train_base = datasets.ImageFolder(root, transform=build_transforms(cfg, train=True))
    eval_base = datasets.ImageFolder(root, transform=build_transforms(cfg, train=False))
    splits = stratified_split(train_base.targets, cfg["data"]["split"], cfg["seed"])
    return {
        "train": Subset(train_base, splits["train"]),
        "val": Subset(eval_base, splits["val"]),
        "test": Subset(eval_base, splits["test"]),
        "classes": train_base.classes,
        "class_counts": Counter(train_base.targets),
        "samples": train_base.samples,
        "split_indices": splits,
    }


def build_loaders(cfg: dict) -> dict:
    ds = build_datasets(cfg)
    bs = cfg["data"]["batch_size"]
    nw = cfg["data"].get("num_workers", 2)
    pin = torch.cuda.is_available()
    loaders = {
        name: DataLoader(
            ds[name],
            batch_size=bs,
            shuffle=(name == "train"),
            num_workers=nw,
            pin_memory=pin,
        )
        for name in ("train", "val", "test")
    }
    loaders["classes"] = ds["classes"]
    return loaders
