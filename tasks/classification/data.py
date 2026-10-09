"""Classification data: an ImageFolder-style directory, split once, reproducibly.

    root/
      class_a/ img1.jpg img2.jpg ...
      class_b/ ...

Class names come from the folder names. Splits are stratified per class and fixed
by `seed`, so every script (train / evaluate / app) sees the same test images.
"""
from __future__ import annotations

from collections import Counter

from torchvision import datasets

from cvkit.data import (ImageTargetDataset, image_transforms, make_loader,
                        require_dir, split_indices)


def build_datasets(cfg: dict) -> dict:
    root = require_dir(cfg["data"]["root"])
    base = datasets.ImageFolder(root)                 # just discovers classes + file list
    samples = [(p, t) for p, t in base.samples]
    splits = split_indices(len(samples), cfg["data"]["split"], cfg["seed"], groups=base.targets)
    size, aug = cfg["data"]["image_size"], cfg["data"].get("augment", {})
    train_tf, eval_tf = image_transforms(size, aug, True), image_transforms(size, aug, False)

    def subset(name, tf):
        return ImageTargetDataset([samples[i] for i in splits[name]], tf)

    return {"train": subset("train", train_tf), "val": subset("val", eval_tf),
            "test": subset("test", eval_tf), "classes": base.classes,
            "class_counts": Counter(base.targets), "samples": samples, "split_indices": splits}


def build_loaders(cfg: dict) -> dict:
    ds = build_datasets(cfg)
    return {"train": make_loader(ds["train"], cfg, shuffle=True),
            "val": make_loader(ds["val"], cfg, shuffle=False),
            "test": make_loader(ds["test"], cfg, shuffle=False),
            "classes": ds["classes"]}


def test_image_paths(cfg: dict, n: int = 3) -> list[str]:
    ds = build_datasets(cfg)
    return [p for p, _ in ds["test"].samples[:n]]
