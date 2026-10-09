"""Regression data: an images folder plus a labels.csv with one numeric target.

    root/
      images/   img_0001.png ...
      labels.csv
          filename,value
          img_0001.png,17.5
          img_0002.png,9.0

Targets stay in their original units here. Training standardizes them with the *train*
split's mean and std (stored in the checkpoint), so the network learns values near 0..1
instead of e.g. 0..400, and predictions are converted back to real units.
"""
from __future__ import annotations

import torch

from cvkit.data import ImageTargetDataset, image_transforms, make_loader, read_image_table, split_indices


def build_datasets(cfg: dict) -> dict:
    d = cfg["data"]
    columns, samples = read_image_table(d["root"], d.get("images_dir", "images"),
                                        d.get("labels_file", "labels.csv"))
    if len(columns) != 1:
        raise ValueError(f"regression expects exactly one target column after 'filename', got {columns}")
    samples = [(p, t[0]) for p, t in samples]
    splits = split_indices(len(samples), d["split"], cfg["seed"])
    aug = d.get("augment", {})
    train_tf, eval_tf = image_transforms(d["image_size"], aug, True), image_transforms(d["image_size"], aug, False)

    def subset(name, tf):
        return ImageTargetDataset([samples[i] for i in splits[name]], tf, target_dtype=torch.float32)

    values = torch.tensor([samples[i][1] for i in splits["train"]])
    stats = {"mean": float(values.mean()), "std": float(values.std().clamp(min=1e-6)) if len(values) > 1 else 1.0,
             "target_name": columns[0]}
    return {"train": subset("train", train_tf), "val": subset("val", eval_tf),
            "test": subset("test", eval_tf), "stats": stats, "samples": samples,
            "split_indices": splits}


def build_loaders(cfg: dict) -> dict:
    ds = build_datasets(cfg)
    return {"train": make_loader(ds["train"], cfg, shuffle=True),
            "val": make_loader(ds["val"], cfg, shuffle=False),
            "test": make_loader(ds["test"], cfg, shuffle=False), "stats": ds["stats"]}


def test_image_paths(cfg: dict, n: int = 3) -> list[str]:
    return [p for p, _ in build_datasets(cfg)["test"].samples[:n]]
