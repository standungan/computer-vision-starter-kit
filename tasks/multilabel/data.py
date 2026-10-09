"""Multi-label data: an images folder plus a labels.csv with one 0/1 column per label.

    root/
      images/   img_0001.jpg ...
      labels.csv
          filename,circle,square,triangle
          img_0001.jpg,1,0,1
          img_0002.jpg,0,1,0

Label names come from the CSV header. The split is random (fixed by `seed`).
"""
from __future__ import annotations

import torch

from cvkit.data import ImageTargetDataset, image_transforms, make_loader, read_image_table, split_indices


def build_datasets(cfg: dict) -> dict:
    d = cfg["data"]
    classes, samples = read_image_table(d["root"], d.get("images_dir", "images"),
                                        d.get("labels_file", "labels.csv"))
    bad = [p for p, t in samples if any(v not in (0.0, 1.0) for v in t)]
    if bad:
        raise ValueError(f"labels must be 0 or 1; first offending image: {bad[0]}")
    splits = split_indices(len(samples), d["split"], cfg["seed"])
    aug = d.get("augment", {})
    train_tf, eval_tf = image_transforms(d["image_size"], aug, True), image_transforms(d["image_size"], aug, False)

    def subset(name, tf):
        return ImageTargetDataset([samples[i] for i in splits[name]], tf, target_dtype=torch.float32)

    pos = torch.tensor([samples[i][1] for i in splits["train"]]).sum(0)
    return {"train": subset("train", train_tf), "val": subset("val", eval_tf),
            "test": subset("test", eval_tf), "classes": classes, "samples": samples,
            "split_indices": splits, "train_positive_counts": pos.tolist(),
            "n_train": len(splits["train"])}


def build_loaders(cfg: dict) -> dict:
    ds = build_datasets(cfg)
    return {"train": make_loader(ds["train"], cfg, shuffle=True),
            "val": make_loader(ds["val"], cfg, shuffle=False),
            "test": make_loader(ds["test"], cfg, shuffle=False),
            "classes": ds["classes"], "train_positive_counts": ds["train_positive_counts"],
            "n_train": ds["n_train"]}


def test_image_paths(cfg: dict, n: int = 3) -> list[str]:
    return [p for p, _ in build_datasets(cfg)["test"].samples[:n]]
