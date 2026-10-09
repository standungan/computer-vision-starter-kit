"""Keypoint data: COCO keypoints, one labelled object per image.

    root/
      images/            img_0001.jpg ...
      annotations.json   COCO: categories[0]["keypoints"] = ["apex", "left", "right"]
                         annotations[i]["keypoints"] = [x1, y1, v1, x2, y2, v2, ...]
                         v = 0 not labelled | 1 labelled but hidden | 2 labelled and visible

Images are resized to a square; the keypoints are rescaled with them. Augmentation (rotation,
shift, zoom, colour) moves the keypoints together with the pixels. Points that leave the frame
are marked "not visible" and ignored by the loss and metrics.
"""
from __future__ import annotations

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from cvkit.coco import load_coco
from cvkit.data import IMAGENET_MEAN, IMAGENET_STD, make_loader, require_dir, split_indices


def keypoint_transforms(size: int, augment: dict | None, train: bool) -> v2.Compose:
    aug = augment or {}
    ops: list = [v2.Resize((size, size))]
    if train:
        if any(aug.get(k) for k in ("rotation", "translate", "scale")):
            tr = aug.get("translate", 0)
            ops.append(v2.RandomAffine(degrees=aug.get("rotation", 0),
                                       translate=(tr, tr) if tr else None,
                                       scale=tuple(aug["scale"]) if aug.get("scale") else None))
        if aug.get("color_jitter", 0):
            j = aug["color_jitter"]
            ops.append(v2.ColorJitter(brightness=j, contrast=j, saturation=j))
    ops += [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return v2.Compose(ops)


class KeypointDataset(Dataset):
    """(image, keypoints (K,2) in resized-pixel coords, visible (K,) bool)."""

    def __init__(self, items: list[dict], size: int, transform):
        self.items, self.size, self.transform = items, size, transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        with Image.open(it["path"]) as im:
            image = im.convert("RGB")
        kp = torch.tensor(it["keypoints"], dtype=torch.float32).view(-1, 3)
        scale = torch.tensor([self.size / it["width"], self.size / it["height"]])
        pts = tv_tensors.KeyPoints(kp[:, :2] * scale, canvas_size=(self.size, self.size))
        image, pts = self.transform(image, pts)
        pts = pts.as_subclass(torch.Tensor)
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < self.size) & (pts[:, 1] >= 0) & (pts[:, 1] < self.size)
        return image, pts, (kp[:, 2] > 0) & inside


def build_datasets(cfg: dict) -> dict:
    d = cfg["data"]
    coco = load_coco(require_dir(d["root"]), d.get("annotations", "annotations.json"), d.get("images_dir", "images"))
    items = []
    for im in coco["images"]:
        anns = [a for a in im["annotations"] if "keypoints" in a]
        if not anns:
            raise ValueError(f"{im['path']} has no keypoint annotation (every image needs exactly one)")
        items.append({**im, "keypoints": anns[0]["keypoints"]})
    n_kp = len(items[0]["keypoints"]) // 3
    if any(len(it["keypoints"]) // 3 != n_kp for it in items):
        raise ValueError("every annotation must list the same number of keypoints")
    names = coco["keypoint_names"] or [f"kp_{i}" for i in range(n_kp)]
    if len(names) != n_kp:
        raise ValueError(f"categories[0]['keypoints'] has {len(names)} names but annotations have {n_kp} points")
    splits = split_indices(len(items), d["split"], cfg["seed"])
    aug, size = d.get("augment", {}), int(d["image_size"])

    def subset(name, train):
        return KeypointDataset([items[i] for i in splits[name]], size, keypoint_transforms(size, aug, train))

    return {"train": subset("train", True), "val": subset("val", False), "test": subset("test", False),
            "keypoint_names": names, "items": items, "split_indices": splits}


def build_loaders(cfg: dict) -> dict:
    ds = build_datasets(cfg)
    return {"train": make_loader(ds["train"], cfg, shuffle=True),
            "val": make_loader(ds["val"], cfg, shuffle=False),
            "test": make_loader(ds["test"], cfg, shuffle=False), "keypoint_names": ds["keypoint_names"]}


def test_image_paths(cfg: dict, n: int = 3) -> list[str]:
    return [it["path"] for it in build_datasets(cfg)["test"].items[:n]]
