"""Data plumbing shared by all tasks: splits, loaders, image-level datasets.

The split logic is the one thing that must be identical everywhere — a leaky
or unreproducible split makes every metric downstream meaningless — so it
lives here once: seeded, optionally stratified by class, and guaranteed to
leave at least one image in val/test whenever a group has 3+ images.
"""
from __future__ import annotations

import csv
import warnings
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def require_dir(path: str | Path, what: str = "data root") -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found: {path}\n"
            "Point `data.root` in your config at your data (see the task README for the "
            "expected layout), or create demo data first:\n"
            "    python -m cvkit <task> demo-data --config <config.yaml>"
        )
    return path


# --------------------------------------------------------------------- splits
def split_indices(n: int, ratios: dict, seed: int, groups: list | None = None) -> dict[str, list[int]]:
    """Split range(n) into train/val/test. Pass `groups` (e.g. class ids) to stratify."""
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    groups = groups if groups is not None else [0] * n
    by_group: dict = {}
    for idx, g in enumerate(groups):
        by_group.setdefault(g, []).append(idx)
    gen = torch.Generator().manual_seed(seed)
    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for g, members in sorted(by_group.items(), key=lambda kv: str(kv[0])):
        if len(members) < 3 and len(by_group) > 1:
            warnings.warn(f"group '{g}' has only {len(members)} item(s) — "
                          "val/test may be empty for it; add more data")
        perm = torch.randperm(len(members), generator=gen).tolist()
        shuffled = [members[i] for i in perm]
        n_train, n_val, n_test = _split_counts(len(shuffled), ratios)
        out["train"] += shuffled[:n_train]
        out["val"] += shuffled[n_train:n_train + n_val]
        out["test"] += shuffled[n_train + n_val:n_train + n_val + n_test]
    return out


def _split_counts(n: int, ratios: dict) -> tuple[int, int, int]:
    n_val, n_test = round(n * ratios["val"]), round(n * ratios["test"])
    if n >= 3:                                   # never starve val/test of a group
        if ratios["val"] > 0:
            n_val = max(n_val, 1)
        if ratios["test"] > 0:
            n_test = max(n_test, 1)
    while n - n_val - n_test < 1 and (n_val or n_test):
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        else:
            n_test -= 1
    return n - n_val - n_test, n_val, n_test


# ------------------------------------------------------------------- datasets
class ImageTargetDataset(Dataset):
    """(image, target) pairs from a list of (path, target). Used by the image-level tasks."""

    def __init__(self, samples: list[tuple[str, object]], transform, target_dtype=torch.long):
        self.samples = samples
        self.transform = transform
        self.target_dtype = target_dtype

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, target = self.samples[i]
        with Image.open(path) as im:
            image = self.transform(im.convert("RGB"))
        return image, torch.as_tensor(target, dtype=self.target_dtype)


def image_transforms(size: int, augment: dict | None, train: bool) -> transforms.Compose:
    """Eval: resize + normalize. Train: adds whichever augmentations the config enables."""
    aug = augment or {}
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
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if train and aug.get("random_erasing", 0):
        ops.append(transforms.RandomErasing(p=aug["random_erasing"]))
    return transforms.Compose(ops)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization so a (3,H,W) tensor can be shown with imshow -> (H,W,3)."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0)


def make_loader(dataset, cfg: dict, shuffle: bool, collate_fn=None) -> DataLoader:
    d = cfg["data"]
    nw = int(d.get("num_workers", 0))
    return DataLoader(dataset, batch_size=int(d["batch_size"]), shuffle=shuffle,
                      num_workers=nw, pin_memory=torch.cuda.is_available(),
                      persistent_workers=nw > 0, collate_fn=collate_fn)


def read_labels_csv(path: str | Path) -> tuple[list[str], list[list[str]]]:
    """Read a labels.csv -> (header, rows). First column is always the image filename."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"labels file not found: {path}\nSee the task README for the expected format.")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise ValueError(f"{path} has no data rows")
    return rows[0], rows[1:]


def read_image_table(root: str | Path, images_dir: str = "images",
                     labels_file: str = "labels.csv") -> tuple[list[str], list[tuple[str, list[float]]]]:
    """Read root/<labels_file> (first column = filename, other columns = numeric targets).

    Returns (target_column_names, [(image_path, [targets...]), ...]). Fails early with a
    readable message if a listed image is missing or a value isn't a number — much easier
    to fix here than as a stack trace in the middle of epoch 1.
    """
    root = require_dir(root)
    header, rows = read_labels_csv(root / labels_file)
    samples, missing = [], []
    for line_no, row in enumerate(rows, start=2):
        if len(row) != len(header):
            raise ValueError(f"{root / labels_file} line {line_no}: expected {len(header)} columns, got {len(row)}")
        path = root / images_dir / row[0]
        if not path.exists():
            missing.append(row[0])
            continue
        try:
            samples.append((str(path), [float(v) for v in row[1:]]))
        except ValueError:
            raise ValueError(f"{root / labels_file} line {line_no}: non-numeric target in {row[1:]}") from None
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) listed in {labels_file} are missing from {root / images_dir}, "
            f"e.g. {missing[:5]}")
    return header[1:], samples
