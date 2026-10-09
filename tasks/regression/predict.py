"""Predict a number from an image.

    python -m cvkit regression predict --config <config.yaml> --input photo.jpg

    predictor = Predictor.from_config(cfg)
    predictor.predict("photo.jpg")      # 17.4   (a float, in the units of your labels.csv)
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from cvkit.checkpoint import checkpoint_path, load_checkpoint, restore_model
from cvkit.data import IMAGE_EXTS, image_transforms
from cvkit.utils import get_device

from .data import test_image_paths
from .model import build_model


class Predictor:
    def __init__(self, ckpt: dict, device: torch.device | None = None):
        self.device = device or get_device()
        cfg = ckpt["config"]
        self.stats = ckpt["meta"]["stats"]
        self.target_name = self.stats["target_name"]
        self.model = restore_model(ckpt, build_model, self.device)
        self.transform = image_transforms(cfg["data"]["image_size"], cfg["data"].get("augment"), False)

    @classmethod
    def from_config(cls, cfg: dict, name: str = "best.pt", device=None) -> "Predictor":
        device = device or get_device(cfg.get("device", "auto"))
        return cls(load_checkpoint(checkpoint_path(cfg, name), device), device)

    @torch.inference_mode()
    def predict(self, image: Image.Image | str | Path) -> float:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        out = float(self.model(x).squeeze().float().cpu())
        return out * self.stats["std"] + self.stats["mean"]


def run(cfg: dict, input_path: str | None) -> None:
    predictor = Predictor.from_config(cfg)
    if input_path is None:
        paths = [Path(p) for p in test_image_paths(cfg, 3)]
        print("(no --input given: using a few images from the test split)")
    else:
        target = Path(input_path)
        paths = ([target] if target.is_file()
                 else sorted(p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTS))
    if not paths:
        raise SystemExit(f"no images found at {input_path}")
    for p in paths:
        print(f"{p.name}: {predictor.target_name} = {predictor.predict(p):.2f}")
