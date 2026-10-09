"""Segment an image with a trained model.

    python -m cvkit segmentation predict --config <config.yaml> --input photo.jpg

    predictor = Predictor.from_config(cfg)
    mask = predictor.predict("photo.jpg")       # numpy uint8 (H, W): class id per pixel, at the photo's size
    predictor.summary(mask)                     # {"background": 0.86, "circle": 0.14, ...}  area fractions
    predictor.visualize("photo.jpg", mask)      # PIL image with the classes tinted on top
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from cvkit.checkpoint import checkpoint_path, load_checkpoint, restore_model
from cvkit.config import output_dir
from cvkit.data import IMAGE_EXTS, image_transforms
from cvkit.utils import get_device
from cvkit.viz import overlay_label_map

from .model import build_model


class Predictor:
    def __init__(self, ckpt: dict, device: torch.device | None = None):
        self.device = device or get_device()
        cfg = ckpt["config"]
        self.classes: list[str] = ckpt["meta"]["classes"]
        self.model = restore_model(ckpt, build_model, self.device)
        self.transform = image_transforms(cfg["data"]["image_size"], None, False)

    @classmethod
    def from_config(cls, cfg: dict, name: str = "best.pt", device=None) -> "Predictor":
        device = device or get_device(cfg.get("device", "auto"))
        return cls(load_checkpoint(checkpoint_path(cfg, name), device), device)

    @torch.inference_mode()
    def predict(self, image: Image.Image | str | Path) -> np.ndarray:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = F.interpolate(self.model(x), size=(image.height, image.width),
                               mode="bilinear", align_corners=False)   # back to the photo's own size
        return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)

    def summary(self, mask: np.ndarray) -> dict[str, float]:
        return {c: float((mask == i).mean()) for i, c in enumerate(self.classes)}

    def visualize(self, image: Image.Image | str | Path, mask: np.ndarray) -> Image.Image:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        return overlay_label_map(image, mask)


def run(cfg: dict, input_path: str | None) -> None:
    from .data import test_image_paths
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
    out_dir = output_dir(cfg) / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        mask = predictor.predict(p)
        share = ", ".join(f"{c} {v:.0%}" for c, v in predictor.summary(mask).items() if v > 0)
        saved = out_dir / f"{p.stem}_pred.png"
        predictor.visualize(p, mask).save(saved)
        print(f"{p.name}: {share}  -> {saved}")
