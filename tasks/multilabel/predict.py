"""Predict labels with a trained multi-label model.

    python -m cvkit multilabel predict --config <config.yaml> --input photo.jpg

    predictor = Predictor.from_config(cfg)
    predictor.predict("photo.jpg", threshold=0.5)
    # [{"label": "circle", "confidence": 0.97, "present": True}, ...]  highest confidence first
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
        self.classes: list[str] = ckpt["meta"]["classes"]
        self.threshold = float(cfg.get("eval", {}).get("threshold", 0.5))
        self.model = restore_model(ckpt, build_model, self.device)
        self.transform = image_transforms(cfg["data"]["image_size"], cfg["data"].get("augment"), False)

    @classmethod
    def from_config(cls, cfg: dict, name: str = "best.pt", device=None) -> "Predictor":
        device = device or get_device(cfg.get("device", "auto"))
        return cls(load_checkpoint(checkpoint_path(cfg, name), device), device)

    @torch.inference_mode()
    def predict(self, image: Image.Image | str | Path, threshold: float | None = None) -> list[dict]:
        threshold = self.threshold if threshold is None else threshold
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.sigmoid(self.model(x))[0].cpu().tolist()
        out = [{"label": c, "confidence": p, "present": p >= threshold}
               for c, p in zip(self.classes, probs)]
        return sorted(out, key=lambda r: -r["confidence"])


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
        preds = predictor.predict(p)
        present = [f"{r['label']} ({r['confidence']:.2f})" for r in preds if r["present"]]
        print(f"{p.name}: {', '.join(present) if present else '(no label above threshold)'}")
