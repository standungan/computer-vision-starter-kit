"""Predict with a trained classifier — from the CLI or from your own code.

    python -m cvkit classification predict --config <config.yaml> --input photo.jpg

    from tasks.classification.predict import Predictor
    predictor = Predictor.from_config(cfg)
    predictor.predict("photo.jpg", top_k=3)   # [{"label": "crazing", "confidence": 0.93}, ...]
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
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
        self.model = restore_model(ckpt, build_model, self.device)
        self.transform = image_transforms(cfg["data"]["image_size"], cfg["data"].get("augment"), False)

    @classmethod
    def from_config(cls, cfg: dict, name: str = "best.pt", device=None) -> "Predictor":
        device = device or get_device(cfg.get("device", "auto"))
        return cls(load_checkpoint(checkpoint_path(cfg, name), device), device)

    @torch.inference_mode()
    def predict(self, image: Image.Image | str | Path, top_k: int = 3) -> list[dict]:
        """Best-first [{"label": ..., "confidence": ...}]. Confidence is a softmax probability."""
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = F.softmax(self.model(x), dim=1)[0]
        conf, idx = probs.topk(min(top_k, len(self.classes)))
        return [{"label": self.classes[i], "confidence": float(c)} for c, i in zip(conf, idx)]


def run(cfg: dict, input_path: str | None, top_k: int = 3) -> None:
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
        preds = predictor.predict(p, top_k=top_k)
        rest = ", ".join(f"{q['label']} {q['confidence']:.2f}" for q in preds[1:])
        print(f"{p.name}: {preds[0]['label']} ({preds[0]['confidence']:.2f})"
              + (f"  [{rest}]" if rest else ""))
