"""Segment object instances with a trained model.

    python -m cvkit instance_segmentation predict --config <config.yaml> --input photo.jpg

    predictor = Predictor.from_config(cfg)
    objs = predictor.predict("photo.jpg")
    # [{"label": "circle", "confidence": 0.98, "box": [x0, y0, x1, y1], "mask": <bool array HxW>}, ...]
    predictor.visualize("photo.jpg", objs)     # PIL image with coloured masks and boxes
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from cvkit.checkpoint import checkpoint_path, load_checkpoint, restore_model
from cvkit.config import output_dir
from cvkit.data import IMAGE_EXTS
from cvkit.utils import get_device
from cvkit.viz import draw_boxes, overlay_masks

from .model import build_model


class Predictor:
    def __init__(self, ckpt: dict, device: torch.device | None = None):
        self.device = device or get_device()
        self.classes: list[str] = ckpt["meta"]["classes"]
        ev = ckpt["config"].get("eval", {})
        self.score_threshold = float(ev.get("score_threshold", 0.5))
        self.mask_threshold = float(ev.get("mask_threshold", 0.5))
        self.model = restore_model(ckpt, build_model, self.device)

    @classmethod
    def from_config(cls, cfg: dict, name: str = "best.pt", device=None) -> "Predictor":
        device = device or get_device(cfg.get("device", "auto"))
        return cls(load_checkpoint(checkpoint_path(cfg, name), device), device)

    @torch.inference_mode()
    def predict(self, image: Image.Image | str | Path, score_threshold: float | None = None) -> list[dict]:
        thr = self.score_threshold if score_threshold is None else score_threshold
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        x = pil_to_tensor(image.convert("RGB")).float().div(255).to(self.device)
        out = self.model([x])[0]
        masks = (out["masks"].squeeze(1) > self.mask_threshold).cpu().numpy()
        return [{"label": self.classes[int(l) - 1], "confidence": float(s),
                 "box": [float(v) for v in b], "mask": m}
                for b, l, s, m in zip(out["boxes"].cpu(), out["labels"].cpu(), out["scores"].cpu(), masks)
                if s >= thr]

    def visualize(self, image: Image.Image | str | Path, objects: list[dict]) -> Image.Image:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        if not objects:
            return image.convert("RGB")
        labels = [self.classes.index(o["label"]) + 1 for o in objects]
        shaded = overlay_masks(image, np.stack([o["mask"] for o in objects]), labels)
        return draw_boxes(shaded, [o["box"] for o in objects], labels,
                          [o["confidence"] for o in objects], self.classes, width=1)


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
        objs = predictor.predict(p)
        summary = ", ".join(f"{o['label']} {o['confidence']:.2f} ({int(o['mask'].sum())} px)" for o in objs)
        saved = out_dir / f"{p.stem}_pred.png"
        predictor.visualize(p, objs).save(saved)
        print(f"{p.name}: {len(objs)} instance(s): {summary or 'nothing above threshold'}  -> {saved}")
