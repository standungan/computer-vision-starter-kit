"""Detect objects with a trained model.

    python -m cvkit detection predict --config <config.yaml> --input photo.jpg

    predictor = Predictor.from_config(cfg)
    dets = predictor.predict("photo.jpg", score_threshold=0.5)
    # [{"label": "circle", "confidence": 0.98, "box": [x0, y0, x1, y1]}, ...]
    predictor.visualize("photo.jpg", dets)      # PIL image with the boxes drawn
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from cvkit.checkpoint import checkpoint_path, load_checkpoint, restore_model
from cvkit.config import output_dir
from cvkit.data import IMAGE_EXTS
from cvkit.utils import get_device
from cvkit.viz import draw_boxes

from .model import build_model


class Predictor:
    def __init__(self, ckpt: dict, device: torch.device | None = None):
        self.device = device or get_device()
        self.classes: list[str] = ckpt["meta"]["classes"]
        self.score_threshold = float(ckpt["config"].get("eval", {}).get("score_threshold", 0.5))
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
        return [{"label": self.classes[int(l) - 1], "confidence": float(s), "box": [float(v) for v in b]}
                for b, l, s in zip(out["boxes"].cpu(), out["labels"].cpu(), out["scores"].cpu())
                if s >= thr]

    def visualize(self, image: Image.Image | str | Path, detections: list[dict]) -> Image.Image:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        labels = [self.classes.index(d["label"]) + 1 for d in detections]
        return draw_boxes(image, [d["box"] for d in detections], labels,
                          [d["confidence"] for d in detections], self.classes)


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
        dets = predictor.predict(p)
        summary = ", ".join(f"{d['label']} {d['confidence']:.2f}" for d in dets) or "nothing above threshold"
        saved = out_dir / f"{p.stem}_pred.png"
        predictor.visualize(p, dets).save(saved)
        print(f"{p.name}: {len(dets)} object(s): {summary}  -> {saved}")
