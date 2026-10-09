"""Locate keypoints with a trained model.

    python -m cvkit keypoints predict --config <config.yaml> --input photo.jpg

    predictor = Predictor.from_config(cfg)
    predictor.predict("photo.jpg")
    # [{"name": "apex", "x": 81.2, "y": 17.9, "confidence": 0.93}, ...]   x, y in the photo's own pixels
    predictor.visualize("photo.jpg", points)
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from cvkit.checkpoint import checkpoint_path, load_checkpoint, restore_model
from cvkit.config import output_dir
from cvkit.data import IMAGE_EXTS, image_transforms
from cvkit.utils import get_device
from cvkit.viz import draw_keypoints

from .model import build_model, decode, to_probs


class Predictor:
    def __init__(self, ckpt: dict, device: torch.device | None = None):
        self.device = device or get_device()
        cfg = ckpt["config"]
        self.names: list[str] = ckpt["meta"]["keypoint_names"]
        self.size = int(cfg["data"]["image_size"])
        self.model = restore_model(ckpt, build_model, self.device)
        self.transform = image_transforms(self.size, None, False)

    @classmethod
    def from_config(cls, cfg: dict, name: str = "best.pt", device=None) -> "Predictor":
        device = device or get_device(cfg.get("device", "auto"))
        return cls(load_checkpoint(checkpoint_path(cfg, name), device), device)

    @torch.inference_mode()
    def predict(self, image: Image.Image | str | Path) -> list[dict]:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        pts, conf = decode(to_probs(self.model(x)))
        sx, sy = image.width / self.size, image.height / self.size      # back to the photo's own pixels
        return [{"name": n, "x": float(p[0]) * sx, "y": float(p[1]) * sy, "confidence": float(c)}
                for n, p, c in zip(self.names, pts[0].cpu(), conf[0].cpu())]

    def visualize(self, image: Image.Image | str | Path, points: list[dict]) -> Image.Image:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        return draw_keypoints(image, [(p["x"], p["y"]) for p in points], [p["name"] for p in points])


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
        pts = predictor.predict(p)
        saved = out_dir / f"{p.stem}_pred.png"
        predictor.visualize(p, pts).save(saved)
        print(f"{p.name}: " + ", ".join(f"{q['name']} ({q['x']:.0f}, {q['y']:.0f})" for q in pts) + f"  -> {saved}")
