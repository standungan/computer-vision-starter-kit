"""Inference on new images — single file, a folder, or from your own code.

    python -m src.inference --config configs/config.yaml --input path/to/image.jpg
    python -m src.inference --config configs/config.yaml --input path/to/folder --top-k 3

`Predictor` is the reusable piece: it loads the best checkpoint once and
exposes `.predict(image)` for scripts, notebooks, and app/app.py alike.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .dataset import build_transforms
from .model import build_model
from .utils import checkpoint_path, get_device, load_config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class Predictor:
    def __init__(self, cfg: dict, device: torch.device | None = None):
        self.device = device or get_device()
        ckpt = torch.load(checkpoint_path(cfg, "best.pt"),
                          map_location=self.device, weights_only=False)
        self.classes: list[str] = ckpt["classes"]
        self.model = build_model(cfg, num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.transform = build_transforms(cfg, train=False)

    @torch.inference_mode()
    def predict(self, image: Image.Image | str | Path, top_k: int = 3) -> list[dict]:
        """Returns [{"label": ..., "confidence": ...}, ...] best first."""
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = F.softmax(self.model(x), dim=1)[0]
        k = min(top_k, len(self.classes))
        conf, idx = probs.topk(k)
        return [{"label": self.classes[i], "confidence": float(c)}
                for c, i in zip(conf, idx)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", required=True, help="image file or folder of images")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    predictor = Predictor(load_config(args.config))
    target = Path(args.input)
    paths = ([target] if target.is_file()
             else sorted(p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTS))
    if not paths:
        raise SystemExit(f"no images found at {target}")

    for p in paths:
        preds = predictor.predict(p, top_k=args.top_k)
        top = preds[0]
        rest = ", ".join(f"{q['label']} {q['confidence']:.2f}" for q in preds[1:])
        print(f"{p.name}: {top['label']} ({top['confidence']:.2f})"
              + (f"  [{rest}]" if rest else ""))


if __name__ == "__main__":
    main()
