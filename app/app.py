"""Deployment step: a Gradio app around the trained model.

    python app/app.py                # http://127.0.0.1:7860
    python app/app.py --share       # temporary public link

Drop in an image, get the predicted class with confidences. Loads the same
best.pt + config.yaml the rest of the pipeline uses — train a new model and
the app picks it up on restart, whatever the dataset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr

# allow `python app/app.py` from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import Predictor          # noqa: E402
from src.utils import load_config            # noqa: E402


def build_app(config_path: str = "configs/config.yaml") -> gr.Interface:
    cfg = load_config(config_path)
    predictor = Predictor(cfg)

    def classify(image):
        if image is None:
            return {}
        preds = predictor.predict(image, top_k=len(predictor.classes))
        return {p["label"]: p["confidence"] for p in preds}

    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="input image"),
        outputs=gr.Label(num_top_classes=3, label="prediction"),
        title=cfg.get("project_name", "computer-vision-starter-kit"),
        description=(
            "Computer Vision Project Starter Kit — trained with "
            "`python -m src.train`. Free kit by @sofyantandungan · NeuronEngine."
        ),
        flagging_mode="never",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_app(args.config).launch(share=args.share)
