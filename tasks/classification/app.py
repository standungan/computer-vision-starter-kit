"""Gradio demo for the trained classifier:  python -m cvkit classification app"""
from __future__ import annotations

from .predict import Predictor


def build_app(cfg: dict):
    import gradio as gr
    predictor = Predictor.from_config(cfg)

    def classify(image):
        if image is None:
            return {}
        preds = predictor.predict(image, top_k=len(predictor.classes))
        return {p["label"]: p["confidence"] for p in preds}

    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="input image"),
        outputs=gr.Label(num_top_classes=3, label="prediction"),
        title=cfg.get("project_name", "image classifier"),
        description="Drop in an image to see the predicted class and its confidence.",
        flagging_mode="never",
    )
