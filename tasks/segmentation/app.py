"""Gradio demo for semantic segmentation:  python -m cvkit segmentation app"""
from __future__ import annotations

from .predict import Predictor


def build_app(cfg: dict):
    import gradio as gr
    predictor = Predictor.from_config(cfg)

    def segment(image):
        if image is None:
            return None, {}
        mask = predictor.predict(image)
        return predictor.visualize(image, mask), predictor.summary(mask)

    return gr.Interface(
        fn=segment,
        inputs=gr.Image(type="pil", label="input image"),
        outputs=[gr.Image(label="segmentation"), gr.Label(label="share of pixels per class")],
        title=cfg.get("project_name", "semantic segmenter"),
        description="Every pixel is assigned a class; the colours show which.",
        flagging_mode="never",
    )
