"""Gradio demo for the multi-label tagger:  python -m cvkit multilabel app"""
from __future__ import annotations

from .predict import Predictor


def build_app(cfg: dict):
    import gradio as gr
    predictor = Predictor.from_config(cfg)

    def tag(image, threshold):
        if image is None:
            return {}, ""
        preds = predictor.predict(image, threshold)
        present = ", ".join(r["label"] for r in preds if r["present"]) or "(none above threshold)"
        return {r["label"]: r["confidence"] for r in preds}, present

    return gr.Interface(
        fn=tag,
        inputs=[gr.Image(type="pil", label="input image"),
                gr.Slider(0.05, 0.95, value=predictor.threshold, step=0.05, label="threshold")],
        outputs=[gr.Label(num_top_classes=len(predictor.classes), label="probability per label"),
                 gr.Textbox(label="labels present")],
        title=cfg.get("project_name", "multi-label tagger"),
        description="Each label gets its own probability. Move the threshold to trade precision for recall.",
        flagging_mode="never",
    )
