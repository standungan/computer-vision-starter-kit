"""Gradio demo for the regressor:  python -m cvkit regression app"""
from __future__ import annotations

from .predict import Predictor


def build_app(cfg: dict):
    import gradio as gr
    predictor = Predictor.from_config(cfg)

    def regress(image):
        return None if image is None else round(predictor.predict(image), 3)

    return gr.Interface(
        fn=regress,
        inputs=gr.Image(type="pil", label="input image"),
        outputs=gr.Number(label=f"predicted {predictor.target_name}"),
        title=cfg.get("project_name", "image regressor"),
        description="Drop in an image to get the predicted value.",
        flagging_mode="never",
    )
