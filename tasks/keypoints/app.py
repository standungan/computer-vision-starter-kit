"""Gradio demo for the keypoint model:  python -m cvkit keypoints app"""
from __future__ import annotations

from .predict import Predictor


def build_app(cfg: dict):
    import gradio as gr
    predictor = Predictor.from_config(cfg)

    def locate(image):
        if image is None:
            return None, ""
        pts = predictor.predict(image)
        text = "\n".join(f"{p['name']}: ({p['x']:.0f}, {p['y']:.0f})  confidence {p['confidence']:.2f}" for p in pts)
        return predictor.visualize(image, pts), text

    return gr.Interface(
        fn=locate,
        inputs=gr.Image(type="pil", label="input image"),
        outputs=[gr.Image(label="keypoints"), gr.Textbox(label="coordinates")],
        title=cfg.get("project_name", "keypoint locator"),
        description="Each landmark is drawn as a coloured dot with its name.",
        flagging_mode="never",
    )
