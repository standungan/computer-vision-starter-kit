"""Gradio demo for the detector:  python -m cvkit detection app"""
from __future__ import annotations

from .predict import Predictor


def build_app(cfg: dict):
    import gradio as gr
    predictor = Predictor.from_config(cfg)

    def detect(image, threshold):
        if image is None:
            return None, ""
        dets = predictor.predict(image, threshold)
        summary = "\n".join(f"{d['label']}  {d['confidence']:.2f}" for d in dets) or "nothing above threshold"
        return predictor.visualize(image, dets), summary

    return gr.Interface(
        fn=detect,
        inputs=[gr.Image(type="pil", label="input image"),
                gr.Slider(0.05, 0.95, value=predictor.score_threshold, step=0.05, label="score threshold")],
        outputs=[gr.Image(label="detections"), gr.Textbox(label="found")],
        title=cfg.get("project_name", "object detector"),
        description="Boxes below the score threshold are hidden. Lower it to find more (and more wrong) objects.",
        flagging_mode="never",
    )
