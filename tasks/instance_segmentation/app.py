"""Gradio demo for instance segmentation:  python -m cvkit instance_segmentation app"""
from __future__ import annotations

from .predict import Predictor


def build_app(cfg: dict):
    import gradio as gr
    predictor = Predictor.from_config(cfg)

    def segment(image, threshold):
        if image is None:
            return None, ""
        objs = predictor.predict(image, threshold)
        summary = "\n".join(f"{o['label']}  {o['confidence']:.2f}  ({int(o['mask'].sum())} px)" for o in objs)
        return predictor.visualize(image, objs), summary or "nothing above threshold"

    return gr.Interface(
        fn=segment,
        inputs=[gr.Image(type="pil", label="input image"),
                gr.Slider(0.05, 0.95, value=predictor.score_threshold, step=0.05, label="score threshold")],
        outputs=[gr.Image(label="instances"), gr.Textbox(label="found")],
        title=cfg.get("project_name", "instance segmenter"),
        description="Each object gets its own colour mask. Lower the threshold to see more candidates.",
        flagging_mode="never",
    )
