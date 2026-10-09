"""Every task must run end to end: demo data -> train -> evaluate -> predict (CPU, seconds).

Parametrized over the folders in tasks/, so a new task is covered the moment it exists.
This is the contract test: if it passes, `python -m cvkit <task> quickstart --quick` works.
"""
import json
from pathlib import Path

import pytest

from cvkit.checkpoint import checkpoint_path, load_checkpoint
from cvkit.cli import task_module
from cvkit.config import output_dir
from cvkit.synthetic import make_demo_data

from conftest import ALL_TASKS


@pytest.mark.parametrize("task", ALL_TASKS)
def test_task_end_to_end(task, smoke_cfg, capsys):
    cfg = smoke_cfg(task)
    make_demo_data(task, cfg)

    result = task_module(task, "train").train(cfg)
    assert result["history"] and result["best_epoch"] >= 1
    assert checkpoint_path(cfg, "best.pt").exists() and checkpoint_path(cfg, "last.pt").exists()
    assert (output_dir(cfg) / "history.csv").exists()

    metrics = task_module(task, "evaluate").evaluate(cfg)
    assert metrics, "evaluate must return its metrics"
    reports = output_dir(cfg) / "reports"
    assert (reports / "metrics.json").exists()
    assert any(reports.glob("*.png")), "evaluate must write at least one plot"
    loaded = json.loads((reports / "metrics.json").read_text())
    assert all(isinstance(v, (int, float)) for v in loaded.values())

    task_module(task, "predict").run(cfg, None)
    assert capsys.readouterr().out.strip()


@pytest.mark.parametrize("task", ALL_TASKS)
def test_checkpoint_survives_config_edit(task, smoke_cfg):
    """Inference must use the checkpoint's own config, not whatever the YAML says now."""
    cfg = smoke_cfg(task)
    make_demo_data(task, cfg)
    task_module(task, "train").train(cfg)
    ckpt = load_checkpoint(checkpoint_path(cfg), "cpu")
    assert ckpt["config"]["model"]["arch"] == cfg["model"]["arch"]
    assert ckpt["meta"], "checkpoints must carry task metadata (classes, ...)"
    edited = {**cfg, "model": {**cfg["model"], "arch": "does_not_exist", "pretrained": True}}
    Predictor = task_module(task, "predict").Predictor
    Predictor.from_config(edited)              # must still load: it rebuilds from ckpt["config"]


def test_cli_lists_tasks_and_rejects_unknown(capsys):
    from cvkit.cli import main
    assert main([]) == 0
    assert "classification" in capsys.readouterr().out
    assert main(["not_a_task", "train"]) == 2


@pytest.mark.parametrize("task", ALL_TASKS)
def test_app_builds_and_answers(task, smoke_cfg):
    """The Gradio demo must load the trained model and produce output for an image."""
    pytest.importorskip("gradio")
    from PIL import Image
    cfg = smoke_cfg(task)
    make_demo_data(task, cfg)
    task_module(task, "train").train(cfg)
    app = task_module(task, "app").build_app(cfg)
    sample = task_module(task, "data").test_image_paths(cfg, 1)[0]
    import inspect
    n_inputs = len(inspect.signature(app.fn).parameters)
    out = app.fn(Image.open(sample).convert("RGB"), *([0.5] * (n_inputs - 1)))
    assert out is not None


@pytest.mark.parametrize("task", ALL_TASKS)
def test_default_config_is_valid_and_trains_a_step(task, tmp_path):
    """default.yaml is the 'real run' preset and never gets exercised by quick runs. It must at least
    parse, build its model, and complete a training step on demo data (pretrained weights are switched
    off here only because the test must not download anything)."""
    from cvkit.cli import task_config_path
    from cvkit.config import load_config
    cfg = load_config(task_config_path(task, quick=False))
    for section in ("seed", "data", "model", "train", "output", "demo"):
        assert section in cfg, f"{task}/default.yaml is missing '{section}'"
    cfg["data"].update(root=str(tmp_path / "data"), num_workers=0, batch_size=2)
    cfg["output"]["dir"] = str(tmp_path / "out")
    cfg["model"]["pretrained"] = False
    cfg["train"].update(epochs=1, limit_batches=1)
    cfg["demo"] = {**cfg["demo"], "n": 12}
    make_demo_data(task, cfg)
    assert task_module(task, "train").train(cfg)["history"]
