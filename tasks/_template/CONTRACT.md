# Adding a new task

A task is a folder in `tasks/`. If it follows the contract below, the CLI
(`python -m cvkit <task> ...`), the smoke test (`tests/test_tasks_smoke.py`) and the docs
pick it up with no other changes.

**Easiest start:** copy the closest existing task and edit it.

```bash
cp -r tasks/classification tasks/my_task        # image in -> label out? copy classification
cp -r tasks/detection      tasks/my_task        # image in -> boxes out? copy detection
```

Then add a demo-data generator (below) and run `python -m pytest tests -k my_task`.

## Required layout

```
tasks/my_task/
├── __init__.py          one-line docstring: what the task is
├── configs/
│   ├── default.yaml     the full config (seed, device, data, model, train, eval, output, demo)
│   └── quick.yaml       `base: default.yaml` + overrides: CPU-friendly, finishes in minutes
├── data.py              build_loaders(cfg), test_image_paths(cfg, n)
├── model.py             build_model(cfg, meta)
├── train.py             train(cfg) -> {"best_epoch", "best_metrics", "history"}
├── evaluate.py          evaluate(cfg) -> dict of metrics
├── predict.py           Predictor.from_config(cfg), run(cfg, input_path_or_None)
├── app.py               build_app(cfg) -> gradio Interface
├── README.md            the standard sections (below)
└── notebooks/           00_quickstart, 01_data, 02_train_and_evaluate, 03_predict_and_deploy
```

## What each file must do

| File | Contract |
|---|---|
| `data.py` | Split with `cvkit.data.split_indices` (never your own split). Fail early with a readable message on bad data (`require_dir`, `read_image_table`, `load_coco` already do). `test_image_paths` returns a few *test-split* image paths. |
| `model.py` | `build_model(cfg, meta)` where `meta` is exactly what `train.py` saved in the checkpoint. It must not need the dataset. Honour `cfg["model"]["pretrained"]`. |
| `train.py` | Build a `cvkit.engine.TrainSpec` (a `train_step`, an `evaluate`, a `monitor` key and `mode`, and a **plain-python** `meta` dict) and call `cvkit.engine.fit`. `evaluate` returns floats only. |
| `evaluate.py` | Load with `load_checkpoint` + `restore_model` (so the checkpoint's config wins). Write into `<output.dir>/reports/`: a `metrics.json` of numbers, and at least one `.png`. Return the metrics. |
| `predict.py` | `Predictor.from_config(cfg)` loads `best.pt`. `run(cfg, None)` predicts a few test images and prints something. `run(cfg, path)` accepts a file or a folder. |
| `app.py` | Import `gradio` *inside* `build_app` so the rest of the task works without it. |

## Demo data

Add a generator to `cvkit/synthetic.py` that writes exactly the layout your task reads, and register it
in `MAKERS`. Keep it deterministic (seeded) and small. The smoke test calls it with `n=24`.

## The standard README sections

1. **What it is** — one input → output example, in plain words
2. **When to use it** — and a "not this, use X instead" line
3. **Data format** — folder tree + 10-line example (link to `docs/data-formats.md`)
4. **Run it** — the quickstart command and what you'll see
5. **How to read the results** — each metric and plot explained, good vs bad
6. **Configuration** — the keys that matter for this task
7. **Common problems**
8. **Go further** — experiments and next tasks

## Check your work

```bash
python -m pytest tests -k my_task          # contract: demo data -> train -> evaluate -> predict
python -m cvkit my_task quickstart --quick # and watch it learn: the metrics should clearly beat chance
```

The smoke test only proves the pipeline *runs*. Before you ship, make sure the quick run actually
*learns* (e.g. accuracy well above chance, mAP clearly above 0): a demo that runs but learns nothing
is a bad first experience.
