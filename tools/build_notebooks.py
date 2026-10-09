"""Generate the four notebooks of every task from one place, so they stay consistent.

    python tools/build_notebooks.py            # (re)writes tasks/<task>/notebooks/*.ipynb, without outputs
    python tools/build_notebooks.py --execute  # ...and runs them so the saved notebooks show their results

The .ipynb files are build artifacts: edit THIS file (the shared skeleton or a task's entry in
SPECS), then rebuild. Each task gets
    00_quickstart                the whole pipeline in ~10 cells
    01_data                      look at the data, the labels, and the augmentation
    02_train_and_evaluate        train, learning curves, metrics, how to read them, experiments
    03_predict_and_deploy        predict on new images, the CLI, the demo app, deployment notes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO_URL = "https://github.com/standungan/computer-vision-starter-kit"

BOOTSTRAP = '''# Run this cell first.
# On Colab it clones the kit and installs what's missing; locally it just finds the repo root.
import os, sys, subprocess
from pathlib import Path

if "google.colab" in sys.modules and not Path("cvkit").exists():
    subprocess.run(["git", "clone", "-q", "<<REPO>>"], check=True)
    os.chdir("computer-vision-starter-kit")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "scikit-learn", "gradio"], check=True)
while not Path("cvkit").exists() and Path.cwd() != Path.cwd().parent:
    os.chdir(Path.cwd().parent)          # notebooks live 3 folders below the repo root
sys.path.insert(0, os.getcwd())
print("working in", Path.cwd().name)'''

SETUP = '''%config InlineBackend.figure_format = "jpeg"
%config InlineBackend.print_figure_kwargs = {"bbox_inches": "tight", "pil_kwargs": {"quality": 90}}   # keeps saved notebooks small

from PIL import Image

from cvkit.config import load_config
from cvkit.synthetic import make_demo_data
from cvkit.utils import seed_everything

# quick.yaml = small synthetic demo, runs on a laptop CPU in minutes.
# Swap in default.yaml (bigger, pretrained) or your own config + data when you're ready.
CONFIG = "tasks/<<TASK>>/configs/quick.yaml"
cfg = load_config(CONFIG)
seed_everything(cfg["seed"])

if not Path(cfg["data"]["root"]).exists():          # first run: create the demo dataset
    make_demo_data("<<TASK>>", cfg)
print("task:", "<<TASK>>", "| data:", cfg["data"]["root"], "| outputs:", cfg["output"]["dir"])'''

SHOW_REPORTS = '''from cvkit.viz import show_image

reports = Path(cfg["output"]["dir"]) / "reports"
for name in <<NAMES>>:
    print(name)
    show_image(reports / name)'''

ENSURE_TRAINED = '''from cvkit.checkpoint import checkpoint_path
from tasks.<<TASK>>.train import train

if not checkpoint_path(cfg, "best.pt").exists():     # this notebook needs a trained model; train one if missing
    train(cfg)
print("model:", checkpoint_path(cfg, "best.pt"))'''


def build(task: str, spec: dict) -> dict[str, list]:
    t = lambda s: s.replace("<<TASK>>", task).replace("<<REPO>>", REPO_URL)          # noqa: E731
    boot, setup = ("code", t(BOOTSTRAP)), ("code", t(SETUP))
    names = repr(spec["reports"])
    show = ("code", t(SHOW_REPORTS).replace("<<NAMES>>", names))
    title = spec["title"]

    nb00 = [
        ("md", f"# {title} — quickstart\n\n{spec['blurb']}\n\n"
               "This notebook runs the **whole pipeline in about ten cells**. The other notebooks slow down and explain each step.\n\n"
               f"**Data → Train → Evaluate → Predict** &nbsp;·&nbsp; runs on CPU in about {spec['minutes']} minutes with the `quick` preset."),
        boot, setup,
        ("md", "## 1 · The data\nA few examples, with their labels:"),
        ("code", spec["peek"]),
        ("md", "## 2 · Train\n`train(cfg)` is the same function the command line calls: "
               f"`python -m cvkit {task} train --quick`. It prints one line per epoch and saves the best model."),
        ("code", t(f"from tasks.<<TASK>>.train import train\n\nresult = train(cfg)")),
        ("md", "## 3 · Evaluate on the held-out test split\nImages the model never trained on, and never used to pick its best epoch."),
        ("code", t("from tasks.<<TASK>>.evaluate import evaluate\n\nmetrics = evaluate(cfg)")),
        show,
        ("md", spec["quick_read"]),
        ("md", "## 4 · Predict\n" + spec["predict_intro"]),
        ("code", t(spec["predict_code"])),
        ("md", "## What next?\n"
               "* **[01_data](01_data.ipynb)** — look at the data and augmentation properly\n"
               "* **[02_train_and_evaluate](02_train_and_evaluate.ipynb)** — learning curves, how to read the metrics, experiments\n"
               "* **[03_predict_and_deploy](03_predict_and_deploy.ipynb)** — your own images, the demo app, deployment\n"
               f"* Read the task's [README](../README.md) · use your own data: [data formats](../../../docs/data-formats.md)"),
    ]

    nb01 = [
        ("md", f"# {title} — look at your data first\n\n"
               "Most failed projects fail here, not in the model. Before training anything, answer:\n"
               + "\n".join(f"{i}. {q}" for i, q in enumerate(spec["questions"], 1))),
        boot, setup,
        ("md", f"## How the data is organised\n{spec['data_md']}"),
    ]
    for md_, code_ in spec["explore"]:
        nb01 += [("md", md_), ("code", t(code_))]
    nb01 += [
        ("md", f"## Augmentation\n{spec['aug_md']}"),
        ("code", t(spec["aug_code"])),
        ("md", "## What the model will actually receive\nOne batch from the training loader, exactly as `train()` gets it:"),
        ("code", t(spec["batch_code"])),
        ("md", "## Before moving on\n" + "\n".join(f"- [ ] {c}" for c in spec["checklist"])
         + "\n\nNext: **[02_train_and_evaluate](02_train_and_evaluate.ipynb)**."),
    ]

    nb02 = [
        ("md", f"# {title} — train and evaluate\n\n"
               "Training is one call; the interesting part is *reading what comes out*. "
               "This notebook trains, plots the learning curves, evaluates on the test split, and shows you how to interpret each number."),
        boot, setup,
        ("md", "## The configuration\nEverything tunable is in one YAML file. These are the sections that matter:"),
        ("code", "import yaml\n\nfor section in ('data', 'model', 'train'):\n    print(section + ':')\n"
                 "    for line in yaml.safe_dump(cfg[section], sort_keys=False).splitlines():\n        print('  ' + line)"),
        ("md", "## What are we about to train?"),
        ("code", t(spec["model_summary"])),
        ("md", "## Train\nWatch the `val` numbers: they are what picks the best epoch."),
        ("code", t("from tasks.<<TASK>>.train import train\n\nresult = train(cfg)\n"
                   "print('\\nbest epoch:', result['best_epoch'], '|', {k: round(v, 4) for k, v in result['best_metrics'].items()})")),
        ("md", "## Learning curves\n" + spec["curves_md"]),
        ("code", "from cvkit.viz import plot_history\n\n_ = plot_history(Path(cfg['output']['dir']) / 'history.csv')"),
        ("md", "## Evaluate on the test split\nVal chose the checkpoint; the **test split** was untouched until now, so it is the honest number. "
               "One rule: look at test results, *don't tune on them* — the moment you tweak settings to chase the test score, it stops measuring generalisation."),
        ("code", t("from tasks.<<TASK>>.evaluate import evaluate\n\nmetrics = evaluate(cfg)")),
        show,
        ("md", "## How to read these results\n" + spec["read_md"]),
        ("md", "## Experiments worth running\n"
               "Each one is a one-line change. Re-run the training cell with a different config and compare the best val numbers "
               "(change `cfg` like this, then re-run from *Train*):\n\n"
               "```python\ncfg['model']['arch'] = 'resnet34'     # or any key, same as --set model.arch=resnet34\n```\n\n"
               + "\n".join(f"{i}. {e}" for i, e in enumerate(spec["experiments"], 1))
               + "\n\nChange **one thing at a time**, keep the seed, and write down what you changed — otherwise you can't tell what helped.\n\n"
               "Next: **[03_predict_and_deploy](03_predict_and_deploy.ipynb)**."),
    ]

    nb03 = [
        ("md", f"# {title} — predict and deploy\n\n"
               "A trained model is a file. This notebook turns it into answers for new images, a command-line tool, and a demo app."),
        boot, setup,
        ("md", "## A trained model\nThis notebook needs `best.pt`; it trains one (quick preset) if you haven't yet."),
        ("code", t(ENSURE_TRAINED)),
        ("md", "## The `Predictor`\nOne object loads `best.pt` once and answers for a file path or a PIL image. "
               "It rebuilds the model from the *checkpoint's own* config, so it keeps working even if you edit the YAML later."),
        ("code", t(spec["predictor_code"])),
        ("md", spec["predict_more_md"]),
        ("code", t(spec["predict_more_code"])),
        ("md", f"## From the command line\n```bash\npython -m cvkit {task} predict --quick --input path/to/image.jpg\n"
               f"python -m cvkit {task} predict --quick --input path/to/folder\n```\n"
               "Without `--input` it predicts a few test images. Annotated copies (where the task has them) are saved under `outputs/<run>/predictions/`."),
        ("md", f"## The demo app\n```bash\npython -m cvkit {task} app --quick            # http://127.0.0.1:7860\n"
               f"python -m cvkit {task} app --quick --share    # a temporary public link\n```\n"
               "Or from a notebook cell: `from tasks." + task + ".app import build_app; build_app(cfg).launch()`.\n\n"
               "**Permanent free hosting:** create a *Gradio* Space on [Hugging Face](https://huggingface.co/spaces), "
               "upload this repo plus `outputs/<run>/checkpoints/best.pt`, and point the Space at the task's `app.py`."),
        ("code", t("from tasks.<<TASK>>.app import build_app\n\napp = build_app(cfg)          # building it proves the app can load the model\n"
                   "print(type(app).__name__, '- call app.launch() to open it in the browser')")),
        ("md", "## Before you ship\n"
               "- [ ] The test metrics are what you'd tell a user to expect (not the val numbers)\n"
               "- [ ] You looked at the worst predictions and know *why* they happen\n"
               "- [ ] You tried images that look different from your training data (new lighting, camera, background)\n"
               + spec["ship_extra"]),
    ]
    return {"00_quickstart": nb00, "01_data": nb01, "02_train_and_evaluate": nb02, "03_predict_and_deploy": nb03}


def to_notebook(cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in cells]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def main() -> None:
    from tools.notebook_specs import SPECS            # per-task content lives next door
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true", help="run each notebook and keep its outputs")
    ap.add_argument("--task", nargs="*", help="only these tasks")
    args = ap.parse_args()
    for task, spec in SPECS.items():
        if args.task and task not in args.task:
            continue
        out = ROOT / "tasks" / task / "notebooks"
        out.mkdir(parents=True, exist_ok=True)
        for name, cells in build(task, spec).items():
            nb = to_notebook(cells)
            if args.execute:
                from nbconvert.preprocessors import ExecutePreprocessor
                print(f"executing {task}/{name} ...", flush=True)
                ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(out)}})
            nbf.write(nb, out / f"{name}.ipynb")
        print("wrote", out)


if __name__ == "__main__":
    main()
