# Getting started

## 1. Install

You need Python 3.10+. A GPU is optional: every task has a `quick` preset that runs on a laptop CPU.

```bash
git clone https://github.com/standungan/computer-vision-starter-kit
cd computer-vision-starter-kit
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **The one rule:** run every command from the repo's top folder (the one containing `cvkit/` and
> `tasks/`). Notebooks handle this for themselves.

**No local setup? Use Google Colab.** Open any notebook under `tasks/<task>/notebooks/` in Colab
(File → Open notebook → GitHub). Its first cell clones this repo and installs what's missing. Switch
on a GPU with *Runtime → Change runtime type*.

## 2. Run a task in one command

```bash
python -m cvkit classification quickstart --quick
```

This makes a small synthetic dataset, trains a model, evaluates it and predicts a few images.
Everything lands in `outputs/classification_quick/`: `checkpoints/best.pt`, `history.csv`, and
`reports/` full of plots. Swap `classification` for any task:

| Task | One-line goal | `quickstart --quick` takes about |
|---|---|---|
| `classification` | which of N classes is this? | 1 min |
| `multilabel` | which of N tags apply? | 1 min |
| `regression` | what number does this image show? | 1 min |
| `keypoints` | where are the landmarks? | 3 min |
| `segmentation` | what class is each pixel? | 3 min |
| `detection` | where are the objects, and what are they? | 5 min |
| `instance_segmentation` | the exact outline of each object | 10 min |

(Times were measured on a 4-core CPU. A GPU is much faster.)

## 3. The commands

Every task answers to the same six commands:

```bash
python -m cvkit <task> demo-data   --quick          # write the synthetic demo dataset to data.root
python -m cvkit <task> train       --quick          # train; saves outputs/.../checkpoints/best.pt
python -m cvkit <task> evaluate    --quick          # test-split metrics + plots in outputs/.../reports/
python -m cvkit <task> predict     --quick --input my_image.jpg       # or a folder; omit --input for samples
python -m cvkit <task> app         --quick          # Gradio demo in your browser
python -m cvkit <task> quickstart  --quick          # the four steps above, in order
```

Leave out `--quick` to use the task's `configs/default.yaml` (bigger images, more epochs, pretrained
weights). Change any config value from the command line without editing the file:

```bash
python -m cvkit classification train --quick --set train.epochs=30 model.arch=resnet34
```

## 4. Use your own data

1. Arrange your files as the task's README describes (all formats: [data-formats.md](data-formats.md)).
2. Point the config at them: `--set data.root=data/my-dataset` (or edit `configs/default.yaml`).
3. Run `train`, then `evaluate`.

Start with `quick` on a small slice of your data to catch format problems in a minute, then switch to
`default`.

**Check your data in seconds, before a long run:** a *debug run* does one epoch of two batches. If it finishes, the data
loads, the labels are valid, and the model, metrics and plots all work on your files:

```bash
python -m cvkit <task> train --set data.root=data/my-dataset train.epochs=1 train.limit_batches=2
python -m cvkit <task> evaluate --set data.root=data/my-dataset
```

## 5. Where to go from here

Each task folder is self-contained and uses the same layout:

```
tasks/<task>/
├── README.md        what it is, data format, how to read the results      ← start here
├── notebooks/       00 quickstart → 01 data → 02 train & evaluate → 03 predict & deploy
├── configs/         quick.yaml (minutes, CPU)  ·  default.yaml (the real run)
├── data.py  model.py  train.py  evaluate.py  predict.py  app.py     ← the code, ~50–150 lines each
```

Shared building blocks (config loading, the training loop, metrics, plotting) live in `cvkit/`.

| You want to… | Do this |
|---|---|
| see it work | `quickstart --quick`, then open the plots in `outputs/<run>/reports/` |
| understand it | read the task's notebooks, then the task's `README.md` ("How to read the results") |
| use your data | section 4 above |
| tune it | notebook 02 lists experiments; every one is a `--set key=value` change |
| change the model or loss | edit `tasks/<task>/model.py` / `train.py`; they're short on purpose |
| add a new task | `tasks/_template/CONTRACT.md` |

Unfamiliar word? [glossary.md](glossary.md). Something broke? [troubleshooting.md](troubleshooting.md).
