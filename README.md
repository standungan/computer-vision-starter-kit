# Computer Vision Starter Kit

A free, hands-on **PyTorch kit for learning computer vision by running it** — seven classic tasks, one consistent
structure, and everything works on a laptop CPU in minutes.

Pick a task, run one command, watch a model learn, read the plots, then point the same code at your own images.

```bash
pip install -r requirements.txt
python -m cvkit classification quickstart --quick      # data → train → evaluate → predict, ~1 minute
```

By [Sofyan Tandungan](https://instagram.com/sofyantandungan) · NeuronEngine · MIT license.

---

## Pick a task

| Task | The question | Input → output | `quickstart --quick` | Try in Colab |
|---|---|---|---|---|
| [**classification**](tasks/classification) | Which class is this image? | image → one label | ~1 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/standungan/computer-vision-starter-kit/blob/main/tasks/classification/notebooks/00_quickstart.ipynb) |
| [**multilabel**](tasks/multilabel) | Which tags apply? | image → any number of labels | ~1 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/standungan/computer-vision-starter-kit/blob/main/tasks/multilabel/notebooks/00_quickstart.ipynb) |
| [**regression**](tasks/regression) | What number does this image show? | image → a value | ~1 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/standungan/computer-vision-starter-kit/blob/main/tasks/regression/notebooks/00_quickstart.ipynb) |
| [**detection**](tasks/detection) | Where are the objects, and what are they? | image → boxes + classes | ~5 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/standungan/computer-vision-starter-kit/blob/main/tasks/detection/notebooks/00_quickstart.ipynb) |
| [**instance_segmentation**](tasks/instance_segmentation) | What is the exact outline of each object? | image → mask per object | ~10 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/standungan/computer-vision-starter-kit/blob/main/tasks/instance_segmentation/notebooks/00_quickstart.ipynb) |
| [**segmentation**](tasks/segmentation) | What class is every pixel? | image → class map | ~5 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/standungan/computer-vision-starter-kit/blob/main/tasks/segmentation/notebooks/00_quickstart.ipynb) |
| [**keypoints**](tasks/keypoints) | Where are the landmarks? | image → named (x, y) points | ~2 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/standungan/computer-vision-starter-kit/blob/main/tasks/keypoints/notebooks/00_quickstart.ipynb) |

Not sure which? Every task README opens with *when to use it* and *what to use instead*. Unfamiliar words: [glossary](docs/glossary.md).
Times were measured on a 4-core CPU; a GPU is much faster.

Every task runs on a **generated demo dataset** (circles, squares, triangles) — no downloads, no accounts — so you see the
whole pipeline work before you bring your own data.

## Four levels, same in every task

| Level | You… | How |
|---|---|---|
| **0 · Run it** | just want to see it work | `python -m cvkit <task> quickstart --quick` or click the Colab badge |
| **1 · Follow it** | want to understand each step | the task's 4 notebooks: **00** quickstart · **01** data · **02** train & evaluate · **03** predict & deploy |
| **2 · Use your data** | have images and labels | arrange them as the task's README says, then `--set data.root=data/mine` |
| **3 · Change it** | want to tune or modify | every setting is a `--set key=value`; the code in `tasks/<task>/` is short and commented |
| **4 · Extend it** | want a new task | [`tasks/_template/CONTRACT.md`](tasks/_template/CONTRACT.md) |

## The commands

Every task answers to the same six:

```bash
python -m cvkit <task> demo-data  --quick     # write the synthetic demo dataset
python -m cvkit <task> train      --quick     # train → outputs/<run>/checkpoints/best.pt
python -m cvkit <task> evaluate   --quick     # test-split metrics + plots → outputs/<run>/reports/
python -m cvkit <task> predict    --quick --input my_image.jpg
python -m cvkit <task> app        --quick     # Gradio demo in your browser
python -m cvkit <task> quickstart --quick     # all of the above, in order
```

Drop `--quick` for the task's real config (bigger images, pretrained weights, more epochs). Override anything inline:
`--set train.epochs=30 model.arch=resnet34 data.root=data/my-dataset`.

## What's inside

```
computer-vision-starter-kit/
│
├── tasks/                       one self-contained folder per task, all with the same layout
│   ├── classification/  multilabel/  regression/  detection/
│   ├── instance_segmentation/  segmentation/  keypoints/
│   │       ├── README.md            what it is · data format · how to read the results · common problems
│   │       ├── notebooks/           00_quickstart · 01_data · 02_train_and_evaluate · 03_predict_and_deploy
│   │       ├── configs/             quick.yaml (minutes, CPU)  ·  default.yaml (the real run)
│   │       └── data.py  model.py  train.py  evaluate.py  predict.py  app.py     ← ~50–150 lines each
│   └── _template/CONTRACT.md    how to add a task
│
├── cvkit/                       the shared core (no task-specific logic)
│   ├── engine.py    the training loop: AdamW · cosine LR · mixed precision · early stopping · checkpoints
│   ├── data.py      seeded, stratified splits · loaders · image transforms
│   ├── backbones.py pretrained ResNet / MobileNet / EfficientNet + a fresh head
│   ├── metrics.py   mAP · IoU / Dice — written out, so you can read them
│   ├── detection.py COCO dataset + prediction/eval shared by detection & instance segmentation
│   ├── coco.py  synthetic.py  viz.py  config.py  checkpoint.py  cli.py  utils.py
│
├── docs/                        getting-started · data-formats · glossary · troubleshooting
├── tests/                       unit tests + one end-to-end smoke test per task (CPU, runs in CI)
├── tools/                       build_notebooks.py — the notebooks are generated; edit the generator
├── requirements.txt  LICENSE
```

The split matters: **notebooks teach, the task folders do the work.** Every notebook step is also a terminal
command, so when you outgrow notebooks the project doesn't change.

## Use your own data

1. Arrange files as the task's README shows. All formats in one page: [docs/data-formats.md](docs/data-formats.md)
   (ImageFolder, `labels.csv`, COCO JSON, mask images — with YOLO / Pascal VOC conversion notes).
2. `python -m cvkit <task> train --set data.root=data/my-dataset`
3. `python -m cvkit <task> evaluate --set data.root=data/my-dataset`

The loaders stop with a specific message on bad data (missing image, wrong column count, RGB mask, class id out of range…)
instead of failing in epoch 1. Class names, class counts, splits, model heads, reports and the app all adapt automatically.

## Outputs

```
outputs/<run>/
├── checkpoints/best.pt     best validation score — used by evaluate / predict / app
├── checkpoints/last.pt
├── history.csv             per-epoch metrics
├── reports/                metrics.json + plots specific to the task
└── predictions/            annotated images from `predict`, where the task has them
```

Checkpoints are self-describing: they store the config they were trained with, and loading uses `weights_only=True`,
so a checkpoint file can't run code and editing the YAML later can't break inference.

## Requirements

Python 3.10+, `torch ≥ 2.8`, `torchvision ≥ 0.23` (developed and tested on Python 3.13, torch 2.14, torchvision 0.29). No GPU needed. On Colab only `pip install pyyaml scikit-learn gradio` is missing
(every notebook's first cell does it).

```bash
pip install -r requirements-dev.txt && python -m pytest        # unit tests + every task end to end
python tools/build_notebooks.py --execute                     # rebuild + re-run the notebooks
```

## FAQ

**Does it need a GPU?** No. `--quick` presets run on CPU. For the real configs a free Colab T4 is plenty.

**Why synthetic demo data?** So the first run needs no download, account or folder wrangling — and the demo doubles as a working
example of each data format (`demo-data` writes it to `data/demo/<task>/`).

**Pretrained or from scratch?** The quick presets train from scratch (works offline, fine for the easy demos). Real data almost always wants
`model.pretrained=true`, and for detection/instance segmentation a COCO-pretrained detector (`model.pretrained_detector=true`).

**What about transformers, CLIP, SAM, anomaly detection, depth?** Not yet — this kit covers the classic tasks first.
Next candidates: anomaly detection, zero-shot classification (CLIP), promptable segmentation (SAM), image retrieval, depth estimation.

**Why notebooks *and* scripts?** Notebooks are where you look at data and results; scripts are what you automate and deploy.
Projects that live only in one notebook die in that notebook.

---

## Want to go deeper?

This kit is the free companion to **Computer Vision with PyTorch — The Practical Masterclass**: 12 chapters, 100% runnable code,
from tensors to deployed projects (object detection, segmentation, GANs, model optimization, and three portfolio capstones).
Available in **English & Bahasa Indonesia** — link in bio at [@sofyantandungan](https://instagram.com/sofyantandungan).

Found a bug or built something with the kit? Tag [@sofyantandungan](https://instagram.com/sofyantandungan) — I feature builds.
