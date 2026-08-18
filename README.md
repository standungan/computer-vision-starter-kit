# Computer Vision Project Starter Kit

A free, reusable **PyTorch project template** for image classification — the structure I
use for real computer-vision projects, stripped to the essentials and documented end to end.

**Point it at any image dataset and you get the full pipeline:**

```
Dataset → EDA → Preprocessing → Augmentation → Training → Evaluation → Error Analysis → Inference → Deployment
```

Ships with a worked example — **steel surface-defect classification** on the NEU dataset
(6 defect classes) — but nothing in the code is specific to it. Change one line in a YAML
config and the same kit trains on your dataset.

By [Sofyan Tandungan](https://instagram.com/sofyantandungan) · NeuronEngine · MIT license.

---

## What's inside

```
computer-vision-starter-kit/
│
├── notebooks/                     ← the guided path (run in order)
│   ├── 01_explore_dataset.ipynb       EDA: classes, balance, look at the images
│   ├── 02_prepare_dataset.ipynb       stratified splits + augmentation, visualized
│   ├── 03_train_model.ipynb           transfer learning + learning curves
│   ├── 04_evaluate_model.ipynb        test metrics, confusion matrix, error analysis
│   └── 05_inference.ipynb             predictions, confidence thresholds, deployment
│
├── src/                           ← the reusable engine (notebooks import from here)
│   ├── dataset.py                     ImageFolder pipeline, stratified splits, transforms
│   ├── model.py                       5 torchvision backbones, head swap, freeze option
│   ├── train.py                       AMP, cosine LR, early stopping, checkpointing
│   ├── evaluate.py                    reports: per-class metrics, confusion matrix, top losses
│   ├── inference.py                   Predictor class + CLI
│   └── utils.py                       config loading, seeding, device
│
├── configs/
│   └── config.yaml                ← the single source of truth: data, model, training
│
├── app/
│   └── app.py                     ← Gradio demo app around the trained model
│
├── requirements.txt
├── README.md
└── LICENSE                        (MIT)
```

The split matters: **notebooks teach, `src/` does the work.** Every notebook step is also
a terminal command, so when you outgrow notebooks the project doesn't change — you just
stop opening them.

## Quickstart

```bash
# 1. install (Colab: torch is preinstalled — `pip install gradio` is enough)
pip install -r requirements.txt

# 2. get the example dataset (or use your own — next section)
#    NEU Surface Defect Database → data/NEU-CLS/<class>/*.jpg
#    https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

# 3. run the pipeline
jupyter lab notebooks/01_explore_dataset.ipynb     # then 02 → 03 → 04 → 05

# — or skip the notebooks entirely —
python -m src.train     --config configs/config.yaml
python -m src.evaluate  --config configs/config.yaml
python -m src.inference --config configs/config.yaml --input path/to/image.jpg
python app/app.py       # demo app at http://127.0.0.1:7860
```

## Use your own dataset

Arrange images as one folder per class — the layout `torchvision.datasets.ImageFolder` expects:

```
data/my-dataset/
├── class_a/  img001.jpg ...
├── class_b/  ...
└── class_c/  ...
```

Then edit **one line** in `configs/config.yaml`:

```yaml
data:
  root: data/my-dataset
```

Class names, class count, splits, the model head, evaluation reports, and the app all
adapt automatically. Rerun notebooks 01 → 05.

## The example project: steel surface defects

The NEU Surface Defect Database is 1,800 grayscale images of hot-rolled steel strip,
300 per class: `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`,
`scratches`. It's small enough to train in minutes on a free Colab GPU and real enough
to be a portfolio piece — visual quality inspection is one of the most common industrial
CV jobs.

Download from [Kaggle](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
(or the original NEU page), and place the class folders under `data/NEU-CLS/`.

## Config reference

Everything tunable lives in [`configs/config.yaml`](configs/config.yaml):

| Section | Keys | Notes |
|---|---|---|
| `data` | `root`, `image_size`, `batch_size`, `split`, `augment` | augmentation applies to the train split only |
| `model` | `arch`, `pretrained`, `freeze_backbone`, `dropout` | archs: `resnet18/34/50`, `mobilenet_v3_small`, `efficientnet_b0` |
| `train` | `epochs`, `lr`, `weight_decay`, `label_smoothing`, `amp`, `early_stopping_patience` | AMP auto-disables on CPU |
| `output` | `dir` | checkpoints + reports land here |

Outputs after a run:

```
outputs/
├── checkpoints/best.pt        highest val accuracy — used by evaluate / inference / app
├── checkpoints/last.pt
├── history.csv                per-epoch metrics
└── reports/                   classification_report.txt, confusion_matrix.png,
                               top_losses.png, predictions.csv
```

## FAQ

**Does it need a GPU?** No — it runs on CPU, just slower. On a free Colab T4 the example
project trains in a few minutes. Mixed precision (AMP) kicks in automatically on CUDA.

**Object detection / segmentation?** This kit is classification. The structure transfers,
but detection needs different datasets, losses, and metrics — that's a chapter-length
topic of its own (see below).

**Why notebooks *and* scripts?** Notebooks are where you look at data and results;
scripts are what you automate and deploy. Projects that live only in one notebook die
in that notebook.

---

## Want to go deeper?

This kit is the free companion to **Computer Vision with PyTorch — The Practical
Masterclass**: 12 chapters, 100% runnable code, from tensors to deployed projects
(object detection, segmentation, GANs, model optimization, and three portfolio
capstones). Available in **English & Bahasa Indonesia** — link in bio at
[@sofyantandungan](https://instagram.com/sofyantandungan).

Found a bug or built something with the kit? Tag [@sofyantandungan](https://instagram.com/sofyantandungan) — I feature builds.
