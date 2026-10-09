# Image classification

**One label per image.** Given a photo, which of N classes is it?

```
image ──► model ──► crazing 0.93 · scratches 0.05 · patches 0.02
```

## When to use it

* Each image is *about one thing* and you want its category: defect type, plant disease, product
  type, ripe / unripe.
* **Not this?** Several things can be present at once → [multilabel](../multilabel). You need to know
  *where* things are → [detection](../detection). You need a number, not a category → [regression](../regression).

## Data format

One folder per class — the folder name is the label:

```
data/my-dataset/
├── cat/   img001.jpg  img002.jpg ...
├── dog/   ...
└── bird/  ...
```

Splits are stratified per class and fixed by `seed`. Details and checks: [data-formats.md](../../docs/data-formats.md#classification--imagefolder).

## Run it

```bash
python -m cvkit classification quickstart --quick      # ~1 min on CPU, makes its own shapes dataset
```

You should see test accuracy around **0.98** and these files in `outputs/classification_quick/`:

```
checkpoints/best.pt        history.csv        reports/{confusion_matrix,top_losses}.png
reports/{classification_report.txt,predictions.csv,metrics.json}
```

On your own data:

```bash
python -m cvkit classification train    --set data.root=data/my-dataset
python -m cvkit classification evaluate --set data.root=data/my-dataset
python -m cvkit classification predict  --input photo.jpg
python -m cvkit classification app
```

**The worked example** is steel-surface defects (the NEU dataset: 1,800 grayscale images, six classes —
`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`). Download it from
[Kaggle](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database), put the six class
folders under `data/NEU-CLS/`, and run the commands above without `--quick` (the default config points there).

## How to read the results

| Output | What it tells you |
|---|---|
| **accuracy** | share of test images correct — quote it with the test-set size |
| **precision / recall / F1** per class | precision: of what was *called* X, how much was X. recall: of what *was* X, how much was found |
| **macro-F1** | F1 averaged over classes equally — honest when classes are imbalanced |
| `confusion_matrix.png` | rows = true class, columns = predicted. Off-diagonal = which classes get mixed up |
| `top_losses.png` | the test images the model was most wrong or unsure about, ✗ = wrong. **Start error analysis here** — wrong label? ambiguous? a real model gap? |

## Configuration that matters

| Key | Meaning |
|---|---|
| `data.root` | your dataset folder |
| `data.image_size` | images are resized to this square (224 for pretrained nets) |
| `data.augment.*` | `hflip`, `vflip`, `rotation`, `color_jitter`, `random_erasing`. Turn `vflip` off when "up" matters |
| `model.arch` | `resnet18/34/50`, `mobilenet_v3_small/large`, `efficientnet_b0` |
| `model.pretrained` / `freeze_backbone` | start from ImageNet weights; train only the new head |
| `train.epochs`, `lr`, `label_smoothing`, `early_stopping_patience` | the usual |

## Common problems

* **Everything is predicted as one class** → check class balance (notebook 01), lower `train.lr`, train longer.
* **Train accuracy ≫ val accuracy** → overfitting: more augmentation, `model.freeze_backbone=true`, fewer epochs.
* **`FileNotFoundError: data root not found`** → fix `data.root`, or `python -m cvkit classification demo-data --quick`.

More: [troubleshooting.md](../../docs/troubleshooting.md) · [glossary.md](../../docs/glossary.md)

## Go further

Notebooks: [00 quickstart](notebooks/00_quickstart.ipynb) → [01 data](notebooks/01_data.ipynb) → [02 train & evaluate](notebooks/02_train_and_evaluate.ipynb) → [03 predict & deploy](notebooks/03_predict_and_deploy.ipynb)

Experiments: compare `resnet18` vs `mobilenet_v3_small`; freeze the backbone vs not; remove each augmentation in turn.
Next task: [multilabel](../multilabel) (several labels per image) or [detection](../detection) (where are they?).
