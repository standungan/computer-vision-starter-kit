# Multi-label classification

**Any number of labels per image** — none, one, or several. Each label is its own yes/no question.

```
image ──► model ──► beach 0.97 ✓ · sunset 0.91 ✓ · people 0.12 ✗      (threshold 0.5)
```

## When to use it

* Tags on photos, findings on a medical image, several defect types on one part, "which of these objects appear".
* **Not this?** Exactly one label per image → [classification](../classification) (its softmax makes
  classes compete; here they don't). Need locations → [detection](../detection).

## Data format

An `images/` folder plus `labels.csv`: first column = file name, then **one 0/1 column per label** (the header gives the label names).

```
data/my-dataset/
├── images/  img_0001.jpg ...
└── labels.csv
```
```csv
filename,beach,sunset,people
img_0001.jpg,1,1,0
img_0002.jpg,0,0,1
```

An image with all zeros is fine ("none of these"). The split is random, fixed by `seed`.
Details: [data-formats.md](../../docs/data-formats.md#multilabel-and-regression--labelscsv).

## Run it

```bash
python -m cvkit multilabel quickstart --quick          # ~1 min on CPU
```

Expect mAP near **1.0** and macro-F1 around **0.99** on the easy shapes demo. Outputs in `outputs/multilabel_quick/`:
`reports/{per_label_metrics,threshold_sweep,worst_images}.png`, `classification_report.txt`, `predictions.csv`, `metrics.json`.

## How to read the results

| Output | What it tells you |
|---|---|
| **mAP** | average precision per label, averaged. Threshold-free: how well images are *ranked* per label. Picks the best epoch |
| **precision / recall / F1** per label | at the config threshold. A rare label typically has high precision, low recall |
| **macro-F1 / micro-F1** | macro: every label counts equally. micro: pooled decisions, dominated by common labels |
| **exact match** | share of images whose *whole* label set is right — harsh by design |
| `threshold_sweep.png` | macro-F1 vs the yes/no cut-off: is 0.5 right for you? Choose it on val, not test |

## Configuration that matters

| Key | Meaning |
|---|---|
| `data.labels_file`, `data.images_dir` | where the CSV and images are |
| `eval.threshold` | a label is "present" when its probability ≥ this |
| `train.pos_weight` | `true` up-weights rare labels (imbalanced tags) |
| `model.*`, `data.image_size`, `data.augment.*` | as in classification |

## Common problems

* **A label is never predicted** → it's probably rare: set `train.pos_weight=true`, lower `eval.threshold`.
* **`labels must be 0 or 1`** → a cell in `labels.csv` has another value; the message names the image.
* **Exact match is low but F1 is high** → normal with many labels: one wrong label fails the image.

## Go further

Notebooks: [00](notebooks/00_quickstart.ipynb) → [01 data](notebooks/01_data.ipynb) → [02 train & evaluate](notebooks/02_train_and_evaluate.ipynb) → [03 predict & deploy](notebooks/03_predict_and_deploy.ipynb)

Experiments: `train.pos_weight`; threshold 0.3 vs 0.7; freeze the backbone. Next task: [detection](../detection).
