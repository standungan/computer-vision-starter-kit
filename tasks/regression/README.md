# Regression from images

**Predict a number from a picture** — size, age, count, quality score, price.

```
image ──► model ──► radius = 21.3 px
```

## When to use it

* The answer is a quantity on a scale, not a category: an object's dimension, a crowd size, a rating.
* **Not this?** Categories → [classification](../classification). *Where* something is → [detection](../detection) or [keypoints](../keypoints).
* A human must be able to estimate it from the image — if not, the model can't either.

## Data format

An `images/` folder plus `labels.csv` with **exactly one numeric column** after the file name:

```
data/my-dataset/
├── images/  img_0001.png ...
└── labels.csv
```
```csv
filename,value
img_0001.png,17.5
img_0002.png,9.0
```

Training standardises the target with the *train* split's mean and std; predictions come back in your units.
Details: [data-formats.md](../../docs/data-formats.md#multilabel-and-regression--labelscsv).

## Run it

```bash
python -m cvkit regression quickstart --quick          # ~1 min on CPU; predicts a circle's radius
```

Expect MAE around **0.4 px** against a "guess the average" baseline of about **7.5 px**, and R² near **1.0**.
Outputs in `outputs/regression_quick/reports/`: `pred_vs_true.png`, `residuals.png`, `worst_errors.png`, `predictions.csv`, `metrics.json`.

## How to read the results

| Output | What it tells you |
|---|---|
| **MAE** | average miss, in your units — the most interpretable number. Compare with the printed *baseline MAE* |
| **RMSE** | like MAE but big misses count more; RMSE ≫ MAE = a heavy tail of bad predictions |
| **R²** | 1 perfect · 0 no better than the average · negative worse |
| `pred_vs_true.png` | dots on the dashed diagonal = perfect. Fanning at the ends = poor at extremes |
| `residuals.png` | should be narrow and centred on 0; shifted = systematic bias |

## Configuration that matters

| Key | Meaning |
|---|---|
| `train.loss` | `huber` (robust to a few bad labels), `mse`, `l1` |
| `data.augment.*` | **must not change the target** — flips and small rotations are fine, zoom/crop usually not |
| `model.arch`, `model.pretrained`, `train.*` | as in classification; best epoch = lowest val MAE |

## Common problems

* **MAE no better than the baseline** → the target may not be visible in the image, or lr is off (try ÷3 / ×3).
* **Predictions squeezed toward the mean** → common with little data; more data / longer training / unfreeze.
* **Terrible on values outside the training range** → regression models don't extrapolate; widen the data.
* **`regression expects exactly one target column`** → keep one numeric column in `labels.csv`.

## Go further

Notebooks: [00](notebooks/00_quickstart.ipynb) → [01 data](notebooks/01_data.ipynb) → [02 train & evaluate](notebooks/02_train_and_evaluate.ipynb) → [03 predict & deploy](notebooks/03_predict_and_deploy.ipynb)

Experiments: swap the loss; try `mobilenet_v3_small`; add label noise and compare `huber` vs `mse`. Next task: [keypoints](../keypoints).
