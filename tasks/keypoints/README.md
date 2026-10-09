# Keypoint estimation

**Find named landmarks.** For each keypoint: an (x, y) position — body joints, facial points, corners of a part.

```
image ──► model ──► apex (81, 18) · left (47, 98) · right (110, 92)
```

## When to use it

* You need the *pose or geometry* of one object: where the joints are, where a part's corners lie, facial landmarks.
* This task expects **one labelled object per image**. Several people per image needs a detector first (detect, then crop, then locate points) or a bottom-up method — see "Go further".
* **Not this?** Whole-object boxes → [detection](../detection). A single number → [regression](../regression).

## Data format

COCO keypoints: `annotations.json` with the point names in `categories[0]["keypoints"]` and, per object, `keypoints: [x1, y1, v1, …]` (`v`: 0 not labelled · 1 hidden · 2 visible).

```
data/my-dataset/
├── images/  img_0001.jpg ...
└── annotations.json
```
```json
{"categories":  [{"id": 1, "name": "face", "keypoints": ["left_eye", "right_eye", "nose"]}],
 "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 40, 120, 140],
                  "keypoints": [80, 90, 2,  130, 92, 2,  105, 120, 2]}]}
```

Details: [data-formats.md](../../docs/data-formats.md#detection-instance-segmentation-keypoints--coco-json).

## How it works (the part worth understanding)

The network doesn't output coordinates. It outputs one **score map per keypoint**, a small image that is high where
that point probably is; the peak is the answer, refined to sub-pixel precision. Training turns each map into a probability
distribution over pixels and teaches it to match a small Gaussian blob at the true position. This beats plain coordinate
regression because it keeps spatial structure. (`model.py` is short and commented. An earlier version used mean-squared
error against the blob and **collapsed to predicting zeros** — [tests/test_task_internals.py](../../tests/test_task_internals.py) keeps that lesson.)

## Run it

```bash
python -m cvkit keypoints quickstart --quick           # ~2 min on a 4-core CPU
```

Expect a mean error of about **1 px** on 128 px images and **PCK@0.05 near 1.0** on the demo triangles. Outputs in
`outputs/keypoints_quick/reports/`: `pck_curve.png`, `error_per_keypoint.png`, `predictions.png`, `worst_images.png`, `metrics.json`.

## How to read the results

| Output | What it tells you |
|---|---|
| **NME** | average distance to the true point as a fraction of the image side (lower is better). Picks the best epoch |
| **PCK@t** | share of points within `t` × image side of the truth ("95% within 5%") |
| `pck_curve.png` | PCK as the allowed distance grows: steep near 0 = precise, slow climb = sloppy |
| `error_per_keypoint.png` | which landmark is hardest (often the symmetric or often-hidden one) |
| `worst_images.png` | truth (white ring) vs prediction (dot) on the worst test images |
| training loss | flattens above 0 on purpose (the blob's entropy is a floor) — watch validation error instead |

## Configuration that matters

| Key | Meaning |
|---|---|
| `data.augment` | `rotation`, `translate`, `scale`, `color_jitter` — keypoints move with the pixels. **No flips** (they'd swap left/right) |
| `model.heatmap_sigma` | blur radius of the training blob, in map pixels (1 = sharp, 3 = soft) |
| `eval.pck_threshold` | the distance (fraction of image side) that counts as "correct" |
| `data.image_size` | must be a multiple of 4; the score maps are ¼ of it |

## Common problems

* **Predictions all at the image centre** → the model isn't learning: lower/raise `train.lr`, train longer, check labels with notebook 01.
* **Left and right swapped** → a flip somewhere (keep flips off) or inconsistent labelling across images.
* **`has no keypoint annotation`** → every image needs exactly one object with `keypoints`.
* **Points snapped to the wrong feature** → more data or `data.image_size` larger for finer maps.

## Go further

Notebooks: [00](notebooks/00_quickstart.ipynb) → [01 data](notebooks/01_data.ipynb) → [02 train & evaluate](notebooks/02_train_and_evaluate.ipynb) → [03 predict & deploy](notebooks/03_predict_and_deploy.ipynb)

Experiments: `heatmap_sigma` 1 vs 3; turn rotation off; `resnet34`. For several objects per image, first run [detection](../detection), then
apply this model to each crop.
