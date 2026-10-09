# Semantic segmentation

**A class for every pixel.** The output is an image the size of the input where each pixel holds a class id.

```
image ──► model ──► mask:  background background circle circle ...
                           background square    circle circle ...
```

## When to use it

* You need *regions* or *amounts*, not individual objects: road vs sidewalk, tumour vs tissue, rust vs clean metal, crop vs soil.
* **Not this?** You must tell touching objects apart or count them → [instance_segmentation](../instance_segmentation). Boxes are enough → [detection](../detection).

## Data format

Photos plus a same-named **mask image** where each pixel's value is its class id, and a `classes.txt`:

```
data/my-dataset/
├── images/       road_001.jpg ...
├── masks/        road_001.png ...      single-channel; pixel value = class id; 255 = ignore
└── classes.txt   background\nroad\ncar     (line n = class id n-1)
```

**Save masks as PNG** (JPEG blurs ids). RGB colour masks are rejected with a message. Details: [data-formats.md](../../docs/data-formats.md#semantic-segmentation--mask-images).

## Run it

```bash
python -m cvkit segmentation quickstart --quick        # ~5 min on a 4-core CPU
```

Expect **mean IoU around 0.9** after 40 epochs. Outputs in `outputs/segmentation_quick/reports/`: `per_class_iou.png`, `confusion_matrix.png`, `predictions.png`, `worst_images.png`, `classification_report.txt`, `metrics.json`; overlays of predicted images in `predictions/`.

> Watch the demo's first epochs: pixel accuracy is already **~89%** while mean IoU is only **~0.22** — the background is ~90% of every
> image, so a model that paints everything "background" scores 89% and is useless. By the end, pixel accuracy is ~99% and mIoU ~0.9.
> That gap is the single most important thing to understand about segmentation metrics.

## How to read the results

| Output | What it tells you |
|---|---|
| **IoU per class** | shared pixels ÷ pixels in either mask. Look at the *worst* class |
| **mean IoU** | IoU averaged over classes — the headline; picks the best epoch |
| **Dice** | close cousin of IoU (always ≥ IoU, same ordering) |
| **pixel accuracy** | misleading when one class dominates — quote it only next to mIoU |
| `confusion_matrix.png` | where each true class's pixels went (row %) — usually object ↔ background at edges |
| `worst_images.png` | image · truth · prediction for the lowest-IoU test images |

## Configuration that matters

| Key | Meaning |
|---|---|
| `model.arch` | `unet_resnet18` (written out in `model.py`, ~40 lines) · `lraspp_mobilenet_v3_large` · `deeplabv3_mobilenet_v3_large` |
| `train.dice_weight` | loss = cross-entropy + this × Dice. Raise it when small classes get ignored; `0` = cross-entropy only |
| `data.ignore_index` | mask value (default 255) skipped by loss and metrics: use for "unlabelled" |
| `data.image_size` | larger = finer borders, slower |

## Common problems

* **Small class never predicted** → raise `train.dice_weight`, check its pixel share in notebook 01, train longer.
* **`contains class id N but classes.txt lists only …`** → mask uses an id beyond `classes.txt`; fix the file or mark those pixels 255.
* **`mask must be single-channel`** → convert RGB masks to class-id PNGs.
* **Image/mask size mismatch** → resize both consistently before training (notebook 01 checks).

## Go further

Notebooks: [00](notebooks/00_quickstart.ipynb) → [01 data](notebooks/01_data.ipynb) → [02 train & evaluate](notebooks/02_train_and_evaluate.ipynb) → [03 predict & deploy](notebooks/03_predict_and_deploy.ipynb)

Experiments: `dice_weight` 0 vs 1; the lighter `lraspp` model; larger `image_size`. Read `model.py`'s `UNet` and change one thing.
