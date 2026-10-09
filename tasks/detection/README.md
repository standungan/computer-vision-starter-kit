# Object detection

**Find every object and box it.** For each object: where it is, what it is, and how sure the model is.

```
image ──► model ──► circle  0.99  [x0,y0,x1,y1]
                    square  0.97  [x0,y0,x1,y1]
                    ...                           (zero, one or many per image)
```

## When to use it

* Count things, locate them, or trigger an action when something appears: products on a shelf, cars in a frame, defects on a part.
* **Not this?** One label for the whole image → [classification](../classification). You need the exact *outline* → [instance_segmentation](../instance_segmentation). A single point per landmark → [keypoints](../keypoints).

## Data format

[COCO](https://cocodataset.org/#format-data) JSON: an `images/` folder and one `annotations.json` (`bbox = [x, y, width, height]` in pixels).

```
data/my-dataset/
├── images/  img_0001.jpg ...
└── annotations.json
```
```json
{"images":      [{"id": 1, "file_name": "img_0001.jpg", "width": 640, "height": 480}],
 "categories":  [{"id": 1, "name": "dog"}, {"id": 2, "name": "cat"}],
 "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [120, 80, 200, 150]}]}
```

Images with no objects are fine. Converting from YOLO / Pascal VOC, and every format check: [data-formats.md](../../docs/data-formats.md#detection-instance-segmentation-keypoints--coco-json).

## Run it

```bash
python -m cvkit detection quickstart --quick           # ~5 min on a 4-core CPU
```

On the shapes demo (trained from scratch in minutes) expect **mAP@.5 near 1.0** and **mAP@[.5:.95] around 0.8**.
Outputs in `outputs/detection_quick/`: `reports/{ap_per_class,predictions,worst_images}.png`, `metrics.json`, and annotated copies of predicted images in `predictions/`.

## How to read the results

| Output | What it tells you |
|---|---|
| **IoU** | overlap of a predicted box and a true box: shared area ÷ combined area |
| **mAP@.5** | detection correct if IoU ≥ 0.5: "found the right object" |
| **mAP@[.5:.95]** | averaged over IoU 0.5…0.95: rewards *tight* boxes. Picks the best epoch |
| **precision / recall** at `eval.score_threshold` | what a user sees: of the boxes shown, how many real; of the real objects, how many found |
| **AP per class** | which class is hard (often the rare or small one) |
| `worst_images.png` | missed objects, phantom boxes, wrong class — the error-analysis starting point |

## Configuration that matters

| Key | Meaning |
|---|---|
| `data.image_size` | the model resizes the shorter side to this; **small objects need it larger** (and slower) |
| `model.arch` | `fasterrcnn_resnet18_fpn` (light, CPU-friendly) · `fasterrcnn_mobilenet_v3_large_320_fpn` / `_fpn` · `fasterrcnn_resnet50_fpn` |
| `model.pretrained` | ImageNet backbone; `model.pretrained_detector` starts from a **COCO-trained** detector (resnet50 / mobilenet archs) — usually the biggest win on real data |
| `eval.score_threshold` | detections below it are hidden in plots and counted as "not detected" |
| `train.grad_clip`, `lr`, `epochs` | detection is touchier than classification: keep clipping on |

## Common problems

* **mAP stays near 0** → boxes are probably in the wrong format (COCO wants pixels, `[x, y, width, height]`). Draw your ground truth in notebook 01 to check. Tiny objects → raise `image_size`.
* **Loss is `nan`** → lower `train.lr` by 3–10×.
* **Duplicate boxes of different classes on one object** → normal early on; train longer or raise `eval.score_threshold`.
* **Too slow on CPU** → use the `quick` preset, `fasterrcnn_resnet18_fpn`, smaller `image_size`, fewer `roi_samples_per_image`.

## Go further

Notebooks: [00](notebooks/00_quickstart.ipynb) → [01 data](notebooks/01_data.ipynb) → [02 train & evaluate](notebooks/02_train_and_evaluate.ipynb) → [03 predict & deploy](notebooks/03_predict_and_deploy.ipynb)

Experiments: a lighter backbone; a bigger `image_size`; COCO-pretrained detector on real data.
Next tasks: [instance_segmentation](../instance_segmentation) (outlines) · [segmentation](../segmentation) (every pixel).
