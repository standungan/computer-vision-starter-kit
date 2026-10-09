# Instance segmentation

**An outline for every object.** For each object: a class, a score, a box *and a pixel mask*. Two touching objects give two separate masks.

```
image ──► model ──► circle 0.99 [box] + mask (1286 px)
                    triangle 0.97 [box] + mask (798 px)
```

## When to use it

* You need an object's **shape or area**, or need to tell apart objects that touch: cells in a microscope image, fruit to pick, parts to grasp.
* **Not this?** Boxes are enough → [detection](../detection) (simpler, faster). You only care about *class per pixel*, not individual objects → [segmentation](../segmentation).

## Data format

The same COCO JSON as detection, but **every annotation also needs `segmentation`** — polygons tracing the object:

```
data/my-dataset/
├── images/  img_0001.jpg ...
└── annotations.json
```
```json
{"annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                  "bbox": [120, 80, 200, 150],
                  "segmentation": [[120,80, 320,80, 320,230, 120,230]]}]}
```

Polygons only (RLE / `iscrowd` masks are skipped or rejected with a clear message). Masks are rasterized on the fly.
Details: [data-formats.md](../../docs/data-formats.md#detection-instance-segmentation-keypoints--coco-json).

## Run it

```bash
python -m cvkit instance_segmentation quickstart --quick     # ~10 min on a 4-core CPU
```

On the shapes demo expect **mask mAP@.5 near 1.0**, **mask mAP@[.5:.95] around 0.85**. Outputs in `outputs/instance_segmentation_quick/`: `reports/{ap_per_class,predictions,worst_images}.png`, `metrics.json`, annotated images in `predictions/`.

## How to read the results

| Output | What it tells you |
|---|---|
| **mask IoU** | shared pixels ÷ pixels in either mask — punishes fat, thin or shifted outlines even when the box is fine |
| **mask mAP@[.5:.95]** | the headline; picks the best epoch. **@.75** gauges outline accuracy |
| **box mAP** | printed alongside. Box fine but mask low → the mask head needs more data or training |
| **precision / recall** | at `eval.score_threshold`, counted on boxes |
| `worst_images.png` | merged instances, split instances, masks bleeding into background, missed small objects |

## Configuration that matters

| Key | Meaning |
|---|---|
| `model.arch` | `maskrcnn_resnet18_fpn` (light) · `maskrcnn_resnet50_fpn` |
| `model.pretrained`, `model.pretrained_detector` | ImageNet backbone / start from COCO-trained Mask R-CNN (resnet50) |
| `data.image_size` | larger = finer masks, slower |
| `eval.score_threshold`, `eval.mask_threshold` | which instances to keep; how confident a pixel must be to belong to the mask |

## Common problems

* **`instance segmentation needs polygon 'segmentation' for every annotation`** → an annotation lacks polygons; add them or drop the box.
* **Masks too fat / too thin** → adjust `eval.mask_threshold` (0.3 fatter, 0.7 thinner), train longer.
* **Very slow on CPU** → expected: this is the heaviest task. Use `quick`, keep `image_size` small, or use a GPU (Colab).

## Go further

Notebooks: [00](notebooks/00_quickstart.ipynb) → [01 data](notebooks/01_data.ipynb) → [02 train & evaluate](notebooks/02_train_and_evaluate.ipynb) → [03 predict & deploy](notebooks/03_predict_and_deploy.ipynb)

Experiments: bigger `image_size`; `mask_threshold` sweep; COCO-pretrained Mask R-CNN on real data.
