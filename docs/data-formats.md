# Data formats

Every task reads one dataset folder, set by `data.root` in its config. Each task's README shows
its layout; this page has all of them in one place, with converters for the common formats you
might already have.

> **Fastest way to see a layout:** `python -m cvkit <task> demo-data --quick` writes a tiny valid
> dataset for that task into `data/demo/<task>/`. Open it next to your own data and compare.

| Task | Layout | Labels live in |
|---|---|---|
| [classification](#classification--imagefolder) | `root/<class>/*.jpg` | the folder names |
| [multilabel](#multilabel-and-regression--labelscsv) | `root/images/*` + `labels.csv` | one 0/1 column per label |
| [regression](#multilabel-and-regression--labelscsv) | `root/images/*` + `labels.csv` | one numeric column |
| [detection](#detection-instance-segmentation-keypoints--coco-json) | `root/images/*` + `annotations.json` | COCO boxes |
| [instance_segmentation](#detection-instance-segmentation-keypoints--coco-json) | `root/images/*` + `annotations.json` | COCO boxes + polygons |
| [keypoints](#detection-instance-segmentation-keypoints--coco-json) | `root/images/*` + `annotations.json` | COCO keypoints |
| [segmentation](#semantic-segmentation--mask-images) | `root/images/*` + `root/masks/*` + `classes.txt` | a mask image per photo |

Image files can be `.jpg .jpeg .png .bmp .webp .tif .tiff`. Colour, grayscale and RGBA are all
converted to RGB when loaded.

---

## Classification — ImageFolder

One folder per class; the folder name *is* the label.

```
data/my-dataset/
├── cat/   img001.jpg  img002.jpg ...
├── dog/   ...
└── bird/  ...
```

Class order is alphabetical by folder name. Splits are stratified, so each class appears in
train, val and test.

## Multilabel and regression — labels.csv

An `images/` folder plus a CSV whose **first column is the file name**.

Multilabel — one 0/1 column per label (the header gives the label names):

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

Regression — exactly one numeric column (the header names the quantity):

```csv
filename,value
img_0001.jpg,17.5
img_0002.jpg,9.0
```

The kit checks the CSV on load and tells you which line or image is wrong (missing image file,
non-numeric value, label other than 0/1, wrong number of columns).

## Detection, instance segmentation, keypoints — COCO JSON

[COCO](https://cocodataset.org/#format-data) is the most common annotation format. The kit
reads the part of it that it needs:

```
data/my-dataset/
├── images/  img_0001.jpg ...
└── annotations.json
```

```json
{
  "images":     [{"id": 1, "file_name": "img_0001.jpg", "width": 640, "height": 480}],
  "categories": [{"id": 1, "name": "dog"}, {"id": 2, "name": "cat"}],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1,
     "bbox": [120, 80, 200, 150],
     "segmentation": [[120,80, 320,80, 320,230, 120,230]]}
  ]
}
```

* `bbox` is `[x, y, width, height]` in pixels, measured from the top-left corner.
* **Detection** needs `bbox` + `category_id`.
* **Instance segmentation** also needs `segmentation` as **polygons** (a list of
  `[x1, y1, x2, y2, ...]` lists). RLE-encoded masks are not supported; the error message says so.
* **Keypoints** also needs `keypoints: [x1, y1, v1, x2, y2, v2, ...]` per object, and the point
  names in `categories[0]["keypoints"]`. `v` is 0 = not labelled, 1 = labelled but hidden,
  2 = labelled and visible. **One labelled object per image.**
* Annotations with `"iscrowd": 1` and boxes smaller than 1 px are skipped.
* Images with no annotations are fine for detection and instance segmentation (they teach
  the model what "nothing here" looks like).
* One JSON file holds every image; the kit does the train/val/test split itself.

```json
"categories": [{"id": 1, "name": "face", "keypoints": ["left_eye", "right_eye", "nose"]}],
"annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 40, 120, 140],
                 "keypoints": [80, 90, 2,  130, 92, 2,  105, 120, 2]}]
```

### Converting from other formats

* **YOLO txt** (`class cx cy w h`, normalized, one `.txt` per image): COCO needs pixels and a
  corner origin. For each line: `x = (cx - w/2) * W`, `y = (cy - h/2) * H`,
  `width = w * W`, `height = h * H`. Roboflow, CVAT and Label Studio can also export COCO directly.
* **Pascal VOC** (XML with `xmin ymin xmax ymax`): `bbox = [xmin, ymin, xmax - xmin, ymax - ymin]`.
* Annotation tools (CVAT, Label Studio, Roboflow, labelme): choose **COCO JSON** on export;
  choose COCO *with polygons* for instance segmentation.

## Semantic segmentation — mask images

Each photo has a mask image with the same file name (any image extension). The mask is
**single-channel and each pixel's value is its class id**.

```
data/my-dataset/
├── images/       road_001.jpg ...
├── masks/        road_001.png ...
└── classes.txt
```
```
background
road
car
```

* Line *n* of `classes.txt` is class id *n − 1*: `background` = 0, `road` = 1, `car` = 2.
  Without `classes.txt` the kit infers the number of classes from the masks and names them
  `class_0, class_1, ...`.
* Pixels with value **255** are *ignored* by training and metrics. Use that for "unlabelled" or
  "don't care" regions.
* **Save masks as PNG**, never JPEG: JPEG compression turns class 2 into 1s and 3s at edges.
* A coloured mask (RGB) will be rejected with a message: convert colours to class ids first.
* A class id that `classes.txt` doesn't list is caught before training starts.
