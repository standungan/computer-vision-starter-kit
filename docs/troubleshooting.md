# Troubleshooting

## Setup and running

**`ModuleNotFoundError: No module named 'cvkit'` (or `tasks`)** — You're not in the repo's top
folder. `cd` there. (Notebooks set this up automatically in their first cell.)

**`FileNotFoundError: data root not found`** — `data.root` points at a folder that doesn't exist.
Check the path, or create demo data: `python -m cvkit <task> demo-data --quick`.

**`FileNotFoundError: checkpoint not found ... best.pt`** — You ran `evaluate`, `predict` or `app`
before `train`, or with a different `output.dir`. Train first, using the same config.

**Downloading pretrained weights fails / you're offline** — `model.pretrained: true` fetches ImageNet
weights the first time. Offline? Set `model.pretrained=false` (it works, but needs more data and
epochs), or download once on a connected machine and copy `~/.cache/torch/hub/checkpoints/`.

**`CUDA out of memory`** — Lower `data.batch_size` (halve until it fits), then `data.image_size`.
For detection / instance segmentation also try a smaller `model.arch`.

**Windows: the DataLoader hangs, or errors mentioning `spawn` / `freeze_support`** — Set
`data.num_workers=0`. (The `quick` presets already do.)

**Training is very slow** — Check the first line of output: `device: cpu` means no GPU was found.
On CPU use the `quick` preset, a small `data.image_size`, and a lighter `model.arch`
(`resnet18`, `mobilenet_v3_small`). On Colab: *Runtime → Change runtime type → GPU*.

**Edited the YAML after training and now `evaluate` / `predict` behave differently?** They shouldn't: they rebuild the model from the config *stored inside the checkpoint*, so changing `model.arch` afterwards can't break them. (Your edits do apply to the next `train`.)

## Results look wrong

**Val accuracy is high, test accuracy is much lower** — The val split was used to pick the epoch
and to tune settings, so it's slightly optimistic; a small gap is normal. A big gap on a small test
set is often just noise (with 20 test images one mistake is 5%). Make the splits bigger or
cross-check with more data.

**Train is great, val is poor (overfitting)** — More data and more augmentation first; then fewer
epochs / early stopping, a smaller model, higher `model.dropout`, or `model.freeze_backbone=true`.

**Both train and val are poor (underfitting)** — Train longer, try a higher or lower `train.lr`
(×3 or ÷3), unfreeze the backbone, use a bigger model, or check that the labels are right.

**The loss is `nan` or explodes in the first epoch** — The learning rate is too high: divide
`train.lr` by 3–10. For detection also keep `train.grad_clip` on.

**Classification: it predicts the same class for everything** — Check class balance (notebook 01).
Check that the folder names are the classes and the images are really different. Lower `lr`
and train longer.

**Detection: mAP stays near 0** — Boxes are probably in the wrong format. COCO is
`[x, y, width, height]` in *pixels*; YOLO's normalized `cx cy w h` must be converted (see
data-formats.md). Draw your ground-truth boxes on a few images in notebook 01 to check them.
Also: small objects need a larger `data.image_size`.

**Detection: many duplicate or low-confidence boxes** — Raise `eval.score_threshold`. Duplicates of
different classes on one object are normal early in training and fade with more epochs.

**Segmentation: `contains class id N but classes.txt lists only …`** — A mask has a pixel value your
`classes.txt` doesn't cover. Fix the file (one name per line, line 1 = id 0) or mark those pixels as 255 (ignore).

**Segmentation: mask must be single-channel** — Your masks are RGB colour images. Convert each
colour to a class id and save as a one-channel PNG.

**Segmentation: high pixel accuracy but low mIoU** — The background dominates. Judge by per-class IoU;
raise `train.dice_weight`.

**Keypoints: the points land on the wrong side (left/right swapped)** — Flipping an image swaps
left and right. The kit doesn't flip for keypoints; if you add flips, swap the matching point
labels too.

## Data errors the kit catches for you

The loaders stop with a specific message for: a missing `labels.csv`; a CSV row with the wrong number of
columns; an image listed in the CSV that isn't on disk; a non-numeric or non-0/1 label; a missing or
RLE-encoded `segmentation`; an image with no mask; RGB masks; out-of-range class ids; a split
whose ratios don't sum to 1. If you hit an error that isn't clear, that's a bug in the kit — please open an issue.
