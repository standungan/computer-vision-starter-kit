# Glossary

Plain-language definitions for the terms the notebooks and READMEs use. Where a term is a
setting, the config key is in `code`.

## Data

**Dataset split** — The images are divided into three groups. **Train** teaches the model.
**Validation (val)** is used during training to pick the best epoch and spot overfitting.
**Test** is touched once, at the end, for the honest score. Looking at test results and then
changing your model to improve them quietly turns test into a second validation set. (`data.split`)

**Stratified split** — Each class is split separately, so every split has the same class mix.
Matters most when some classes are rare.

**Augmentation** — Random changes to training images (flips, small rotations, colour shifts) so
the model sees more variety than you collected. It must never change the answer: flipping a
cat is still a cat, but flipping a "6" might make a "9". Applied to the train split only.
(`data.augment`)

**Normalization** — Subtracting a mean and dividing by a standard deviation from each colour
channel. The pretrained models were trained with ImageNet's values, so we use the same.

**Class imbalance** — Some classes have far more images than others. A model can score well by
mostly guessing the common class; look at per-class numbers, not just overall accuracy.

**Annotation / label** — The answer attached to an image: a class name, a box, a mask, a number.

## Models and training

**Transfer learning / pretrained** — Starting from a network already trained on millions of images
(ImageNet) instead of from random numbers. It already knows edges, textures and shapes, so you
need far less data and time. (`model.pretrained`)

**Backbone** — The part of the network that turns pixels into features. You usually keep it and
replace the last layer(s), the **head**, to fit your task.

**Freezing the backbone** — Train only the new head and leave the backbone as-is. Faster and safer
on tiny datasets; "unfreezing" later adapts the whole network. (`model.freeze_backbone`)

**Epoch** — One full pass over the training images. (`train.epochs`)

**Batch / batch size** — How many images the model looks at before updating. Bigger = faster per
epoch but needs more memory. (`data.batch_size`)

**Loss** — One number measuring how wrong the model currently is. Training tries to push it
down. Different tasks use different losses; they are not comparable across tasks.

**Learning rate** — The size of each update step. Too high: the loss bounces around or explodes.
Too low: learning crawls. (`train.lr`)

**Cosine schedule** — The learning rate starts at `lr` and glides smoothly down toward 0 over
training, which helps the model settle into a good solution.

**Overfitting** — The model memorizes the training images instead of learning the pattern: train
score keeps improving while val score stalls or drops. Fixes: more data, more augmentation, less
training, a smaller model, dropout.

**Underfitting** — Both train and val scores are poor: the model hasn't learned enough. Fixes:
train longer, higher learning rate, a bigger model, unfreeze the backbone.

**Early stopping** — Stop when val score hasn't improved for N epochs. (`train.early_stopping_patience`)

**Checkpoint** — A saved copy of the model. `best.pt` is the epoch with the best val score;
`last.pt` is the final epoch.

**Dropout** — Randomly switching off some units during training so the network can't lean on any
single one. (`model.dropout`)

**Label smoothing** — Training toward "95% sure" instead of "100% sure". Reduces overconfidence.

**Mixed precision (AMP)** — Using 16-bit numbers where safe: faster and lighter on GPUs, same
accuracy. Ignored on CPU. (`train.amp`)

**Seed** — A number that fixes the random choices, so the same run gives the same split and
(nearly) the same result. (`seed`)

## Metrics

**Accuracy** — Share of predictions that are right. Misleading when classes are imbalanced.

**Precision** — Of everything the model *said* was X, how much really was X? High precision =
few false alarms.

**Recall** — Of everything that really *was* X, how much did the model find? High recall = few misses.

**F1** — A single score that balances precision and recall (their harmonic mean). **Macro** F1
averages over classes equally; **micro** F1 pools all predictions.

**Confusion matrix** — A table of true class (rows) against predicted class (columns). Off-diagonal
cells show which classes get mixed up.

**Confidence** — The probability the model attaches to its own prediction. It is *not* a guarantee:
models can be confidently wrong. Check it on the test set before trusting a threshold.

**Threshold** — A cut-off turning a score into a yes/no ("present if probability ≥ 0.5").
Raising it trades recall for precision.

**MAE / RMSE / R²** — Regression errors. MAE: average miss, in the units of your target. RMSE:
like MAE but big misses count extra. R²: 1 = perfect, 0 = no better than always guessing the
average.

**IoU (intersection over union)** — Overlap of two regions (boxes or masks): shared area divided
by combined area. 1 = identical, 0 = no overlap.

**AP / mAP (average precision)** — For detection: sort the model's detections by confidence, walk
down the list, and measure precision vs recall; AP is the area under that curve, mAP its average over
classes. **mAP@.5** counts a detection as correct at IoU ≥ 0.5; **mAP@[.5:.95]** averages IoU
thresholds from 0.5 to 0.95, so it rewards tight boxes.

**mIoU / Dice** — For segmentation: IoU (or Dice, a close cousin) averaged over classes.

**PCK / NME** — For keypoints. PCK: share of keypoints within a distance of the truth. NME: the
average distance, as a fraction of the image side.

## Task vocabulary

**Class / label** — A category the model can output ("cat", "scratch").

**Bounding box** — A rectangle around an object: `[x_min, y_min, x_max, y_max]` (or COCO's
`[x, y, width, height]`).

**Mask** — A per-pixel yes/no (instance segmentation: one per object) or class id (semantic
segmentation: one per image).

**Heatmap** — An image that glows where something probably is. The keypoint model predicts one per
landmark.

**Anchor / proposal / NMS** — Detection internals: the model suggests many candidate boxes
(proposals); non-maximum suppression (NMS) removes overlapping duplicates, keeping the best.

**Foundation model / zero-shot** — A large model trained on huge data that can do tasks it was never
explicitly trained on, e.g. classify with a text prompt. Not in this kit yet.
