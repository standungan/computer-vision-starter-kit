"""Per-task content for tools/build_notebooks.py. Code cells use `cfg` and the imports shown in each cell."""

CURVES_GENERIC = ("**Gap widening** between train and val → overfitting (more augmentation, fewer epochs, smaller model). "
                  "**Both flat and poor** → underfitting (train longer, raise the learning rate, unfreeze the backbone). "
                  "**Val bouncing hard** → learning rate too high, or the val split is small. "
                  "Remember the quick preset trains for only a few epochs, so expect curves still on their way up.")

# ======================================================================= classification
CLASSIFICATION = dict(
    title="Image classification",
    blurb="**Classification** assigns one label to a whole image: *which of these N classes is this?* "
          "Here the classes are three shapes; the same code classifies steel defects, plant diseases, or anything you can sort into folders.",
    minutes=1,
    peek='''from PIL import Image
from cvkit.viz import image_grid
from tasks.classification.data import build_datasets

ds = build_datasets(cfg)
classes, samples = ds["classes"], ds["samples"]
picks = samples[::max(len(samples) // 8, 1)][:8]
_ = image_grid([Image.open(p) for p, _ in picks], [classes[t] for _, t in picks], cols=4)''',
    quick_read="**Reading it:** the confusion matrix puts the true class on the rows and the predicted class on the columns — a bright diagonal means correct. "
               "`top_losses.png` shows the test images the model was most wrong (or least sure) about; ✗ marks real mistakes.",
    predict_intro="`Predictor` loads the saved model and returns the top classes with probabilities:",
    predict_code='''from tasks.classification.predict import Predictor
from tasks.classification.data import test_image_paths

predictor = Predictor.from_config(cfg)
for path in test_image_paths(cfg, 4):
    top = predictor.predict(path, top_k=3)
    print(Path(path).name, "->", ", ".join(f"{p['label']} {p['confidence']:.2f}" for p in top))''',
    questions=["**What classes exist, and how many images in each?** (imbalance changes everything downstream)",
               "**What do the images actually look like?** (resolution, colour, background, artifacts)",
               "**Could a human tell the classes apart?** (if not, check the labels before blaming the model)"],
    data_md="One folder per class — the folder name is the label (`root/<class>/*.jpg`). Splits are made per class (*stratified*), "
            "so every split keeps the class balance, and a fixed `seed` makes them identical on every run.",
    explore=[
        ("## Class inventory",
         '''import matplotlib.pyplot as plt
from cvkit.viz import BLUE
from tasks.<<TASK>>.data import build_datasets

ds = build_datasets(cfg)
classes = ds["classes"]
counts = [ds["class_counts"][i] for i in range(len(classes))]

fig, ax = plt.subplots(figsize=(7, 0.6 * len(classes) + 1.5))
ax.barh(classes, counts, color=BLUE)
ax.invert_yaxis()
for i, c in enumerate(counts):
    ax.text(c, i, f" {c}", va="center")
ax.set_title("images per class")
plt.show()
print(f"{len(classes)} classes, {sum(counts)} images")
if max(counts) > 2 * min(counts):
    print("WARNING: more than 2x class imbalance - look at per-class metrics, consider a weighted loss or oversampling")'''),
        ("## Look at the images\nOne row per class, straight off disk with no transforms. This is the single highest-value cell in the notebook.",
         '''import random
from PIL import Image
from cvkit.viz import image_grid

rng = random.Random(cfg["seed"])
per_class, images, titles = 5, [], []
for ci, name in enumerate(classes):
    paths = [p for p, t in ds["samples"] if t == ci]
    for p in rng.sample(paths, min(per_class, len(paths))):
        images.append(Image.open(p)); titles.append(name)
_ = image_grid(images, titles, cols=per_class, cell=2.0)'''),
        ("## Image geometry\nMixed sizes are fine (everything is resized to `image_size`), but you want to *know* before choosing it.",
         '''from collections import Counter

sizes = Counter()
for path, _ in ds["samples"][:300]:
    with Image.open(path) as im:
        sizes[im.size + (im.mode,)] += 1
for (w, h, mode), n in sizes.most_common(5):
    print(f"{w}x{h} {mode}: {n} images")
print(f"\\nconfig image_size: {cfg['data']['image_size']} (every image is resized to this square)")'''),
    ],
    aug_md="Same image, eight draws of the *training* transform, next to the deterministic *evaluation* transform. "
           "If an augmented image could plausibly change class, the augmentation is too aggressive.",
    aug_code='''from cvkit.data import denormalize, image_transforms
from cvkit.viz import image_grid
from tasks.<<TASK>>.data import build_datasets

ds = build_datasets(cfg)
size, aug = cfg["data"]["image_size"], cfg["data"]["augment"]
train_tf, eval_tf = image_transforms(size, aug, True), image_transforms(size, aug, False)
path, target = ds["samples"][0]
img = Image.open(path).convert("RGB")
shown = [denormalize(eval_tf(img))] + [denormalize(train_tf(img)) for _ in range(7)]
_ = image_grid(shown, ["eval (fixed)"] + [f"train draw {i}" for i in range(1, 8)], cols=4,
               suptitle=f"class: {ds['classes'][target]}")''',
    batch_code='''from cvkit.data import denormalize
from cvkit.viz import image_grid
from tasks.<<TASK>>.data import build_loaders

loaders = build_loaders(cfg)
images, labels = next(iter(loaders["train"]))
print("images:", tuple(images.shape), images.dtype, "| labels:", tuple(labels.shape), labels.dtype)
_ = image_grid([denormalize(i) for i in images[:8]], [loaders["classes"][l] for l in labels[:8]], cols=4,
               suptitle="one training batch (un-normalised for display)")''',
    checklist=["Every class folder has images, and the names make sense", "No class is drowning the others (or you have a plan)",
               "You looked at real images and the labels match what you see", "`image_size` suits the native resolution"],
    model_summary='''from cvkit.utils import count_parameters
from tasks.<<TASK>>.data import build_loaders
from tasks.<<TASK>>.model import build_model

classes = build_loaders(cfg)["classes"]
model = build_model(cfg, {"classes": classes})
p = count_parameters(model)
print(f"arch {cfg['model']['arch']} | pretrained {cfg['model']['pretrained']} | frozen backbone {cfg['model']['freeze_backbone']}")
print(f"{p['total']:,} parameters, {p['trainable']:,} trainable ({p['trainable_pct']:.0f}%) | {len(classes)} output classes")
del model''',
    curves_md=CURVES_GENERIC,
    reports=["confusion_matrix.png", "top_losses.png"],
    read_md="* **Accuracy** — share of test images labelled correctly. Quote it with the test-set size: on 66 images one mistake is 1.5 points.\n"
            "* **Per-class precision / recall / F1** (in the printed report) — precision: of what the model *called* class X, how much was X; "
            "recall: of what *was* X, how much it found. A class with high precision but low recall is being under-predicted.\n"
            "* **Macro-F1** — F1 averaged over classes equally; stays honest when classes are imbalanced, where accuracy flatters.\n"
            "* **Confusion matrix** — find the off-diagonal cells: those are *which* classes get mixed up. Two classes that confuse each other "
            "often look alike to a human too, or are labelled inconsistently.\n"
            "* **Top losses** — the error-analysis step most people skip. For each image ask: is the **label wrong** (fix the data)? "
            "is it **genuinely ambiguous** (the ceiling is the labels, not the model)? or is the model **missing something obvious** (more data, targeted augmentation)?",
    experiments=["`model.freeze_backbone = True` — how far does feature extraction alone get you?",
                 "`model.arch = 'resnet34'` or `'mobilenet_v3_small'` — does capacity help, or just cost time? how much does a deploy-friendly model give up?",
                 "`train.epochs` ×2 — is the curve still climbing at the end?",
                 "`data.augment.rotation = 0` and `data.augment.vflip = False` — what does augmentation buy you on this data?"],
    predictor_code='''from tasks.<<TASK>>.predict import Predictor
from tasks.<<TASK>>.data import build_datasets
from cvkit.viz import image_grid

predictor = Predictor.from_config(cfg)
test = build_datasets(cfg)["test"]
images, titles = [], []
for path, target in test.samples[:8]:
    top = predictor.predict(path, top_k=1)[0]
    mark = "✓" if top["label"] == predictor.classes[target] else "✗"
    images.append(Image.open(path))
    titles.append(f"{mark} {top['label']} ({top['confidence']:.2f})\\ntrue: {predictor.classes[target]}")
_ = image_grid(images, titles, cols=4)''',
    predict_more_md="## A confidence threshold for the real world\n"
                    "A softmax probability is not a guarantee, but on held-out data it is usually informative. "
                    "Check where the *wrong* predictions sit, then abstain below a threshold and send those to a human:",
    predict_more_code='''import csv
import matplotlib.pyplot as plt
from cvkit.viz import BLUE, ORANGE

with open(Path(cfg["output"]["dir"]) / "reports" / "predictions.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
right = [float(r["confidence"]) for r in rows if r["true"] == r["pred"]]
wrong = [float(r["confidence"]) for r in rows if r["true"] != r["pred"]]
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(right, bins=20, range=(0, 1), color=BLUE, alpha=0.9, label=f"correct ({len(right)})")
if wrong:
    ax.hist(wrong, bins=20, range=(0, 1), color=ORANGE, alpha=0.9, label=f"wrong ({len(wrong)})")
ax.set_xlabel("confidence"); ax.set_ylabel("test images"); ax.legend(frameon=False)
ax.set_title("confidence: correct vs wrong")
plt.show()

THRESHOLD = 0.8
def predict_or_abstain(image, threshold=THRESHOLD):
    top = predictor.predict(image, top_k=1)[0]
    return top if top["confidence"] >= threshold else {**top, "label": "uncertain - needs human review"}

print(predict_or_abstain(test.samples[0][0]))''',
    ship_extra="- [ ] You chose a confidence threshold from the histogram above (and decided what happens below it)\n",
)

# ========================================================================== multilabel
MULTILABEL = dict(
    title="Multi-label classification",
    blurb="**Multi-label classification** tags an image with *any number* of labels — none, one, or several — instead of forcing one. "
          "Photo tagging, defect types that co-occur, and medical findings are all multi-label. Each label is its own yes/no question.",
    minutes=1,
    peek='''from PIL import Image
from cvkit.viz import image_grid
from tasks.multilabel.data import build_datasets

ds = build_datasets(cfg)
classes, samples = ds["classes"], ds["samples"]
picks = samples[:8]
_ = image_grid([Image.open(p) for p, _ in picks],
               [", ".join(c for c, v in zip(classes, t) if v) or "(none)" for _, t in picks], cols=4)''',
    quick_read="**Reading it:** `per_label_metrics.png` shows precision, recall and F1 for each label — a weak label stands out. "
               "`threshold_sweep.png` shows macro-F1 as the yes/no cut-off moves, so you can see whether 0.5 was a good choice.",
    predict_intro="`Predictor.predict` returns *every* label with its probability and whether it clears the threshold:",
    predict_code='''from tasks.multilabel.predict import Predictor
from tasks.multilabel.data import test_image_paths

predictor = Predictor.from_config(cfg)
for path in test_image_paths(cfg, 4):
    preds = predictor.predict(path)
    present = [f"{r['label']} {r['confidence']:.2f}" for r in preds if r["present"]]
    print(Path(path).name, "->", ", ".join(present) or "(nothing above threshold)")''',
    questions=["**How common is each label?** (a rare label is hard to learn and easy to ignore)",
               "**How many labels does a typical image have?** (including images with none)",
               "**Which labels appear together?** (co-occurrence the model can exploit — or be fooled by)"],
    data_md="An `images/` folder plus `labels.csv`: the first column is the file name, then one **0/1 column per label** "
            "(the header gives the label names). The split is random, fixed by `seed`.",
    explore=[
        ("## How common is each label?",
         '''import numpy as np
import matplotlib.pyplot as plt
from cvkit.viz import BLUE
from tasks.<<TASK>>.data import build_datasets

ds = build_datasets(cfg)
classes, samples = ds["classes"], ds["samples"]
Y = np.array([t for _, t in samples])                  # (images, labels) of 0/1
freq = Y.mean(0)

fig, ax = plt.subplots(figsize=(7, 0.6 * len(classes) + 1.5))
ax.barh(classes, freq, color=BLUE); ax.invert_yaxis(); ax.set_xlim(0, 1)
for i, f in enumerate(freq):
    ax.text(f, i, f" {f:.0%}", va="center")
ax.set_title("share of images that have each label")
plt.show()
print("rarest label:", classes[int(freq.argmin())], f"({freq.min():.0%}) -> consider train.pos_weight=true")'''),
        ("## Labels per image",
         '''per_image = Y.sum(1).astype(int)
fig, ax = plt.subplots(figsize=(6, 3))
vals = np.bincount(per_image)
ax.bar(range(len(vals)), vals, color=BLUE)
ax.set_xlabel("labels on one image"); ax.set_ylabel("images"); ax.set_xticks(range(len(vals)))
plt.show()
print(f"{int((per_image == 0).sum())} images have no label at all")'''),
        ("## Which labels appear together?\nThe diagonal is how often each label occurs; off-diagonal cells count images having both.",
         '''from cvkit.viz import plot_confusion_matrix

co = (Y.T @ Y).astype(int)
_ = plot_confusion_matrix(co, classes, title="co-occurrence (images having both labels)")'''),
    ],
    aug_md="Flips are safe for most tags; avoid anything that could *remove* a label from the picture (an aggressive crop can cut out the very thing you're tagging).",
    aug_code='''from PIL import Image
from cvkit.data import denormalize, image_transforms
from cvkit.viz import image_grid

size, aug = cfg["data"]["image_size"], cfg["data"]["augment"]
train_tf, eval_tf = image_transforms(size, aug, True), image_transforms(size, aug, False)
path, target = samples[0]
img = Image.open(path).convert("RGB")
shown = [denormalize(eval_tf(img))] + [denormalize(train_tf(img)) for _ in range(7)]
_ = image_grid(shown, ["eval (fixed)"] + [f"train draw {i}" for i in range(1, 8)], cols=4,
               suptitle="labels: " + (", ".join(c for c, v in zip(classes, target) if v) or "none"))''',
    batch_code='''from cvkit.data import denormalize
from cvkit.viz import image_grid
from tasks.<<TASK>>.data import build_loaders

loaders = build_loaders(cfg)
images, labels = next(iter(loaders["train"]))
print("images:", tuple(images.shape), "| labels:", tuple(labels.shape), labels.dtype, "- one 0/1 per label per image")
names = [", ".join(c for c, v in zip(loaders["classes"], l) if v) or "(none)" for l in labels[:8]]
_ = image_grid([denormalize(i) for i in images[:8]], names, cols=4, suptitle="one training batch")''',
    checklist=["Label names in the CSV header are what you intend", "No label is so rare the model can't learn it (or you set `train.pos_weight`)",
               "You looked at a few images and the 0/1 labels agree with what you see"],
    model_summary='''from cvkit.utils import count_parameters
from tasks.<<TASK>>.data import build_loaders
from tasks.<<TASK>>.model import build_model

classes = build_loaders(cfg)["classes"]
model = build_model(cfg, {"classes": classes})
p = count_parameters(model)
print(f"arch {cfg['model']['arch']} | pretrained {cfg['model']['pretrained']} | {len(classes)} labels: {classes}")
print(f"{p['total']:,} parameters, {p['trainable']:,} trainable ({p['trainable_pct']:.0f}%)")
print("loss: BCEWithLogits - an independent sigmoid per label (single-label classification uses a softmax instead)")
del model''',
    curves_md=CURVES_GENERIC + " Here **map** is average precision (higher is better) and `label_acc` is per-label accuracy at the threshold.",
    reports=["per_label_metrics.png", "threshold_sweep.png", "worst_images.png"],
    read_md="* **mAP** — average precision per label, averaged. It does not depend on the 0.5 cut-off: it measures how well the model *ranks* images "
            "for each label. Use it to compare models; it picked the best epoch.\n"
            "* **Precision / recall / F1 per label** — at the configured threshold. A rare label usually shows high precision, low recall.\n"
            "* **Macro-F1 vs micro-F1** — macro averages labels equally (a rare label matters as much as a common one); micro pools every decision "
            "(dominated by the common labels).\n"
            "* **Exact match** — the share of images where the *whole label set* is right. Harsh: one wrong label out of five fails the image.\n"
            "* **Threshold sweep** — if the curve peaks away from 0.5, change `eval.threshold`. Pick it on *val*, not on test.\n"
            "* **Worst images** — compare true and predicted sets: are missed labels small or occluded? are false labels look-alikes?",
    experiments=["`train.pos_weight = True` — up-weights rare labels; watch recall on the rarest label",
                 "`eval.threshold = 0.3` vs `0.7` — see the precision/recall trade in the report",
                 "`model.arch = 'resnet34'` — does capacity help?",
                 "`model.freeze_backbone = True` — feature extraction only"],
    predictor_code='''from tasks.<<TASK>>.predict import Predictor
from tasks.<<TASK>>.data import build_datasets
from cvkit.viz import image_grid

predictor = Predictor.from_config(cfg)
test = build_datasets(cfg)["test"]
images, titles = [], []
for path, target in test.samples[:8]:
    preds = predictor.predict(path)
    got = {r["label"] for r in preds if r["present"]}
    true = {c for c, v in zip(predictor.classes, target) if v}
    images.append(Image.open(path))
    titles.append(("✓ " if got == true else "✗ ") + "pred: " + (", ".join(sorted(got)) or "-") + "\\ntrue: " + (", ".join(sorted(true)) or "-"))
_ = image_grid(images, titles, cols=4)''',
    predict_more_md="## Moving the threshold\nThe model outputs a probability per label; the threshold turns it into yes/no. The same image at three thresholds:",
    predict_more_code='''path = test.samples[0][0]
for thr in (0.2, 0.5, 0.8):
    preds = predictor.predict(path, threshold=thr)
    print(f"threshold {thr}:", [r["label"] for r in preds if r["present"]])
print("probabilities:", {r["label"]: round(r["confidence"], 2) for r in predictor.predict(path)})''',
    ship_extra="- [ ] You picked the threshold on validation data, for the error you can afford (missed labels vs false alarms)\n",
)

# ========================================================================== regression
REGRESSION = dict(
    title="Regression from images",
    blurb="**Image regression** predicts a *number* from a picture — an object's size, a person's age, a quality score, how many things there are. "
          "The demo predicts the radius (in pixels) of a circle.",
    minutes=1,
    peek='''from PIL import Image
from cvkit.viz import image_grid
from tasks.regression.data import build_datasets

ds = build_datasets(cfg)
samples = ds["samples"]
picks = samples[:8]
_ = image_grid([Image.open(p) for p, _ in picks], [f"{ds['stats']['target_name']} = {v:.1f}" for _, v in picks], cols=4)''',
    quick_read="**Reading it:** in `pred_vs_true.png` a perfect model puts every dot on the dashed diagonal. `residuals.png` should be a narrow bump centred on 0 — "
               "off-centre means a systematic bias. The printed *baseline MAE* is what you'd get by always guessing the average: a useful model must beat it clearly.",
    predict_intro="`Predictor.predict` returns a single float, in the same units as your `labels.csv`:",
    predict_code='''from tasks.regression.predict import Predictor
from tasks.regression.data import build_datasets

predictor = Predictor.from_config(cfg)
test = build_datasets(cfg)["test"]
for path, true in test.samples[:5]:
    print(f"{Path(path).name}: predicted {predictor.predict(path):6.2f} | true {true:6.2f}")''',
    questions=["**What range and spread does the target have?** (and how well would guessing the average already do?)",
               "**Are there outliers or impossible values?** (a typo in `labels.csv` can dominate the loss)",
               "**Is the target visible in the image at all?** (if a human couldn't estimate it, the model won't either)"],
    data_md="An `images/` folder plus `labels.csv` with exactly one numeric column after the file name. Training standardises the target "
            "(subtract the *train* mean, divide by the *train* std) so the network learns values near 0 instead of, say, 0–400; predictions are converted back to your units.",
    explore=[
        ("## The target distribution",
         '''import numpy as np
import matplotlib.pyplot as plt
from cvkit.viz import BLUE, ORANGE
from tasks.<<TASK>>.data import build_datasets

ds = build_datasets(cfg)
samples = ds["samples"]
values = np.array([v for _, v in samples])
name = ds["stats"]["target_name"]

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(values, bins=25, color=BLUE, edgecolor="#fcfcfb")
ax.axvline(values.mean(), color=ORANGE, linestyle="--")
ax.set_xlabel(name); ax.set_ylabel("images"); ax.set_title(f"{name}: mean {values.mean():.1f}, std {values.std():.1f}")
plt.show()
print(f"range {values.min():.1f} .. {values.max():.1f}")
print(f"baseline: always predicting the mean would be off by {np.abs(values - values.mean()).mean():.2f} on average (MAE)")'''),
        ("## Low, middle and high values side by side\nCan *you* see the difference? If not, the model will struggle too.",
         '''from PIL import Image
from cvkit.viz import image_grid

order = np.argsort(values)
idx = [order[int(q * (len(order) - 1))] for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
_ = image_grid([Image.open(samples[i][0]) for i in idx], [f"{name} = {values[i]:.1f}" for i in idx], cols=5, cell=2.2)'''),
    ],
    aug_md="For regression, augmentation **must not change the target.** Flips and small rotations keep a circle's radius; zooming or cropping would not, "
           "so the config has no zoom. Ask of every augmentation: *does the answer stay the same?*",
    aug_code='''from cvkit.data import denormalize, image_transforms
from cvkit.viz import image_grid

size, aug = cfg["data"]["image_size"], cfg["data"]["augment"]
train_tf, eval_tf = image_transforms(size, aug, True), image_transforms(size, aug, False)
path, value = samples[0]
img = Image.open(path).convert("RGB")
shown = [denormalize(eval_tf(img))] + [denormalize(train_tf(img)) for _ in range(7)]
_ = image_grid(shown, ["eval (fixed)"] + [f"train draw {i}" for i in range(1, 8)], cols=4, suptitle=f"{name} = {value:.1f} in every draw")''',
    batch_code='''from cvkit.data import denormalize
from cvkit.viz import image_grid
from tasks.<<TASK>>.data import build_loaders

loaders = build_loaders(cfg)
images, targets = next(iter(loaders["train"]))
print("images:", tuple(images.shape), "| targets:", tuple(targets.shape), targets.dtype)
print("train stats used to standardise:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in loaders["stats"].items()})
_ = image_grid([denormalize(i) for i in images[:8]], [f"{float(t):.1f}" for t in targets[:8]], cols=4, suptitle="one training batch")''',
    checklist=["The target's range and units make sense", "You checked a few of the largest and smallest values by eye",
               "No augmentation changes the target", "You know the baseline MAE the model must beat"],
    model_summary='''from cvkit.utils import count_parameters
from tasks.<<TASK>>.data import build_loaders
from tasks.<<TASK>>.model import build_model

stats = build_loaders(cfg)["stats"]
model = build_model(cfg, {})
p = count_parameters(model)
print(f"arch {cfg['model']['arch']} | pretrained {cfg['model']['pretrained']} | loss: {cfg['train']['loss']}")
print(f"{p['total']:,} parameters, {p['trainable']:,} trainable | target '{stats['target_name']}': mean {stats['mean']:.2f}, std {stats['std']:.2f}")
del model''',
    curves_md=CURVES_GENERIC + " Here **mae** is the average miss in your units (lower is better).",
    reports=["pred_vs_true.png", "residuals.png", "worst_errors.png"],
    read_md="* **MAE** — on average the prediction is this far off, in your target's units. The most interpretable number; compare it to the *baseline MAE* printed above.\n"
            "* **RMSE** — like MAE but squares errors first, so a few big misses weigh more. RMSE much larger than MAE means a heavy tail of bad predictions.\n"
            "* **R²** — 1 is perfect; 0 means no better than always predicting the mean; negative means worse than that.\n"
            "* **Predicted vs true** — dots fanning out at the ends means the model is poor at extremes; a slope flatter than the diagonal means it squeezes predictions toward the mean.\n"
            "* **Residuals** — centred on zero? Narrow? Skewed? A shifted centre is a bias you can sometimes fix.\n"
            "* **Worst errors** — are these mislabelled, unusual images, or edge-of-range values?",
    experiments=["`train.loss = 'mse'` or `'l1'` instead of `'huber'` — how do outliers change things?",
                 "`model.arch = 'resnet34'`, then `'mobilenet_v3_small'`",
                 "`train.epochs` ×2 — is val MAE still falling?",
                 "Add noise to a few labels in `labels.csv` and see which loss copes best"],
    predictor_code='''from tasks.<<TASK>>.predict import Predictor
from tasks.<<TASK>>.data import build_datasets
from cvkit.viz import image_grid

predictor = Predictor.from_config(cfg)
test = build_datasets(cfg)["test"]
images, titles = [], []
for path, true in test.samples[:8]:
    pred = predictor.predict(path)
    images.append(Image.open(path)); titles.append(f"pred {pred:.1f}  true {true:.1f}  (off by {abs(pred - true):.1f})")
_ = image_grid(images, titles, cols=4)''',
    predict_more_md="## A sanity check: does the model respect what shouldn't matter?\nFlipping a circle doesn't change its radius, so predictions on the original and the flipped image should agree. "
                    "Cheap *invariance tests* like this catch models that learned the wrong cue.",
    predict_more_code='''from PIL import ImageOps

for path, true in test.samples[:5]:
    img = Image.open(path)
    print(f"true {true:6.2f} | original {predictor.predict(img):6.2f} | mirrored {predictor.predict(ImageOps.mirror(img)):6.2f}")''',
    ship_extra="- [ ] You know the units, and what error size is acceptable to the person using the prediction\n"
               "- [ ] You tested inputs outside the training range: regression models don't extrapolate\n",
)

# ========================================================================== detection
DET_ANN = '''from PIL import Image
from cvkit.coco import load_coco
from cvkit.viz import draw_boxes

coco = load_coco(cfg["data"]["root"], cfg["data"].get("annotations", "annotations.json"), cfg["data"].get("images_dir", "images"))
classes, items = coco["classes"], coco["images"]'''

DETECTION = dict(
    title="Object detection",
    blurb="**Object detection** finds every object in an image and returns, for each, a **box** and a **class** (plus a confidence score). "
          "Unlike classification, an image can hold zero, one or many objects. The demo detects circles, squares and triangles.",
    minutes=5,
    peek=DET_ANN + '''
from cvkit.viz import image_grid

shown = [draw_boxes(Image.open(im["path"]), [a["bbox"] for a in im["annotations"]], [a["label"] for a in im["annotations"]], classes=classes)
         for im in items[:8]]
_ = image_grid(shown, [f"{len(im['annotations'])} object(s)" for im in items[:8]], cols=4, cell=2.4)''',
    quick_read="**Reading it:** `predictions.png` puts the ground truth next to the model's boxes. `ap_per_class.png` shows how well each class is detected. "
               "`worst_images.png` shows the images with the most missed and false detections — start error analysis there.",
    predict_intro="`Predictor.predict` returns a list of detections (label, confidence, box); `visualize` draws them:",
    predict_code='''from cvkit.viz import image_grid
from tasks.detection.predict import Predictor
from tasks.detection.data import test_image_paths

predictor = Predictor.from_config(cfg)
paths = test_image_paths(cfg, 4)
dets = [predictor.predict(p) for p in paths]
_ = image_grid([predictor.visualize(p, d) for p, d in zip(paths, dets)], [f"{len(d)} detected" for d in dets], cols=4, cell=2.6)''',
    questions=["**How many objects per image, and of which classes?** (empty images and crowded images need different care)",
               "**How big are the objects?** (tiny objects need a larger `image_size`)",
               "**Do the boxes actually sit on the objects?** (wrong box format is the #1 detection bug — draw them and look)"],
    data_md="COCO format: an `images/` folder and one `annotations.json` listing every image, class and box (`bbox = [x, y, width, height]` in pixels). "
            "See [docs/data-formats.md](../../../docs/data-formats.md) for a worked example and YOLO / Pascal VOC conversion.",
    explore=[
        ("## Objects per image and per class",
         DET_ANN + '''
import numpy as np
import matplotlib.pyplot as plt
from cvkit.viz import BLUE

per_image = np.array([len(im["annotations"]) for im in items])
per_class = np.bincount([a["label"] - 1 for im in items for a in im["annotations"]], minlength=len(classes))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2))
vals = np.bincount(per_image)
a1.bar(range(len(vals)), vals, color=BLUE); a1.set_xlabel("objects in an image"); a1.set_ylabel("images"); a1.set_title("objects per image")
a2.bar(classes, per_class, color=BLUE); a2.set_ylabel("boxes"); a2.set_title("boxes per class")
plt.tight_layout(); plt.show()
print(f"{len(items)} images, {int(per_class.sum())} boxes, {int((per_image == 0).sum())} images with no objects")'''),
        ("## How big are the objects?\nSize relative to the image decides how large `image_size` must be — a 10-pixel object shrunk to a 5-pixel one can't be found.",
         '''sizes = np.array([[(a["bbox"][2] - a["bbox"][0]) / im["width"], (a["bbox"][3] - a["bbox"][1]) / im["height"]]
                  for im in items for a in im["annotations"]])
fig, ax = plt.subplots(figsize=(4.6, 4))
ax.scatter(sizes[:, 0], sizes[:, 1], s=14, color=BLUE, alpha=0.7)
ax.set_xlabel("box width / image width"); ax.set_ylabel("box height / image height"); ax.set_title("box sizes")
plt.show()
print(f"median object covers {np.median(sizes[:, 0] * sizes[:, 1]):.1%} of the image area")'''),
        ("## Check the boxes sit on the objects\nThe most valuable cell here. Wrong box format shows up instantly as boxes that are shifted, too small, or outside the image.",
         '''import random
from cvkit.viz import image_grid

rng = random.Random(0)
pick = rng.sample(items, min(8, len(items)))
_ = image_grid([draw_boxes(Image.open(im["path"]), [a["bbox"] for a in im["annotations"]], [a["label"] for a in im["annotations"]], classes=classes) for im in pick],
               [Path(im["path"]).name for im in pick], cols=4, cell=2.4)'''),
    ],
    aug_md="Detection augmentation has to move the **boxes together with the pixels** — a flipped image with unflipped boxes is garbage. "
           "The kit uses torchvision's v2 transforms, which do exactly that. Same image, four draws:",
    aug_code='''import torch
from torchvision.transforms.functional import to_pil_image
from cvkit.detection import build_coco_datasets
from cvkit.viz import draw_boxes, image_grid

train_ds = build_coco_datasets(cfg)["train"]
shown = []
for _ in range(4):
    img, tgt = train_ds[0]                          # a fresh random draw each time
    shown.append(draw_boxes(to_pil_image(img), tgt["boxes"].as_subclass(torch.Tensor).numpy(), tgt["labels"].numpy(), classes=train_ds.images and classes))
_ = image_grid(shown, [f"draw {i}" for i in range(1, 5)], cols=4, cell=2.6, suptitle="same image, random augmentation")''',
    batch_code='''from cvkit.detection import build_coco_loaders

loaders = build_coco_loaders(cfg)
images, targets = next(iter(loaders["train"]))
print(f"a batch is a tuple of {len(images)} images (not one stacked tensor: sizes may differ)")
print("image:", tuple(images[0].shape), images[0].dtype, "| target keys:", list(targets[0]))
print("boxes:", tuple(targets[0]["boxes"].shape), "(N, 4) as x0, y0, x1, y1 | labels:", targets[0]["labels"].tolist(), "(0 is reserved for background)")''',
    checklist=["Boxes are drawn on the objects, not beside them", "You know the smallest object size and set `image_size` accordingly",
               "Every class you care about has enough boxes (dozens at least)", "Empty images are intentional"],
    model_summary='''from cvkit.utils import count_parameters
from cvkit.detection import build_coco_loaders
from tasks.<<TASK>>.model import build_model

classes = build_coco_loaders(cfg)["classes"]
model = build_model(cfg, {"classes": classes})
p = count_parameters(model)
print(f"arch {cfg['model']['arch']} | pretrained backbone {cfg['model']['pretrained']} | {len(classes)} classes (+1 background)")
print(f"{p['total']:,} parameters, {p['trainable']:,} trainable ({p['trainable_pct']:.0f}%)")
del model''',
    curves_md="Detection has no validation *loss*; the right-hand panels are validation **mAP** (higher is better). The left panel is the training loss. "
              "Expect mAP to climb in steps. " + CURVES_GENERIC,
    reports=["ap_per_class.png", "predictions.png", "worst_images.png"],
    read_md="* **IoU** (intersection over union) — how much a predicted box overlaps a true box: shared area ÷ combined area. 1 = identical.\n"
            "* **mAP@.5** — a detection is *correct* if it has IoU ≥ 0.5 with an unmatched true box of the same class. Generous: roughly \"found the right object\".\n"
            "* **mAP@[.5:.95]** — the same, averaged over IoU thresholds 0.5, 0.55, … 0.95. Strict: rewards *tight* boxes. This one picked the best epoch. "
            "A big gap between mAP@.5 and mAP@[.5:.95] means the model finds objects but boxes them loosely.\n"
            "* **Precision / recall at the score threshold** — what a user sees: precision = of the boxes shown, how many were real; recall = of the real objects, how many got a box. "
            "Raise `eval.score_threshold` for fewer false boxes, lower it for fewer misses.\n"
            "* **AP per class** — which class is hard? Often the rare or the small one.\n"
            "* **Worst images** — missed object (recall problem), phantom box (precision problem), or right object wrong class? Duplicate boxes of different classes on one object are common early on.",
    experiments=["`data.image_size = 192` — does a larger input help small objects, and what does it cost in time?",
                 "`train.epochs` ×2 — the mAP curve is probably still climbing",
                 "`data.augment.hflip = False` — what does the flip contribute?",
                 "`model.arch = 'fasterrcnn_mobilenet_v3_large_320_fpn'` — a lighter, faster detector",
                 "On real data, `model.pretrained = True` (and, for resnet50/mobilenet archs, `model.pretrained_detector = True` to start from COCO) usually matters more than any other change"],
    predictor_code='''from cvkit.viz import image_grid
from tasks.<<TASK>>.predict import Predictor
from tasks.<<TASK>>.data import test_image_paths

predictor = Predictor.from_config(cfg)
paths = test_image_paths(cfg, 8)
dets = [predictor.predict(p) for p in paths]
_ = image_grid([predictor.visualize(p, d) for p, d in zip(paths, dets)], [f"{len(d)} detected" for d in dets], cols=4, cell=2.5)
print(dets[0])''',
    predict_more_md="## The score threshold\nEvery box has a confidence score. Where you cut decides how many boxes you see — "
                    "low finds more objects (and more false ones), high is cautious.",
    predict_more_code='''from cvkit.viz import image_grid

path = paths[0]
thresholds = (0.2, 0.5, 0.8)
results = [predictor.predict(path, score_threshold=t) for t in thresholds]
_ = image_grid([predictor.visualize(path, r) for r in results], [f"threshold {t}: {len(r)} boxes" for t, r in zip(thresholds, results)], cols=3, cell=3)''',
    ship_extra="- [ ] You chose a score threshold on validation data for the mistakes you can afford (missed vs phantom objects)\n"
               "- [ ] You tested on images with *no* target objects, to see how many phantom boxes appear\n",
)

# ================================================================ instance segmentation
INST_PRELUDE = '''import numpy as np
from PIL import Image
from cvkit.coco import load_coco, polygons_to_mask
from cvkit.viz import draw_boxes, image_grid, overlay_masks

coco = load_coco(cfg["data"]["root"], cfg["data"].get("annotations", "annotations.json"), cfg["data"].get("images_dir", "images"))
classes, items = coco["classes"], coco["images"]

def show(im):
    anns = im["annotations"]
    masks = np.stack([polygons_to_mask(a["segmentation"], im["height"], im["width"]) for a in anns])
    shaded = overlay_masks(Image.open(im["path"]), masks, [a["label"] for a in anns])
    return draw_boxes(shaded, [a["bbox"] for a in anns], [a["label"] for a in anns], classes=classes, width=1)'''

INSTANCE = dict(
    title="Instance segmentation",
    blurb="**Instance segmentation** outlines every object *individually*: for each one you get a class, a score, a box and a **pixel mask**. "
          "Two touching circles give two separate masks — that's what separates it from semantic segmentation, which only says \"circle pixels\".",
    minutes=10,
    peek=INST_PRELUDE + '''
_ = image_grid([show(im) for im in items[:8]], [f"{len(im['annotations'])} instance(s)" for im in items[:8]], cols=4, cell=2.4)''',
    quick_read="**Reading it:** `predictions.png` shows true masks next to predicted masks (one colour per class). The headline number is **mask mAP**: overlap is measured between masks, "
               "so it checks the *outline*, not just the bounding rectangle. It is always lower than box mAP.",
    predict_intro="`Predictor.predict` returns one dict per object, including a boolean `mask` the size of the image:",
    predict_code='''from cvkit.viz import image_grid
from tasks.instance_segmentation.predict import Predictor
from tasks.instance_segmentation.data import test_image_paths

predictor = Predictor.from_config(cfg)
paths = test_image_paths(cfg, 4)
objs = [predictor.predict(p) for p in paths]
_ = image_grid([predictor.visualize(p, o) for p, o in zip(paths, objs)], [f"{len(o)} instance(s)" for o in objs], cols=4, cell=2.6)''',
    questions=["**How many objects per image, and do they overlap or touch?** (touching objects are why instance masks exist)",
               "**Do the polygons trace the objects?** (draw the masks and look)",
               "**How big are the objects?** (masks of tiny objects are only a few pixels)"],
    data_md="Same COCO file as detection, but **every annotation also needs `segmentation`** — one or more polygons `[x1, y1, x2, y2, …]` tracing the object. "
            "The kit rasterizes the polygons into masks on the fly. RLE masks (`iscrowd`) are not supported.",
    explore=[
        ("## Objects and mask sizes",
         INST_PRELUDE + '''
import matplotlib.pyplot as plt
from cvkit.viz import BLUE

per_image = np.array([len(im["annotations"]) for im in items])
areas = np.array([polygons_to_mask(a["segmentation"], im["height"], im["width"]).mean() for im in items for a in im["annotations"]])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2))
vals = np.bincount(per_image); a1.bar(range(len(vals)), vals, color=BLUE); a1.set_xlabel("objects in an image"); a1.set_ylabel("images")
a2.hist(areas * 100, bins=20, color=BLUE, edgecolor="#fcfcfb"); a2.set_xlabel("mask area (% of image)"); a2.set_ylabel("objects")
plt.tight_layout(); plt.show()
print(f"{len(items)} images, {len(areas)} instances; smallest mask covers {areas.min():.2%} of its image")'''),
        ("## Do the masks match the objects?",
         '''import random
rng = random.Random(1)
pick = rng.sample(items, min(8, len(items)))
_ = image_grid([show(im) for im in pick], [Path(im["path"]).name for im in pick], cols=4, cell=2.4)'''),
    ],
    aug_md="Flips must move the **masks and boxes with the pixels**, which torchvision's v2 transforms do. Same image, four random draws:",
    aug_code='''import torch
from torchvision.transforms.functional import to_pil_image
from cvkit.detection import build_coco_datasets
from cvkit.viz import draw_boxes, image_grid, overlay_masks

train_ds = build_coco_datasets(cfg, with_masks=True)["train"]
shown = []
for _ in range(4):
    img, tgt = train_ds[0]
    shaded = overlay_masks(to_pil_image(img), tgt["masks"].as_subclass(torch.Tensor).numpy(), tgt["labels"].numpy())
    shown.append(draw_boxes(shaded, tgt["boxes"].as_subclass(torch.Tensor).numpy(), tgt["labels"].numpy(), classes=classes, width=1))
_ = image_grid(shown, [f"draw {i}" for i in range(1, 5)], cols=4, cell=2.6, suptitle="same image, random augmentation")''',
    batch_code='''from cvkit.detection import build_coco_loaders

loaders = build_coco_loaders(cfg, with_masks=True)
images, targets = next(iter(loaders["train"]))
t = targets[0]
print(f"a batch is a tuple of {len(images)} images; each target has: {list(t)}")
print("boxes", tuple(t["boxes"].shape), "| masks", tuple(t["masks"].shape), "(one H x W mask per object) | labels", t["labels"].tolist())''',
    checklist=["Polygons trace the objects", "You know whether objects overlap (that's the case instance masks are for)",
               "Every annotation has a `segmentation`", "Small objects are large enough to survive `image_size`"],
    model_summary='''from cvkit.utils import count_parameters
from cvkit.detection import build_coco_loaders
from tasks.<<TASK>>.model import build_model

classes = build_coco_loaders(cfg, with_masks=True)["classes"]
model = build_model(cfg, {"classes": classes})
p = count_parameters(model)
print(f"arch {cfg['model']['arch']} | pretrained backbone {cfg['model']['pretrained']} | {len(classes)} classes (+1 background)")
print(f"{p['total']:,} parameters, {p['trainable']:,} trainable ({p['trainable_pct']:.0f}%)")
print("Mask R-CNN = Faster R-CNN (boxes + classes) + a mask head that predicts the object's outline inside each box")
del model''',
    curves_md="The validation panels show mask and box **mAP** (higher is better); the training loss is on the left. Mask mAP picked the best epoch. " + CURVES_GENERIC,
    reports=["ap_per_class.png", "predictions.png", "worst_images.png"],
    read_md="* **Mask IoU** — shared pixels ÷ pixels covered by either mask. It punishes a mask that's too fat, too thin or shifted, even when the box is fine.\n"
            "* **Mask mAP@[.5:.95] / @.5 / @.75** — as for detection, but with mask overlap. This is the headline number. Mask mAP@.75 is a good \"are the outlines accurate?\" gauge.\n"
            "* **Box mAP** — printed alongside; if box mAP is fine but mask mAP is low, detection works and the mask head needs more data or training.\n"
            "* **Precision / recall at the score threshold** — what a user sees (counted on boxes).\n"
            "* **Worst images** — merged instances (two objects, one mask), split instances, masks bleeding into the background, or missed small objects.",
    experiments=["`train.epochs` ×2 — mask quality keeps improving for longer than box quality",
                 "`data.image_size = 192` — finer masks, slower training",
                 "`eval.mask_threshold = 0.3` vs `0.7` — fatter or thinner masks from the same model",
                 "On real data, start from a COCO-trained model: `model.arch = 'maskrcnn_resnet50_fpn'`, `model.pretrained_detector = True`"],
    predictor_code='''from cvkit.viz import image_grid
from tasks.<<TASK>>.predict import Predictor
from tasks.<<TASK>>.data import test_image_paths

predictor = Predictor.from_config(cfg)
paths = test_image_paths(cfg, 8)
objs = [predictor.predict(p) for p in paths]
_ = image_grid([predictor.visualize(p, o) for p, o in zip(paths, objs)], [f"{len(o)} instance(s)" for o in objs], cols=4, cell=2.5)
first = next((o for per_image in objs for o in per_image), None)
if first is None:
    print("no instance above the score threshold - train longer, or lower the threshold")
else:
    print({k: (v if k != "mask" else f"bool array {v.shape}, {int(v.sum())} px") for k, v in first.items()})''',
    predict_more_md="## What you can compute from masks\nBoxes can't give you area or shape; masks can. For example, the area of each detected object:",
    predict_more_code='''for obj in objs[0]:
    print(f"{obj['label']:<9} score {obj['confidence']:.2f}  area {int(obj['mask'].sum()):5d} px  ({obj['mask'].mean():.1%} of the image)")''',
    ship_extra="- [ ] You checked mask quality at the sizes that matter (small, touching and partially hidden objects)\n",
)

# ========================================================================= segmentation
SEGMENTATION = dict(
    title="Semantic segmentation",
    blurb="**Semantic segmentation** classifies every *pixel*: road, sky, tumour, defect, background. The output is an image the same size as the input, "
          "where each pixel holds a class id. It doesn't separate individual objects — two touching circles are one 'circle' region.",
    minutes=5,
    peek='''import numpy as np
from PIL import Image
from cvkit.viz import image_grid, overlay_label_map
from tasks.segmentation.data import build_datasets, read_mask

ds = build_datasets(cfg)
classes, pairs = ds["classes"], ds["pairs"]
print("classes:", dict(enumerate(classes)))
shown, titles = [], []
for img_path, mask_path in pairs[:4]:
    im = Image.open(img_path).convert("RGB")
    shown += [im, overlay_label_map(im, read_mask(mask_path).numpy())]
    titles += ["image", "mask overlay"]
_ = image_grid(shown, titles, cols=4, cell=2.4)''',
    quick_read="**Reading it:** `predictions.png` shows image | ground truth | prediction. **IoU per class** is the headline: shared pixels ÷ pixels either mask covers. "
               "Pixel accuracy alone can look great when the background is most of the image.",
    predict_intro="`Predictor.predict` returns a `(H, W)` array of class ids at the image's own size:",
    predict_code='''from cvkit.viz import image_grid
from tasks.segmentation.predict import Predictor
from tasks.segmentation.data import test_image_paths

predictor = Predictor.from_config(cfg)
paths = test_image_paths(cfg, 4)
masks = [predictor.predict(p) for p in paths]
print("mask:", masks[0].shape, masks[0].dtype, "| class ids present:", sorted(set(masks[0].ravel().tolist())))
_ = image_grid([predictor.visualize(p, m) for p, m in zip(paths, masks)], [str(sorted(set(m.ravel().tolist()))) for m in masks], cols=4, cell=2.6)''',
    questions=["**How much of the data is each class?** (a tiny object class next to a huge background is the usual problem)",
               "**Are masks the right size, and aligned with their images?**",
               "**Is the class id numbering what you expect?** (`classes.txt`, and 255 for \"ignore\")"],
    data_md="An `images/` folder, a `masks/` folder with a same-named single-channel PNG per image where **each pixel's value is its class id**, and `classes.txt` "
            "(line *n* names class id *n − 1*; pixels with 255 are ignored). Image and mask are always transformed *together*.",
    explore=[
        ("## How many pixels does each class get?\nThis chart decides your loss function. A class with 1% of pixels is easily ignored by plain cross-entropy — hence the Dice term in the loss.",
         '''import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cvkit.viz import BLUE, image_grid, overlay_label_map
from tasks.segmentation.data import build_datasets, read_mask

ds = build_datasets(cfg)
classes, pairs = ds["classes"], ds["pairs"]
print("classes:", dict(enumerate(classes)))

counts = np.zeros(len(classes))
for _, mask_path in pairs:
    m = read_mask(mask_path).numpy()
    counts += np.bincount(m[m < len(classes)].ravel(), minlength=len(classes))
share = counts / counts.sum()
fig, ax = plt.subplots(figsize=(7, 0.6 * len(classes) + 1.5))
ax.barh(classes, share, color=BLUE); ax.invert_yaxis(); ax.set_xlim(0, 1)
for i, s in enumerate(share):
    ax.text(s, i, f" {s:.1%}", va="center")
ax.set_title("share of all labelled pixels")
plt.show()
if share.max() > 0.8:
    print("NOTE: one class holds >80% of pixels. Accuracy will look great even for a useless model; judge by per-class IoU.")'''),
        ("## Do masks line up with the images, and do sizes agree?",
         '''bad = []
for img_path, mask_path in pairs:
    with Image.open(img_path) as im, Image.open(mask_path) as mk:
        if im.size != mk.size:
            bad.append((Path(img_path).name, im.size, mk.size))
print(f"{len(pairs)} image/mask pairs; {len(bad)} with mismatched sizes", bad[:3])
print("(mismatched sizes are resized independently - they must be the same shape in the first place)" if bad else "all pairs have matching sizes")'''),
    ],
    aug_md="Image and mask go through the **same** random transform: a flipped image must come with a flipped mask. Masks are resized with *nearest-neighbour* so class ids are never blended into meaningless in-between values.",
    aug_code='''import torch
from cvkit.data import denormalize
from cvkit.viz import image_grid, overlay_label_map
from torchvision.transforms.functional import to_pil_image

train_ds = ds["train"]
shown, titles = [], []
for i in range(4):
    img, mask = train_ds[0]                      # same sample, fresh random augmentation
    pil = to_pil_image(denormalize(img).permute(2, 0, 1))
    shown += [pil, overlay_label_map(pil, mask.numpy().astype("uint8"))]
    titles += [f"draw {i + 1}", "its mask"]
_ = image_grid(shown, titles, cols=4, cell=2.4)''',
    batch_code='''from tasks.<<TASK>>.data import build_loaders

loaders = build_loaders(cfg)
images, masks = next(iter(loaders["train"]))
print("images:", tuple(images.shape), images.dtype, "| masks:", tuple(masks.shape), masks.dtype, "(one class id per pixel)")
print("class ids in this batch:", sorted(masks.unique().tolist()))''',
    checklist=["Image/mask pairs line up and have the same size", "`classes.txt` matches the ids actually used in the masks",
               "You know which classes are rare, and kept `train.dice_weight` > 0 if one is tiny", "Unlabelled areas are marked 255, not 0"],
    model_summary='''import torch
from cvkit.utils import count_parameters
from tasks.<<TASK>>.data import build_loaders
from tasks.<<TASK>>.model import build_model

classes = build_loaders(cfg)["classes"]
model = build_model(cfg, {"classes": classes})
p = count_parameters(model)
size = cfg["data"]["image_size"]
out = model.eval()(torch.randn(1, 3, size, size))
print(f"arch {cfg['model']['arch']} | pretrained {cfg['model']['pretrained']} | {len(classes)} classes")
print(f"{p['total']:,} parameters, {p['trainable']:,} trainable | input (1,3,{size},{size}) -> output {tuple(out.shape)}: a score per class per pixel")
del model''',
    curves_md="Validation panels show **miou** (mean IoU, the headline — it picked the best epoch), `mdice` and `pixel_acc`. " + CURVES_GENERIC,
    reports=["per_class_iou.png", "confusion_matrix.png", "predictions.png", "worst_images.png"],
    read_md="* **IoU per class** — shared pixels ÷ pixels in either mask (1 = perfect outline). Look at the *worst* class, not the average.\n"
            "* **Mean IoU (mIoU)** — IoU averaged over classes: the standard single number.\n"
            "* **Dice** — a close relative of IoU (2·overlap ÷ total area of both); always ≥ IoU, same ordering.\n"
            "* **Pixel accuracy** — share of pixels correct. *Misleading* when the background dominates: predicting \"all background\" can score 90%+. Quote it only next to mIoU.\n"
            "* **Confusion matrix (pixels)** — each row shows where a true class's pixels went. Large off-diagonal cells show which classes bleed into each other (often object ↔ background at edges).\n"
            "* **Worst images** — thin or tiny objects missed, ragged borders, a class painted on the wrong region. Ragged borders usually mean more resolution or training is needed.",
    experiments=["`train.dice_weight = 0` vs `1.0` — what does the Dice term do for the rarest class?",
                 "`model.arch = 'lraspp_mobilenet_v3_large'` — a lighter model; compare mIoU and time",
                 "`data.image_size = 192` — finer outlines, slower",
                 "`train.epochs` ×2 — is mIoU still rising?"],
    predictor_code='''from cvkit.viz import image_grid
from tasks.<<TASK>>.predict import Predictor
from tasks.<<TASK>>.data import test_image_paths

predictor = Predictor.from_config(cfg)
paths = test_image_paths(cfg, 8)
masks = [predictor.predict(p) for p in paths]
_ = image_grid([predictor.visualize(p, m) for p, m in zip(paths, masks)], [", ".join(f"{k} {v:.0%}" for k, v in predictor.summary(m).items() if v) for m in masks], cols=4, cell=2.5)''',
    predict_more_md="## Turning a mask into numbers\nThe point of segmentation is usually a measurement: how much of the image is each class?",
    predict_more_code='''print(predictor.summary(masks[0]))                         # area fraction per class
print("pixels of each class:", {c: int((masks[0] == i).sum()) for i, c in enumerate(predictor.classes)})''',
    ship_extra="- [ ] You checked thin/small structures and object borders, where segmentation usually fails\n",
)

# ========================================================================== keypoints
KEYPOINTS = dict(
    title="Keypoint estimation",
    blurb="**Keypoint estimation** finds named landmarks — body joints, facial points, the corners of a part — and returns an (x, y) position for each. "
          "The demo locates the three corners of a triangle (`apex`, `left`, `right`). This task expects **one labelled object per image**.",
    minutes=2,
    peek='''from PIL import Image
from cvkit.viz import draw_keypoints, image_grid
from tasks.keypoints.data import build_datasets

ds = build_datasets(cfg)
names, items = ds["keypoint_names"], ds["items"]
print("keypoints:", names)
shown = [draw_keypoints(Image.open(it["path"]), [(it["keypoints"][i], it["keypoints"][i + 1]) for i in range(0, len(it["keypoints"]), 3)], names) for it in items[:8]]
_ = image_grid(shown, [Path(it["path"]).name for it in items[:8]], cols=4, cell=2.4)''',
    quick_read="**Reading it:** `predictions.png` draws the prediction as coloured dots and the truth as white rings — a dot inside its ring is a hit. "
               "`pck_curve.png` shows the share of keypoints within a given distance of the truth, so you see the whole accuracy picture at once.",
    predict_intro="`Predictor.predict` returns one dict per keypoint, with x and y in the image's own pixels:",
    predict_code='''from cvkit.viz import image_grid
from tasks.keypoints.predict import Predictor
from tasks.keypoints.data import test_image_paths

predictor = Predictor.from_config(cfg)
paths = test_image_paths(cfg, 4)
pts = [predictor.predict(p) for p in paths]
print(pts[0])
_ = image_grid([predictor.visualize(p, q) for p, q in zip(paths, pts)], [Path(p).name for p in paths], cols=4, cell=2.6)''',
    questions=["**Which keypoints are labelled, and are they visible?** (hidden points teach the model to guess)",
               "**Where in the image do the keypoints fall?** (the spread of positions sets how much the model must generalise)",
               "**Is the keypoint order consistent?** (\"left\" must always mean the same thing — a swap poisons training)"],
    data_md="COCO keypoints: `annotations.json` with `categories[0][\"keypoints\"]` = the point names and, per object, `keypoints: [x1, y1, v1, x2, y2, v2, …]` "
            "where `v` is 0 not labelled · 1 labelled but hidden · 2 visible. **One object per image.**",
    explore=[
        ("## Where do the keypoints fall?\nEvery keypoint of every image, normalised to the image frame. A tight cluster means the model can lean on position; a wide spread means it must really look.",
         '''from tasks.keypoints.data import build_datasets

ds = build_datasets(cfg)
names, items = ds["keypoint_names"], ds["items"]

import numpy as np
import matplotlib.pyplot as plt
from cvkit.viz import CLASS_COLORS

fig, ax = plt.subplots(figsize=(4.6, 4.6))
for k, name in enumerate(names):
    xy = np.array([[it["keypoints"][3 * k] / it["width"], it["keypoints"][3 * k + 1] / it["height"]] for it in items])
    ax.scatter(xy[:, 0], xy[:, 1], s=12, color=CLASS_COLORS[k], alpha=0.7, label=name)
ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal"); ax.legend(frameon=False, loc="upper right")
ax.set_title("keypoint positions (image frame)")
plt.show()
vis = np.array([[it["keypoints"][3 * k + 2] for k in range(len(names))] for it in items])
print({n: f"{(vis[:, k] > 0).mean():.0%} labelled" for k, n in enumerate(names)})'''),
        ("## The model's training target: heatmaps\nInstead of regressing (x, y) directly, the network predicts one *heatmap* per keypoint — an image that glows where that point probably is — and we read the position off the brightest pixel. "
         "Each heatmap below is the target the model learns for one keypoint.",
         '''import torch
from cvkit.data import denormalize
from cvkit.viz import image_grid
from tasks.keypoints.model import make_heatmaps

image, points, visible = ds["train"][0]
size = cfg["data"]["image_size"]
hm = make_heatmaps(points[None], visible[None].float(), size, cfg["model"]["heatmap_sigma"])[0]
_ = image_grid([denormalize(image)] + [hm[k] for k in range(len(names))], ["image"] + [f"target: {n}" for n in names], cols=len(names) + 1, cell=2.4,
               suptitle=f"heatmaps are {tuple(hm.shape[1:])} - 1/4 of the {size}px input")'''),
    ],
    aug_md="Rotation, shift and zoom move the keypoints **together with the pixels** (torchvision's v2 transforms handle that). "
           "There are deliberately **no flips**: a mirrored face swaps \"left eye\" and \"right eye\", so the labels would be wrong unless you also swap them. "
           "A point pushed out of the frame is marked *not visible* and ignored.",
    aug_code='''from tasks.keypoints.data import build_datasets
from cvkit.data import denormalize
from cvkit.viz import draw_keypoints, image_grid
from torchvision.transforms.functional import to_pil_image

train_ds = build_datasets(cfg)["train"]
shown = []
for i in range(4):
    img, pts, vis = train_ds[0]
    shown.append(draw_keypoints(to_pil_image(denormalize(img).permute(2, 0, 1)), pts.numpy(), names, visible=vis.tolist()))
_ = image_grid(shown, [f"draw {i + 1}" for i in range(4)], cols=4, cell=2.6, suptitle="same image, random augmentation")''',
    batch_code='''from tasks.<<TASK>>.data import build_loaders

loaders = build_loaders(cfg)
images, points, visible = next(iter(loaders["train"]))
print("images:", tuple(images.shape), "| keypoints:", tuple(points.shape), "(batch, K, x/y) in resized pixels | visible:", tuple(visible.shape), visible.dtype)''',
    checklist=["Keypoint names and order are consistent across all images", "The points sit on the landmarks (draw them and look)",
               "You know which keypoints are often hidden", "`image_size` is a multiple of 4 and large enough for the precision you need"],
    model_summary='''import torch
from cvkit.utils import count_parameters
from tasks.<<TASK>>.data import build_loaders
from tasks.<<TASK>>.model import build_model

names = build_loaders(cfg)["keypoint_names"]
model = build_model(cfg, {"keypoint_names": names})
p = count_parameters(model)
size = cfg["data"]["image_size"]
out = model.eval()(torch.randn(1, 3, size, size))
print(f"arch {cfg['model']['arch']} | pretrained {cfg['model']['pretrained']} | {len(names)} keypoints: {names}")
print(f"{p['total']:,} parameters | input (1,3,{size},{size}) -> heatmaps {tuple(out.shape)}")
del model''',
    curves_md="Validation panels show **nme** (normalised error: lower is better, it picked the best epoch), **pck** (higher is better) and the mean error in pixels. The training loss is on the left. "
              "Note it flattens at a value well above 0 (about 4 here): cross-entropy against a blob has a *floor* equal to the blob's own entropy, so a flat loss does not mean learning stopped — watch the validation error instead. " + CURVES_GENERIC,
    reports=["pck_curve.png", "error_per_keypoint.png", "predictions.png", "worst_images.png"],
    read_md="* **NME (normalised mean error)** — average distance between predicted and true points, as a fraction of the image side. Resolution-independent; lower is better.\n"
            "* **Mean error in pixels** — the same distance in input pixels, easier to picture.\n"
            "* **PCK@threshold** — the share of keypoints within `eval.pck_threshold` × image side of the truth. \"95% of points are within 5% of the image\" is an easy sentence to say.\n"
            "* **PCK curve** — PCK as the allowed distance grows. A curve that rises steeply near zero means precise points; a slow climb means sloppy ones.\n"
            "* **Error per keypoint** — one landmark is usually the hard one (often the symmetric or the often-hidden one).\n"
            "* **Worst images** — left/right confusions, points snapped to the wrong similar-looking feature, or objects partly out of frame.",
    experiments=["`data.augment.rotation = 0` — what does rotation augmentation buy?",
                 "`model.heatmap_sigma = 1` vs `3` — sharper or softer targets",
                 "`data.image_size = 192` — finer heatmaps, slower",
                 "`model.arch = 'resnet34'` — does capacity help?"],
    predictor_code='''from cvkit.viz import image_grid
from tasks.<<TASK>>.predict import Predictor
from tasks.<<TASK>>.data import test_image_paths

predictor = Predictor.from_config(cfg)
paths = test_image_paths(cfg, 8)
pts = [predictor.predict(p) for p in paths]
_ = image_grid([predictor.visualize(p, q) for p, q in zip(paths, pts)], [Path(p).name for p in paths], cols=4, cell=2.5)''',
    predict_more_md="## Comparing with the truth\nThe `confidence` of each point is the peak of its heatmap: low confidence flags a point the model is unsure about. Let's check it against the actual error.",
    predict_more_code='''import numpy as np
from tasks.<<TASK>>.data import build_datasets

test = build_datasets(cfg)["test"]
rows = []
for it in test.items[:30]:
    for k, p in enumerate(predictor.predict(it["path"])):
        err = np.hypot(p["x"] - it["keypoints"][3 * k], p["y"] - it["keypoints"][3 * k + 1])
        rows.append((p["confidence"], err))
rows = np.array(rows)
lo, hi = rows[rows[:, 0] < np.median(rows[:, 0])], rows[rows[:, 0] >= np.median(rows[:, 0])]
print(f"mean error when confidence is low : {lo[:, 1].mean():.2f}px")
print(f"mean error when confidence is high: {hi[:, 1].mean():.2f}px")''',
    ship_extra="- [ ] You decided what to do with low-confidence keypoints (hide them, or flag the image)\n",
)

SPECS = {"classification": CLASSIFICATION, "multilabel": MULTILABEL, "regression": REGRESSION,
         "detection": DETECTION, "instance_segmentation": INSTANCE,
         "segmentation": SEGMENTATION, "keypoints": KEYPOINTS}
