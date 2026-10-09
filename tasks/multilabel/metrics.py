"""Multi-label metrics.

Each label is its own yes/no question, so we score every label separately and then average:

    precision  of the labels we said were present, how many really were?
    recall     of the labels that really were present, how many did we find?
    F1         the balance of the two
    AP / mAP   average precision per label, then the mean: threshold-free, so it
               measures how well the model *ranks* images, independent of the 0.5 cut-off
    exact      fraction of images where the whole label set is exactly right (harsh)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def multilabel_metrics(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> dict:
    pred = probs >= threshold
    true = targets > 0.5
    tp = (pred & true).sum(0).astype(float)
    fp = (pred & ~true).sum(0).astype(float)
    fn = (~pred & true).sum(0).astype(float)
    f1_per = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    micro = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)
    aps = [average_precision_score(true[:, k], probs[:, k]) for k in range(true.shape[1]) if true[:, k].any()]
    return {"f1_macro": float(f1_per.mean()), "f1_micro": float(micro),
            "map": float(np.mean(aps)) if aps else 0.0,
            "exact": float((pred == true).all(1).mean())}
