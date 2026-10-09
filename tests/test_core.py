"""Unit tests for the shared core: config, splits, metrics, synthetic data, COCO parsing."""
import json

import numpy as np
import pytest
import torch
from torchvision.ops import box_iou

from cvkit.coco import load_coco, polygons_to_mask
from cvkit.config import apply_override, deep_merge, load_config
from cvkit.data import split_indices
from cvkit.metrics import iou_from_confusion, mean_average_precision, mask_iou, update_confusion
from cvkit import synthetic


# ------------------------------------------------------------------ config
def test_base_inheritance_and_override(tmp_path):
    (tmp_path / "a.yaml").write_text("x: 1\nnested: {p: 1, q: 2}\n")
    (tmp_path / "b.yaml").write_text("base: a.yaml\nnested: {q: 20}\n")
    cfg = load_config(tmp_path / "b.yaml", ["nested.r=3", "x=7"])
    assert cfg == {"x": 7, "nested": {"p": 1, "q": 20, "r": 3}}


def test_load_config_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "c.yaml").write_text("output: {dir: outputs/x}\n")
    load_config(tmp_path / "c.yaml")
    assert not (tmp_path / "outputs").exists()


def test_bad_override_rejected():
    with pytest.raises(ValueError):
        apply_override({}, "no_equals_sign")


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1}}
    deep_merge(base, {"a": {"b": 2}})
    assert base == {"a": {"b": 1}}


# ------------------------------------------------------------------ splits
RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def test_split_is_disjoint_complete_and_reproducible():
    groups = [i % 3 for i in range(90)]
    a = split_indices(90, RATIOS, seed=1, groups=groups)
    b = split_indices(90, RATIOS, seed=1, groups=groups)
    assert a == b
    allidx = a["train"] + a["val"] + a["test"]
    assert sorted(allidx) == list(range(90))
    for name in ("train", "val", "test"):          # stratified: every class in every split
        assert {groups[i] for i in a[name]} == {0, 1, 2}


def test_small_groups_still_reach_val_and_test():
    groups = [0] * 4 + [1] * 30
    s = split_indices(len(groups), RATIOS, seed=0, groups=groups)
    assert any(groups[i] == 0 for i in s["val"]) and any(groups[i] == 0 for i in s["test"])
    assert any(groups[i] == 0 for i in s["train"])


def test_bad_ratios():
    with pytest.raises(ValueError):
        split_indices(10, {"train": 0.5, "val": 0.2, "test": 0.2}, 0)


# ------------------------------------------------------------------ metrics
def test_miou_known_values():
    cm = torch.zeros(3, 3, dtype=torch.long)
    target = torch.tensor([[[0, 1], [2, 2]]])
    pred = torch.tensor([[[0, 1], [2, 1]]])
    res = iou_from_confusion(update_confusion(cm, pred, target, 3))
    assert res["iou"] == pytest.approx([1.0, 0.5, 0.5])
    assert res["pixel_acc"] == pytest.approx(0.75)


def test_ignore_index_is_skipped():
    cm = torch.zeros(2, 2, dtype=torch.long)
    update_confusion(cm, torch.tensor([[0, 1]]), torch.tensor([[255, 1]]), 2)
    assert cm.sum() == 1


def _box_fn(a, b):
    return box_iou(a["boxes"], b["boxes"])


def test_map_perfect_and_half():
    gt = [{"boxes": torch.tensor([[0, 0, 10, 10.], [20, 20, 40, 40]]), "labels": torch.tensor([1, 2])}]
    perfect = [{**gt[0], "scores": torch.tensor([0.9, 0.8])}]
    assert mean_average_precision(perfect, gt, 2, _box_fn)["map"] == pytest.approx(1.0)
    half = [{"boxes": torch.tensor([[0, 0, 10, 10.], [100, 100, 120, 120]]),
             "labels": torch.tensor([1, 2]), "scores": torch.tensor([0.9, 0.8])}]
    assert mean_average_precision(half, gt, 2, _box_fn)["map50"] == pytest.approx(0.5)


def test_map_penalises_duplicates_and_low_iou():
    gt = [{"boxes": torch.tensor([[0, 0, 10, 10.]]), "labels": torch.tensor([1])}]
    dup = [{"boxes": torch.tensor([[0, 0, 10, 10.], [0, 0, 10, 10.]]), "labels": torch.tensor([1, 1]),
            "scores": torch.tensor([0.9, 0.8])}]
    # the duplicate is a false positive *after* the true positive: AP stays 1.0 (recall done first)
    assert mean_average_precision(dup, gt, 1, _box_fn)["map50"] == pytest.approx(1.0)
    loose = [{"boxes": torch.tensor([[0, 0, 10, 5.]]), "labels": torch.tensor([1]),
              "scores": torch.tensor([0.9])}]               # IoU 0.5 exactly
    res = mean_average_precision(loose, gt, 1, _box_fn)
    assert res["map50"] == pytest.approx(1.0) and res["map75"] == 0.0


def test_mask_iou():
    a = torch.zeros(1, 4, 4, dtype=torch.bool); a[0, :2] = True
    b = torch.zeros(1, 4, 4, dtype=torch.bool); b[0, :, :2] = True
    assert mask_iou(a, b)[0, 0].item() == pytest.approx(2 / 6)


# ------------------------------------------------------- synthetic + coco
def test_synthetic_makers_cover_all_tasks_and_are_reproducible(tmp_path):
    for task, maker in synthetic.MAKERS.items():
        maker(tmp_path / "a" / task, n=6, seed=5)
        maker(tmp_path / "b" / task, n=6, seed=5)
    for f in (tmp_path / "a").rglob("*.png"):
        twin = tmp_path / "b" / f.relative_to(tmp_path / "a")
        assert f.read_bytes() == twin.read_bytes()


def test_coco_polygons_match_boxes(tmp_path):
    synthetic.make_coco_shapes(tmp_path, n=10, size=96, seed=2)
    coco = load_coco(tmp_path)
    assert coco["classes"] == synthetic.SHAPES
    for im in coco["images"]:
        for a in im["annotations"]:
            mask = polygons_to_mask(a["segmentation"], im["height"], im["width"])
            ys, xs = np.where(mask)
            x0, y0, x1, y1 = a["bbox"]
            assert abs(xs.min() - x0) <= 2 and abs(xs.max() - x1) <= 2
            assert abs(ys.min() - y0) <= 2 and abs(ys.max() - y1) <= 2


def test_coco_rejects_rle_and_skips_crowd(tmp_path):
    synthetic.make_coco_shapes(tmp_path, n=2, seed=0)
    data = json.loads((tmp_path / "annotations.json").read_text())
    data["annotations"][0]["iscrowd"] = 1
    data["annotations"][1]["segmentation"] = {"counts": "abc", "size": [96, 96]}
    (tmp_path / "annotations.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="RLE"):
        load_coco(tmp_path)


def test_missing_data_message_is_helpful(tmp_path):
    from cvkit.data import require_dir
    with pytest.raises(FileNotFoundError, match="demo-data"):
        require_dir(tmp_path / "nope")
