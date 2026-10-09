"""Focused tests for task-specific math that the smoke tests can't judge (the smoke test only
proves the pipeline runs, not that it's right)."""
import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import tv_tensors

from cvkit.backbones import ARCHS, build_image_model


def test_every_image_arch_builds_and_freezes_the_whole_head():
    for arch in ARCHS:
        m = build_image_model(arch, 5, pretrained=False, freeze_backbone=True, dropout=0.1)
        assert m(torch.randn(2, 3, 64, 64)).shape == (2, 5)
        head = m.fc if hasattr(m, "fc") else m.classifier
        assert all(p.requires_grad for p in head.parameters())
        body = [p for n, p in m.named_parameters() if not n.startswith(("fc", "classifier"))]
        assert body and not any(p.requires_grad for p in body)


def test_unknown_arch_message():
    with pytest.raises(ValueError, match="choose from"):
        build_image_model("vgg_nope", 3, pretrained=False)


# ---------------------------------------------------------------- keypoints
def test_heatmap_encode_decode_roundtrip_is_subpixel():
    from tasks.keypoints.model import decode, make_heatmaps
    torch.manual_seed(0)
    pts = torch.rand(4, 6, 2) * 100 + 10
    dec, conf = decode(make_heatmaps(pts, torch.ones(4, 6), 128, 2.0))
    assert (dec - pts).abs().max() < 0.05
    assert ((conf > 0.2) & (conf <= 1.0)).all()


def test_decode_confidence_is_high_for_sharp_and_low_for_smeared_maps():
    from tasks.keypoints.model import decode, make_heatmaps, to_probs
    pts, ones = torch.tensor([[[60.0, 60.0]]]), torch.ones(1, 1)
    sharp = make_heatmaps(pts, ones, 128, 1.0)
    smeared = make_heatmaps(pts, ones, 128, 6.0)
    assert decode(sharp)[1] > decode(smeared)[1]
    flat = to_probs(torch.zeros(1, 1, 32, 32))                       # no idea where: uniform
    assert decode(flat)[1] < 0.02


def test_keypoint_loss_does_not_collapse_to_all_zero_maps():
    """The failure that motivated cross-entropy: with MSE, 'predict zeros' is a strong local optimum.
    Here the loss of an all-zero map must be clearly worse than that of a map peaked at the truth."""
    from tasks.keypoints.model import make_heatmaps
    pts, vis = torch.tensor([[[60.0, 60.0]]]), torch.ones(1, 1)
    target = make_heatmaps(pts, vis, 128, 2.0).flatten(2)
    target = target / target.sum(-1, keepdim=True)
    ce = lambda logits: -(target * logits.flatten(2).log_softmax(-1)).sum()          # noqa: E731
    zeros = torch.zeros(1, 1, 32, 32)
    peaked = torch.log(make_heatmaps(pts, vis, 128, 2.0) + 1e-6)
    assert ce(peaked) < ce(zeros) - 1.0


def test_hidden_keypoints_get_empty_heatmaps():
    from tasks.keypoints.model import make_heatmaps
    hm = make_heatmaps(torch.tensor([[[30.0, 30.0], [60.0, 60.0]]]), torch.tensor([[1.0, 0.0]]), 128, 2.0)
    assert hm[0, 0].max() > 0.9 and hm[0, 1].max() == 0


def test_affine_augmentation_moves_keypoints_with_the_pixels():
    """A bright blob and its keypoint must stay glued together through rotation/zoom/shift."""
    from tasks.keypoints.data import keypoint_transforms
    size = 64
    arr = np.zeros((size, size, 3), np.uint8)
    arr[40:43, 20:23] = 255
    tf = keypoint_transforms(size, {"rotation": 25, "translate": 0.1, "scale": [0.8, 1.2]}, train=True)
    checked = 0
    for seed in range(15):
        torch.manual_seed(seed)
        img, pt = tf(Image.fromarray(arr), tv_tensors.KeyPoints(torch.tensor([[21.0, 41.0]]), canvas_size=(size, size)))
        gray = img.mean(0)
        if gray.max() - gray.min() < 1:           # blob rotated/shifted out of frame
            continue
        ys, xs = torch.nonzero(gray > gray.min() + 0.5 * (gray.max() - gray.min()), as_tuple=True)
        assert abs(xs.float().mean() - pt[0, 0]) < 1.0 and abs(ys.float().mean() - pt[0, 1]) < 1.0
        checked += 1
    assert checked >= 8


def test_keypoint_metrics():
    from tasks.keypoints.metrics import keypoint_metrics
    true = torch.tensor([[[10.0, 10.0], [50.0, 50.0]]])
    pred = true + torch.tensor([[[0.0, 0.0], [10.0, 0.0]]])           # second point off by 10px of 100
    m = keypoint_metrics(pred, true, torch.ones(1, 2, dtype=torch.bool), size=100, threshold=0.05)
    assert m["nme"] == pytest.approx(0.05) and m["pck"] == pytest.approx(0.5)
    hidden = keypoint_metrics(pred, true, torch.tensor([[True, False]]), size=100, threshold=0.05)
    assert hidden["nme"] == 0.0                                        # the bad point is not visible: ignored


# ------------------------------------------------------------- segmentation
def test_unet_shapes_for_awkward_sizes():
    from tasks.segmentation.model import UNet
    net = UNet(4, "resnet18", pretrained=False).eval()
    for hw in ((64, 64), (96, 128), (100, 75)):                       # not multiples of 32
        assert net(torch.randn(1, 3, *hw)).shape == (1, 4, *hw)


def test_dice_loss_perfect_vs_wrong_and_ignore():
    from tasks.segmentation.train import dice_loss
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :4] = 1
    perfect = torch.zeros(1, 2, 8, 8)
    perfect[:, 1, :4] = 20
    perfect[:, 0, 4:] = 20
    assert dice_loss(perfect, target, 255) < 0.01
    assert dice_loss(-perfect, target, 255) > 0.9
    ignored = target.clone()
    ignored[:] = 255
    assert dice_loss(perfect, ignored, 255) == 0                       # nothing to score -> no gradient signal


def test_mask_reader_rejects_rgb_and_keeps_palette_ids(tmp_path):
    from tasks.segmentation.data import read_mask
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(tmp_path / "rgb.png")
    with pytest.raises(ValueError, match="single-channel"):
        read_mask(tmp_path / "rgb.png")
    pal = Image.fromarray(np.array([[0, 1], [2, 3]], np.uint8), mode="P")
    pal.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255] + [0] * 756)
    pal.save(tmp_path / "pal.png")
    assert read_mask(tmp_path / "pal.png").tolist() == [[0, 1], [2, 3]]    # ids, not RGB-converted greys


def test_segmentation_data_errors_are_specific(tmp_path, smoke_cfg):
    from cvkit.synthetic import make_demo_data
    from tasks.segmentation.data import build_datasets
    cfg = smoke_cfg("segmentation")
    make_demo_data("segmentation", cfg)
    next((tmp_path / "data" / "masks").glob("*.png")).unlink()
    with pytest.raises(FileNotFoundError, match="no mask"):
        build_datasets(cfg)


# ----------------------------------------------------------- image-level tasks
def test_labels_csv_errors_are_specific(tmp_path):
    from cvkit.data import read_image_table
    (tmp_path / "images").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "images" / "a.png")
    (tmp_path / "labels.csv").write_text("filename,x\na.png,1\nmissing.png,0\n")
    with pytest.raises(FileNotFoundError, match="missing"):
        read_image_table(tmp_path)
    (tmp_path / "labels.csv").write_text("filename,x\na.png,notanumber\n")
    with pytest.raises(ValueError, match="non-numeric"):
        read_image_table(tmp_path)
    (tmp_path / "labels.csv").write_text("filename,x\na.png,1,2\n")
    with pytest.raises(ValueError, match="columns"):
        read_image_table(tmp_path)


def test_multilabel_metrics_known_values():
    from tasks.multilabel.metrics import multilabel_metrics
    probs = np.array([[0.9, 0.1], [0.8, 0.7], [0.2, 0.9]])
    true = np.array([[1, 0], [1, 1], [0, 1]], dtype=float)
    m = multilabel_metrics(probs, true, 0.5)
    assert m["exact"] == 1.0 and m["f1_macro"] == 1.0 and m["map"] == 1.0
    m2 = multilabel_metrics(probs, 1 - true, 0.5)
    assert m2["exact"] == 0.0 and m2["f1_macro"] == 0.0


def test_regression_metrics_known_values():
    from tasks.regression.metrics import regression_metrics
    true = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(true + 0.5, true)
    assert m["mae"] == pytest.approx(0.5) and m["rmse"] == pytest.approx(0.5)
    assert regression_metrics(np.full(4, true.mean()), true)["r2"] == pytest.approx(0.0)
