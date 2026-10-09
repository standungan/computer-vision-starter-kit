"""Keypoint model: predict one heatmap per keypoint ("SimpleBaseline").

Instead of regressing coordinates directly, the network outputs K small images, one per keypoint,
that glow where that keypoint probably is. The position of the brightest pixel is the answer.
Heatmaps keep the spatial structure that a plain "x, y" output layer throws away, and they work
much better.

    image (3,S,S) -> ResNet encoder (stride 32) -> 3 x [upsample x2 + conv] -> (K, S/4, S/4)

`decode` turns heatmaps back into coordinates, refining the peak to sub-pixel precision with a
a parabola fit over its neighbours.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from cvkit.backbones import resnet_stages

STRIDE = 4          # heatmap pixel = 4 input pixels


class HeatmapNet(nn.Module):
    def __init__(self, num_keypoints: int, arch: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.encoder, widths = resnet_stages(arch, pretrained)
        layers, cin = [], widths[-1]
        for _ in range(3):                                     # /32 -> /16 -> /8 -> /4
            layers += [nn.ConvTranspose2d(cin, 256, 4, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
            cin = 256
        self.deconv = nn.Sequential(*layers)
        self.head = nn.Conv2d(256, num_keypoints, 1)

    def forward(self, x):
        for stage in self.encoder:
            x = stage(x)
        return self.head(self.deconv(x))


def build_model(cfg: dict, meta: dict) -> nn.Module:
    m = cfg["model"]
    return HeatmapNet(len(meta["keypoint_names"]), m.get("arch", "resnet18"), m.get("pretrained", True))


def to_probs(scores: torch.Tensor) -> torch.Tensor:
    """(B,K,h,w) raw scores -> per-keypoint probability maps that each sum to 1."""
    B, K, h, w = scores.shape
    return scores.float().flatten(2).softmax(-1).view(B, K, h, w)


def make_heatmaps(points: torch.Tensor, visible: torch.Tensor, size: int, sigma: float) -> torch.Tensor:
    """Gaussian blobs: (B,K,2) pixel coords -> (B,K,size/STRIDE,size/STRIDE). Hidden points stay all-zero."""
    hm = size // STRIDE
    idx = torch.arange(hm, dtype=torch.float32, device=points.device)
    cx = (points[..., 0] / STRIDE - 0.5)[..., None]           # position in heatmap-pixel units
    cy = (points[..., 1] / STRIDE - 0.5)[..., None]
    gx = torch.exp(-((idx - cx) ** 2) / (2 * sigma ** 2))     # (B,K,hm)
    gy = torch.exp(-((idx - cy) ** 2) / (2 * sigma ** 2))
    return gy[..., :, None] * gx[..., None, :] * visible[..., None, None]


def decode(heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,K,h,w) probability maps (or Gaussian heatmaps) -> ((B,K,2) pixel coords in the
    model-input frame, (B,K) confidence).

    The brightest pixel only gives a position to the nearest 4 input pixels. Because a Gaussian's
    logarithm is a parabola, fitting a parabola through the log-values of the peak and its
    left/right (and up/down) neighbours recovers the true centre to a fraction of a pixel.
    Confidence is the share of the map's total mass inside the 3x3 window around the peak: close
    to 1 for a sharp, certain peak; small when the map is smeared out or has several candidates.
    """
    B, K, h, w = heatmaps.shape
    _, flat = heatmaps.flatten(2).max(2)
    iy, ix = flat // w, flat % w
    eps = 1e-6
    pad = nn.functional.pad(heatmaps.float().clamp(min=eps), (1, 1, 1, 1), value=eps).flatten(2)

    def raw(dy: int, dx: int) -> torch.Tensor:
        return torch.gather(pad, 2, ((iy + 1 + dy) * (w + 2) + (ix + 1 + dx))[..., None])[..., 0]

    def refine(minus: torch.Tensor, centre: torch.Tensor, plus: torch.Tensor) -> torch.Tensor:
        minus, centre, plus = minus.log(), centre.log(), plus.log()
        curvature = minus - 2 * centre + plus                       # < 0 at a real peak
        offset = (minus - plus) / (2 * curvature.clamp(max=-1e-6))
        return offset.clamp(-0.5, 0.5)

    c = raw(0, 0)
    fx = ix.float() + refine(raw(0, -1), c, raw(0, 1))
    fy = iy.float() + refine(raw(-1, 0), c, raw(1, 0))
    window = sum(raw(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    confidence = (window / pad.sum(2).clamp(min=eps)).clamp(max=1.0)
    return torch.stack([(fx + 0.5) * STRIDE, (fy + 0.5) * STRIDE], -1), confidence
