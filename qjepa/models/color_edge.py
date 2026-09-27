"""Split an RGB image into colour and edges, and put it back together.

Luminance Y (BT.601) carries nearly all of an image's edges; chroma (Cb, Cr) is
its colour. The split decoder restores the two apart: colour at reduced
resolution, where a mean loss is exactly what stable colour needs, and the
luminance edges at full resolution.

Three pieces:

* ``base``         -- the image the colour branch can represent: RGB averaged
                      down by ``color_scale`` and back up. Its chroma is the
                      output's chroma.
* ``illumination`` -- the base's luminance averaged down by ``illumination_scale``
                      and back up: brightness at periods of that many pixels and
                      longer, no edges.
* ``detail``       -- everything else in luminance: every edge, at full resolution.

``compose`` adds ``illumination + detail - Y(base)`` to all three RGB channels.
The luminance weights sum to 1, so adding the same value to R, G and B shifts Y by
that value and leaves Cb and Cr untouched: the edge branch writes luminance and
cannot disturb colour.

Measured on 48 clean TartanAir frames with perfect edges, colour at 128x128
(``color_scale=2``) caps the recombined image at 34.9 dB, 64x64 at 31.9 dB and
32x32 at 30.0 dB; 128x128 is the chroma resolution JPEG 4:2:0 uses.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def luminance(rgb: torch.Tensor) -> torch.Tensor:
    return rgb[:, 0:1] * 0.299 + rgb[:, 1:2] * 0.587 + rgb[:, 2:3] * 0.114


def chroma(rgb: torch.Tensor) -> torch.Tensor:
    y = luminance(rgb)
    return torch.cat(((rgb[:, 2:3] - y) * 0.564, (rgb[:, 0:1] - y) * 0.713), dim=1)


def downsample(x: torch.Tensor, scale: int) -> torch.Tensor:
    return F.avg_pool2d(x, scale)


def upsample(x: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
    return F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False)


def color_base(rgb: torch.Tensor, scale: int) -> torch.Tensor:
    """What the colour branch can represent of ``rgb``."""
    return upsample(downsample(rgb, scale), rgb.shape[-2:])


def illumination(rgb: torch.Tensor, scale: int) -> torch.Tensor:
    y = luminance(rgb)
    return upsample(downsample(y, scale), y.shape[-2:])


def compose(base: torch.Tensor, light: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
    """Colour from ``base``, luminance = ``light + detail``."""
    return base + (light + detail - luminance(base))


def split_targets(
    clean: torch.Tensor, color_scale: int, illumination_scale: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three pieces of ``clean``; ``compose`` of them has Y exactly ``Y(clean)``."""
    base = color_base(clean, color_scale)
    light = illumination(base, illumination_scale)
    return base, light, luminance(clean) - light


def luminance_gradient_l1(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between horizontal and vertical luminance differences -- edge slopes."""
    a, b = luminance(predicted), luminance(target)
    return (F.l1_loss(a.diff(dim=-1), b.diff(dim=-1))
            + F.l1_loss(a.diff(dim=-2), b.diff(dim=-2)))


def color_error(predicted: torch.Tensor, target: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """Mean chroma difference after averaging ``scale`` x ``scale`` blocks.

    Averaging first keeps edge placement out of it: this is the colour cast and
    colour noise a viewer sees, not sharpness.
    """
    return (chroma(downsample(predicted, scale)) - chroma(downsample(target, scale))).abs().mean()


def excess_roughness(predicted: torch.Tensor, reference: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
    """Gradient the prediction has BEYOND the reference, weighted towards flat areas.

    mean over both axes of relu(|grad pred| - |grad ref|) * exp(-|grad ref| / sigma).
    Zero for the reference itself and for anything as smooth as it; grain, ringing
    around edges and blur spilling into flat areas all add to it; real texture the
    reference has is not penalised. A plain TV term would be, and on dark inputs --
    whose gradients are small anyway -- it cannot tell the input from the clean
    frame (measured on noise B: 0.0021 vs 0.0019). This one reads 0 for clean,
    0.0070 for 3/255 grain, 0.0025 for ringing, 0.0014 for a 3x3 blur.
    """
    total = predicted.new_zeros(())
    for dim in (-1, -2):
        grad_pred = predicted.diff(dim=dim).abs()
        grad_ref = reference.diff(dim=dim).abs()
        total = total + (F.relu(grad_pred - grad_ref) * torch.exp(-grad_ref / sigma)).mean()
    return total / 2
