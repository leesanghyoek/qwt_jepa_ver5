"""One-level dual-tree quaternion wavelet transform for RGB.

Each RGB channel is analysed by four separable trees. Along each axis the signal
passes through one of two filter banks, ``A`` or ``B``, and the four combinations
are packed as the quaternion components ``(real, i, j, k)``:

    q = f_AA + i * f_BA + j * f_AB + k * f_BB

which is the standard dual-tree construction: ``i`` carries the Hilbert transform
along x, ``j`` along y, and ``k`` along both. Packing is
RGB x 4 bands x 4 components = 48 real channels.

With ``image_input="luminance"`` the transform first collapses RGB to luminance Y
(BT.601, the weights of qjepa/models/color_edge.py) and analyses that one channel:
16 real channels, edges without colour. Synthesis then returns Y, not RGB -- the
colour is gone from the coefficients by construction, so whoever needs it must
keep the frame itself.

What decides whether this is a quaternion wavelet transform at all is whether
tree B's wavelet is the Hilbert transform of tree A's. If it is, the quaternion
modulus is nearly shift invariant and the three phases encode sub-pixel
displacement -- the property that makes the transform worth its four-fold cost,
and the one that links it to an IMU, which measures exactly that displacement.

Two backends are available and both are measured, never assumed:

``qwt_dualtree_db4``
    The original: one db4 filter used in both trees, offset by one sample. This
    is *not* a Hilbert pair. Two trees running the same filter at an integer
    offset satisfy ``W_B(w) = W_A(w) e^{-jwd}``, so the analytic residual is fixed
    by the shift alone and is identical for every orthonormal wavelet -- measured
    at 0.1804 negative-frequency energy for db2 through db20, every symlet and
    every coiflet tried. It is a structural ceiling, not a tuning problem. Kept
    selectable so the new transform has a control arm to be compared against.

``qwt_dualtree_hilbert``
    A filter pair designed for this transform by direct minimisation of
    negative-frequency energy over orthonormal filter banks; see
    ``tools/design_hilbert_pair.py`` for the design and
    ``tests/test_qwt_analyticity.py`` for the measurement.

Both backends are orthonormal per tree under periodic extension, so each tree
inverts exactly and synthesis averages the four reconstructions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layout import TransformLayout, fp32_transform

BAND_ORDER = ("approx", "detail_y", "detail_x", "detail_xy")
COMPONENT_ORDER = ("real", "i", "j", "k")
CHANNEL_ORDER = ("R", "G", "B")
IMAGE_INPUTS = ("rgb", "luminance")
# BT.601, cung trong so voi qjepa/models/color_edge.luminance.
LUMINANCE_WEIGHTS = (0.299, 0.587, 0.114)

DB4_H0 = (
    0.23037781330885523,
    0.7148465705525415,
    0.6308807679295904,
    -0.02798376941698385,
    -0.18703481171888114,
    0.030841381835986965,
    0.032883011666982945,
    -0.010597401784997278,
)

# Designed Hilbert pair, 14 taps, measured negative-frequency energy 0.0677 --
# 2.68x below the 0.1814 ceiling that any integer-shifted single filter is stuck
# at. Regenerate with `python3 tools/design_hilbert_pair.py --orders 4 6 7 8`.
#
# Tree B came out of the search as the exact time-reversal of tree A, which is the
# q-shift / Farras structure arrived at independently: reversing a spectral factor
# negates its phase response, and that is what supplies the half-sample delay an
# integer shift cannot. Both trees are orthonormal, so each inverts on its own.
HILBERT_TREE_A = (
    0.017376403426887545,
    0.056826755006918264,
    0.05875724493357679,
    0.13753686415006175,
    0.5098526634073375,
    0.7315488782878448,
    0.25879152712708775,
    -0.2711118340076547,
    -0.17919011734188073,
    0.06347856039627707,
    0.04670174870514969,
    -0.012757197858304687,
    -0.005182689071610881,
    0.001584755211404701,
)
HILBERT_TREE_B = tuple(reversed(HILBERT_TREE_A))
HILBERT_OFFSETS = (0, 2)

# backend -> (tree A lowpass, tree B lowpass, (offset A, offset B), revision)
FILTER_BANKS: dict[str, tuple[tuple[float, ...], tuple[float, ...], tuple[int, int], str]] = {
    "qwt_dualtree_db4": (DB4_H0, DB4_H0, (0, 1), "1.0.0"),
    "qwt_dualtree_hilbert": (HILBERT_TREE_A, HILBERT_TREE_B, HILBERT_OFFSETS, "2.0.0"),
}
DEFAULT_BACKEND = "qwt_dualtree_hilbert"
QWT_BACKENDS = tuple(FILTER_BANKS)

# Retained so existing imports keep resolving; the active backend is per-instance.
BACKEND = DEFAULT_BACKEND
REVISION = FILTER_BANKS[DEFAULT_BACKEND][3]


def _qmf(h0: torch.Tensor) -> torch.Tensor:
    sign = torch.tensor([(-1.0) ** n for n in range(h0.numel())], dtype=h0.dtype)
    return sign * h0.flip(0)


def _analysis_1d(x: torch.Tensor, filt: torch.Tensor, offset: int) -> torch.Tensor:
    n, taps = x.shape[-1], filt.numel()
    shape = x.shape
    flat = x.reshape(-1, 1, n)
    offset = offset % n
    needed = offset + taps - 1
    repetitions = math.ceil(needed / n) + 2
    periodic = flat.repeat(1, 1, repetitions)[..., : n + needed]
    if offset:
        periodic = periodic[..., offset:]
    out = F.conv1d(periodic, filt.view(1, 1, taps), stride=2)
    return out.reshape(*shape[:-1], n // 2)


def _synthesis_1d(x: torch.Tensor, filt: torch.Tensor, offset: int, n: int) -> torch.Tensor:
    shape = x.shape
    offset = offset % n
    wide = F.conv_transpose1d(
        x.reshape(-1, 1, shape[-1]), filt.view(1, 1, -1), stride=2
    )
    out = wide.new_zeros(wide.shape[0], 1, n)
    for start in range(0, wide.shape[-1], n):
        chunk = wide[..., start : start + n]
        index = (torch.arange(chunk.shape[-1], device=x.device) + start + offset) % n
        out.index_add_(-1, index, chunk)
    return out.reshape(*shape[:-1], n)


def _analysis_axis(x: torch.Tensor, h0: torch.Tensor, h1: torch.Tensor, offset: int, axis: int):
    if axis == -2:
        x = x.transpose(-1, -2)
    low, high = _analysis_1d(x, h0, offset), _analysis_1d(x, h1, offset)
    if axis == -2:
        low, high = low.transpose(-1, -2), high.transpose(-1, -2)
    return low, high


def _synthesis_axis(
    low: torch.Tensor,
    high: torch.Tensor,
    h0: torch.Tensor,
    h1: torch.Tensor,
    offset: int,
    size: int,
    axis: int,
) -> torch.Tensor:
    if axis == -2:
        low, high = low.transpose(-1, -2), high.transpose(-1, -2)
    out = _synthesis_1d(low, h0, offset, size) + _synthesis_1d(high, h1, offset, size)
    return out.transpose(-1, -2) if axis == -2 else out


class QuaternionWaveletTransform2D(nn.Module):
    def __init__(self, levels: int = 1, backend: str = DEFAULT_BACKEND, image_input: str = "rgb") -> None:
        super().__init__()
        if levels != 1:
            raise ValueError("Only one QWT level is supported")
        if image_input not in IMAGE_INPUTS:
            raise ValueError(f"image_input must be one of {IMAGE_INPUTS}, got {image_input!r}")
        self.image_input = image_input
        self.input_channels = 3 if image_input == "rgb" else 1
        if backend not in FILTER_BANKS:
            raise ValueError(f"Unknown QWT backend {backend!r}; have {sorted(FILTER_BANKS)}")
        tree_a, tree_b, offsets, revision = FILTER_BANKS[backend]
        if len(tree_a) != len(tree_b):
            raise ValueError("Both trees must use filters of the same length")
        # Keep canonical taps in float64 so numerical reference tests do not
        # inherit float32 rounding. They are cast to the input dtype per call.
        for name, taps in (("a", tree_a), ("b", tree_b)):
            h0 = torch.tensor(taps, dtype=torch.float64)
            self.register_buffer(f"h0_{name}", h0)
            self.register_buffer(f"h1_{name}", _qmf(h0))
        self.backend = backend
        self.revision = revision
        self.offsets = tuple(int(value) for value in offsets)
        self.levels = levels

    @property
    def coeff_channels(self) -> int:
        return 16 * self.input_channels

    def prepare(self, x: torch.Tensor) -> torch.Tensor:
        """The signal this transform analyses: the RGB frame itself, or its luminance.

        In luminance mode a one-channel input is taken as luminance already, so
        ``analysis(synthesis(c))`` stays well defined.
        """
        if self.image_input == "rgb" or (x.ndim == 4 and x.shape[1] == 1):
            return x
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected [B,3,H,W] or [B,1,H,W], got {tuple(x.shape)}")
        r, g, b = LUMINANCE_WEIGHTS
        return x[:, 0:1] * r + x[:, 1:2] * g + x[:, 2:3] * b

    def _trees(self, dtype: torch.dtype):
        """(h0, h1, offset) for tree A then tree B, cast to the working dtype."""
        return (
            (self.h0_a.to(dtype), self.h1_a.to(dtype), self.offsets[0]),
            (self.h0_b.to(dtype), self.h1_b.to(dtype), self.offsets[1]),
        )

    def layout_for(self, shape: tuple[int, int, int, int]) -> TransformLayout:
        """Layout of the PREPARED signal: [B,3,H,W] for RGB, [B,1,H,W] for luminance."""
        b, c, h, w = shape
        if c != self.input_channels or h % 2 or w % 2:
            raise ValueError(f"Expected [B,{self.input_channels},even H,even W], got {shape}")
        return TransformLayout(
            backend=self.backend,
            revision=self.revision,
            original_shape=shape,
            coefficient_shape=(b, 16 * c, h // 2, w // 2),
            levels=1,
            boundary_mode="periodic",
            scale_convention="orthonormal_mean_of_four_trees",
            band_order=BAND_ORDER,
            component_order=COMPONENT_ORDER,
            channel_order=CHANNEL_ORDER if c == 3 else ("Y",),
            extra={"tree_offsets": list(self.offsets), "taps": int(self.h0_a.numel())},
        )

    @fp32_transform
    def analysis(self, x: torch.Tensor) -> tuple[torch.Tensor, TransformLayout]:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,3,H,W], got {tuple(x.shape)}")
        x = self.prepare(x)
        layout = self.layout_for(tuple(x.shape))
        b, c, h, w = x.shape
        trees = self._trees(x.dtype)
        out = x.new_empty(b, c, 4, 4, h // 2, w // 2)
        for tree_x, (h0x, h1x, offset_x) in enumerate(trees):
            low_x, high_x = _analysis_axis(x, h0x, h1x, offset_x, -1)
            for tree_y, (h0y, h1y, offset_y) in enumerate(trees):
                ll, lh = _analysis_axis(low_x, h0y, h1y, offset_y, -2)
                hl, hh = _analysis_axis(high_x, h0y, h1y, offset_y, -2)
                component = tree_x + 2 * tree_y
                out[:, :, 0, component] = ll
                out[:, :, 1, component] = lh
                out[:, :, 2, component] = hl
                out[:, :, 3, component] = hh
        return out.reshape(b, 16 * c, h // 2, w // 2), layout

    @fp32_transform
    def synthesis(self, coeff: torch.Tensor, layout: TransformLayout) -> torch.Tensor:
        layout.require(self.backend, self.revision)
        b, c, h, w = layout.original_shape
        if c != self.input_channels or tuple(coeff.shape) != (b, 16 * c, h // 2, w // 2):
            raise ValueError(f"Coefficient shape {tuple(coeff.shape)} does not match layout")
        trees = self._trees(coeff.dtype)
        packed = coeff.reshape(b, c, 4, 4, h // 2, w // 2)
        result = torch.zeros((b, c, h, w), dtype=coeff.dtype, device=coeff.device)
        for tree_x, (h0x, h1x, offset_x) in enumerate(trees):
            for tree_y, (h0y, h1y, offset_y) in enumerate(trees):
                component = tree_x + 2 * tree_y
                ll = packed[:, :, 0, component]
                lh = packed[:, :, 1, component]
                hl = packed[:, :, 2, component]
                hh = packed[:, :, 3, component]
                low_x = _synthesis_axis(ll, lh, h0y, h1y, offset_y, h, -2)
                high_x = _synthesis_axis(hl, hh, h0y, h1y, offset_y, h, -2)
                result.add_(_synthesis_axis(low_x, high_x, h0x, h1x, offset_x, w, -1))
        return result / 4.0


def quaternion_modulus(coefficients: torch.Tensor) -> torch.Tensor:
    """Quaternion modulus per colour and band: ``sqrt(sum of the four components)``.

    Input ``[B,48,H,W]`` packed as ``[colour, band, component]``; output
    ``[B,3,4,H,W]``. For a genuine Hilbert pair this is the nearly shift-invariant
    edge strength, which is what makes it usable as a restoration target.
    """
    b, channels, h, w = coefficients.shape
    if channels % 16:
        raise ValueError(f"Expected 16 bands*components per colour, got {channels}")
    packed = coefficients.reshape(b, channels // 16, 4, 4, h, w)
    return packed.square().sum(dim=3).sqrt()


def quaternion_phases(coefficients: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """The two shift-encoding phases ``(phi_x, phi_y)`` per colour and band.

    ``phi_x`` comes from the (real, i) pair and advances with horizontal
    displacement; ``phi_y`` from (real, j) and advances with vertical
    displacement. Output ``[B,3,4,2,H,W]``.
    """
    b, channels, h, w = coefficients.shape
    if channels % 16:
        raise ValueError(f"Expected 16 bands*components per colour, got {channels}")
    packed = coefficients.reshape(b, channels // 16, 4, 4, h, w)
    real = packed[:, :, :, 0]
    phi_x = torch.atan2(packed[:, :, :, 1], real + eps)
    phi_y = torch.atan2(packed[:, :, :, 2], real + eps)
    return torch.stack((phi_x, phi_y), dim=3)
