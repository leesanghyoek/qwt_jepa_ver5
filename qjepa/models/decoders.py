"""Fresh phase-2 decoders driven by ZI and ZU, optionally correcting the input."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Stage, Upsample, initialize_trainable, resize
from .color_edge import color_base, compose, downsample, illumination, luminance, upsample

IMAGE_DECODERS = ("qwt_coefficients", "resnet_pixel", "split_color_edge")
# Decoders that produce pixels; RestorationSystem.decode drives them.
PIXEL_IMAGE_DECODERS = ("resnet_pixel", "split_color_edge")


class SkipMerge(nn.Module):
    """Noi feature encoder vao duong decoder.

    Skip mang duong net cua anh NHIEU: sac net nhung khong dang tin, vi trong do
    co ca canh that lan hat nhieu. Latent moi la thu biet canh nao la that — no
    duoc huan luyen de doan latent cua anh SACH, cong voi neo ep no giu he so sach
    o bang chi tiet. Nen phan cong dung la: skip cap do phan giai, latent quyet
    dinh giu cai gi.

    `gated=False` KHONG lam duoc viec do. Mot conv pointwise tren tensor noi chi
    hoc duoc mot TI LE PHA TRON co dinh theo kenh — sau khi train, kenh nao lay
    bao nhieu skip la co dinh o moi vi tri, moi anh. No khong the nhin latent de
    quyet dinh "cho nay canh that, cho qua; cho kia la hat nhieu, chan lai".

    `gated=True` sinh cong TU DUONG LATENT, nen cong phu thuoc tung vi tri va
    tung kenh. Dung mau va quy uoc cua SharedGatedFusion: bias -2.0 cho
    sigmoid(-2) ~ 0.12, tuc luc bat dau cong gan nhu dong va mo dan theo muc skip
    to ra huu ich.

    Ca hai che do deu bat dau o identity tren duong decoder, nen san identity cua
    residual (head zero-init) con nguyen o update 0.

    Danh doi phai noi ro: net di qua duong nay den tu ANH DAU VAO, khong phai tu
    latent. Do la ly do `encoder_skips` phai bat tuong minh, va la ly do
    delta_report.py co --ablate-latent.
    """

    def __init__(
        self, channels: int, skip_channels: int, *, dim: int,
        gated: bool = True, gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        conv = nn.Conv1d if dim == 1 else nn.Conv2d
        self.dim = dim
        self.gated = gated
        if not gated:
            self.project = conv(channels + skip_channels, channels, 1)
            nn.init.zeros_(self.project.bias)
            with torch.no_grad():
                self.project.weight.zero_()
                for index in range(channels):
                    self.project.weight[index, index] = 1.0
            return
        self.gate = conv(channels, channels, 1)
        self.skip_project = conv(skip_channels, channels, 1)
        # Cong khoi tao tu bias thuan tuy: trong so 0 nen luc dau cong khong phu
        # thuoc noi dung, va skip_project zero-init nen dong gop ban dau dung bang 0.
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)
        nn.init.zeros_(self.skip_project.weight)
        nn.init.zeros_(self.skip_project.bias)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if skip.shape[2:] != x.shape[2:]:
            skip = resize(skip, tuple(x.shape[2:]), dim=self.dim)
        if not self.gated:
            return self.project(torch.cat((x, skip), dim=1))
        return x + torch.sigmoid(self.gate(x)) * self.skip_project(skip)


class LatentCoefficientDecoder(nn.Module):
    def __init__(
        self,
        output_channels: int,
        output_size: tuple[int, ...],
        channels: tuple[int, int, int, int] = (32, 64, 96, 128),
        *,
        dim: int,
        groups: int = 8,
        residual: bool = False,
        skip_channels: tuple[int, ...] | None = None,
        skip_gating: bool = True,
        sees_input: bool = True,
    ) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels
        self.dim = dim
        self.residual = residual
        self.sees_input = bool(residual and sees_input)
        self.uses_skips = skip_channels is not None
        self.output_size = output_size
        sizes = []
        for divisor in (4, 2, 1):
            sizes.append(tuple(max(1, (value + divisor - 1) // divisor) for value in output_size))
        self.sizes = tuple(sizes)
        self.shuffle2 = Upsample(c3, dim=dim)
        self.up2 = Stage(c3, c2, dim=dim, groups=groups)
        self.shuffle1 = Upsample(c2, dim=dim)
        self.up1 = Stage(c2, c1, dim=dim, groups=groups)
        self.shuffle0 = Upsample(c1, dim=dim)
        self.up0 = Stage(c1, c0, dim=dim, groups=groups)
        conv = nn.Conv1d if dim == 1 else nn.Conv2d
        # Voi sees_input, head doc CA duong latent lan he so dau vao. Khong co no,
        # delta = f(Z) va decoder khong the bieu dien mot phep khu nhieu: muon tru
        # bot phan nhieu thi phai DOC duoc no, ma latent duoc huan luyen de doan
        # latent cua tin hieu SACH, tuc duoc day de vut bo hien thuc cua nhieu.
        # Voi he so wavelet, co bien phep co gian — delta = -alpha * C_in tren nua
        # bang detail — nam trong tam mot conv duy nhat.
        head_in = c0 + output_channels if self.sees_input else c0
        self.head = conv(head_in, output_channels, 3, padding=1)
        initialize_trainable(self)
        # Sau initialize_trainable, vi SkipMerge tu dat trong so identity+zero cua
        # no va khong duoc Kaiming ghi de.
        if skip_channels is not None:
            if len(skip_channels) != 3:
                raise ValueError(f"Expected three skip levels, got {len(skip_channels)}")
            self.merge2 = SkipMerge(c2, skip_channels[0], dim=dim, gated=skip_gating)
            self.merge1 = SkipMerge(c1, skip_channels[1], dim=dim, gated=skip_gating)
            self.merge0 = SkipMerge(c0, skip_channels[2], dim=dim, gated=skip_gating)
        if residual:
            # Bat dau o dung identity: update 0 tra lai chinh he so dau vao, nen
            # model khong the te hon input va moi buoc chi co the di len.
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(
        self,
        latent: torch.Tensor,
        base: torch.Tensor | None = None,
        skips: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        if self.uses_skips and skips is None:
            raise ValueError("Decoder was built with skips; pass the encoder stages")
        if skips is not None and not self.uses_skips:
            raise ValueError("Decoder was built without skips; refusing to use them")
        x = self.up2(self.shuffle2(latent))
        if skips is not None:
            x = self.merge2(x, skips[0])
        x = self.up1(self.shuffle1(x))
        if skips is not None:
            x = self.merge1(x, skips[1])
        x = self.up0(self.shuffle0(x))
        if skips is not None:
            x = self.merge0(x, skips[2])
        if tuple(x.shape[2:]) != tuple(self.output_size):
            # Luoi khong chia het cho 8; chi con lai phan le sau ba lan nhan doi.
            x = resize(x, self.output_size, dim=self.dim)
        if not self.residual:
            return self.head(x)
        if base is None:
            raise ValueError("Residual decoding needs the input coefficients as base")
        predicted = self.head(torch.cat((x, base), dim=1) if self.sees_input else x)
        return base + predicted


class _ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.first = nn.Conv2d(width, width, 3, padding=1)
        self.second = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.second(F.relu(self.first(x)))


class PixelResNetDecoder(nn.Module):
    """Restore the image in PIXELS: blurry input + JEPA latent -> residual.

    The coefficient decoder reaches the image only through 48 QWT channels built
    up from a 16x16 latent, and three variants of it (latent only, no skips,
    full-resolution skips) all stopped at the same detail floor. This one works
    on the blurry image itself at full resolution: a head at 256x256, a residual
    trunk at 128x128 into which the upsampled latent is fused, pixel-shuffle
    back to 256x256, a U-Net skip from the head, and a residual added to the
    BLURRY INPUT. The latent says what the clean scene should be; the pixels say
    where its edges are.

    Local A/B, same phase 1 and loss, 600 updates: clean edge content reproduced
    in place (4-16 px periods) 0.309 with the coefficient decoder, 0.359 with
    this one; edge-band error 0.520 -> 0.465.

    The last conv is zero-initialised, so update 0 returns the input exactly --
    the same identity floor as the coefficient residual.
    """

    def __init__(
        self, latent_channels: int = 128, width: int = 64, blocks: int = 8,
        in_channels: int = 3, out_channels: int = 3,
    ) -> None:
        super().__init__()
        # Layer names are part of every saved checkpoint: keep them as they are.
        self.head = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU())
        self.down = nn.Sequential(nn.Conv2d(32, width, 4, stride=2, padding=1), nn.ReLU())
        self.latent = nn.Conv2d(latent_channels, width, 1)
        self.fuse = nn.Conv2d(2 * width, width, 3, padding=1)
        self.trunk = nn.Sequential(*[_ResidualBlock(width) for _ in range(blocks)])
        self.up = nn.Sequential(nn.Conv2d(width, 32 * 4, 3, padding=1), nn.PixelShuffle(2), nn.ReLU())
        self.tail = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, out_channels, 3, padding=1))
        nn.init.zeros_(self.tail[-1].weight)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(
        self, latent: torch.Tensor, image: torch.Tensor, base: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``image + delta``; with ``base`` given, ``base + delta`` instead."""
        if image.shape[-2] % 2 or image.shape[-1] % 2:
            raise ValueError(f"Pixel decoder needs even image sides, got {tuple(image.shape[-2:])}")
        head = self.head(image)
        x = self.down(head)
        z = F.interpolate(self.latent(latent), size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.trunk(self.fuse(torch.cat((x, z), dim=1)))
        return (image if base is None else base) + self.tail(torch.cat((self.up(x), head), dim=1))


class GlobalToneColor(nn.Module):
    """One tone curve and one colour matrix per image, set from the whole frame.

    The low-light corruption is global: a gain per channel (white balance), an
    exposure gain and a tone gamma, the same everywhere in the frame. The gamma
    alone washes colours out -- x^0.6 pulls the channels of a pixel towards each
    other. Undoing it needs the whole frame: the colour trunk sees about 66 px,
    too little to tell a dim red wall from a white one under a red cast, and
    where it cannot tell, L1 answers with the grey in between. Here a small
    funnel and the latent are pooled into one vector per image, which sets
    ``out = M . x^p + b``: ``p`` undoes the gamma, ``M`` the channel gains.
    Zero-initialised: it starts as the identity.
    """

    def __init__(self, latent_channels: int = 128, width: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.latent = nn.Conv2d(latent_channels, width, 1)
        # + per-channel mean and std of the input: the statistics a gain and a gamma move.
        self.head = nn.Sequential(nn.Linear(2 * width + 6, width), nn.ReLU(), nn.Linear(width, 13))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def parameters_for(self, latent: torch.Tensor, image: torch.Tensor):
        """(exponent [B], matrix [B,3,3], bias [B,3]) for each image."""
        pooled = torch.cat((
            self.features(image).mean(dim=(-2, -1)),
            F.relu(self.latent(latent)).mean(dim=(-2, -1)),
            image.mean(dim=(-2, -1)), image.flatten(2).std(dim=-1),
        ), dim=1)
        raw = self.head(pooled)
        exponent = torch.exp(raw[:, 0].clamp(-1.0, 1.0))        # 0.37 .. 2.7; gamma 0.46..0.79 needs 1.3..2.2
        matrix = torch.eye(3, dtype=raw.dtype, device=raw.device) + raw[:, 1:10].view(-1, 3, 3)
        return exponent, matrix, raw[:, 10:13]

    def forward(self, latent: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        exponent, matrix, bias = self.parameters_for(latent, image)
        curved = image.clamp_min(1e-4).pow(exponent[:, None, None, None])
        return torch.einsum("bij,bjhw->bihw", matrix, curved) + bias[:, :, None, None]


class ColorBranch(nn.Module):
    """Colour and brightness at reduced resolution: a residual on the averaged input.

    Trained with L1 and kept away from edges. Working at reduced resolution also
    averages away most of the colour noise a dark frame carries. Zero-initialised
    tail: it starts as the averaged input. L1's answer is the median colour, which
    is the right one only where the hue is certain; where it is not, the median
    leans to grey -- p8's restored colours were washed out. ``global_tone`` puts a
    GlobalToneColor in front, which removes the part of that uncertainty that is
    one global setting per image.
    """

    def __init__(self, latent_channels: int = 128, width: int = 32, blocks: int = 6,
                 global_tone: bool = False) -> None:
        super().__init__()
        # Only when asked: p8 checkpoints have no such layers.
        self.global_tone = GlobalToneColor(latent_channels, width) if global_tone else None
        self.head = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.ReLU())
        self.latent = nn.Conv2d(latent_channels, width, 1)
        self.fuse = nn.Conv2d(2 * width, width, 3, padding=1)
        self.trunk = nn.Sequential(*[_ResidualBlock(width) for _ in range(blocks)])
        self.tail = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, latent: torch.Tensor, small: torch.Tensor) -> torch.Tensor:
        if self.global_tone is not None:
            small = self.global_tone(latent, small)
        x = self.head(small)
        z = upsample(self.latent(latent), x.shape[-2:])
        return small + self.tail(self.trunk(self.fuse(torch.cat((x, z), dim=1))))


class UNetBranch(nn.Module):
    """Funnel and loudspeaker: a U-Net whose bottom sits on the latent's grid.

    The funnel halves the resolution level by level (conv 4x4 stride 2), so each
    level sees a wider area: near features at the top, deep features -- layout,
    objects, overall exposure -- at the bottom. The bottom level has the same
    grid as ZI (16x16 for a 256x256 image), so the JEPA latent joins there, where
    it belongs, instead of being stretched to the working resolution. The
    loudspeaker doubles the resolution back (sub-pixel conv) and, at every level,
    takes the funnel's features of that level through a skip, so the near
    features lost on the way down come back. Residual on ``base`` with a
    zero-initialised last conv: it starts as the identity.

    ``widths[i]`` is the channel count at level i; level 0 is the input
    resolution and each next level is half of it.
    """

    def __init__(self, in_channels: int, out_channels: int, latent_channels: int = 128,
                 widths: tuple[int, ...] = (24, 32, 64, 96, 128), blocks: int = 1) -> None:
        super().__init__()
        if len(widths) < 2:
            raise ValueError("A funnel needs at least two levels")
        self.levels = len(widths)
        self.stem = nn.Conv2d(in_channels, widths[0], 3, padding=1)
        self.down_blocks = nn.ModuleList(
            nn.Sequential(*[_ResidualBlock(w) for _ in range(blocks)]) for w in widths[:-1])
        self.downs = nn.ModuleList(
            nn.Conv2d(widths[i], widths[i + 1], 4, stride=2, padding=1) for i in range(len(widths) - 1))
        self.latent = nn.Conv2d(latent_channels, widths[-1], 1)
        self.bottom = nn.Sequential(nn.Conv2d(2 * widths[-1], widths[-1], 3, padding=1), nn.ReLU(),
                                    *[_ResidualBlock(widths[-1]) for _ in range(blocks)])
        self.ups = nn.ModuleList(
            nn.Sequential(nn.Conv2d(widths[i + 1], widths[i] * 4, 3, padding=1), nn.PixelShuffle(2))
            for i in range(len(widths) - 1))
        self.merges = nn.ModuleList(
            nn.Sequential(nn.Conv2d(2 * w, w, 3, padding=1), nn.ReLU(),
                          *[_ResidualBlock(w) for _ in range(blocks)]) for w in widths[:-1])
        self.tail = nn.Conv2d(widths[0], out_channels, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, latent: torch.Tensor, x: torch.Tensor, base: torch.Tensor | None = None) -> torch.Tensor:
        step = 2 ** (self.levels - 1)
        if x.shape[-2] % step or x.shape[-1] % step:
            raise ValueError(f"U-Net with {self.levels} levels needs sides divisible by {step}")
        h = F.relu(self.stem(x))
        skips = []
        for block, down in zip(self.down_blocks, self.downs):   # funnel
            h = block(h)
            skips.append(h)
            h = F.relu(down(h))
        z = self.latent(latent)
        if z.shape[-2:] != h.shape[-2:]:
            z = F.interpolate(z, size=h.shape[-2:], mode="bilinear", align_corners=False)
        h = self.bottom(torch.cat((h, z), dim=1))
        for up, merge, skip in zip(reversed(self.ups), reversed(self.merges), reversed(skips)):  # loudspeaker
            h = merge(torch.cat((F.relu(up(h)), skip), dim=1))
        return (x if base is None else base) + self.tail(h)


class LayerNorm2d(nn.Module):
    """LayerNorm over the channels of every pixel, as NAFNet uses it."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Statistics in fp32 even under fp16 autocast: a half-precision variance of
        # 48-160 channels loses the small differences the normalisation divides by.
        x32 = x.float()
        mean = x32.mean(dim=1, keepdim=True)
        variance = (x32 - mean).square().mean(dim=1, keepdim=True)
        normalised = (x32 - mean) / torch.sqrt(variance + self.eps)
        return (self.weight * normalised + self.bias).to(x.dtype)


class NAFBlock(nn.Module):
    """NAFNet's block (Chen et al., "Simple Baselines for Image Restoration", ECCV 2022).

    Norm -> 1x1 -> depthwise 3x3 -> SimpleGate (one half of the channels times the
    other, in place of an activation) -> simplified channel attention (a 1x1 conv
    on the image-wide mean: every channel sees the whole frame) -> 1x1, then a gated
    feed-forward the same way. Both paths are scaled by zero-initialised
    per-channel factors, so a fresh block is the identity.
    """

    def __init__(self, channels: int, expand: int = 2) -> None:
        super().__init__()
        hidden = channels * expand
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden // 2, hidden // 2, 1))
        self.conv3 = nn.Conv2d(hidden // 2, channels, 1)
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, hidden, 1)
        self.conv5 = nn.Conv2d(hidden // 2, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    @staticmethod
    def _gate(x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=1)
        return first * second

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._gate(self.conv2(self.conv1(self.norm1(x))))
        y = x + self.conv3(h * self.attention(h)) * self.beta
        return y + self.conv5(self._gate(self.conv4(self.norm2(y)))) * self.gamma


class NAFNetBranch(nn.Module):
    """NAFNet's U-Net for the edge branch, with the JEPA latent at its bottom.

    Level i works at 1/2^i of the input with ``widths[i]`` channels: the top
    levels see single edges and small objects, the bottom one (16x16 for a 256x256
    frame, ZI's grid) layout and what the objects are. Depthwise convolutions make
    blocks cheap, so the bottom can be deep (``middle_blocks``). Down: 2x2 stride-2
    conv; up: 1x1 conv + PixelShuffle; skips are added, as in NAFNet.

    ``aux_factors`` adds a head on the decoder level at 1/f resolution that
    predicts the output downsampled by f (MIMO-UNet's coarse-to-fine supervision):
    the deep levels must fix large blur themselves instead of leaving it to the top.
    All heads and the last conv are zero-initialised: the branch starts as the
    identity on ``base``.
    """

    def __init__(self, in_channels: int, out_channels: int, latent_channels: int,
                 widths: tuple[int, ...], enc_blocks: tuple[int, ...], middle_blocks: int,
                 dec_blocks: tuple[int, ...], aux_factors: tuple[int, ...] = ()) -> None:
        super().__init__()
        levels = len(widths)
        if levels < 2 or len(enc_blocks) != levels - 1 or len(dec_blocks) != levels - 1:
            raise ValueError("NAFNet needs len(enc_blocks) == len(dec_blocks) == len(widths) - 1")
        for factor in aux_factors:
            level = int(factor).bit_length() - 1
            if factor < 2 or 2 ** level != factor or level > levels - 2:
                raise ValueError(f"aux factor {factor} must be a power of two with a decoder level")
        self.levels = levels
        self.intro = nn.Conv2d(in_channels, widths[0], 3, padding=1)
        self.encoders = nn.ModuleList(
            nn.Sequential(*[NAFBlock(w) for _ in range(n)]) for w, n in zip(widths[:-1], enc_blocks))
        self.downs = nn.ModuleList(nn.Conv2d(widths[i], widths[i + 1], 2, stride=2) for i in range(levels - 1))
        self.latent = nn.Conv2d(latent_channels, widths[-1], 1)
        self.middle = nn.Sequential(*[NAFBlock(widths[-1]) for _ in range(middle_blocks)])
        self.ups = nn.ModuleList(
            nn.Sequential(nn.Conv2d(widths[i + 1], 4 * widths[i], 1, bias=False), nn.PixelShuffle(2))
            for i in range(levels - 1))
        self.decoders = nn.ModuleList(
            nn.Sequential(*[NAFBlock(w) for _ in range(n)]) for w, n in zip(widths[:-1], dec_blocks))
        self.ending = nn.Conv2d(widths[0], out_channels, 3, padding=1)
        self.aux_heads = nn.ModuleDict({
            str(f): nn.Conv2d(widths[int(f).bit_length() - 1], out_channels, 3, padding=1) for f in aux_factors})
        for head in (self.ending, *self.aux_heads.values()):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward_with_aux(
        self, latent: torch.Tensor, x: torch.Tensor, base: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        step = 2 ** (self.levels - 1)
        if x.shape[-2] % step or x.shape[-1] % step:
            raise ValueError(f"NAFNet with {self.levels} levels needs sides divisible by {step}")
        base = x if base is None else base
        h = self.intro(x)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            h = encoder(h)
            skips.append(h)
            h = down(h)
        z = self.latent(latent)
        if z.shape[-2:] != h.shape[-2:]:
            z = F.interpolate(z, size=h.shape[-2:], mode="bilinear", align_corners=False)
        h = self.middle(h + z)
        aux = {}
        for level in reversed(range(self.levels - 1)):
            h = self.decoders[level](self.ups[level](h) + skips[level])
            key = str(2 ** level)
            if key in self.aux_heads:
                aux[2 ** level] = F.avg_pool2d(base, 2 ** level) + self.aux_heads[key](h)
        return base + self.ending(h), aux

    def forward(self, latent: torch.Tensor, x: torch.Tensor, base: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward_with_aux(latent, x, base)[0]


class EdgeRefiner(nn.Module):
    """A small full-resolution CNN that sharpens and smooths the edge map.

    It runs after the edge branch, on the luminance detail alone -- colour has
    already been split off, so nothing here can move it. Input: the branch's
    detail, the blurry luminance and the illumination; no down-sampling, so every
    layer works on the 2-4 px scale where small objects live. ``blocks`` residual
    blocks of ``width`` channels, a receptive field of 4 * blocks + 5 px. The last
    conv is zero-initialised: the refiner starts as the identity on the detail.
    """

    def __init__(self, in_channels: int = 3, width: int = 32, blocks: int = 4) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(in_channels, width, 3, padding=1), nn.ReLU())
        self.trunk = nn.Sequential(*[_ResidualBlock(width) for _ in range(blocks)])
        self.tail = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, detail: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return detail + self.tail(self.trunk(self.head(torch.cat((detail, context), dim=1))))


class OvercompleteRefiner(nn.Module):
    """Loudspeaker, then funnel: enlarge the edge map, sharpen it there, shrink it back.

    The U-Net's order reversed (an overcomplete CNN, as in KiU-Net). Same place,
    inputs and output as EdgeRefiner. Loudspeaker: conv 3x3 + PixelShuffle enlarge
    the input ``scale`` times. ``blocks`` residual blocks of ``width`` channels then
    work on the enlarged grid, where a 3x3 kernel spans 3 / scale input pixels, so
    an edge can be placed and steepened between two input pixels rather than on
    one of them. Funnel: PixelUnshuffle folds each scale x scale block back into
    channels -- nothing is averaged away -- and a zero-initialised conv 3x3 at the
    input resolution learns the shrinking filter, like supersampling with a learned
    kernel instead of a box. It starts as the identity on the detail. The loss
    only sees the shrunk output: the enlarged grid is an internal working space,
    not a super-resolved image.
    """

    def __init__(self, in_channels: int = 3, width: int = 16, blocks: int = 4, scale: int = 2) -> None:
        super().__init__()
        if scale < 2:
            raise ValueError("An overcomplete refiner needs scale >= 2; scale 1 is EdgeRefiner")
        self.enlarge = nn.Sequential(nn.Conv2d(in_channels, width * scale ** 2, 3, padding=1),
                                     nn.PixelShuffle(scale), nn.ReLU())
        self.trunk = nn.Sequential(*[_ResidualBlock(width) for _ in range(blocks)])
        self.shrink = nn.PixelUnshuffle(scale)
        self.tail = nn.Conv2d(width * scale ** 2, 1, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, detail: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.trunk(self.enlarge(torch.cat((detail, context), dim=1)))
        return detail + self.tail(self.shrink(h))


class SplitColorEdgeDecoder(nn.Module):
    """Restore colour and edges apart, then put them back together.

    * Colour branch: the blurry image averaged down by ``color_scale`` plus the
      latent -> the colour base, and from it the illumination (luminance at
      periods of ``illumination_scale`` px and longer).
    * Edge branch: blurry LUMINANCE at full resolution plus the latent -> the
      luminance detail, i.e. every edge. It also sees the predicted illumination
      (detached), so it knows how bright the clean frame is and how strong its
      edges should be, without its loss steering the colour branch.
    * ``compose``: chroma from the base, luminance = illumination + detail.

    At initialisation both residuals are zero: the output has exactly the input's
    luminance and the input's colour at ``color_scale`` resolution.
    See qjepa/models/color_edge.py for why the split is exact.
    """

    def __init__(
        self, latent_channels: int = 128, *, color_width: int = 32, color_blocks: int = 6,
        edge_width: int = 64, edge_blocks: int = 6, color_scale: int = 2, illumination_scale: int = 8,
        branch_arch: str = "resnet", color_widths: tuple[int, ...] = (12, 16, 24, 32),
        edge_widths: tuple[int, ...] = (16, 24, 32, 48, 56), unet_blocks: int = 1,
        color_global: bool = False, naf: dict | None = None, refiner: dict | None = None,
    ) -> None:
        super().__init__()
        # Only when asked: earlier checkpoints have no refiner layers. Scale 1 -- p15/p16,
        # and every config without the key -- keeps EdgeRefiner and its layer names.
        self.refiner = None
        if refiner:
            width, blocks, scale = int(refiner["width"]), int(refiner["blocks"]), int(refiner.get("scale", 1))
            self.refiner = (EdgeRefiner(3, width, blocks) if scale == 1
                            else OvercompleteRefiner(3, width, blocks, scale))
        if color_global and branch_arch == "unet":
            raise ValueError("color_global works on the ResNet colour branch (resnet or unet_edge)")
        self.color_scale = int(color_scale)
        self.illumination_scale = int(illumination_scale)
        self.branch_arch = branch_arch
        if branch_arch == "unet":
            # Funnel and loudspeaker on both branches, latent at the bottom.
            self.color = UNetBranch(3, 3, latent_channels, tuple(color_widths), unet_blocks)
            self.edge = UNetBranch(2, 1, latent_channels, tuple(edge_widths), unet_blocks)
        elif branch_arch == "unet_edge":
            # Multi-scale CNN where the edges are: near features (single edges,
            # small objects) at the top levels, overall features (layout, what an
            # object is, exposure) at the bottom. Colour keeps p8's branch and names.
            self.color = ColorBranch(latent_channels, color_width, color_blocks, color_global)
            self.edge = UNetBranch(2, 1, latent_channels, tuple(edge_widths), unet_blocks)
        elif branch_arch == "nafnet_edge":
            # NAFNet on the edges (deep and cheap: depthwise blocks, channel attention),
            # p8's colour branch; ``naf`` holds widths, block counts and aux factors.
            if not naf:
                raise ValueError("nafnet_edge needs its naf settings")
            self.color = ColorBranch(latent_channels, color_width, color_blocks, color_global)
            self.edge = NAFNetBranch(2, 1, latent_channels, tuple(naf["widths"]), tuple(naf["enc_blocks"]),
                                     int(naf["middle_blocks"]), tuple(naf["dec_blocks"]),
                                     tuple(naf.get("aux_factors", ())))
        elif branch_arch == "resnet":
            # p8: layer names are part of its checkpoints; keep them.
            self.color = ColorBranch(latent_channels, color_width, color_blocks, color_global)
            self.edge = PixelResNetDecoder(latent_channels, edge_width, edge_blocks,
                                           in_channels=2, out_channels=1)
        else:
            raise ValueError("branch_arch must be resnet, unet, unet_edge or nafnet_edge")

    def forward(self, latent: torch.Tensor, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        size = image.shape[-2:]
        if size[0] % self.illumination_scale or size[1] % self.illumination_scale:
            raise ValueError(f"Image sides {tuple(size)} must divide by {self.illumination_scale}")
        base = upsample(self.color(latent, downsample(image, self.color_scale)), size)
        light = illumination(base, self.illumination_scale)
        y = luminance(image)
        detail_in = y - illumination(color_base(image, self.color_scale), self.illumination_scale)
        edge_in = torch.cat((y, light.detach()), dim=1)
        aux = {}
        if isinstance(self.edge, NAFNetBranch):
            detail, aux = self.edge.forward_with_aux(latent, edge_in, base=detail_in)
        else:
            detail = self.edge(latent, edge_in, base=detail_in)
        parts = {}
        if self.refiner is not None:
            parts["image_detail_stage1"] = detail
            detail = self.refiner(detail, edge_in)
        parts.update(image_color_base=base, image_illumination=light, image_detail=detail)
        # Flat keys: only tensors cross DataParallel's gather.
        parts.update({f"image_detail_aux{factor}": value for factor, value in aux.items()})
        return compose(base, light, detail), parts


class LatentDecoders(nn.Module):
    """Decode ZI/ZU into wavelet coefficients, optionally with input paths.

    With ``image_decoder="resnet_pixel"`` the image branch is a PixelResNetDecoder
    and produces pixels, not coefficients; RestorationSystem.decode drives it. The
    IMU branch is the coefficient decoder either way.
    """

    def __init__(
        self,
        image_coefficient_size: tuple[int, int] = (128, 128),
        imu_coefficient_length: int = 64,
        channels: tuple[int, int, int, int] = (32, 64, 96, 128),
        groups: int = 8,
        residual: bool = False,
        skip_channels: tuple[int, ...] | None = None,
        skip_gating: bool = True,
        sees_input: bool = True,
        image_decoder: str = "qwt_coefficients",
        resnet_width: int = 64,
        resnet_blocks: int = 8,
        split: dict | None = None,
    ) -> None:
        super().__init__()
        if image_decoder not in IMAGE_DECODERS:
            raise ValueError(f"image_decoder must be one of {IMAGE_DECODERS}")
        self.residual = residual
        self.uses_skips = skip_channels is not None
        self.image_decoder = image_decoder
        if image_decoder == "resnet_pixel":
            self.image = PixelResNetDecoder(channels[3], resnet_width, resnet_blocks)
        elif image_decoder == "split_color_edge":
            self.image = SplitColorEdgeDecoder(channels[3], **(split or {}))
        else:
            self.image = LatentCoefficientDecoder(
                48, image_coefficient_size, channels, dim=2, groups=groups,
                residual=residual, skip_channels=skip_channels, skip_gating=skip_gating,
                sees_input=sees_input,
            )
        self.imu = LatentCoefficientDecoder(
            12, (imu_coefficient_length,), channels, dim=1, groups=groups,
            residual=residual, skip_channels=skip_channels, skip_gating=skip_gating,
            sees_input=sees_input,
        )

    def forward(
        self,
        ZI: torch.Tensor,
        ZU: torch.Tensor,
        image_base: torch.Tensor | None = None,
        imu_base: torch.Tensor | None = None,
        image_skips: tuple[torch.Tensor, ...] | None = None,
        imu_skips: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.image_decoder in PIXEL_IMAGE_DECODERS:
            raise ValueError("A pixel image decoder is driven by RestorationSystem.decode")
        return (
            self.image(ZI, image_base, image_skips),
            self.imu(ZU, imu_base, imu_skips),
        )

