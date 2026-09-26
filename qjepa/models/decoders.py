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


class ColorBranch(nn.Module):
    """Colour and brightness at reduced resolution: a residual on the averaged input.

    Colour is what a mean loss restores well -- the average of the plausible
    colours IS the right colour -- so this branch is trained with L1 and kept
    away from edges. Working at reduced resolution also averages away most of
    the colour noise a dark frame carries. Zero-initialised tail: it starts as
    the averaged input.
    """

    def __init__(self, latent_channels: int = 128, width: int = 32, blocks: int = 6) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.ReLU())
        self.latent = nn.Conv2d(latent_channels, width, 1)
        self.fuse = nn.Conv2d(2 * width, width, 3, padding=1)
        self.trunk = nn.Sequential(*[_ResidualBlock(width) for _ in range(blocks)])
        self.tail = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, latent: torch.Tensor, small: torch.Tensor) -> torch.Tensor:
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
    ) -> None:
        super().__init__()
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
            self.color = ColorBranch(latent_channels, color_width, color_blocks)
            self.edge = UNetBranch(2, 1, latent_channels, tuple(edge_widths), unet_blocks)
        elif branch_arch == "resnet":
            # p8: layer names are part of its checkpoints; keep them.
            self.color = ColorBranch(latent_channels, color_width, color_blocks)
            self.edge = PixelResNetDecoder(latent_channels, edge_width, edge_blocks,
                                           in_channels=2, out_channels=1)
        else:
            raise ValueError("branch_arch must be resnet, unet or unet_edge")

    def forward(self, latent: torch.Tensor, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        size = image.shape[-2:]
        if size[0] % self.illumination_scale or size[1] % self.illumination_scale:
            raise ValueError(f"Image sides {tuple(size)} must divide by {self.illumination_scale}")
        base = upsample(self.color(latent, downsample(image, self.color_scale)), size)
        light = illumination(base, self.illumination_scale)
        y = luminance(image)
        detail_in = y - illumination(color_base(image, self.color_scale), self.illumination_scale)
        detail = self.edge(latent, torch.cat((y, light.detach()), dim=1), base=detail_in)
        parts = {"image_color_base": base, "image_illumination": light, "image_detail": detail}
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

