"""Restormer (Zamir et al., CVPR 2022, arXiv 2111.09881) as phase 2's restoration network, from its released weights.

Ported from github.com/swz30/Restormer, basicsr/models/archs/restormer_arch.py (class Restormer with the motion-
deblurring defaults: dim 48, blocks 4/6/6/8, 4 refinement blocks, heads 1/2/4/8, FFN x2.66, no bias, WithBias
LayerNorm, no dual-pixel skip). Parameter names are the original's, so ``motion_deblurring.pth`` (GoPro) loads with
its 494 tensors; einops and pdb dropped. tests/test_restormer.py pins the names and shapes; the port was checked once
against the original file: same output on shared random weights.

p51 (phase2.restore_net: restormer): RestoreRGBDecoder's network. Phase 2 trained its NAFNet from zero on ~32 000
frames (p49: 4 000 updates x 8), where deblurring networks train for 300 000 iterations; it got back about a quarter
of the edges a blur took (Cell 11, D1, blur_only: edge strength 0.139 -> 0.167, clean 0.248). The flare branch, a
released Uformer fine-tuned, did well (p49 flare stage: VisDrone flare region 17.4 -> 23.4 dB); this does the same for
blur. Two inputs of the repo join through zero-initialised layers, so at update 0 the network IS the released model:
``extra_embed`` (the other channels of the decoder's input: the frame without its flare, the gyro path) added to the
patch embedding, and ``latent_in`` (ZI, resized bilinearly -- not by a 16-px nearest step, which drew the blocks of
p49 -- to the bottom level, 1/8 of the frame) added before the latent blocks.

Original code: MIT License, Copyright (c) 2021 Syed Waqas Zamir. Weights: github.com/swz30/Restormer releases v1.0.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

# Ten -> (URL, sha256) cua trong so phat hanh (GitHub release v1.0 cua swz30/Restormer).
RESTORMER_WEIGHTS = {
    "motion_deblurring": ("https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth",
                          "194e38fb5b607c9dc5a5b3e08e65b2e79ee2bf0ef5048e0612f6b2ff2f79da31"),
}
STEP = 8           # ba lan giam 2: canh anh phai chia het (anh khac co duoc dem phan xa roi cat lai)
EXTRA_LAYERS = ("extra_embed", "latent_in")   # lop cua repo (khoi tao 0), khong co trong trong so phat hanh


class LayerNorm(nn.Module):
    """WithBias_LayerNorm over the channels of a [B, C, H, W] map (the original's to_3d / to_4d round trip)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.body = _Affine(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],), self.body.weight, self.body.bias, 1e-5)
        return y.permute(0, 3, 1, 2)


class _Affine(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))


class FeedForward(nn.Module):
    """Gated-Dconv feed-forward network (GDFN)."""

    def __init__(self, dim: int, expansion: float) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=False)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=False)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class Attention(nn.Module):
    """Multi-Dconv head transposed self-attention (MDTA): attention across channels, linear in the pixel count."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q, k, v = (t.reshape(b, self.heads, c // self.heads, h * w) for t in (q, k, v))
        # Chuan hoa va softmax o fp32: o fp16 tong binh phuong tren 65 536 pixel tran so.
        q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature.float()).softmax(dim=-1)
        out = (attn.to(v.dtype) @ v).reshape(b, c, h, w)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, expansion: float) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, inputs: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(inputs, dim, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(features, features // 2, 3, padding=1, bias=False), nn.PixelUnshuffle(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(features, features * 2, 3, padding=1, bias=False), nn.PixelShuffle(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class RestormerNet(nn.Module):
    """Restormer on the decoder's input ``x`` [B, in_channels, H, W]: channels 0-2 are the frame it restores (the
    released model's input), the rest go in through ``extra_embed``. ``latent`` (ZI, or None) joins at the bottom.
    Returns ``base`` (default: channels 0-2) plus the network's answer, as the original returns input + answer.

    Training holds only each block's input and recomputes the block in the backward pass (activation checkpointing):
    on a T4 the full Restormer at 256 x 256 needs about 4 GB per frame otherwise. ``checkpointing`` False (p52, which
    trains on 128 x 128 crops, a quarter of that) skips the recompute: about a quarter of the training step."""

    def __init__(self, in_channels: int, latent_channels: int, *, dim: int = 48, blocks=(4, 6, 6, 8),
                 refinement: int = 4, heads=(1, 2, 4, 8), expansion: float = 2.66, checkpointing: bool = True) -> None:
        super().__init__()
        if in_channels < 3:
            raise ValueError("RestormerNet restores an RGB frame: in_channels >= 3")
        self.in_channels = int(in_channels)
        self.checkpointing = bool(checkpointing)

        def level(width: int, count: int, head: int) -> nn.Sequential:
            return nn.Sequential(*[TransformerBlock(width, head, expansion) for _ in range(count)])

        self.patch_embed = OverlapPatchEmbed(3, dim)
        self.encoder_level1 = level(dim, blocks[0], heads[0])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = level(dim * 2, blocks[1], heads[1])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = level(dim * 4, blocks[2], heads[2])
        self.down3_4 = Downsample(dim * 4)
        self.latent = level(dim * 8, blocks[3], heads[3])
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=False)
        self.decoder_level3 = level(dim * 4, blocks[2], heads[2])
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)
        self.decoder_level2 = level(dim * 2, blocks[1], heads[1])
        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = level(dim * 2, blocks[0], heads[0])
        self.refinement = level(dim * 2, refinement, heads[0])
        self.output = nn.Conv2d(dim * 2, 3, 3, padding=1, bias=False)
        # Cua repo, khoi tao 0: luc dau mang chinh la ban phat hanh.
        self.extra_embed = nn.Conv2d(in_channels - 3, dim, 3, padding=1, bias=False) if in_channels > 3 else None
        self.latent_in = nn.Conv2d(latent_channels, dim * 8, 1)
        for layer in (self.extra_embed, self.latent_in):
            if layer is not None:
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def pretrained_parameters(self) -> list[nn.Parameter]:
        """The parameters the released weights fill (phase2.restore_pretrained_lr_scale trains them slower)."""
        return [p for name, p in self.named_parameters() if not name.startswith(EXTRA_LAYERS)]

    def _blocks(self, blocks: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        if not (self.checkpointing and self.training and torch.is_grad_enabled()):
            return blocks(x)
        for block in blocks:
            x = checkpoint(block, x, use_reentrant=False)
        return x

    def forward(self, latent: torch.Tensor | None, x: torch.Tensor, base: torch.Tensor | None = None) -> torch.Tensor:
        height, width = x.shape[-2:]
        base = x[:, :3] if base is None else base
        pad_h, pad_w = -height % STEP, -width % STEP
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        level1 = self.patch_embed(x[:, :3])
        if self.extra_embed is not None:
            level1 = level1 + self.extra_embed(x[:, 3:])
        out1 = self._blocks(self.encoder_level1, level1)
        out2 = self._blocks(self.encoder_level2, self.down1_2(out1))
        out3 = self._blocks(self.encoder_level3, self.down2_3(out2))
        bottom = self.down3_4(out3)
        if latent is not None:
            bottom = bottom + self.latent_in(F.interpolate(latent.to(bottom.dtype), size=bottom.shape[-2:],
                                                           mode="bilinear", align_corners=False))
        bottom = self._blocks(self.latent, bottom)
        dec3 = self._blocks(self.decoder_level3, self.reduce_chan_level3(torch.cat((self.up4_3(bottom), out3), 1)))
        dec2 = self._blocks(self.decoder_level2, self.reduce_chan_level2(torch.cat((self.up3_2(dec3), out2), 1)))
        dec1 = self._blocks(self.decoder_level1, torch.cat((self.up2_1(dec2), out1), 1))
        answer = self.output(self._blocks(self.refinement, dec1))
        return answer[..., :height, :width] + base


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_restormer_into(net: RestormerNet, path, name: str) -> str:
    """Released weights ``name`` from ``path`` into ``net``: its sha256 must be the release's, every released tensor
    loads and only the repo's zero-initialised layers stay as they are. Returns the sha256."""
    if name not in RESTORMER_WEIGHTS:
        raise ValueError(f"Restormer weights {name!r} unknown; known: {sorted(RESTORMER_WEIGHTS)}")
    digest = file_sha256(path)
    if digest != RESTORMER_WEIGHTS[name][1]:
        raise ValueError(f"{path}: sha256 {digest[:12]}..., not the released {name}.pth")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("params", payload) if isinstance(payload, dict) else payload
    missing, unexpected = net.load_state_dict(state, strict=False)
    extra = [key for key in missing if not key.startswith(EXTRA_LAYERS)]
    if extra or unexpected:
        raise ValueError(f"{path}: not this Restormer (missing {extra[:4]}, unexpected {list(unexpected)[:4]})")
    return digest


def find_restormer(input_root, name: str, skip=(), max_depth: int = 6) -> Path | None:
    """``<name>.pth`` with the release's sha256 anywhere in ``input_root`` (a Kaggle Input), or None. Image folders
    and ``skip`` (dataset shards) are not walked."""
    from .uformer_flare import SCAN_PRUNE, walk_input
    root, top, skip = Path(input_root), len(Path(input_root).parts), {str(path) for path in skip}
    for directory, children, names in walk_input(root):
        here = Path(directory)
        children[:] = [] if len(here.parts) - top >= max_depth else sorted(
            child for child in children if not child.startswith(SCAN_PRUNE) and str(here / child) not in skip)
        if f"{name}.pth" in names and file_sha256(here / f"{name}.pth") == RESTORMER_WEIGHTS[name][1]:
            return here / f"{name}.pth"
    return None


def download_restormer(cache_dir, name: str) -> Path:
    """The released ``<name>.pth`` (about 100 MB) from GitHub into ``cache_dir``, checked by its sha256."""
    url, digest = RESTORMER_WEIGHTS[name]
    target = Path(cache_dir) / f"{name}.pth"
    if target.is_file() and file_sha256(target) == digest:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.{os.getpid()}.part")
    print(f"Tải trọng số Restormer {name} từ GitHub...", flush=True)
    urllib.request.urlretrieve(url, partial)
    if file_sha256(partial) != digest:
        partial.unlink(missing_ok=True)
        raise ValueError(f"{url}: tải về sai sha256")
    os.replace(partial, target)
    return target
