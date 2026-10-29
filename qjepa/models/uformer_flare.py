"""Uformer of Flare7K++ (Dai et al., NeurIPS 2022 / arXiv 2306.04236), for inference with its released checkpoint.

Ported from github.com/ykdai/Flare7K, basicsr/archs/uformer_arch.py (class Uformer and what it uses with its defaults:
linear token projection, "ffn" MLP, no SE layer), which builds on Uformer (Wang et al., arXiv 2106.03106). Parameter
and buffer names are the original's, so the released ``net_g_last.pth`` loads with strict=True. Dropped: timm and
einops (DropPath is the identity at inference; the bias "repeat" has ratio 1), the dilated windows and every unused
variant. tests/test_uformer_flare.py checks the port against the original file, output for output.

Flare7K++'s model (``--flare7kpp`` in its test_large.py) maps a flared sRGB image to 6 channels in [0, 1]: the
flare-free image WITH its light source (channels 0-2) and the flare (3-5). It was trained on 512x512 crops, and its
attention assumes square inputs: ``remove_flare`` resizes the short side to 512 and runs overlapping 512-px tiles along
the long side, blended linearly, as test_large.py does.

Original code: Copyright 2022 S-Lab, S-Lab License 1.0 -- redistribution and use for NON-COMMERCIAL purposes, in source
or binary form, with or without modification, provided the copyright notice and the license conditions are retained.
Commercial use needs the contributors' permission.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from PIL import Image
from torch import nn

TILE = 512


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)

    def forward(self, x):
        b, n, c = x.shape
        q = self.to_q(x).reshape(b, n, 1, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)[0]
        kv = self.to_kv(x).reshape(b, n, 2, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        return q, kv[0], kv[1]


class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, qkv_bias=True):
        super().__init__()
        self.win_size = win_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))
        coords = torch.stack(torch.meshgrid([torch.arange(win_size[0]), torch.arange(win_size[1])], indexing="ij"))
        coords = torch.flatten(coords, 1)
        relative = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        relative[:, :, 0] += win_size[0] - 1
        relative[:, :, 1] += win_size[1] - 1
        relative[:, :, 0] *= 2 * win_size[1] - 1
        self.register_buffer("relative_position_index", relative.sum(-1))
        self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, n, c = x.shape
        q, k, v = self.qkv(x)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        area = self.win_size[0] * self.win_size[1]
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(area, area, -1)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            windows = mask.shape[0]
            attn = attn.view(b // windows, windows, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = attn.softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def window_partition(x, win_size):
    b, h, w, c = x.shape
    x = x.view(b, h // win_size, win_size, w // win_size, win_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, c)


def window_reverse(windows, win_size, h, w):
    b = int(windows.shape[0] / (h * w / win_size / win_size))
    x = windows.view(b, h // win_size, w // win_size, win_size, win_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


def _square(tokens: int) -> int:
    side = int(math.sqrt(tokens))
    if side * side != tokens:
        raise ValueError(f"Uformer needs square inputs (got {tokens} tokens): run it on 512x512 tiles")
    return side


class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1))

    def forward(self, x):
        b, n, c = x.shape
        side = _square(n)
        return self.conv(x.transpose(1, 2).contiguous().view(b, c, side, side)).flatten(2).transpose(1, 2).contiguous()


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.deconv = nn.Sequential(nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2))

    def forward(self, x):
        b, n, c = x.shape
        side = _square(n)
        return self.deconv(x.transpose(1, 2).contiguous().view(b, c, side, side)).flatten(2).transpose(1, 2).contiguous()


class InputProj(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2).contiguous()


class OutputProj(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        b, n, c = x.shape
        side = _square(n)
        return self.proj(x.transpose(1, 2).view(b, c, side, side))


class LeWinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        self.win_size = win_size
        self.shift_size = shift_size
        if min(input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(input_resolution)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, (self.win_size, self.win_size), num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def _shift_mask(self, h, w, like):
        mask = torch.zeros((1, h, w, 1), dtype=like.dtype, device=like.device)
        spans = (slice(0, -self.win_size), slice(-self.win_size, -self.shift_size), slice(-self.shift_size, None))
        count = 0
        for rows in spans:
            for cols in spans:
                mask[:, rows, cols, :] = count
                count += 1
        windows = window_partition(mask, self.win_size).view(-1, self.win_size * self.win_size)
        mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, x):
        b, n, c = x.shape
        side = _square(n)
        mask = self._shift_mask(side, side, x) if self.shift_size > 0 else None
        shortcut = x
        x = self.norm1(x).view(b, side, side, c)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        windows = window_partition(x, self.win_size).view(-1, self.win_size * self.win_size, c)
        windows = self.attn(windows, mask=mask).view(-1, self.win_size, self.win_size, c)
        x = window_reverse(windows, self.win_size, side, side)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + x.view(b, n, c)
        return x + self.mlp(self.norm2(x))


class BasicUformerLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, win_size, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        self.checkpointing = False
        self.blocks = nn.ModuleList([
            LeWinTransformerBlock(dim, input_resolution, num_heads, win_size=win_size,
                                  shift_size=0 if i % 2 == 0 else win_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
            for i in range(depth)])

    def forward(self, x):
        for block in self.blocks:
            if self.checkpointing and torch.is_grad_enabled():
                # Fine-tuning at 512x512 on a 16 GB T4: recompute each block in backward instead of keeping it.
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


class Uformer(nn.Module):
    """Flare7K++'s Uformer with its defaults (embed 32, depths 2 x 9, heads 1-16, window 8)."""

    def __init__(self, img_size=TILE, img_ch=3, output_ch=6, embed_dim=32, depths=(2,) * 9,
                 num_heads=(1, 2, 4, 8, 16, 16, 8, 4, 2), win_size=8, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        e = embed_dim
        layer = lambda dim, scale, k: BasicUformerLayer(dim, (img_size // scale, img_size // scale), depths[k],
                                                        num_heads[k], win_size, mlp_ratio, qkv_bias)
        self.input_proj = InputProj(img_ch, e)
        self.output_proj = OutputProj(2 * e, output_ch)
        self.encoderlayer_0, self.dowsample_0 = layer(e, 1, 0), Downsample(e, 2 * e)
        self.encoderlayer_1, self.dowsample_1 = layer(2 * e, 2, 1), Downsample(2 * e, 4 * e)
        self.encoderlayer_2, self.dowsample_2 = layer(4 * e, 4, 2), Downsample(4 * e, 8 * e)
        self.encoderlayer_3, self.dowsample_3 = layer(8 * e, 8, 3), Downsample(8 * e, 16 * e)
        self.conv = layer(16 * e, 16, 4)
        self.upsample_0, self.decoderlayer_0 = Upsample(16 * e, 8 * e), layer(16 * e, 8, 5)
        self.upsample_1, self.decoderlayer_1 = Upsample(16 * e, 4 * e), layer(8 * e, 4, 6)
        self.upsample_2, self.decoderlayer_2 = Upsample(8 * e, 2 * e), layer(4 * e, 2, 7)
        self.upsample_3, self.decoderlayer_3 = Upsample(4 * e, e), layer(2 * e, 1, 8)

    def set_checkpointing(self, enabled: bool) -> None:
        """Gradient checkpointing per transformer block (same outputs and gradients, less memory, ~30% slower)."""
        for module in self.modules():
            if isinstance(module, BasicUformerLayer):
                module.checkpointing = bool(enabled)

    def forward(self, x):
        y = self.input_proj(x)
        conv0 = self.encoderlayer_0(y)
        conv1 = self.encoderlayer_1(self.dowsample_0(conv0))
        conv2 = self.encoderlayer_2(self.dowsample_1(conv1))
        conv3 = self.encoderlayer_3(self.dowsample_2(conv2))
        conv4 = self.conv(self.dowsample_3(conv3))
        d0 = self.decoderlayer_0(torch.cat([self.upsample_0(conv4), conv3], -1))
        d1 = self.decoderlayer_1(torch.cat([self.upsample_1(d0), conv2], -1))
        d2 = self.decoderlayer_2(torch.cat([self.upsample_2(d1), conv1], -1))
        d3 = self.decoderlayer_3(torch.cat([self.upsample_3(d2), conv0], -1))
        return torch.sigmoid(self.output_proj(d3))


def load_flare7kpp(path, device="cpu") -> Uformer:
    """The released Flare7K++ checkpoint (``net_g_last.pth``: {'params_ema' | 'params': state dict} or the state dict),
    strict, in eval mode. weights_only: a downloaded file is data, not code."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("params_ema", payload.get("params", payload)) if isinstance(payload, dict) else payload
    model = Uformer()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """PIL bicubic, as Flare7K++ resizes (float32 channels, so no uint8 rounding in between)."""
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(image[..., k]), mode="F").resize(
        (width, height), Image.Resampling.BICUBIC)) for k in range(image.shape[-1])], axis=-1)


@torch.no_grad()
def remove_flare(model: Uformer, image: np.ndarray) -> np.ndarray:
    """Flare-free image (light source kept) of ``image`` (H x W x 3 sRGB float in [0, 1]) at its own size.

    Flare7K++'s test_large.py with ``--flare7kpp``: the short side resized to 512; 512-px tiles every 256 px along the
    long side, the last one flush with the end; each tile's first ``overlap`` columns (rows) blended linearly with the
    previous tile's last ones; the flare at 512 (|input - flare-free|) resized back to H x W and taken off the input
    (|input - flare|), so the frame keeps its own resolution outside the flare."""
    device = next(model.parameters()).device
    h, w = image.shape[:2]
    if w < h:
        new_w, new_h = TILE, int(TILE / (w / h))
    else:
        new_w, new_h = int(TILE * (w / h)), TILE
    small = np.clip(_resize(image.astype(np.float32), new_w, new_h), 0.0, 1.0)
    tensor = torch.from_numpy(small).permute(2, 0, 1)[None].to(device)
    wide = new_w > TILE
    length = new_w if wide else new_h
    ends = [min(end, length) for end in range(TILE, length + TILE // 2, TILE // 2)]
    segments = []
    for end in ends:
        tile = tensor[..., :, end - TILE:end] if wide else tensor[..., end - TILE:end, :]
        segments.append(model(tile)[0])
    overlaps = [0] + [previous - current + TILE for previous, current in zip(ends[:-1], ends[1:])]
    for i in range(1, len(segments)):
        overlap = overlaps[i]
        alpha = torch.linspace(0, 1, steps=overlap, device=device)
        alpha = alpha.view(1, 1, -1) if wide else alpha.view(1, -1, 1)
        if wide:
            segments[i][:, :, :overlap] = alpha * segments[i][:, :, :overlap] + (1 - alpha) * segments[i - 1][:, :, -overlap:]
        else:
            segments[i][:, :overlap, :] = alpha * segments[i][:, :overlap, :] + (1 - alpha) * segments[i - 1][:, -overlap:, :]
    kept = [segment[:, :, :-overlap] if wide else segment[:, :-overlap, :]
            for segment, overlap in zip(segments[:-1], overlaps[1:])] + [segments[-1]]
    merged = torch.cat(kept, dim=2 if wide else 1)[:3].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    flare = np.abs(small - merged)
    return np.clip(np.abs(image - _resize(flare, w, h)), 0.0, 1.0).astype(np.float32)


# Four halvings down to a window of 8: every side the Uformer sees is a multiple of 128.
SIDE_MULTIPLE = 128


def _square_tiles(net, x: torch.Tensor) -> torch.Tensor:
    """``net`` (square inputs only) over a B x C x H x W batch whose sides are multiples of SIDE_MULTIPLE: a square
    frame in one pass; otherwise square tiles of the short side every half side along the long one, the last flush
    with the end, each tile's overlap blended linearly into the previous one -- test_large.py's scheme, in torch so
    gradients pass."""
    h, w = x.shape[-2:]
    if h == w:
        return net(x)
    wide, side, length = w > h, min(h, w), max(h, w)
    dim = -1 if wide else -2
    ends = [min(end, length) for end in range(side, length + side // 2, side // 2)]
    segments = [net(x.narrow(dim, end - side, side)) for end in ends]
    overlaps = [0] + [previous - current + side for previous, current in zip(ends[:-1], ends[1:])]
    for i in range(1, len(segments)):
        overlap = overlaps[i]
        shape = [1] * segments[i].dim()
        shape[dim] = overlap
        alpha = torch.linspace(0, 1, overlap, device=x.device, dtype=segments[i].dtype).view(shape)
        head = alpha * segments[i].narrow(dim, 0, overlap) + (1 - alpha) * segments[i - 1].narrow(dim, side - overlap, overlap)
        segments[i] = torch.cat([head, segments[i].narrow(dim, overlap, side - overlap)], dim)
    pieces = [segment.narrow(dim, 0, side - overlap) for segment, overlap in zip(segments[:-1], overlaps[1:])]
    return torch.cat(pieces + [segments[-1]], dim)


class UformerFlareLayer(nn.Module):
    """p44 (flare.model / phase2.split_flare_model: uformer_flare7kpp): Flare7K++'s Uformer in the flare layer's place.

    Same contract as decoders.FlareLayer with uses_latent False: ``forward(None, image)`` gives (image without the
    flare, the flare layer). The frame is upsampled by ``scale`` (bicubic; 2 takes a 256 crop to the 512 the
    checkpoint was trained at), padded to multiples of 128 (replicate), run in square tiles, and the flare -- that
    upsampled frame minus the Uformer's flare-free frame (light source kept) -- is brought back down (bilinear,
    antialiased) and taken off the input, so the frame keeps its own detail outside the flare. Always fp32, as the
    p42 layer: phase 2 runs it frozen under fp16 autocast.

    p53 (``veil_grid``, flare.veil_grid): a bilateral grid of colour affines (decoders.BilateralGridTone, no latent)
    then reads the frame the Uformer left -- at 4 x veil_grid px, with a global context of the whole frame, which the
    Uformer's 8 x 8 windows never see -- and takes off the veil: the smooth glow and haze spread over a large part of the
    frame (on p49's stage, 64% of the error left inside the flare of the FlareImage valid pairs is smoother than a
    Gaussian of 8 px; 90% on HALO, where the Uformer takes off 29% of it). What the grid takes off is added to the
    flare, so the contract is unchanged. Zero-initialised: the layer gives exactly the Uformer's answer until trained.
    """

    uses_latent = False

    def __init__(self, scale: float = 2.0, veil_grid: int | None = None) -> None:
        super().__init__()
        self.scale = float(scale)
        self.net = Uformer()
        self.veil = None
        if veil_grid:
            from .decoders import BilateralGridTone
            self.veil = BilateralGridTone(0, size=int(veil_grid))

    def _deflare(self, tile: torch.Tensor) -> torch.Tensor:
        return self.net(tile)[:, :3]

    def forward(self, latent: torch.Tensor | None, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=image.device.type, enabled=False):
            frame = image.float()
            h, w = frame.shape[-2:]
            big = (F.interpolate(frame, scale_factor=self.scale, mode="bicubic", align_corners=False)
                   if self.scale != 1 else frame)
            bh, bw = big.shape[-2:]
            pad_h, pad_w = (-bh) % SIDE_MULTIPLE, (-bw) % SIDE_MULTIPLE
            padded = F.pad(big, (0, pad_w, 0, pad_h), mode="replicate") if pad_h or pad_w else big
            clean = _square_tiles(self._deflare, padded)[..., :bh, :bw]
            flare = big - clean
            if (bh, bw) != (h, w):
                flare = F.interpolate(flare, size=(h, w), mode="bilinear", align_corners=False,
                                      antialias=self.scale > 1)
            if self.veil is not None:
                deflared = (frame - flare).clamp(0.0, 1.0)
                flare = flare + (deflared - self.veil(None, deflared))
        return image - flare.to(image.dtype), flare


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_flare7kpp_into(layer: UformerFlareLayer, path) -> str:
    """The released checkpoint into ``layer.net`` (strict); its sha256, which the flare stage records."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("params_ema", payload.get("params", payload)) if isinstance(payload, dict) else payload
    layer.net.load_state_dict(state, strict=True)
    return file_sha256(path)


# README of github.com/ykdai/Flare7K, "Pretrained Model", row Flare7K++ (official Google Drive link).
FLARE7KPP_DRIVE_ID = "17AX9BJ-GS0in9Ey7vw3BVPISm67Rpzho"
CHECKPOINT_NAME = "net_g_last.pth"
SCAN_PRUNE = ("image_", "depth_", "seg_", "flow_", "imu", "Flare7K", "Flare-R", "flare_dataset", "tartanair640",
              "__pycache__", ".git")


def walk_input(root):
    """os.walk(root, followlinks=True), each real directory once. Kaggle mounts some Input (a notebook's Output, a
    dataset of symlinked trajectories) through symlinked directories, which a plain os.walk lists but never enters: p49's
    Cell 4 missed p46's flare/best.pt in an attached notebook Output (27/10/2026). A directory already walked (a link
    back up, two links to one place) is skipped with its subtree. The caller may prune ``children`` as with os.walk."""
    seen: set[str] = set()
    for directory, children, names in os.walk(root, followlinks=True):
        real = os.path.realpath(directory)
        if real in seen:
            children[:] = []
            continue
        seen.add(real)
        yield directory, children, names


def _pick_member(bundle: zipfile.ZipFile) -> str | None:
    members = [m for m in bundle.namelist() if m.endswith(".pth")]
    ranked = sorted(members, key=lambda m: ("flare7kpp" not in m.lower(), not m.endswith(CHECKPOINT_NAME)))
    return ranked[0] if ranked else None


def _extract(bundle: zipfile.ZipFile, member: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with bundle.open(member) as source, target.open("wb") as sink:
        shutil.copyfileobj(source, sink)
    return target


def find_flare7kpp(input_root, cache_dir, skip=(), max_depth: int = 12) -> Path | None:
    """net_g_last.pth in ``cache_dir`` or anywhere under ``input_root`` (Kaggle Input): a file, or a member of a zip
    (extracted to ``cache_dir``), paths with "flare7kpp" first. Image folders and ``skip`` (dataset shards) are not
    walked: listing them on Kaggle's network disk takes minutes."""
    cached = Path(cache_dir) / CHECKPOINT_NAME
    if cached.is_file():
        return cached
    files, archives = [], []
    root = Path(input_root)
    skip = {str(path) for path in skip}
    top = len(root.parts)
    for directory, children, names in walk_input(root):
        here = Path(directory)
        if len(here.parts) - top >= max_depth:
            children[:] = []
        else:
            children[:] = sorted(c for c in children if not c.startswith(SCAN_PRUNE) and str(here / c) not in skip)
        for name in sorted(names):
            if name == CHECKPOINT_NAME:
                files.append(here / name)
            elif name.endswith(".zip"):
                archives.append(here / name)
    if files:
        return sorted(files, key=lambda path: "flare7kpp" not in str(path).lower())[0]
    for archive in archives:
        try:
            with zipfile.ZipFile(archive) as bundle:
                member = _pick_member(bundle)
                if member and member.endswith(CHECKPOINT_NAME):
                    print("Giải nén", member, "từ", archive, flush=True)
                    return _extract(bundle, member, cached)
        except (OSError, zipfile.BadZipFile):
            continue
    return None


def download_flare7kpp(cache_dir, drive_id: str | None = None) -> Path | None:
    """The Flare7K++ checkpoint from Google Drive (gdown) into ``cache_dir``: ``drive_id`` (a copy in your own Drive,
    when the official file hits Drive's download quota) or the official one. A zip gives its net_g_last.pth. None,
    with a hint, when Drive refuses."""
    target = Path(cache_dir) / CHECKPOINT_NAME
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = target.parent / "download.bin"
    try:
        import gdown
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        import gdown
    print("Tải checkpoint Flare7K++ từ Google Drive...", flush=True)
    try:
        gdown.download(f"https://drive.google.com/uc?id={drive_id or FLARE7KPP_DRIVE_ID}", str(raw), quiet=False)
    except Exception as error:                              # quota, permission, network
        print("⚠ Tải lỗi:", " ".join(str(error).split())[:400], flush=True)
    if not raw.is_file() or raw.stat().st_size < 1_000_000:
        raw.unlink(missing_ok=True)
        print("⚠ Không tải được (Google Drive hay báo hết hạn mức). Tạo bản sao trong Drive của bạn (Colab: Drive API\n"
              "  files().copy, xem notebook Cell 12e), chia sẻ 'bất kỳ ai có đường liên kết' và đặt ID của nó; hoặc tải\n"
              "  tay rồi tải lên Kaggle thành dataset và gắn làm Input.", flush=True)
        return None
    if zipfile.is_zipfile(raw):
        with zipfile.ZipFile(raw) as bundle:
            member = _pick_member(bundle)
            if member is None:
                raise ValueError(f"the downloaded zip holds no .pth: {bundle.namelist()[:10]}")
            _extract(bundle, member, target)
            print("Lấy", member, "từ zip tải về", flush=True)
        raw.unlink()
    else:
        raw.rename(target)
    print(f"Đã tải {target} ({target.stat().st_size / 2**20:.1f} MiB)", flush=True)
    return target


def _is_uformer_stage(path) -> bool:
    """``path`` is a flare-stage checkpoint (train-flare's last.pt/best.pt) holding a UformerFlareLayer."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return (isinstance(payload, dict) and payload.get("metadata", {}).get("phase") == "flare_stage"
            and "net.input_proj.proj.0.weight" in payload.get("model", {}))


def find_stage_checkpoint(input_root, cache_dir, prefer: str = "", skip=(), max_depth: int = 12) -> Path | None:
    """p45 (flare.uformer_init: stage): a previous run's flare/best.pt holding the fine-tuned Uformer, under
    ``input_root`` (Kaggle Input) -- a file .../flare/best.pt, or a member .../flare/best.pt of a zip (extracted to
    ``cache_dir``) -- paths containing ``prefer`` (e.g. "p44_uformer") first. Shards and image folders are not walked."""
    root, skip = Path(input_root), {str(path) for path in skip}
    files, archives = [], []
    top = len(root.parts)
    for directory, children, names in walk_input(root):
        here = Path(directory)
        if len(here.parts) - top >= max_depth:
            children[:] = []
        else:
            children[:] = sorted(c for c in children if not c.startswith(SCAN_PRUNE) and str(here / c) not in skip)
        for name in sorted(names):
            if name == "best.pt" and here.name == "flare":
                files.append(here / name)
            elif name.endswith(".zip"):
                archives.append(here / name)
    rank = lambda text: (prefer not in str(text)) if prefer else False
    for path in sorted(files, key=rank):
        if _is_uformer_stage(path):
            return path
    for archive in sorted(archives, key=rank):
        try:
            with zipfile.ZipFile(archive) as bundle:
                members = sorted((m for m in bundle.namelist() if m.endswith("flare/best.pt")), key=rank)
                for member in members:
                    target = Path(cache_dir) / member.replace("/", "__")
                    _extract(bundle, member, target)
                    if _is_uformer_stage(target):
                        print("Giải nén", member, "từ", archive, flush=True)
                        return target
                    target.unlink(missing_ok=True)
        except (OSError, zipfile.BadZipFile):
            continue
    return None
