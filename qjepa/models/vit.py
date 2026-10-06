"""Vision Transformer over wavelet coefficients: the encoder of I-JEPA (Assran 2023).

One ViT reads the QWT (image) and Haar (IMU) coefficients together, in patches of
``PATCH`` coefficients a side. The transforms halve the input once, so one
token covers 16x16 pixels (16 IMU samples): the 16x16 token grid of a 256 frame
and the 8 tokens of a 128 window are the grids the CNN encoders gave, and every
phase-2 decoder reads ZI/ZU on them unchanged.

As in I-JEPA: fixed sine-cosine positions, pre-norm blocks (LayerNorm eps 1e-6,
qkv bias, MLP x4, GELU), a final LayerNorm, truncated-normal init (std 0.02) with
each block's output projections scaled by 1/sqrt(2 * layer). ``keep_image`` and
``keep_imu`` run the encoder on those tokens only -- the context encoder never sees the rest.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

# Coefficients per token side; the transform's own halving makes it 16 px / 16 samples.
PATCH = 8
TOKEN_STRIDE = 2 * PATCH
INIT_STD = 0.02


def _sincos_1d(dim: int, positions: torch.Tensor) -> torch.Tensor:
    omega = 1.0 / 10000 ** (torch.arange(dim // 2, dtype=torch.float64) / (dim / 2.0))
    angles = positions.double().reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)


@lru_cache(maxsize=16)
def _sincos(dim: int, grid: tuple[int, ...]) -> torch.Tensor:
    if len(grid) == 1:
        if dim % 2:
            raise ValueError("A 1-D sine-cosine embedding needs an even width")
        return _sincos_1d(dim, torch.arange(grid[0])).float()
    if dim % 4:
        raise ValueError("A 2-D sine-cosine embedding needs a width divisible by 4")
    rows, columns = torch.meshgrid(torch.arange(grid[0]), torch.arange(grid[1]), indexing="ij")
    return torch.cat((_sincos_1d(dim // 2, rows), _sincos_1d(dim // 2, columns)), dim=1).float()


def sincos_embedding(dim: int, grid: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
    """[prod(grid), dim] fixed positions, row-major like ``flatten``; I-JEPA's get_2d_sincos_pos_embed."""
    return _sincos(dim, tuple(int(side) for side in grid)).to(device=like.device, dtype=like.dtype)


def gather_tokens(tokens: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """tokens [B, N, D], index [B, ...] -> [B, ..., D]."""
    flat = index.reshape(index.shape[0], -1)
    picked = torch.gather(tokens, 1, flat.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
    return picked.reshape(*index.shape, tokens.shape[-1])


class Block(nn.Module):
    """Pre-norm transformer block, as timm's and I-JEPA's."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"Width {dim} is not divisible by {heads} heads")
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        q, k, v = (self.qkv(self.norm1(x)).reshape(batch, tokens, 3, self.heads, dim // self.heads)
                   .permute(2, 0, 3, 1, 4))
        attended = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(batch, tokens, dim)
        x = x + self.proj(attended)
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


def init_transformer(module: nn.Module, blocks: nn.ModuleList) -> None:
    """I-JEPA's _init_weights then fix_init_weight."""
    for layer in module.modules():
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.trunc_normal_(layer.weight, std=INIT_STD)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
    for depth, block in enumerate(blocks, start=1):
        block.proj.weight.data.div_(math.sqrt(2.0 * depth))
        block.fc2.weight.data.div_(math.sqrt(2.0 * depth))


class JointCoefficientViT(nn.Module):
    """Image and IMU tokens in one transformer: attention between them is the fusion.

    Each modality has its own patch embedding (2-D over the QWT grid, 1-D over the Haar
    sequence), its own fixed sine-cosine positions and a learned type embedding that tells
    the two apart; the blocks and the final LayerNorm are shared. Every image token can
    attend to every IMU token and back, so ZI carries what the IMU says and ZU what the
    frame says, learnt by the I-JEPA loss alone.

    Dense (no ``keep_*``): FI [B, dim, H, W] and FU [B, dim, L], the shapes the CNN encoders
    returned. ``keep_image`` [B, Ki] and ``keep_imu`` [B, Ku]: the context tokens of both,
    [B, Ki, dim] and [B, Ku, dim], computed from those tokens alone.
    """

    def __init__(self, image_channels: int, imu_channels: int, dim: int, *, depth: int, heads: int) -> None:
        super().__init__()
        self.image_embed = nn.Conv2d(image_channels, dim, PATCH, stride=PATCH)
        self.imu_embed = nn.Conv1d(imu_channels, dim, PATCH, stride=PATCH)
        self.image_type = nn.Parameter(torch.zeros(dim))
        self.imu_type = nn.Parameter(torch.zeros(dim))
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.out_channels = dim
        # One resolution: nothing for a decoder's skips or a finer JEPA target.
        self.skip_channels: tuple[int, ...] = ()
        init_transformer(self, self.blocks)
        nn.init.trunc_normal_(self.image_type, std=INIT_STD)
        nn.init.trunc_normal_(self.imu_type, std=INIT_STD)

    @staticmethod
    def token_grid(coefficient_shape: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(int(side) // PATCH for side in coefficient_shape[2:])

    def forward(self, image_coefficients: torch.Tensor, imu_coefficients: torch.Tensor, *,
                keep_image: torch.Tensor | None = None, keep_imu: torch.Tensor | None = None,
                return_stages: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if return_stages:
            raise ValueError("A ViT encoder has one resolution: there are no encoder stages")
        if (keep_image is None) != (keep_imu is None):
            raise ValueError("Keep the context of both modalities, or of neither")
        parts = []
        for embed, type_embedding, coefficients, keep in (
                (self.image_embed, self.image_type, image_coefficients, keep_image),
                (self.imu_embed, self.imu_type, imu_coefficients, keep_imu)):
            patches = embed(coefficients)
            tokens = patches.flatten(2).transpose(1, 2)
            tokens = tokens + sincos_embedding(tokens.shape[-1], tuple(patches.shape[2:]), tokens) + type_embedding
            parts.append((gather_tokens(tokens, keep) if keep is not None else tokens, tuple(patches.shape[2:])))
        (image, image_grid), (imu, imu_grid) = parts
        tokens = torch.cat((image, imu), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        image, imu = tokens[:, :image.shape[1]], tokens[:, image.shape[1]:]
        if keep_image is not None:
            return image, imu
        return (image.transpose(1, 2).reshape(image.shape[0], image.shape[2], *image_grid),
                imu.transpose(1, 2).reshape(imu.shape[0], imu.shape[2], *imu_grid))
