from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PREDICTOR_TYPES = ("token", "spatial")


def image_tokens(feature: torch.Tensor) -> torch.Tensor:
    return feature.flatten(2).transpose(1, 2)


def imu_tokens(feature: torch.Tensor) -> torch.Tensor:
    return feature.transpose(1, 2)


class LatentPredictor(nn.Module):
    """Token-wise MLP: each prediction sees its own token and nothing else."""

    def __init__(self, dim: int = 128, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class SpatialPredictor(nn.Module):
    """The token MLP, with its hidden layer also mixing each token's neighbours.

    Blur spreads one edge over neighbouring 16x16-pixel tokens, and a token-wise
    predictor cannot gather it back. Here the hidden layer passes through
    depthwise convolutions over the token grid (image) or sequence (IMU): with
    ``layers=2`` and ``kernel=3`` each prediction reads a 5x5 token neighbourhood.
    Depthwise keeps the capacity at the token MLP's (71k vs 66k parameters at
    128/256), so it gathers context without becoming a denoiser that would do
    the encoder's work for it -- the predictor is thrown away after phase 1.

    ``mask`` [B, N] swaps those tokens, after the input LayerNorm, for one learned
    token: their predictions must come from the neighbours alone.

    ``fine_channels`` adds a head that predicts a map at twice the grid's
    resolution (the teacher's previous encoder stage) from the same hidden grid.
    """

    def __init__(
        self,
        dim: int = 128,
        hidden: int = 256,
        *,
        spatial_dims: int,
        kernel: int = 3,
        layers: int = 2,
        fine_channels: int = 0,
    ) -> None:
        super().__init__()
        if spatial_dims not in (1, 2):
            raise ValueError("spatial_dims must be 1 (IMU) or 2 (image)")
        if kernel < 1 or kernel % 2 == 0 or layers < 1:
            raise ValueError("kernel must be odd and layers positive")
        if fine_channels and spatial_dims != 2:
            raise ValueError("The fine-scale head exists for the image grid only")
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv1d
        self.norm = nn.LayerNorm(dim)
        # After the LayerNorm, so zero is an ordinary normalized token, not 0/sqrt(eps).
        self.mask_token = nn.Parameter(torch.zeros(dim))
        self.expand = nn.Linear(dim, hidden)
        self.mix = nn.ModuleList(
            conv(hidden, hidden, kernel, padding=kernel // 2, groups=hidden) for _ in range(layers)
        )
        self.project = nn.Linear(hidden, dim)
        self.fine = None
        if fine_channels:
            self.fine = nn.Sequential(
                nn.Conv2d(hidden, 4 * fine_channels, 1),
                nn.PixelShuffle(2),
                # Smooths the 2x2 blocks PixelShuffle leaves behind.
                nn.Conv2d(fine_channels, fine_channels, 3, padding=1, groups=fine_channels),
            )

    def forward(
        self, dense: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``dense`` [B,D,H,W] or [B,D,L] -> (tokens [B,N,D], fine map or None)."""
        tokens = self.norm(dense.flatten(2).transpose(1, 2))
        if mask is not None:
            tokens = torch.where(mask.unsqueeze(-1), self.mask_token.to(tokens.dtype), tokens)
        hidden = F.gelu(self.expand(tokens))
        grid = hidden.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[2], *dense.shape[2:])
        for layer in self.mix:
            grid = grid + F.gelu(layer(grid))
        fine = self.fine(grid) if self.fine is not None else None
        return self.project(grid.flatten(2).transpose(1, 2)), fine

