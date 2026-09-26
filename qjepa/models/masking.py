"""Which latent tokens the phase-1 predictor is not allowed to see.

Masking happens on the latent, not the input: the encoder still reads the whole
noisy image, so ZI keeps everything phase 2 needs, but the predictor must infer
a hidden token's clean target from its neighbours. That rewards tokens that
carry their surroundings -- shapes and structure -- not only their own patch.

Image masks are rectangles of tokens (I-JEPA uses blocks, since a lone hidden
token is trivially interpolated from its neighbours); IMU masks are contiguous
time spans. Every mask comes from its own integer seed, so one sample gets the
same mask whether the batch runs on one GPU or is split over two.
"""

from __future__ import annotations

import hashlib

import torch


def _generator(seed: int, kind: str) -> torch.Generator:
    raw = f"{seed}|{kind}".encode()
    return torch.Generator(device="cpu").manual_seed(int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") >> 1)


def _side(low: int, high: int, size: int, generator: torch.Generator) -> int:
    # Never the whole axis: a block as wide as the grid leaves no neighbour to read.
    high = max(1, min(high, size - 1))
    low = max(1, min(low, high))
    return int(torch.randint(low, high + 1, (1,), generator=generator))


def _count(ratio: float, total: int) -> int:
    return min(total - 1, max(1, round(ratio * total))) if ratio > 0 and total > 1 else 0


def block_mask(height: int, width: int, ratio: float, block: tuple[int, int], seed: int) -> torch.Tensor:
    """[height * width] bool, True = hidden: rectangles of ``block`` tokens a side."""
    generator = _generator(seed, "image")
    mask = torch.zeros(height, width, dtype=torch.bool)
    wanted = _count(ratio, height * width)
    while int(mask.sum()) < wanted:
        rows, cols = _side(*block, height, generator), _side(*block, width, generator)
        top = int(torch.randint(0, height - rows + 1, (1,), generator=generator))
        left = int(torch.randint(0, width - cols + 1, (1,), generator=generator))
        mask[top:top + rows, left:left + cols] = True
    if bool(mask.all()):
        mask.view(-1)[int(torch.randint(0, height * width, (1,), generator=generator))] = False
    return mask.flatten()


def span_mask(length: int, ratio: float, span: tuple[int, int], seed: int) -> torch.Tensor:
    """[length] bool, True = hidden: contiguous runs of ``span`` tokens."""
    generator = _generator(seed, "imu")
    mask = torch.zeros(length, dtype=torch.bool)
    wanted = _count(ratio, length)
    while int(mask.sum()) < wanted:
        size = _side(*span, length, generator)
        start = int(torch.randint(0, length - size + 1, (1,), generator=generator))
        mask[start:start + size] = True
    if bool(mask.all()):
        mask[int(torch.randint(0, length, (1,), generator=generator))] = False
    return mask


def sample_seeds(seed: int, update: int, batch: int) -> torch.Tensor:
    """One mask seed per sample of an update, reproducible on resume."""
    raw = f"{seed}|{update}|mask".encode()
    base = int.from_bytes(hashlib.sha256(raw).digest()[:6], "big")
    return torch.arange(batch, dtype=torch.long) + base
