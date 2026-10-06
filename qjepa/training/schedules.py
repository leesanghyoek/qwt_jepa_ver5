from __future__ import annotations

import math


def warmup_cosine_lr(
    update: int,
    total_updates: int,
    warmup_updates: int,
    peak_lr: float,
    minimum_lr: float,
) -> float:
    if update < warmup_updates:
        return peak_lr * (update + 1) / max(1, warmup_updates)
    progress = (update - warmup_updates) / max(1, total_updates - warmup_updates - 1)
    progress = min(max(progress, 0.0), 1.0)
    return minimum_lr + 0.5 * (peak_lr - minimum_lr) * (1.0 + math.cos(math.pi * progress))


def ema_momentum(update: int, total_updates: int, start: float, end: float) -> float:
    progress = min(max(update / max(1, total_updates - 1), 0.0), 1.0)
    return start + 0.5 * (end - start) * (1.0 - math.cos(math.pi * progress))


def linear_momentum(update: int, total_updates: int, start: float, end: float) -> float:
    """I-JEPA's EMA schedule: start + (end - start) * update / total. Below 1 for every update < total."""
    progress = min(max(update / max(1, total_updates), 0.0), 1.0)
    return start + (end - start) * progress


def cosine_weight_decay(update: int, total_updates: int, start: float, end: float) -> float:
    """I-JEPA's CosineWDSchedule: from ``start`` at update 0 to ``end`` at the last update."""
    progress = min(max(update / max(1, total_updates), 0.0), 1.0)
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * progress))
