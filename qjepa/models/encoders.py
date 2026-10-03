from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import Stage

DEFAULT_CHANNELS = (32, 64, 96, 128)


class DenseCoefficientEncoder(nn.Module):
    """Four-stage CNN returning the dense final feature.

    Intermediate tensors are returned only when the caller asks for them, so a
    decoder cannot pick up skip connections by accident — it has to request them,
    and `phase2.encoder_skips` is the single place that decides.
    """

    def __init__(
        self,
        in_channels: int,
        channels: tuple[int, int, int, int] = DEFAULT_CHANNELS,
        *,
        dim: int,
        groups: int = 8,
        norm: str = "group",
    ) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels
        self.stages = nn.Sequential(
            Stage(in_channels, c0, dim=dim, groups=groups, norm=norm),
            Stage(c0, c1, dim=dim, stride=2, groups=groups, norm=norm),
            Stage(c1, c2, dim=dim, stride=2, groups=groups, norm=norm),
            Stage(c2, c3, dim=dim, stride=2, groups=groups, norm=norm),
        )
        self.out_channels = c3
        # Do phan giai giam mot nua sau moi stage tru stage 0, nen day chinh la
        # danh sach kenh ma decoder co the noi vao, theo thu tu tu tho den min.
        self.skip_channels = (c2, c1, c0)

    def forward(
        self, coefficients: torch.Tensor, *, return_stages: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if not return_stages:
            return self.stages(coefficients)
        outputs = []
        x = coefficients
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        # Bo stage cuoi: no chinh la FI/FU, da di qua fusion roi.
        return x, tuple(reversed(outputs[:-1]))

