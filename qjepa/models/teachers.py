"""EMA clean-target encoders used only during latent pretraining."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .backbone import MultimodalBackbone


class EMATeachers(nn.Module):
    def __init__(self, online: MultimodalBackbone) -> None:
        super().__init__()
        self.image_encoder = copy.deepcopy(online.image_encoder)
        self.imu_encoder = copy.deepcopy(online.imu_encoder)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    @torch.no_grad()
    def encode_clean(
        self,
        online: MultimodalBackbone,
        image_clean: torch.Tensor,
        imu_clean_normalized: torch.Tensor,
        *,
        image_fine: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """TI, TU; with ``image_fine`` also the image encoder's previous stage,
        which the same forward computes anyway (twice TI's resolution)."""
        image_coeff, _ = online.image_transform.analysis(image_clean)
        imu_coeff, _ = online.imu_transform.analysis(imu_clean_normalized)
        if not image_fine:
            return self.image_encoder(image_coeff), self.imu_encoder(imu_coeff)
        target_image, stages = self.image_encoder(image_coeff, return_stages=True)
        return target_image, self.imu_encoder(imu_coeff), stages[0]

    @torch.no_grad()
    def update(self, online: MultimodalBackbone, momentum: float) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError("EMA momentum must be in [0,1)")
        for target, source in (
            (self.image_encoder, online.image_encoder),
            (self.imu_encoder, online.imu_encoder),
        ):
            for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
                target_parameter.mul_(momentum).add_(source_parameter.detach(), alpha=1.0 - momentum)
            for target_buffer, source_buffer in zip(target.buffers(), source.buffers()):
                target_buffer.copy_(source_buffer)

