"""Online QWT-JEPA backbone. This module contains no decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..transforms import (
    DEFAULT_QWT_BACKEND,
    HaarTransform1D,
    QuaternionWaveletTransform2D,
    TransformLayout,
)
from .encoders import DEFAULT_CHANNELS, DenseCoefficientEncoder
from .fusion import SharedGatedFusion, build_time_metadata
from .vit import JointCoefficientViT

ENCODER_TYPES = ("cnn", "vit")


@dataclass
class LatentBatch:
    FI: torch.Tensor
    FU: torch.Tensor
    ZI: torch.Tensor
    ZU: torch.Tensor
    image_layout: TransformLayout
    imu_layout: TransformLayout
    # He so cua chinh dau vao. Phase 2 dung chung lam nen cho residual; phase 1
    # bo qua, vi neo phai ep thong tin VAO latent chu khong duoc lay duong vong.
    image_coefficients: torch.Tensor | None = None
    imu_coefficients: torch.Tensor | None = None
    # Feature trung gian cua encoder. Chi duoc dien khi phase2.encoder_skips bat;
    # neo phase 1 luon bo qua, vi neo phai ep thong tin VAO latent chu khong duoc
    # thoa man bang mot duong vong quanh no.
    image_skips: tuple[torch.Tensor, ...] | None = None
    imu_skips: tuple[torch.Tensor, ...] | None = None
    # Anh RGB dau vao, chi giu khi QWT doc kenh sang (model.image_input: luminance):
    # he so khi do khong con mau, nen decoder phase 2 phai doc mau tu chinh anh.
    image_rgb: torch.Tensor | None = None


class MultimodalBackbone(nn.Module):
    def __init__(
        self,
        channels: tuple[int, int, int, int] = DEFAULT_CHANNELS,
        embedding_dim: int = 128,
        fusion_hidden: int = 256,
        imu_summary_bins: int = 4,
        time_metadata_dim: int = 3,
        gate_bias: float = -2.0,
        groups: int = 8,
        image_transform: str = DEFAULT_QWT_BACKEND,
        encoder_norm: str = "group",
        image_input: str = "rgb",
        encoder_type: str = "cnn",
        vit_depth: int = 6,
        vit_heads: int = 4,
    ) -> None:
        super().__init__()
        if channels[-1] != embedding_dim:
            raise ValueError("The final encoder width must equal embedding_dim")
        if encoder_type not in ENCODER_TYPES:
            raise ValueError(f"encoder_type must be one of {ENCODER_TYPES}")
        self.encoder_type = encoder_type
        # "luminance": QWT, encoder va teacher chi thay kenh sang Y -- JEPA hoc duong net,
        # mau di duong rieng o phase 2. Encoder dau vao 16 kenh thay vi 48.
        self.image_transform = QuaternionWaveletTransform2D(backend=image_transform, image_input=image_input)
        self.imu_transform = HaarTransform1D(channels=6)
        if encoder_type == "vit":
            # I-JEPA (phase1.objective: ijepa): MOT ViT chung cho token anh va token IMU, cung luoi
            # token voi CNN. Attention giua hai loai token chinh la fusion: ZI = FI, ZU = FU.
            self.joint_encoder = JointCoefficientViT(
                self.image_transform.coeff_channels, self.imu_transform.coeff_channels, embedding_dim,
                depth=vit_depth, heads=vit_heads)
            self.fusion = None
            return
        self.image_encoder = DenseCoefficientEncoder(
            self.image_transform.coeff_channels, channels, dim=2, groups=groups, norm=encoder_norm
        )
        self.imu_encoder = DenseCoefficientEncoder(
            self.imu_transform.coeff_channels, channels, dim=1, groups=groups, norm=encoder_norm
        )
        self.fusion = SharedGatedFusion(
            embedding_dim,
            fusion_hidden,
            imu_bins=imu_summary_bins,
            metadata_dim=time_metadata_dim,
            gate_bias=gate_bias,
        )

    def encode_image_dense(self, image: torch.Tensor) -> torch.Tensor:
        coefficients, _ = self.image_transform.analysis(image)
        return self.image_encoder(coefficients)

    def encode_imu_dense(self, imu_normalized: torch.Tensor) -> torch.Tensor:
        coefficients, _ = self.imu_transform.analysis(imu_normalized)
        return self.imu_encoder(coefficients)

    def encode_online(
        self,
        image: torch.Tensor,
        imu_normalized: torch.Tensor,
        image_time: torch.Tensor,
        imu_times: torch.Tensor,
        *,
        with_skips: bool = False,
    ) -> LatentBatch:
        image_coeff, image_layout = self.image_transform.analysis(image)
        imu_coeff, imu_layout = self.imu_transform.analysis(imu_normalized)
        image_skips = imu_skips = None
        if self.encoder_type == "vit":
            # with_skips: the joint ViT raises -- one resolution, no stages.
            fi, fu = self.joint_encoder(image_coeff, imu_coeff, return_stages=with_skips)
            image_rgb = image if self.image_transform.image_input == "luminance" else None
            return LatentBatch(fi, fu, fi, fu, image_layout, imu_layout, image_coeff, imu_coeff,
                               None, None, image_rgb)
        if with_skips:
            fi, image_skips = self.image_encoder(image_coeff, return_stages=True)
            fu, imu_skips = self.imu_encoder(imu_coeff, return_stages=True)
        else:
            fi = self.image_encoder(image_coeff)
            fu = self.imu_encoder(imu_coeff)
        zi, zu = self.fusion(fi, fu, build_time_metadata(image_time, imu_times))
        image_rgb = image if self.image_transform.image_input == "luminance" else None
        return LatentBatch(fi, fu, zi, zu, image_layout, imu_layout, image_coeff, imu_coeff,
                           image_skips, imu_skips, image_rgb)
