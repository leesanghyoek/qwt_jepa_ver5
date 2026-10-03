"""Phase-specific wrappers that make forbidden data paths impossible by API."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.normalize import ImuNormalizer
from .backbone import LatentBatch, MultimodalBackbone
from .decoders import PIXEL_IMAGE_DECODERS, LatentDecoders
from .predictors import (PREDICTOR_TYPES, DegradationHead, LatentPredictor, SpatialPredictor, image_tokens,
                         imu_tokens)
from .teachers import EMATeachers


class LatentPretrainingModel(nn.Module):
    """Phase 1 owns backbone, predictors and teachers.

    It may also own an auxiliary decoder. That decoder never ships: phase 2
    builds its own from scratch. Its only job is to keep the latent decodable,
    because every other phase-1 term is self-referential and can be satisfied by
    a representation that has thrown the signal away.
    """

    def __init__(
        self,
        backbone: MultimodalBackbone | None = None,
        normalizer: ImuNormalizer | None = None,
        predictor_hidden: int = 256,
        decoders: LatentDecoders | None = None,
        *,
        predictor_type: str = "token",
        predictor_kernel: int = 3,
        predictor_layers: int = 2,
        fine_scale: bool = False,
        finer_scale: bool = False,
        degradation_outputs: int = 0,
        degradation_condition: bool = False,
    ) -> None:
        super().__init__()
        if predictor_type not in PREDICTOR_TYPES:
            raise ValueError(f"predictor_type must be one of {PREDICTOR_TYPES}")
        if fine_scale and predictor_type != "spatial":
            raise ValueError("The fine-scale JEPA target needs the spatial predictor")
        if finer_scale and not fine_scale:
            raise ValueError("The finer JEPA target grows from the fine one: enable fine_scale too")
        if degradation_condition and (predictor_type != "spatial" or not degradation_outputs):
            raise ValueError("The degradation condition needs the spatial predictor and the degradation head")
        self.backbone = backbone or MultimodalBackbone()
        self.normalizer = normalizer or ImuNormalizer()
        embedding_dim = self.backbone.image_encoder.out_channels
        self.predictor_type = predictor_type
        self.fine_scale = fine_scale
        self.finer_scale = finer_scale
        if predictor_type == "token":
            # Layer names unchanged: checkpoints up to p9 load strictly.
            self.image_predictor = LatentPredictor(embedding_dim, predictor_hidden)
            self.imu_predictor = LatentPredictor(embedding_dim, predictor_hidden)
        else:
            # The fine target is the stage before the last: skip_channels[0].
            fine_channels = self.backbone.image_encoder.skip_channels[0] if fine_scale else 0
            # And the stage before it for the finer target: skip_channels[1].
            finer_channels = self.backbone.image_encoder.skip_channels[1] if finer_scale else 0
            self.image_predictor = SpatialPredictor(
                embedding_dim, predictor_hidden, spatial_dims=2, kernel=predictor_kernel,
                layers=predictor_layers, fine_channels=fine_channels, finer_channels=finer_channels,
                condition_dim=degradation_outputs if degradation_condition else 0,
            )
            self.imu_predictor = SpatialPredictor(
                embedding_dim, predictor_hidden, spatial_dims=1, kernel=predictor_kernel,
                layers=predictor_layers,
            )
        # None: no parameters, so phase-1 checkpoints from before the head load as they are.
        self.degradation_head = DegradationHead(embedding_dim, degradation_outputs) if degradation_outputs else None
        self.degradation_condition = degradation_condition
        self.teachers = EMATeachers(self.backbone)
        self.decoders = decoders
        if decoders is not None and decoders.imu_refiner is not None:
            # build_decoders reads phase2.imu_refiner_*, so the anchor carries the phase-2
            # IMU refiner too; phase 1 never runs it. Frozen, DDP stops waiting for its
            # gradients (no gradient ever came, so one-process training is unchanged), and
            # the parameters stay in the state dict so older phase-1 checkpoints still load.
            decoders.imu_refiner.requires_grad_(False)
        self.register_buffer("decoder_forward_calls", torch.zeros((), dtype=torch.long))

    @property
    def reconstructs(self) -> bool:
        return self.decoders is not None

    def online_parameters(self):
        parameters = list(self.backbone.parameters())
        parameters += list(self.image_predictor.parameters())
        parameters += list(self.imu_predictor.parameters())
        if self.degradation_head is not None:
            parameters += list(self.degradation_head.parameters())
        if self.decoders is not None:
            parameters += list(self.decoders.parameters())
        return parameters

    def reconstruct(self, latent: LatentBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict clean coefficients from the noisy latent, as phase 2 will."""
        if self.decoders is None:
            raise ValueError("Phase-1 reconstruction requires decoder_enabled")
        return self.decoders(latent.ZI, latent.ZU)

    def encode_online(
        self, image: torch.Tensor, imu_phys: torch.Tensor, image_time: torch.Tensor, imu_times: torch.Tensor
    ) -> LatentBatch:
        return self.backbone.encode_online(
            image, self.normalizer.normalize(imu_phys), image_time, imu_times
        )

    def predictions(
        self,
        latent: LatentBatch,
        image_mask: torch.Tensor | None = None,
        imu_mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Predicted tokens (image, IMU) and, with ``fine_scale``, the fine image map.

        Masks [B, N] hide tokens from the predictor; only the spatial predictor
        can take them, as a token-wise one would have nothing left to predict from.
        ``condition`` is the degradation estimate the image predictor is conditioned on.
        """
        if self.predictor_type == "token":
            if image_mask is not None or imu_mask is not None:
                raise ValueError("Token masking needs the spatial predictor")
            return (self.image_predictor(image_tokens(latent.ZI)),
                    self.imu_predictor(imu_tokens(latent.ZU)), None)
        image, fine = self.image_predictor(latent.ZI, image_mask, condition)
        imu, _ = self.imu_predictor(latent.ZU, imu_mask)
        return image, imu, fine

    @torch.no_grad()
    def targets(self, image_clean: torch.Tensor, imu_clean_phys: torch.Tensor, *, fine: bool = False,
                finer: bool = False):
        """Teacher TI, TU from the clean pair; with ``fine``, also the fine image target."""
        return self.teachers.encode_clean(
            self.backbone, image_clean, self.normalizer.normalize(imu_clean_phys), image_fine=fine,
            image_finer=finer,
        )


@dataclass
class RestoredBatch:
    image: torch.Tensor
    imu_normalized: torch.Tensor
    imu_physical: torch.Tensor
    image_coefficients: torch.Tensor
    imu_coefficients: torch.Tensor
    # Intermediate pieces some image decoders expose for their own loss terms.
    image_parts: dict[str, torch.Tensor] | None = None


class RestorationSystem(nn.Module):
    """Phase 2/inference: frozen backbone followed by latent-only decoders.

    With ``latent_predictor`` (phase2.decoder_predictor_input) the phase-1 image
    predictor -- the part of JEPA trained to turn the noisy latent into the clean
    teacher's -- stays, frozen, and its prediction joins ZI through the decoders'
    zero-initialised ``predictor_merge``. ``degradation_head`` comes along when the
    predictor is conditioned on its estimate.
    """

    def __init__(
        self,
        backbone: MultimodalBackbone,
        normalizer: ImuNormalizer,
        decoders: LatentDecoders | None = None,
        latent_predictor: SpatialPredictor | None = None,
        degradation_head: DegradationHead | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.normalizer = normalizer
        self.decoders = decoders or LatentDecoders()
        if (latent_predictor is None) != (self.decoders.predictor_merge is None):
            raise ValueError("A latent predictor and the decoders' predictor_merge come together")
        if latent_predictor is not None and (latent_predictor.condition is None) != (degradation_head is None):
            raise ValueError("A conditioned predictor needs its degradation head, and only then")
        if (backbone.image_transform.image_input == "luminance"
                and self.decoders.image_decoder not in PIXEL_IMAGE_DECODERS):
            # Synthesis of luminance coefficients gives Y back, never RGB.
            raise ValueError("A luminance backbone needs a pixel image decoder (resnet_pixel or split_color_edge)")
        self.latent_predictor = latent_predictor
        self.degradation_head = degradation_head
        self.backbone_trainable = False
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False).eval()
        self.normalizer.requires_grad_(False).eval()
        for module in (self.latent_predictor, self.degradation_head):
            if module is not None:
                module.requires_grad_(False).eval()
        self.backbone_trainable = False

    def unfreeze_backbone(self) -> None:
        """LP-FT's second stage: the encoder trains too. It stays in eval mode
        (no batch statistics in it); the IMU normalizer and predictor stay frozen."""
        self.backbone.requires_grad_(True)
        self.backbone_trainable = True

    def train(self, mode: bool = True):
        self.training = mode
        self.backbone.eval()
        self.normalizer.eval()
        for module in (self.latent_predictor, self.degradation_head):
            if module is not None:
                module.eval()
        self.decoders.train(mode)
        return self

    def encode(
        self, image_noisy: torch.Tensor, imu_noisy_phys: torch.Tensor, image_time: torch.Tensor, imu_times: torch.Tensor
    ) -> LatentBatch:
        self.backbone.eval()
        with torch.set_grad_enabled(self.backbone_trainable and torch.is_grad_enabled()):
            return self.backbone.encode_online(
                image_noisy,
                self.normalizer.normalize(imu_noisy_phys),
                image_time,
                imu_times,
                with_skips=self.decoders.uses_skips,
            )

    def image_latent(self, latent: LatentBatch) -> torch.Tensor:
        """ZI, plus the frozen predictor's clean-latent prediction when there is one."""
        if self.latent_predictor is None:
            return latent.ZI
        condition = self.degradation_head(latent.ZI) if self.degradation_head is not None else None
        tokens, _ = self.latent_predictor(latent.ZI, None, condition)
        # The JEPA loss compares LayerNorm'd tokens, so that is the scale the prediction means.
        tokens = F.layer_norm(tokens, (tokens.shape[-1],))
        predicted = tokens.transpose(1, 2).reshape(latent.ZI.shape)
        return latent.ZI + self.decoders.predictor_merge(predicted)

    def decode(self, latent: LatentBatch) -> RestoredBatch:
        transform = self.backbone.image_transform
        parts = None
        if self.decoders.image_decoder in PIXEL_IMAGE_DECODERS:
            # The QWT reconstructs perfectly, so this IS the blurry input image;
            # decode(latent) keeps its signature for the ZI-ablation tools. A luminance
            # QWT holds no colour, so the backbone kept the RGB frame for the decoder.
            if latent.image_rgb is not None:
                blurry = latent.image_rgb
            else:
                blurry = transform.synthesis(latent.image_coefficients, latent.image_layout)
            zi = self.image_latent(latent)
            if getattr(self.decoders.image, "uses_stages", False):
                # The JEPA encoder's finer stages (image_skips: 1/8, 1/4, 1/2 of the frame).
                image = self.decoders.image(zi, blurry, stages=latent.image_skips)
            else:
                image = self.decoders.image(zi, blurry)
            if isinstance(image, tuple):
                image, parts = image
            # Coefficients OF the image, so nothing downstream can score energy
            # that synthesis would throw away.
            image_coeff, _ = transform.analysis(image)
            imu_coeff = self.decoders.imu(latent.ZU, latent.imu_coefficients, latent.imu_skips)
        else:
            image_coeff, imu_coeff = self.decoders(
                latent.ZI, latent.ZU, latent.image_coefficients, latent.imu_coefficients,
                latent.image_skips, latent.imu_skips,
            )
            image = transform.synthesis(image_coeff, latent.image_layout)
        imu_norm = self.backbone.imu_transform.synthesis(imu_coeff, latent.imu_layout)
        if self.decoders.imu_refiner is not None:
            noisy = self.backbone.imu_transform.synthesis(latent.imu_coefficients, latent.imu_layout)
            imu_norm = self.decoders.imu_refiner(imu_norm, noisy)
            # Coefficients OF the smoothed signal: the Haar detail terms score what is output.
            imu_coeff, _ = self.backbone.imu_transform.analysis(imu_norm)
        return RestoredBatch(
            image=image,
            imu_normalized=imu_norm,
            imu_physical=self.normalizer.denormalize(imu_norm),
            image_coefficients=image_coeff,
            imu_coefficients=imu_coeff,
            image_parts=parts,
        )

    def forward(
        self, image_noisy: torch.Tensor, imu_noisy_phys: torch.Tensor, image_time: torch.Tensor, imu_times: torch.Tensor
    ) -> RestoredBatch:
        return self.decode(self.encode(image_noisy, imu_noisy_phys, image_time, imu_times))
