"""I-JEPA (Assran et al., CVPR 2023) as phase 1: multi-block masks, ViT predictor, EMA targets.

From one view, a context encoder sees a large block of tokens with the target
blocks cut out; a narrow ViT predictor, given the context tokens and a mask token
at each target position, predicts what the EMA target encoder makes of the
targets from the whole view. The loss is the distance between the prediction and
the LayerNorm'd target -- nothing else: no VICReg, no reconstruction, no
contrastive term. Collapse is kept off by the predictor and the slow EMA alone.

Image and IMU are one set of tokens: one ViT encodes both, so attention between
the two kinds of token is the fusion, and one predictor predicts the target blocks
of either from the context of both. Masks are drawn per modality (blocks on the
image grid, spans on the IMU sequence); the context is the union. The one departure
from the paper, chosen for this restoration task: ``context_input: noisy`` gives the
context encoder the corrupted pair while the target encoder reads the clean one.

Masks follow I-JEPA's MaskCollator: per batch one target size (scale 0.15-0.2,
aspect 0.75-1.5) and one context size (scale 0.85-1.0, square); per sample four
target blocks (they may overlap each other) and one context block with the
targets removed; every mask cut to the shortest in the batch so they stack. On a
1-D IMU grid a block is a span and the aspect ratio has no meaning.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
import torch.nn as nn

from ..data.normalize import ImuNormalizer
from .backbone import LatentBatch, MultimodalBackbone
from .teachers import EMATeachers
from .vit import INIT_STD, TOKEN_STRIDE, Block, gather_tokens, init_transformer, sincos_embedding

CONTEXT_INPUTS = ("noisy", "clean")
# I-JEPA's MaskCollator: tries per constraint level before one target may overlap the context.
_TIMEOUT = 20


def mask_seed(*context: object) -> int:
    raw = "|".join(str(item) for item in context).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") >> 1


def _uniform(low: float, high: float, generator: torch.Generator) -> float:
    return low + float(torch.rand((), generator=generator)) * (high - low)


def block_size(grid: tuple[int, ...], scale: tuple[float, float], aspect: tuple[float, float],
               generator: torch.Generator) -> tuple[int, ...]:
    """Sides of a block covering ``scale`` of the grid; never a whole axis (I-JEPA's h < height)."""
    keep = int(math.prod(grid) * _uniform(*scale, generator))
    if len(grid) == 1:
        sides = (keep,)
    else:
        ratio = _uniform(*aspect, generator)
        sides = (int(round(math.sqrt(keep * ratio))), int(round(math.sqrt(keep / ratio))))
    return tuple(max(1, min(side, size - 1 if size > 1 else 1)) for side, size in zip(sides, grid))


def _block(grid: tuple[int, ...], sides: tuple[int, ...], generator: torch.Generator,
           acceptable: list[torch.Tensor] | None = None, min_keep: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """(sorted token indices, complement) of one block placed at random, minus ``acceptable``'s holes.

    As I-JEPA: after ``_TIMEOUT`` placements that keep fewer than ``min_keep`` tokens,
    one constraint is dropped (the context may then overlap a target).
    """
    acceptable = acceptable or []
    dropped, timeout = 0, _TIMEOUT
    while True:
        mask = torch.zeros(grid, dtype=torch.bool)
        starts = [int(torch.randint(0, size - side + 1, (1,), generator=generator))
                  for side, size in zip(sides, grid)]
        mask[tuple(slice(start, start + side) for start, side in zip(starts, sides))] = True
        allowed = mask.clone()
        for region in acceptable[:len(acceptable) - dropped]:
            allowed &= region
        index = allowed.flatten().nonzero().flatten()
        if index.numel() >= max(min_keep, 1):
            return index, ~mask
        if dropped == len(acceptable):
            raise ValueError(f"A block of {sides} tokens cannot keep {min_keep} tokens on grid {grid}")
        timeout -= 1
        if timeout == 0:
            dropped, timeout = dropped + 1, _TIMEOUT


def multiblock_masks(grid: tuple[int, ...], batch: int, seed: int, *, targets: int,
                     target_scale: tuple[float, float], target_aspect: tuple[float, float],
                     context_scale: tuple[float, float], min_keep: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Context [B, Kc] and targets [B, M, Kt] token indices (row-major) for one batch."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    target_sides = block_size(grid, target_scale, target_aspect, generator)
    context_sides = block_size(grid, context_scale, (1.0, 1.0), generator)
    contexts, all_targets = [], []
    for _ in range(batch):
        blocks, complements = [], []
        for _ in range(targets):
            index, complement = _block(grid, target_sides, generator)
            blocks.append(index)
            complements.append(complement)
        context, _ = _block(grid, context_sides, generator, acceptable=complements, min_keep=min_keep)
        contexts.append(context)
        all_targets.append(blocks)
    keep_context = min(len(context) for context in contexts)
    keep_target = min(len(block) for blocks in all_targets for block in blocks)
    return (torch.stack([context[:keep_context] for context in contexts]),
            torch.stack([torch.stack([block[:keep_target] for block in blocks]) for blocks in all_targets]))


class IJEPAPredictor(nn.Module):
    """I-JEPA's VisionTransformerPredictor over both modalities: a narrow ViT reads the context
    tokens of the image and of the IMU, plus a mask token at each position of one target block.

    Each token carries its modality's fixed positions and a learned type embedding, as in the
    joint encoder, so an image target can be predicted from IMU context and back.
    """

    def __init__(self, dim: int, predictor_dim: int, *, depth: int, heads: int) -> None:
        super().__init__()
        self.embed = nn.Linear(dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, predictor_dim))
        self.image_type = nn.Parameter(torch.zeros(predictor_dim))
        self.imu_type = nn.Parameter(torch.zeros(predictor_dim))
        self.blocks = nn.ModuleList(Block(predictor_dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(predictor_dim, eps=1e-6)
        self.project = nn.Linear(predictor_dim, dim)
        init_transformer(self, self.blocks)
        for parameter in (self.mask_token, self.image_type, self.imu_type):
            nn.init.trunc_normal_(parameter, std=INIT_STD)

    def forward(self, context: dict[str, torch.Tensor], masks: dict[str, torch.Tensor],
                grids: dict[str, tuple[int, ...]]) -> tuple[torch.Tensor, torch.Tensor]:
        """context: image [B, Ki, D] and imu [B, Ku, D] at masks' image_context / imu_context;
        -> predictions at image_targets [B, M, Kt, D] and imu_targets [B, M', Kt', D].

        Each target block is predicted in its own pass over the shared context of both modalities.
        """
        width = self.embed.out_features
        positions = {name: sincos_embedding(width, grids[name], context[name]) for name in ("image", "imu")}
        types = {"image": self.image_type, "imu": self.imu_type}
        tokens = torch.cat([self.embed(context[name]) + positions[name][masks[f"{name}_context"]] + types[name]
                            for name in ("image", "imu")], dim=1)
        predictions = []
        for name in ("image", "imu"):
            target_index = masks[f"{name}_targets"]
            batch, blocks, length = target_index.shape
            queries = self.mask_token.to(tokens.dtype) + positions[name][target_index] + types[name]
            sequence = torch.cat((tokens.unsqueeze(1).expand(-1, blocks, -1, -1), queries), dim=2).flatten(0, 1)
            for block in self.blocks:
                sequence = block(sequence)
            predicted = self.project(self.norm(sequence[:, -length:]))
            predictions.append(predicted.reshape(batch, blocks, length, -1))
        return predictions[0], predictions[1]


class IJEPAPretrainingModel(nn.Module):
    """Phase 1 as I-JEPA: joint ViT backbone (online = context encoder), one predictor, EMA teacher."""

    objective = "ijepa"
    reconstructs = False

    def __init__(self, backbone: MultimodalBackbone, normalizer: ImuNormalizer | None = None, *,
                 predictor_dim: int, predictor_depth: int, predictor_heads: int,
                 masking: dict[str, Any], context_input: str) -> None:
        super().__init__()
        if backbone.encoder_type != "vit":
            raise ValueError("I-JEPA needs the ViT backbone (model.encoder_type: vit)")
        if context_input not in CONTEXT_INPUTS:
            raise ValueError(f"context_input must be one of {CONTEXT_INPUTS}")
        self.backbone = backbone
        self.normalizer = normalizer or ImuNormalizer()
        self.predictor = IJEPAPredictor(backbone.joint_encoder.out_channels, predictor_dim,
                                        depth=predictor_depth, heads=predictor_heads)
        self.teachers = EMATeachers(backbone)
        # targets, target_scale, target_aspect, context_scale, image_min_keep, imu_min_keep
        self.masking = dict(masking)
        self.context_input = context_input
        # What the CLI and phase 2 ask of every phase-1 model; I-JEPA has none of them.
        self.decoders = None
        self.degradation_head = None
        self.degradation_condition = False

    @torch.no_grad()
    def calibrate_vit_inputs(self, image_noisy: torch.Tensor, imu_noisy_phys: torch.Tensor,
                             image_clean: torch.Tensor, imu_clean_phys: torch.Tensor) -> dict[str, list[float]]:
        """model.vit_input_standardize: set the ViT's per-channel input gains on this batch
        (JointCoefficientViT.calibrate_inputs), then give the EMA teacher the same ones.

        Noisy and clean together: the context encoder reads the noisy pair and the teacher, with
        the same gains, the clean one. Fresh runs only; a checkpoint carries its gains."""
        backbone = self.backbone
        image_coeff, _ = backbone.image_transform.analysis(torch.cat((image_noisy, image_clean)))
        imu_coeff, _ = backbone.imu_transform.analysis(self.normalizer.normalize(torch.cat((imu_noisy_phys,
                                                                                           imu_clean_phys))))
        measured = backbone.joint_encoder.calibrate_inputs(image_coeff, imu_coeff)
        self.teachers.joint_encoder.load_state_dict(backbone.joint_encoder.state_dict())
        return measured

    def online_parameters(self):
        return [*self.backbone.parameters(), *self.predictor.parameters()]

    def token_grids(self, image_shape: tuple[int, ...], imu_length: int) -> tuple[tuple[int, ...], tuple[int]]:
        """Token grids of a [B, C, H, W] frame and an IMU window: 16 px / 16 samples per token."""
        return (image_shape[-2] // TOKEN_STRIDE, image_shape[-1] // TOKEN_STRIDE), (imu_length // TOKEN_STRIDE,)

    def sample_masks(self, image_grid: tuple[int, ...], imu_grid: tuple[int], batch: int,
                     *seed_context: object) -> dict[str, torch.Tensor]:
        """Context and target indices for both modalities, reproducible from ``seed_context``."""
        settings = {key: self.masking[key] for key in ("targets", "target_scale", "target_aspect", "context_scale")}
        image_context, image_targets = multiblock_masks(
            image_grid, batch, mask_seed(*seed_context, "image"), min_keep=self.masking["image_min_keep"],
            **settings)
        imu_context, imu_targets = multiblock_masks(
            imu_grid, batch, mask_seed(*seed_context, "imu"), min_keep=self.masking["imu_min_keep"], **settings)
        return {"image_context": image_context, "image_targets": image_targets,
                "imu_context": imu_context, "imu_targets": imu_targets}

    def encode_online(self, image: torch.Tensor, imu_phys: torch.Tensor, image_time: torch.Tensor,
                      imu_times: torch.Tensor) -> LatentBatch:
        return self.backbone.encode_online(image, self.normalizer.normalize(imu_phys), image_time, imu_times)

    @torch.no_grad()
    def targets(self, image_clean: torch.Tensor, imu_clean_phys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The EMA target encoder on the whole clean pair: dense TI, TU."""
        return self.teachers.encode_clean(self.backbone, image_clean, self.normalizer.normalize(imu_clean_phys))

    def predict(self, image: torch.Tensor, imu_phys: torch.Tensor,
                masks: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Context-encode the given pair -- both modalities' context tokens in one pass -- and
        predict every target block: [B, M, Kt, D] for the image and for the IMU."""
        backbone = self.backbone
        encoder = backbone.joint_encoder
        image_coeff, _ = backbone.image_transform.analysis(image)
        imu_coeff, _ = backbone.imu_transform.analysis(self.normalizer.normalize(imu_phys))
        image, imu = encoder(image_coeff, imu_coeff, keep_image=masks["image_context"], keep_imu=masks["imu_context"])
        grids = {"image": encoder.token_grid(tuple(image_coeff.shape)), "imu": encoder.token_grid(tuple(imu_coeff.shape))}
        return self.predictor({"image": image, "imu": imu}, masks, grids)

    @staticmethod
    def target_tokens(dense: torch.Tensor, target_index: torch.Tensor) -> torch.Tensor:
        """I-JEPA's h = layer_norm(target_encoder(x)) at the target blocks: [B, M, Kt, D]."""
        tokens = dense.flatten(2).transpose(1, 2)
        tokens = nn.functional.layer_norm(tokens, (tokens.shape[-1],))
        return gather_tokens(tokens, target_index)
