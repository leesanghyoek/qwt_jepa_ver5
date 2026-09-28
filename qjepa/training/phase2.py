"""Phase 2 trainer: frozen backbone and fresh latent-only decoders."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from ..distributed import gather_shares, rank_and_world
from ..models.pipeline import RestorationSystem
from ..execution import RestorationForward, execution_metadata, parallel_forward
from .checkpoints import configuration_hash, rng_state, state_dict_hash
from ..models.decoders import PIXEL_IMAGE_DECODERS
from .losses import color_edge_split_loss, invisible_detail_fraction, phase2_reconstruction_loss
from .perceptual import PerceptualLoss
from .phase1 import _finite_gradients, _to_device
from .schedules import warmup_cosine_lr


def _grad_scaler(enabled: bool):
    """torch.amp.GradScaler where it exists (torch >= 2.3), else the cuda one."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


class Phase2Trainer:
    def __init__(
        self,
        system: RestorationSystem,
        config: dict[str, Any],
        device: torch.device,
        parent_checkpoint: str,
        manifest_hash: str = "unknown",
    ) -> None:
        self.system = system.to(device)
        self.system.freeze_backbone()
        self.config = config
        self.phase = config["phase2"]
        self.device = device
        self.forward_model, self.device_ids = parallel_forward(
            RestorationForward(self.system), device, config["runtime"].get("gpu_count", "auto")
        )
        self.rank, self.world = rank_and_world()
        if self.world > 1:
            # One process per GPU (qjepa.distributed). The backbone is frozen and the
            # decoders hold no running statistics: no buffers to broadcast.
            self.forward_model = nn.parallel.DistributedDataParallel(
                self.forward_model, device_ids=[torch.cuda.current_device()] if self.device.type == "cuda" else None,
                broadcast_buffers=False)
        self.parent_checkpoint = parent_checkpoint
        self.manifest_hash = manifest_hash
        self.successful_updates = 0
        self.parameters = list(self.system.decoders.parameters())
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=self.phase["learning_rate"],
            weight_decay=self.phase["weight_decay"],
        )
        # Frozen VGG16 feature loss; held by the trainer, not the system, so it never
        # reaches a checkpoint. Absent from configs written before the key existed.
        self.perceptual_weight = float(self.phase.get("perceptual_weight", 0.0))
        self.perceptual = (PerceptualLoss(crop=int(self.phase.get("perceptual_crop", 0))).to(device)
                           if self.perceptual_weight > 0 else None)
        # Mixed precision: convs and matmuls in fp16 on the T4's tensor cores, weights,
        # losses and optimizer in fp32. CUDA only; on CPU the same config runs in fp32.
        self.amp = self.phase.get("precision", "fp32") == "amp_fp16" and torch.device(device).type == "cuda"
        getattr(self.forward_model, "module", self.forward_model).amp = self.amp
        self.scaler = _grad_scaler(self.amp)
        self.frozen_backbone_hash = state_dict_hash(self.system.backbone)
        self.frozen_normalizer_hash = state_dict_hash(self.system.normalizer)
        self.decoder_initialization_hash = state_dict_hash(self.system.decoders)

    def _set_lr(self) -> float:
        lr = warmup_cosine_lr(
            self.successful_updates,
            self.phase["max_successful_updates"],
            self.phase["warmup_updates"],
            self.phase["learning_rate"],
            self.phase["minimum_lr"],
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    @property
    def evaluation_model(self):
        """The forward for validation: under DDP only rank 0 validates, on its own GPU."""
        return self.forward_model.module if self.world > 1 else self.forward_model

    def assert_backbone_frozen(self) -> None:
        if state_dict_hash(self.system.backbone) != self.frozen_backbone_hash:
            raise RuntimeError("Frozen backbone changed during phase 2")
        if state_dict_hash(self.system.normalizer) != self.frozen_normalizer_hash:
            raise RuntimeError("Frozen IMU normalizer changed during phase 2")

    def step(self, raw_batches: dict[str, Any] | list[dict[str, Any]]) -> dict[str, float | bool | str]:
        microbatches = raw_batches if isinstance(raw_batches, list) else [raw_batches]
        expected = int(self.phase["gradient_accumulation"])
        if len(microbatches) != expected:
            raise ValueError(f"Phase 2 expects {expected} microbatches per update, got {len(microbatches)}")
        self.system.train(True)
        self.optimizer.zero_grad(set_to_none=True)
        lr = self._set_lr()
        totals: dict[str, float] = {"loss": 0.0, "image_l1": 0.0, "imu_accel_smooth_l1": 0.0,
                                    "imu_gyro_smooth_l1": 0.0}
        for index, raw_batch in enumerate(microbatches):
            batch = _to_device(raw_batch, self.device)
            # DDP averages gradients only on the last microbatch of the update.
            accumulating = self.world > 1 and index < expected - 1
            with self.forward_model.no_sync() if accumulating else nullcontext():
                restored = self.forward_model(
                    batch["image_noisy"], batch["imu_noisy_phys"], batch["image_time"], batch["imu_times"]
                )
                if self.world > 1:
                    # This rank ran the model on its share; the loss sees the whole microbatch.
                    restored = {key: gather_shares(value, keep_graph=True) for key, value in restored.items()}
                    batch = {key: gather_shares(value, keep_graph=False) if isinstance(value, torch.Tensor)
                             else value for key, value in batch.items()}
                clean_imu = self.system.normalizer.normalize(batch["imu_clean_phys"])
                detail_weight = float(self.phase.get("reconstruction_detail_weight", 0.0))
                image_transform = self.system.backbone.image_transform
                image_target = imu_target = None
                if detail_weight > 0:
                    with torch.no_grad():
                        image_target, _ = image_transform.analysis(batch["image_clean"])
                        imu_target, _ = self.system.backbone.imu_transform.analysis(clean_imu)
                # The decoder's 48 channels are 4x redundant; synthesis averages the
                # trees, so a detail term scored on them can be lowered in the null
                # space with the image unchanged. Re-analysing the restored image
                # scores only what reaches the pixels. Absent from configs written
                # before the key existed, which keep scoring the decoder output.
                scores_image = self.phase.get("image_detail_source", "decoder_coefficients") == "restored_image"
                if self.system.decoders.image_decoder in PIXEL_IMAGE_DECODERS:
                    # Already analysis(image), computed with the graph in decode.
                    visible_coefficients = restored["image_coefficients"]
                else:
                    with torch.set_grad_enabled(scores_image):
                        visible_coefficients, _ = image_transform.analysis(restored["image"])
                image_coefficients = visible_coefficients if scores_image else restored["image_coefficients"]
                loss, parts = phase2_reconstruction_loss(
                    restored["image"],
                    batch["image_clean"],
                    restored["imu_normalized"],
                    clean_imu,
                    beta=self.phase["smooth_l1_beta"],
                    image_coefficients=image_coefficients,
                    image_coefficient_target=image_target,
                    imu_coefficients=restored["imu_coefficients"],
                    imu_coefficient_target=imu_target,
                    detail_weight=detail_weight,
                    variation_weight=float(self.phase.get("imu_variation_weight", 0.0)),
                    detail_energy_weight=float(self.phase.get("detail_energy_weight", 0.0)),
                    # Absent from configs written before the term existed, which must
                    # keep meaning what they meant when they were trained.
                    image_detail_loss=str(self.phase.get("image_detail_loss", "coefficient")),
                )
                if "image_detail" in restored:
                    split_loss, split_parts = color_edge_split_loss(
                        restored["image_color_base"], restored["image_illumination"],
                        restored["image_detail"], restored["image"], batch["image_clean"],
                        color_scale=int(self.phase["split_color_scale"]),
                        illumination_scale=int(self.phase["split_illumination_scale"]),
                        color_weight=float(self.phase["split_color_weight"]),
                        edge_weight=float(self.phase["split_edge_weight"]),
                        gradient_weight=float(self.phase["split_gradient_weight"]),
                        stats_weight=float(self.phase.get("split_color_stats_weight", 0.0)),
                        fft_weight=float(self.phase.get("split_edge_fft_weight", 0.0)),
                        aux_details={int(key[len("image_detail_aux"):]): value for key, value in restored.items()
                                     if key.startswith("image_detail_aux")},
                        aux_weight=float(self.phase.get("split_edge_aux_weight", 0.0)),
                        detail_stage1=restored.get("image_detail_stage1"),
                        stage1_weight=float(self.phase.get("split_edge_stage1_weight", 0.0)),
                        smooth_weight=float(self.phase.get("split_edge_smooth_weight", 0.0)),
                    )
                    loss = loss + split_loss
                    parts.update(split_parts)
                if self.perceptual is not None:
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                        perceptual = self.perceptual(restored["image"], batch["image_clean"])
                    loss = loss + self.perceptual_weight * perceptual
                    parts["image_perceptual"] = perceptual
                with torch.no_grad():
                    parts["image_detail_invisible_fraction"] = invisible_detail_fraction(
                        restored["image_coefficients"], visible_coefficients)
                loss = self.phase["reconstruction_loss_weight"] * loss
                # x world: only this rank's share carries a graph, and DDP averages over ranks.
                self.scaler.scale(loss * self.world / expected).backward()
                totals["loss"] += float(loss.detach()) / expected
                for key, value in parts.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach()) / expected
        self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.phase["gradient_clip_norm"])
        amp_report: dict[str, float | bool] = {}
        if self.amp:
            # An fp16 overflow is routine while the scale settles: the scaler skips
            # that optimizer step and lowers the scale. The data is still consumed,
            # so the update counts; the log says it was an overflow.
            overflow = not bool(torch.isfinite(gradient_norm))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            amp_report = {"amp_overflow": overflow, "amp_scale": float(self.scaler.get_scale())}
        else:
            if not torch.isfinite(gradient_norm) or not _finite_gradients(self.parameters):
                self.optimizer.zero_grad(set_to_none=True)
                return {"skipped": True, "reason": "non_finite_gradient", **totals}
            self.optimizer.step()
        self.successful_updates += 1
        return {
            "skipped": False,
            **totals,
            **amp_report,
            "gradient_norm": float(gradient_norm),
            "learning_rate": lr,
            "successful_updates": self.successful_updates,
        }

    def checkpoint_payload(self, config: dict[str, Any]) -> dict[str, Any]:
        self.assert_backbone_frozen()
        return {
            "metadata": {
                "pipeline_version": 3,
                "phase": "latent_decoder_train",
                "decoder_input": self.phase["decoder_input"],
                "image_decoder": self.system.decoders.image_decoder,
                "output_coefficients": self.phase["output_coefficients"],
                "successful_updates": self.successful_updates,
                "data_microbatches_consumed": self.successful_updates * self.phase["gradient_accumulation"],
                "parent_phase1_checkpoint": self.parent_checkpoint,
                "frozen_backbone_hash": self.frozen_backbone_hash,
                "frozen_normalizer_hash": self.frozen_normalizer_hash,
                "decoder_initialization_hash": self.decoder_initialization_hash,
                "decoder_current_hash": state_dict_hash(self.system.decoders),
                "configuration_hash": configuration_hash(config, "phase2"),
                "manifest_hash": self.manifest_hash,
                "execution": execution_metadata(self.device, self.device_ids, self.world),
            },
            "system": self.system.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "successful_updates": self.successful_updates,
            "config": config,
            "rng": rng_state(),
        }
