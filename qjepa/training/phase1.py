"""Phase 1 trainer: noisy-to-clean JEPA latent learning, never reconstruction."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from ..models.masking import sample_seeds
from ..models.pipeline import LatentPretrainingModel
from ..execution import Phase1Forward, execution_metadata, parallel_forward
from .checkpoints import configuration_hash, rng_state, state_dict_hash
from .losses import (
    dense_positions,
    jepa_coarse_loss,
    jepa_diagnostics,
    jepa_fine_loss,
    jepa_latent_loss,
    phase1_reconstruction_loss,
    variance_covariance_loss,
)
from .schedules import ema_momentum, warmup_cosine_lr
from .sensitivity import (
    corruption_direction,
    detail_direction,
    directional_gain,
    make_probe,
    sensitivity_ratio_loss,
    sensitivity_weight,
)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _position_indices(total: int, maximum: int, seed: int, *context: object) -> torch.Tensor:
    if maximum >= total:
        return torch.arange(total)
    raw = "|".join(str(item) for item in (seed, *context)).encode()
    derived = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") >> 1
    generator = torch.Generator(device="cpu").manual_seed(derived)
    return torch.randperm(total, generator=generator)[:maximum].sort().values


def phase1_masking(phase: dict[str, Any]) -> dict[str, Any] | None:
    """The Phase1Forward masking settings, or None when nothing is masked."""
    image_ratio = float(phase.get("image_mask_ratio", 0.0))
    imu_ratio = float(phase.get("imu_mask_ratio", 0.0))
    if image_ratio <= 0 and imu_ratio <= 0:
        return None
    return {"image_ratio": image_ratio, "image_block": tuple(phase.get("image_mask_block", (2, 4))),
            "imu_ratio": imu_ratio, "imu_span": tuple(phase.get("imu_mask_span", (1, 2)))}


def jepa_report_terms(features: dict[str, torch.Tensor]) -> dict[str, float]:
    """Normalized JEPA, cosine and the visible/masked split, for the log only."""
    report: dict[str, float] = {}
    for name, prediction, target, mask in (
        ("image", features["prediction_i"], features["target_i"], features.get("image_mask")),
        ("imu", features["prediction_u"], features["target_u"], features.get("imu_mask")),
    ):
        for key, value in jepa_diagnostics(prediction, target, mask).items():
            report[f"jepa_{name}_{key}"] = float(value)
    if "prediction_i_fine" in features:
        fine_tokens = features["prediction_i_fine"].flatten(2).transpose(1, 2)
        diagnostics = jepa_diagnostics(fine_tokens, features["target_i_fine"])
        report["jepa_image_fine_normalized"] = float(diagnostics["normalized"])
    return report


def _finite_gradients(parameters: list[torch.nn.Parameter]) -> bool:
    # One device sync for all gradients, not one per parameter tensor (hundreds).
    checks = [torch.isfinite(parameter.grad).all() for parameter in parameters if parameter.grad is not None]
    return not checks or bool(torch.stack(checks).all())


class Phase1Trainer:
    LOSS_KEYS = {
        "loss",
        "jepa",
        "jepa_image",
        "jepa_imu",
        "jepa_image_normalized",
        "jepa_imu_normalized",
        "jepa_image_cosine",
        "jepa_imu_cosine",
        "jepa_image_visible",
        "jepa_image_masked",
        "jepa_imu_visible",
        "jepa_imu_masked",
        "jepa_image_fine",
        "jepa_image_fine_normalized",
        "jepa_image_coarse",
        "variance",
        "covariance",
        "encoder_sensitivity",
        "encoder_sensitivity_weight",
        "sensitivity_noise_gain",
        "sensitivity_signal_gain",
        "sensitivity_ratio",
        "sensitivity_valid_fraction",
        "reconstruction",
        "reconstruction_image",
        "reconstruction_image_detail",
        "reconstruction_imu",
        "reconstruction_imu_detail",
    }

    def __init__(
        self,
        model: LatentPretrainingModel,
        config: dict[str, Any],
        device: torch.device,
        manifest_hash: str = "unknown",
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.phase = config["phase1"]
        self.sensitivity = config["encoder_sensitivity"]
        self.device = device
        self.masking = phase1_masking(self.phase)
        self.forward_model, self.device_ids = parallel_forward(
            Phase1Forward(self.model, self.masking), device, config["runtime"].get("gpu_count", "auto")
        )
        self.manifest_hash = manifest_hash
        self.decoder_forward_calls = 0
        self.successful_updates = 0
        self.parameters = list(model.online_parameters())
        teacher_ids = {id(parameter) for parameter in model.teachers.parameters()}
        if any(id(parameter) in teacher_ids for parameter in self.parameters):
            raise RuntimeError("Teacher parameters leaked into the phase-1 optimizer")
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=self.phase["learning_rate"],
            weight_decay=self.phase["weight_decay"],
        )
        self.initialization_hash = state_dict_hash(model.backbone)

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

    def step(self, raw_batch: dict[str, Any]) -> dict[str, float | str | bool]:
        batch = _to_device(raw_batch, self.device)
        batch_size = int(batch["image_noisy"].shape[0])
        minimum_batch = int(self.phase["minimum_statistics_batch"])
        if batch_size < minimum_batch:
            raise ValueError(
                f"Phase-1 batch {batch_size} is below minimum_statistics_batch={minimum_batch}"
            )
        self.model.train(True)
        self.model.normalizer.eval()
        self.optimizer.zero_grad(set_to_none=True)
        lr = self._set_lr()

        encoder_weight = sensitivity_weight(
            self.successful_updates,
            start_after=self.sensitivity["start_after_updates"],
            ramp_updates=self.sensitivity["ramp_updates"],
            maximum=self.sensitivity["weight_max"],
        ) if self.sensitivity["enabled"] else 0.0
        source = "off"
        probe_noise = probe_signal = probe_valid = None
        noise_energy = signal_energy = None
        clipped_fraction = 0.0
        if encoder_weight > 0:
            source = "image" if self.successful_updates % 2 == 0 else "imu"
            if source == "image":
                base_input = batch["image_clean"]
                noisy_input = batch["image_noisy"]
            else:
                base_input = self.model.normalizer.normalize(batch["imu_clean_phys"])
                noisy_input = self.model.normalizer.normalize(batch["imu_noisy_phys"])
            # Both probes start from the CLEAN operating point, whose features the
            # variance term already computes, so the base costs nothing extra.
            probe_kwargs = dict(
                image_epsilon=self.sensitivity["image_epsilon"],
                imu_epsilon=self.sensitivity["imu_normalized_epsilon"],
                alpha=self.sensitivity["alpha"],
                minimum_energy=self.sensitivity["minimum_energy"],
            )
            noise_direction, noise_valid = corruption_direction(base_input, noisy_input)
            signal_direction, signal_valid = detail_direction(base_input)
            # A sample the corruptor left clean has no noise direction to measure.
            probe_valid = noise_valid & signal_valid
            probe_noise, noise_energy, clipped_fraction = make_probe(
                base_input, source, noise_direction, **probe_kwargs
            )
            probe_signal, signal_energy, _ = make_probe(
                base_input, source, signal_direction, **probe_kwargs
            )
        mask_seeds = None
        if self.masking is not None:
            mask_seeds = sample_seeds(self.phase["position_seed"], self.successful_updates, batch_size)
        # Gather dense features, not per-device scalar losses. Statistics below
        # see all B samples even when each GPU processed only B/2 samples.
        features = self.forward_model(
            batch["image_noisy"], batch["imu_noisy_phys"], batch["image_clean"],
            batch["imu_clean_phys"], batch["image_time"], batch["imu_times"],
            probe_noise=probe_noise, probe_signal=probe_signal, probe_source=source,
            mask_seeds=mask_seeds,
        )
        # Masked tokens count like any other: the loss is still the mean over all
        # tokens, only the predictor's view of them changed.
        jepa, jepa_image, jepa_imu = jepa_latent_loss(
            features["prediction_i"], features["prediction_u"], features["target_i"], features["target_u"]
        )
        jepa_report = jepa_report_terms(features)
        multiscale = jepa.new_zeros(())
        fine_weight = float(self.phase.get("multiscale_fine_weight", 0.0))
        coarse_weight = float(self.phase.get("multiscale_coarse_weight", 0.0))
        if fine_weight > 0:
            fine = jepa_fine_loss(features["prediction_i_fine"], features["target_i_fine"])
            multiscale = multiscale + fine_weight * fine
            jepa_report["jepa_image_fine"] = float(fine.detach())
        if coarse_weight > 0:
            coarse = jepa_coarse_loss(features["prediction_i"], features["target_i"],
                                      int(self.phase["multiscale_coarse_pool"]))
            multiscale = multiscale + coarse_weight * coarse
            jepa_report["jepa_image_coarse"] = float(coarse.detach())

        image_total = features["FI"].shape[-2] * features["FI"].shape[-1]
        image_indices = _position_indices(
            image_total,
            self.phase["image_positions_per_update"],
            self.phase["position_seed"],
            self.successful_updates,
            "image",
        )
        imu_indices = _position_indices(
            features["FU"].shape[-1],
            self.phase["imu_positions_per_update"],
            self.phase["position_seed"],
            self.successful_updates,
            "imu",
        )
        maps = (
            (features["FI"], image_indices),
            (features["FU"], imu_indices),
            (features["ZI"], image_indices),
            (features["ZU"], imu_indices),
            (features["FI_clean"], image_indices),
            (features["FU_clean"], imu_indices),
            (features["ZI_clean"], image_indices),
            (features["ZU_clean"], imu_indices),
        )
        # Absent from configs written before the key existed: they used the
        # per-position covariance, and resuming them must keep it.
        pooled = self.phase.get("covariance_pooling", "per_position") == "pooled"
        regularizers = [
            variance_covariance_loss(
                dense_positions(feature, indices),
                gamma=self.phase["variance_gamma"],
                eps=self.phase["variance_eps"],
                pooled_covariance=pooled,
            )
            for feature, indices in maps
        ]
        variance = torch.stack([item[0] for item in regularizers]).mean()
        covariance = torch.stack([item[1] for item in regularizers]).mean()

        reconstruction = jepa.new_zeros(())
        reconstruction_parts: dict[str, float] = {}
        reconstruction_weight = float(self.phase.get("coefficient_reconstruction_loss_weight", 0.0))
        if reconstruction_weight > 0:
            reconstruction, parts = phase1_reconstruction_loss(
                features["reconstruction_image"],
                features["reconstruction_image_target"],
                features["reconstruction_imu"],
                features["reconstruction_imu_target"],
                detail_weight=float(self.phase.get("reconstruction_detail_weight", 0.5)),
            )
            reconstruction_parts = {key: float(value.detach()) for key, value in parts.items()}
            reconstruction_parts["reconstruction"] = float(reconstruction.detach())
            self.decoder_forward_calls += 1

        encoder_term = jepa.new_zeros(())
        sensitivity_report: dict[str, float] = {}
        if encoder_weight > 0:
            # Base point is the clean feature, matching the clean probes above.
            base = features["FI_clean" if source == "image" else "FU_clean"]
            eps = self.sensitivity["layer_norm_eps"]
            noise_gain = directional_gain(
                base, features["probe_noise_feature"], noise_energy, eps=eps
            )
            signal_gain = directional_gain(
                base, features["probe_signal_feature"], signal_energy, eps=eps
            )
            encoder_term, sensitivity_report = sensitivity_ratio_loss(
                noise_gain, signal_gain, probe_valid,
                floor_log_ratio=self.sensitivity.get("floor_log_ratio"),
            )

        total = (
            self.phase["jepa_weight"] * jepa
            + multiscale
            + self.phase["variance_weight"] * variance
            + self.phase["covariance_weight"] * covariance
            + encoder_weight * encoder_term
            + reconstruction_weight * reconstruction
        )
        if not torch.isfinite(total):
            self.optimizer.zero_grad(set_to_none=True)
            return {"skipped": True, "reason": "non_finite_loss", "loss": float(total.detach())}
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.phase["gradient_clip_norm"])
        if not torch.isfinite(gradient_norm) or not _finite_gradients(self.parameters):
            self.optimizer.zero_grad(set_to_none=True)
            return {"skipped": True, "reason": "non_finite_gradient", "loss": float(total.detach())}
        self.optimizer.step()
        momentum = ema_momentum(
            self.successful_updates,
            self.phase["max_successful_updates"],
            self.phase["teacher_momentum_start"],
            self.phase["teacher_momentum_end"],
        )
        self.model.teachers.update(self.model.backbone, momentum)
        self.successful_updates += 1
        return {
            "skipped": False,
            "loss": float(total.detach()),
            "jepa": float(jepa.detach()),
            "jepa_image": float(jepa_image.detach()),
            "jepa_imu": float(jepa_imu.detach()),
            **jepa_report,
            "variance": float(variance.detach()),
            "covariance": float(covariance.detach()),
            "encoder_sensitivity": float(encoder_term.detach()),
            **reconstruction_parts,
            "encoder_sensitivity_weight": encoder_weight,
            **sensitivity_report,
            "encoder_source": source,
            "probe_clipped_fraction": clipped_fraction,
            "gradient_norm": float(gradient_norm),
            "teacher_momentum": momentum,
            "learning_rate": lr,
            "successful_updates": self.successful_updates,
        }

    def checkpoint_payload(
        self,
        config: dict[str, Any],
        latent_gate_status: str = "NOT_EVALUATED",
        latent_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "metadata": {
                "pipeline_version": 3,
                "phase": "latent_pretrain",
                "trained_with_reconstruction": bool(self.decoder_forward_calls),
                "phase1_decoder_forward_calls": self.decoder_forward_calls,
                "successful_updates": self.successful_updates,
                "data_microbatches_consumed": self.successful_updates,
                "manifest_hash": self.manifest_hash,
                "initialization_hash": self.initialization_hash,
                "backbone_hash": state_dict_hash(self.model.backbone),
                "normalizer_hash": state_dict_hash(self.model.normalizer),
                "configuration_hash": configuration_hash(config, "phase1"),
                "latent_gate_status": latent_gate_status,
                "execution": execution_metadata(self.device, self.device_ids),
            },
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "successful_updates": self.successful_updates,
            "config": config,
            "rng": rng_state(),
            "latent_metrics": latent_metrics or {},
        }
