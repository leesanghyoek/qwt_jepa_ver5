"""Phase 1 trainer for phase1.objective: ijepa -- I-JEPA's loss and nothing else.

Per update, as I-JEPA's train.py: masks for the batch; the EMA target encoder on
the whole clean pair, LayerNorm'd, at the target blocks; the context encoder on
the context tokens and the predictor at the targets; loss = smooth L1 between
them (I-JEPA's code; the paper writes L2, the same up to 1/2 while |error| < 1);
AdamW with weight decay on the matrices only, its rate on a cosine from
``weight_decay`` to ``weight_decay_end``; then the EMA step with the momentum
rising linearly from ``teacher_momentum_start`` to ``teacher_momentum_end``.

Image and IMU each contribute half the loss. ``gradient_clip_norm: null`` leaves
the gradients alone as I-JEPA does; the norm is still logged.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..distributed import gather_shares, rank_and_world
from ..execution import IJEPAForward, execution_metadata, parallel_forward
from ..models.ijepa import IJEPAPretrainingModel
from .checkpoints import configuration_hash, rng_state, state_dict_hash, target_backbone_hash
from .phase1 import _finite_gradients, _to_device
from .schedules import cosine_weight_decay, linear_momentum, warmup_cosine_lr

MASK_KEYS = ("image_context", "image_targets", "imu_context", "imu_targets")


def ijepa_loss(features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(total, image, IMU): smooth L1 between predictions and targets, the modalities averaged."""
    image = F.smooth_l1_loss(features["prediction_i"], features["target_i"].detach())
    imu = F.smooth_l1_loss(features["prediction_u"], features["target_u"].detach())
    return 0.5 * (image + imu), image, imu


def ijepa_parameter_groups(model: IJEPAPretrainingModel) -> list[dict[str, Any]]:
    """I-JEPA's two groups: weight decay on weights, none on biases and 1-D (norm) parameters."""
    online = {id(parameter) for parameter in model.online_parameters()}
    decay, plain = [], []
    for name, parameter in model.named_parameters():
        if id(parameter) not in online:
            continue
        (plain if "bias" in name or parameter.ndim == 1 else decay).append(parameter)
    return [{"params": decay, "decayed": True}, {"params": plain, "decayed": False, "weight_decay": 0.0}]


class IJEPATrainer:
    LOSS_KEYS = {"loss", "jepa", "jepa_image", "jepa_imu"}

    def __init__(self, model: IJEPAPretrainingModel, config: dict[str, Any], device: torch.device,
                 manifest_hash: str = "unknown") -> None:
        self.model = model.to(device)
        self.config = config
        self.phase = config["phase1"]
        self.device = device
        self.forward_model, self.device_ids = parallel_forward(
            IJEPAForward(self.model), device, config["runtime"].get("gpu_count", "auto"))
        self.rank, self.world = rank_and_world()
        if self.world > 1:
            # As phase 1's other objective: one process per GPU, teachers move by EMA alike.
            self.forward_model = nn.parallel.DistributedDataParallel(
                self.forward_model, device_ids=[torch.cuda.current_device()] if self.device.type == "cuda" else None,
                broadcast_buffers=False)
        self.manifest_hash = manifest_hash
        self.decoder_forward_calls = 0
        self.successful_updates = 0
        self.parameters = list(model.online_parameters())
        teacher_ids = {id(parameter) for parameter in model.teachers.parameters()}
        if any(id(parameter) in teacher_ids for parameter in self.parameters):
            raise RuntimeError("Teacher parameters leaked into the phase-1 optimizer")
        self.optimizer = torch.optim.AdamW(ijepa_parameter_groups(model), lr=self.phase["learning_rate"],
                                           weight_decay=self.phase["weight_decay"])
        self.initialization_hash = state_dict_hash(model.backbone)

    @property
    def evaluation_model(self):
        return self.forward_model.module if self.world > 1 else self.forward_model

    def _schedules(self) -> tuple[float, float]:
        update, total = self.successful_updates, self.phase["max_successful_updates"]
        lr = warmup_cosine_lr(update, total, self.phase["warmup_updates"], self.phase["learning_rate"],
                              self.phase["minimum_lr"])
        decay = cosine_weight_decay(update, total, self.phase["weight_decay"], self.phase["weight_decay_end"])
        for group in self.optimizer.param_groups:
            group["lr"] = lr
            if group["decayed"]:
                group["weight_decay"] = decay
        return lr, decay

    def masks(self, image_shape: tuple[int, ...], imu_length: int, batch_size: int) -> dict[str, torch.Tensor]:
        """This update's masks for the whole batch: the same on every rank, and on resume."""
        image_grid, imu_grid = self.model.token_grids(image_shape, imu_length)
        return self.model.sample_masks(image_grid, imu_grid, batch_size,
                                       self.phase["position_seed"], self.successful_updates, "ijepa")

    def step(self, raw_batch: dict[str, Any]) -> dict[str, float | str | bool]:
        batch = _to_device(raw_batch, self.device)
        share = int(batch["image_noisy"].shape[0])
        batch_size = share * self.world
        self.model.train(True)
        self.model.normalizer.eval()
        self.optimizer.zero_grad(set_to_none=True)
        lr, decay = self._schedules()
        masks = self.masks(tuple(batch["image_noisy"].shape), int(batch["imu_noisy_phys"].shape[-1]), batch_size)
        sizes = {f"{key}_tokens": int(value.shape[-1]) for key, value in masks.items()}
        # This rank's samples keep the masks they have in the whole batch.
        masks = {key: value[self.rank * share:(self.rank + 1) * share].to(self.device) for key, value in masks.items()}
        features = self.forward_model(
            batch["image_noisy"], batch["imu_noisy_phys"], batch["image_clean"], batch["imu_clean_phys"],
            batch["image_time"], batch["imu_times"], **masks)
        if self.world > 1:
            features = {key: gather_shares(value, keep_graph=key.startswith("prediction"))
                        for key, value in features.items()}
        total, image, imu = ijepa_loss(features)
        if not torch.isfinite(total):
            self.optimizer.zero_grad(set_to_none=True)
            return {"skipped": True, "reason": "non_finite_loss", "loss": float(total.detach())}
        # x world: only this rank's share carries a graph, and DDP averages over ranks.
        (total * self.world).backward()
        clip = self.phase.get("gradient_clip_norm")
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.parameters, float("inf") if clip is None else clip)
        if not torch.isfinite(gradient_norm) or not _finite_gradients(self.parameters):
            self.optimizer.zero_grad(set_to_none=True)
            return {"skipped": True, "reason": "non_finite_gradient", "loss": float(total.detach())}
        self.optimizer.step()
        momentum = linear_momentum(self.successful_updates, self.phase["max_successful_updates"],
                                   self.phase["teacher_momentum_start"], self.phase["teacher_momentum_end"])
        self.model.teachers.update(self.model.backbone, momentum)
        self.successful_updates += 1
        return {
            "skipped": False,
            "loss": float(total.detach()),
            "jepa": float(total.detach()),
            "jepa_image": float(image.detach()),
            "jepa_imu": float(imu.detach()),
            **sizes,
            "gradient_norm": float(gradient_norm),
            "teacher_momentum": momentum,
            "learning_rate": lr,
            "weight_decay": decay,
            "successful_updates": self.successful_updates,
        }

    def checkpoint_payload(self, config: dict[str, Any], latent_gate_status: str = "NOT_EVALUATED",
                           latent_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "metadata": {
                "pipeline_version": 3,
                "phase": "latent_pretrain",
                "phase1_objective": "ijepa",
                "trained_with_reconstruction": False,
                "phase1_decoder_forward_calls": 0,
                "successful_updates": self.successful_updates,
                "data_microbatches_consumed": self.successful_updates,
                "manifest_hash": self.manifest_hash,
                "initialization_hash": self.initialization_hash,
                "backbone_hash": state_dict_hash(self.model.backbone),
                # phase2.backbone_weights: target freezes this one instead.
                "target_backbone_hash": target_backbone_hash(self.model),
                "normalizer_hash": state_dict_hash(self.model.normalizer),
                "configuration_hash": configuration_hash(config, "phase1"),
                "latent_gate_status": latent_gate_status,
                "execution": execution_metadata(self.device, self.device_ids, self.world),
            },
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "successful_updates": self.successful_updates,
            "config": config,
            "rng": rng_state(),
            "latent_metrics": latent_metrics or {},
        }
