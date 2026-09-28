"""One-process execution on CPU, one GPU or two GPUs with global-batch losses.

Only tensor dictionaries cross DataParallel's gather boundary. Losses (especially
variance/covariance) are computed after gathering the entire batch, not per GPU.
The original unwrapped modules own optimizer parameters and checkpoint state.
"""

from __future__ import annotations

import os

import torch
from torch import nn

from .models.masking import block_mask, span_mask
from .models.pipeline import LatentPretrainingModel, RestorationSystem


def select_device_ids(device: torch.device, gpu_count: str | int = "auto") -> list[int]:
    if str(gpu_count) not in {"auto", "1", "2"}:
        raise ValueError("gpu_count must be auto, 1 or 2")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Use python -m qjepa once; this backend uses DataParallel, not torchrun")
    if device.type != "cuda":
        if str(gpu_count) == "2":
            raise ValueError("Two GPUs require a CUDA device")
        return []
    available = torch.cuda.device_count()
    primary = device.index if device.index is not None else torch.cuda.current_device()
    count = min(2, available) if str(gpu_count) == "auto" else int(gpu_count)
    if available < count or primary >= available or available == 0:
        raise ValueError(f"Requested {count} GPU(s) with primary cuda:{primary}; only {available} visible")
    return ([primary] + [index for index in range(available) if index != primary])[:count]


def parallel_forward(module: nn.Module, device: torch.device, gpu_count: str | int = "auto") -> tuple[nn.Module, list[int]]:
    ids = select_device_ids(device, gpu_count)
    module.to(device)
    if len(ids) > 1:
        return nn.DataParallel(module, device_ids=ids, output_device=ids[0]), ids
    return module, ids


def execution_metadata(device: torch.device, ids: list[int]) -> dict:
    return {"backend": "data_parallel" if len(ids) > 1 else "single_device",
            "device": str(device), "device_ids": ids, "gpu_count": len(ids),
            "loss_batch": "global_gathered_batch"}


class Phase1Forward(nn.Module):
    def __init__(self, model: LatentPretrainingModel, masking: dict | None = None):
        super().__init__()
        self.model = model
        # image_ratio, image_block, imu_ratio, imu_span; None = no masking.
        self.masking = masking

    def _masks(self, mask_seeds, latent):
        """Masks built here, per sample, because only here is the token grid known."""
        if mask_seeds is None or self.masking is None:
            return None, None
        seeds = mask_seeds.tolist()
        device = latent.ZI.device
        image_mask = imu_mask = None
        if self.masking["image_ratio"] > 0:
            height, width = latent.ZI.shape[-2:]
            image_mask = torch.stack([block_mask(height, width, self.masking["image_ratio"],
                                                 self.masking["image_block"], seed) for seed in seeds]).to(device)
        if self.masking["imu_ratio"] > 0:
            length = latent.ZU.shape[-1]
            imu_mask = torch.stack([span_mask(length, self.masking["imu_ratio"],
                                              self.masking["imu_span"], seed) for seed in seeds]).to(device)
        return image_mask, imu_mask

    def forward(self, image_noisy, imu_noisy_phys, image_clean, imu_clean_phys,
                image_time, imu_times, probe_noise=None, probe_signal=None,
                probe_source="off", mask_seeds=None) -> dict[str, torch.Tensor]:
        noisy = self.model.encode_online(image_noisy, imu_noisy_phys, image_time, imu_times)
        clean = self.model.encode_online(image_clean, imu_clean_phys, image_time, imu_times)
        image_mask, imu_mask = self._masks(mask_seeds, noisy)
        prediction_i, prediction_u, prediction_i_fine = self.model.predictions(noisy, image_mask, imu_mask)
        targets = self.model.targets(image_clean, imu_clean_phys, fine=self.model.fine_scale)
        result = {"prediction_i": prediction_i, "prediction_u": prediction_u,
                  "target_i": targets[0], "target_u": targets[1]}
        if self.model.fine_scale:
            result["prediction_i_fine"] = prediction_i_fine
            result["target_i_fine"] = targets[2]
        for name, mask in (("image_mask", image_mask), ("imu_mask", imu_mask)):
            if mask is not None:
                result[name] = mask
        for name in ("FI", "FU", "ZI", "ZU"):
            result[name] = getattr(noisy, name)
            result[name + "_clean"] = getattr(clean, name)
        if self.model.reconstructs:
            # Giai ma tu latent NHIEU ra he so SACH — dung nhiem vu cua phase 2.
            image_coefficients, imu_coefficients = self.model.reconstruct(noisy)
            backbone = self.model.backbone
            target_image, _ = backbone.image_transform.analysis(image_clean)
            target_imu, _ = backbone.imu_transform.analysis(
                self.model.normalizer.normalize(imu_clean_phys)
            )
            result["reconstruction_image"] = image_coefficients
            result["reconstruction_imu"] = imu_coefficients
            result["reconstruction_image_target"] = target_image
            result["reconstruction_imu_target"] = target_imu
        # Two probes share one encoder forward each; the base point they are
        # measured against is FI_clean/FU_clean above, already computed.
        probes = {"probe_noise_feature": probe_noise, "probe_signal_feature": probe_signal}
        for name, probe in probes.items():
            if probe is None:
                continue
            if probe_source == "image":
                result[name] = self.model.backbone.encode_image_dense(probe)
            elif probe_source == "imu":
                result[name] = self.model.backbone.encode_imu_dense(probe)
            else:
                raise ValueError("A probe requires image or imu probe_source")
        return result


class RestorationForward(nn.Module):
    def __init__(self, system: RestorationSystem):
        super().__init__()
        self.system = system
        # Set by Phase2Trainer. Autocast state is per thread and DataParallel runs
        # each replica in its own thread, so it must be entered here, not around the call.
        self.amp = False

    def forward(self, image_noisy, imu_noisy_phys, image_time, imu_times) -> dict[str, torch.Tensor]:
        with torch.autocast(device_type=image_noisy.device.type, dtype=torch.float16, enabled=self.amp):
            result = self.system(image_noisy, imu_noisy_phys, image_time, imu_times)
        outputs = {"image": result.image, "imu_normalized": result.imu_normalized,
                   "imu_physical": result.imu_physical,
                   "image_coefficients": result.image_coefficients,
                   "imu_coefficients": result.imu_coefficients,
                   # Only tensors cross DataParallel's gather, so parts are flattened in.
                   **(result.image_parts or {})}
        # Losses and metrics always read fp32: wavelet detail, FFT and PSNR need it.
        return {key: value.float() if value.is_floating_point() else value for key, value in outputs.items()}
