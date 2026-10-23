"""Phase-2 validation on FlareImage's held-out pairs: is the flare being learned while phase 2 trains?

The TartanAir validation bank holds no flare since p36 (its "loe" kind is empty), so until now the flare was scored
only after training (tools/flare_image_test.py, Cell 12d). Here each validation also runs every FlareImage valid pair
as a training pair is built (PairedCameraImuDataset._flare_item: same crop rule, camera blur and grain, borrowed IMU,
phase-2 target), and scores PSNR in -> out over the whole crop, INSIDE the flare -- pixels where the raw flared photo
differs from its clean photo by more than FLARE_THRESHOLD (mean of 3 channels), as in tools/flare_image_test.py -- and
OUTSIDE it, where the model should leave the frame alone (p40 lost 4-5 dB there). Log only: no checkpoint choice
reads it.

p46: the scene pairs' valid split (data.scene_pairs_*, e.g. VisDrone test-dev scenes with held-out flare layers) is
scored the same way, its flared photo always the source ("VisDrone valid" line).

p47: the real low-light pairs' valid split (data.lowlight_pairs_*, LOL) over the whole crop and its brightness
("LOL valid" line).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ..data.dataset import PairedCameraImuDataset, collate_paired

FLARE_THRESHOLD = 0.05


def flare_validation_dataset(dataset: PairedCameraImuDataset) -> PairedCameraImuDataset | None:
    """``dataset`` (built with the FlareImage bank and flare_layer=True, so each item carries its raw flare) reading
    the valid pairs; None when there is no bank or no valid pair."""
    if dataset.flare_bank is None:
        return None
    dataset.flare_pairs_split = "valid"
    return dataset if dataset.flare_bank.pairs("valid") else None


def _psnr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    error = (a - b).square()
    if mask is not None:
        if not bool(mask.any()):
            return float("nan")
        error = error[:, mask]
    return float(-10.0 * torch.log10(error.mean().clamp_min(1e-12)))


def _mean(values: list[float]) -> float:
    kept = [value for value in values if math.isfinite(value)]
    return sum(kept) / len(kept) if kept else float("nan")


def scene_validation_dataset(dataset: PairedCameraImuDataset) -> PairedCameraImuDataset | None:
    """``dataset`` (built with the scene bank and flare_layer=True) reading the scene pairs' valid split; None when
    there is no bank."""
    if dataset.scene_bank is None:
        return None
    dataset.scene_pairs_split = "valid"
    return dataset if dataset.scene_bank.pairs("valid") else None


@torch.no_grad()
def evaluate_flare_pairs(dataset: PairedCameraImuDataset, forward_model: torch.nn.Module, device: torch.device,
                         batch_size: int) -> dict[str, float]:
    """Every valid pair once (draws 0..n-1 of the fixed realization: the same crops each validation)."""
    count = len(dataset.flare_bank.pairs(dataset.flare_pairs_split))
    return _evaluate_pairs(dataset._flare_item, count, forward_model, device, batch_size)


@torch.no_grad()
def evaluate_scene_pairs(dataset: PairedCameraImuDataset, forward_model: torch.nn.Module, device: torch.device,
                         batch_size: int) -> dict[str, float]:
    """p46: every valid scene pair once, its flared photo as the source (draws 0..n-1 of the fixed realization)."""
    count = len(dataset.scene_bank.pairs(dataset.scene_pairs_split))
    return _evaluate_pairs(lambda draw: dataset._scene_item(draw, flared=True), count, forward_model, device,
                           batch_size)


def _evaluate_pairs(item, count: int, forward_model: torch.nn.Module, device: torch.device,
                    batch_size: int) -> dict[str, float]:
    rows: dict[str, list[float]] = {key: [] for key in (
        "psnr_in", "psnr_out", "flare_in", "flare_out", "rest_in", "rest_out", "flare_fraction")}
    for start in range(0, count, batch_size):
        batch = collate_paired([item(draw) for draw in range(start, min(count, start + batch_size))])
        restored = forward_model(batch["image_noisy"].to(device), batch["imu_noisy_phys"].to(device),
                                 batch["image_time"].to(device), batch["imu_times"].to(device))
        out = restored["image"].float().clamp(0, 1).cpu()
        noisy, clean = batch["image_noisy"].float().clamp(0, 1), batch["image_clean"].float().clamp(0, 1)
        for k in range(out.shape[0]):
            mask = batch["image_flare_target"][k].abs().mean(0) > FLARE_THRESHOLD
            rows["flare_fraction"].append(float(mask.float().mean()))
            rows["psnr_in"].append(_psnr(noisy[k], clean[k]))
            rows["psnr_out"].append(_psnr(out[k], clean[k]))
            rows["flare_in"].append(_psnr(noisy[k], clean[k], mask))
            rows["flare_out"].append(_psnr(out[k], clean[k], mask))
            rows["rest_in"].append(_psnr(noisy[k], clean[k], ~mask))
            rows["rest_out"].append(_psnr(out[k], clean[k], ~mask))
    report = {key: _mean(values) for key, values in rows.items()}
    report["count"] = count
    report["worse_fraction"] = _mean([float(b < a) for a, b in zip(rows["psnr_in"], rows["psnr_out"])])
    report["rest_worse_fraction"] = _mean([float(b < a - 0.5) for a, b in zip(rows["rest_in"], rows["rest_out"])
                                           if math.isfinite(a) and math.isfinite(b)])
    return report


def flare_validation_text(report: dict[str, Any], name: str = "FlareImage") -> str:
    return (f"{name} valid n={int(report['count'])}: ca anh {report['psnr_in']:.2f} -> {report['psnr_out']:.2f}"
            f" ({100 * report['worse_fraction']:.0f}% te hon vao) | vung loe ({100 * report['flare_fraction']:.0f}% khung)"
            f" {report['flare_in']:.2f} -> {report['flare_out']:.2f} | ngoai loe {report['rest_in']:.2f} ->"
            f" {report['rest_out']:.2f} ({100 * report['rest_worse_fraction']:.0f}% tut > 0,5 dB)")


def scene_validation_text(report: dict[str, Any]) -> str:
    return flare_validation_text(report, "VisDrone")


def lowlight_validation_dataset(dataset: PairedCameraImuDataset) -> PairedCameraImuDataset | None:
    """p47: ``dataset`` (built with the low-light bank) reading the real low-light pairs' valid split; None without."""
    if dataset.lowlight_bank is None:
        return None
    dataset.lowlight_pairs_split = "valid"
    return dataset if dataset.lowlight_bank.pairs("valid") else None


@torch.no_grad()
def evaluate_lowlight_pairs(dataset: PairedCameraImuDataset, forward_model: torch.nn.Module, device: torch.device,
                            batch_size: int) -> dict[str, float]:
    """p47: every valid low-light pair once (draws 0..n-1 of the fixed realization), PSNR in -> out of the whole crop,
    and the mean brightness (luma) of input, output and target: is a real dark photo brought up to the right level?"""
    count = len(dataset.lowlight_bank.pairs(dataset.lowlight_pairs_split))
    rows: dict[str, list[float]] = {key: [] for key in ("psnr_in", "psnr_out", "luma_in", "luma_out", "luma_target")}
    luma = lambda image: float((0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]).mean())
    for start in range(0, count, batch_size):
        batch = collate_paired([dataset._lowlight_item(draw) for draw in range(start, min(count, start + batch_size))])
        restored = forward_model(batch["image_noisy"].to(device), batch["imu_noisy_phys"].to(device),
                                 batch["image_time"].to(device), batch["imu_times"].to(device))
        out = restored["image"].float().clamp(0, 1).cpu()
        noisy, clean = batch["image_noisy"].float().clamp(0, 1), batch["image_clean"].float().clamp(0, 1)
        for k in range(out.shape[0]):
            rows["psnr_in"].append(_psnr(noisy[k], clean[k]))
            rows["psnr_out"].append(_psnr(out[k], clean[k]))
            rows["luma_in"].append(luma(noisy[k]))
            rows["luma_out"].append(luma(out[k]))
            rows["luma_target"].append(luma(clean[k]))
    report = {key: _mean(values) for key, values in rows.items()}
    report["count"] = count
    report["worse_fraction"] = _mean([float(b < a) for a, b in zip(rows["psnr_in"], rows["psnr_out"])])
    return report


def lowlight_validation_text(report: dict[str, Any]) -> str:
    return (f"LOL valid n={int(report['count'])}: ca anh {report['psnr_in']:.2f} -> {report['psnr_out']:.2f}"
            f" ({100 * report['worse_fraction']:.0f}% te hon vao) | do sang vao {report['luma_in']:.3f} -> ra"
            f" {report['luma_out']:.3f} (dich {report['luma_target']:.3f})")
