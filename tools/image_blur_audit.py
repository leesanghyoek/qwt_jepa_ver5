"""Diagnose image blur on an existing phase-2 checkpoint; no training.

The same fixed validation samples are evaluated in three scenarios. Pixel error,
edge-aligned gradient error, and wavelet-band error are reported separately so
an exposure improvement cannot masquerade as sharper restored edges.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qjepa.cli import _dataset, _loader, _system_from_phase2, _to_device
from qjepa.data import read_manifest


SCENARIOS = {
    "blur_only": ("blur_only", "clean"),
    "low_light_only": ("low_light_only", "clean"),
    "full_full": ("full", "full"),
}
BANDS = ("LL", "LH", "HL", "HH")


def _luma(image: torch.Tensor) -> torch.Tensor:
    return (image[:, :1] * 0.299 + image[:, 1:2] * 0.587 + image[:, 2:3] * 0.114)


def _gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grey = _luma(image.clamp(0, 1))
    gx = F.pad(grey[..., 1:] - grey[..., :-1], (0, 1))
    gy = F.pad(grey[..., 1:, :] - grey[..., :-1, :], (0, 0, 0, 1))
    return gx, gy, gx.abs() + gy.abs()


def edge_components(
    clean: torch.Tensor, noisy: torch.Tensor, restored: torch.Tensor,
) -> dict[str, float]:
    """Sum/count contributions, using the strongest 10% of CLEAN edges per frame."""
    gx_c, gy_c, magnitude_c = _gradients(clean)
    gx_n, gy_n, magnitude_n = _gradients(noisy)
    gx_r, gy_r, magnitude_r = _gradients(restored)
    threshold = torch.quantile(magnitude_c.flatten(1), 0.9, dim=1).view(-1, 1, 1, 1)
    mask = magnitude_c > threshold
    edge_count = mask.sum().clamp_min(1)
    smooth_count = (~mask).sum().clamp_min(1)
    reference = (gx_c.abs() + gy_c.abs()) * mask
    noisy_error = ((gx_n - gx_c).abs() + (gy_n - gy_c).abs()) * mask
    restored_error = ((gx_r - gx_c).abs() + (gy_r - gy_c).abs()) * mask
    return {
        "edge_count": int(edge_count),
        "smooth_count": int(smooth_count),
        "edge_reference_sum": float(reference.sum()),
        "edge_input_sum": float((magnitude_n * mask).sum()),
        "edge_restored_sum": float((magnitude_r * mask).sum()),
        "edge_input_error_sum": float(noisy_error.sum()),
        "edge_restored_error_sum": float(restored_error.sum()),
        "smooth_reference_sum": float((magnitude_c * ~mask).sum()),
        "smooth_input_sum": float((magnitude_n * ~mask).sum()),
        "smooth_restored_sum": float((magnitude_r * ~mask).sum()),
    }


def _accumulate(target: dict[str, float], values: dict[str, float]) -> None:
    for key, value in values.items():
        target[key] = target.get(key, 0.0) + value


def _band_sums(
    clean: torch.Tensor, noisy: torch.Tensor, restored: torch.Tensor,
) -> dict[str, list[float] | int]:
    if clean.shape != noisy.shape or clean.shape != restored.shape or clean.shape[1] % 16:
        raise ValueError("Expected matching [B, 16*colours, H, W] image coefficients")
    shape = (clean.shape[0], clean.shape[1] // 16, 4, 4, *clean.shape[2:])
    clean, noisy, restored = (tensor.reshape(shape) for tensor in (clean, noisy, restored))
    axes = (0, 1, 3, 4, 5)
    return {
        "coefficient_count_per_band": clean[:, :, 0].numel(),
        "clean_squared": clean.square().sum(axes).cpu().tolist(),
        "input_squared": noisy.square().sum(axes).cpu().tolist(),
        "restored_squared": restored.square().sum(axes).cpu().tolist(),
        "input_error_squared": (noisy - clean).square().sum(axes).cpu().tolist(),
        "restored_error_squared": (restored - clean).square().sum(axes).cpu().tolist(),
    }


def _add_bands(target: dict, values: dict) -> None:
    target["coefficient_count_per_band"] += values["coefficient_count_per_band"]
    for key in ("clean_squared", "input_squared", "restored_squared",
                "input_error_squared", "restored_error_squared"):
        target[key] = [a + b for a, b in zip(target[key], values[key])]


def _empty_bands() -> dict:
    bands = {"coefficient_count_per_band": 0}
    for key in ("clean_squared", "input_squared", "restored_squared",
                "input_error_squared", "restored_error_squared"):
        bands[key] = [0.0] * 4
    return bands


def _spread_indices(total: int, count: int) -> list[int]:
    """Deterministically cover the entire split instead of its first trajectory."""
    if total < 1 or count < 1:
        raise ValueError("total and count must be positive")
    count = min(count, total)
    if count == 1:
        return [total // 2]
    return [round(index * (total - 1) / (count - 1)) for index in range(count)]


def _record(
    stats: dict, bands: dict, clean: torch.Tensor, noisy: torch.Tensor,
    restored: torch.Tensor, clean_coefficients: torch.Tensor,
    input_coefficients: torch.Tensor, output_coefficients: torch.Tensor,
    zeroed: torch.Tensor | None = None,
) -> None:
    input_error = noisy - clean
    restored_error = restored - clean
    _accumulate(stats, {
        "pixel_input_error_sum": float(input_error.abs().sum()),
        "pixel_restored_error_sum": float(restored_error.abs().sum()),
        "pixel_input_squared_error_sum": float(input_error.square().sum()),
        "pixel_restored_squared_error_sum": float(restored_error.square().sum()),
    })
    _accumulate(stats, edge_components(clean, noisy, restored))
    _add_bands(bands, _band_sums(clean_coefficients, input_coefficients,
                                output_coefficients))
    if zeroed is not None:
        _accumulate(stats, {
            "ablation_count": clean.numel(),
            "ablation_full_mae_sum": float(restored_error.abs().sum()),
            "ablation_zero_zi_mae_sum": float((zeroed - clean).abs().sum()),
        })


def _finish(stats: dict, bands: dict, samples: int, pixels: int) -> dict:
    edges = max(stats["edge_count"], 1)
    smooth = max(stats["smooth_count"], 1)
    clean_edge = stats["edge_reference_sum"] / edges
    input_edge_error = stats["edge_input_error_sum"] / edges
    restored_edge_error = stats["edge_restored_error_sum"] / edges
    result = {
        "samples": samples,
        "image_mae_input": stats["pixel_input_error_sum"] / pixels,
        "image_mae_restored": stats["pixel_restored_error_sum"] / pixels,
        "image_psnr_input_db": -10 * math.log10(max(stats["pixel_input_squared_error_sum"] / pixels, 1e-12)),
        "image_psnr_restored_db": -10 * math.log10(max(stats["pixel_restored_squared_error_sum"] / pixels, 1e-12)),
        "clean_strong_edge_magnitude": clean_edge,
        "input_strong_edge_magnitude": stats["edge_input_sum"] / edges,
        "restored_strong_edge_magnitude": stats["edge_restored_sum"] / edges,
        "strong_edge_gradient_mae_input": input_edge_error,
        "strong_edge_gradient_mae_restored": restored_edge_error,
        "strong_edge_gradient_error_reduction_pct":
            100 * (1 - restored_edge_error / max(input_edge_error, 1e-12)),
        "smooth_region_edge_magnitude_clean": stats["smooth_reference_sum"] / smooth,
        "smooth_region_edge_magnitude_input": stats["smooth_input_sum"] / smooth,
        "smooth_region_edge_magnitude_restored": stats["smooth_restored_sum"] / smooth,
        "bands": {},
    }
    count = max(bands["coefficient_count_per_band"], 1)
    for index, name in enumerate(BANDS):
        result["bands"][name] = {
            "rms_clean": math.sqrt(bands["clean_squared"][index] / count),
            "rms_input": math.sqrt(bands["input_squared"][index] / count),
            "rms_restored": math.sqrt(bands["restored_squared"][index] / count),
            "rmse_input_to_clean": math.sqrt(bands["input_error_squared"][index] / count),
            "rmse_restored_to_clean": math.sqrt(bands["restored_error_squared"][index] / count),
        }
    if stats.get("ablation_count", 0):
        n = stats["ablation_count"]
        zero_mae = stats["ablation_zero_zi_mae_sum"] / n
        full_mae = stats["ablation_full_mae_sum"] / n
        result["zero_zi_ablation"] = {
            "mae_full": full_mae,
            "mae_zero_zi": zero_mae,
            "mae_increase_pct_when_zeroed": 100 * (zero_mae / max(full_mae, 1e-12) - 1),
        }
    return result


def audit(checkpoint: str | Path, manifest_path: str | Path, *, output: str | Path,
          split: str = "valid", batches: int = 16, samples: int | None = None,
          device: str = "cuda") -> dict:
    if batches < 1:
        raise ValueError("batches must be positive")
    if samples is not None and samples < 1:
        raise ValueError("samples must be positive")
    device_obj = torch.device(device)
    system, config = _system_from_phase2(str(checkpoint), device_obj)
    manifest = read_manifest(manifest_path)
    indices = _spread_indices(len(manifest["samples"][split]), samples) if samples else None
    # Diagnostics should not spawn DataLoader workers or retain image batches.
    config["data"] = dict(config["data"], num_workers=0, pin_memory=False)
    report: dict[str, dict] = {}
    first_sample_ids = None
    for scenario, (image_mode, imu_mode) in SCENARIOS.items():
        dataset = _dataset(config, manifest, split, fixed_realization=True,
                           image_mode=image_mode, imu_mode=imu_mode)
        selected = Subset(dataset, indices) if indices is not None else dataset
        loader = _loader(config, selected, config["phase2"]["batch_size"], train=False)
        stats: dict[str, float] = {}
        bands = _empty_bands()
        active_stats: dict[str, float] = {}
        active_bands = _empty_bands()
        active_samples = active_pixels = 0
        seen_samples = pixels = 0
        scenario_ids = []
        for index, raw in enumerate(loader):
            if indices is None and index >= batches:
                break
            scenario_ids.extend(raw["sample_id"])
            batch = _to_device(raw, device_obj)
            with torch.no_grad():
                latent = system.encode(batch["image_noisy"], batch["imu_noisy_phys"],
                                       batch["image_time"], batch["imu_times"])
                restored = system.decode(latent, system.motion_for(batch))
                clean_coefficients, _ = system.backbone.image_transform.analysis(batch["image_clean"])
                clean, noisy, output_image = (
                    batch["image_clean"].clamp(0, 1),
                    batch["image_noisy"].clamp(0, 1),
                    restored.image.clamp(0, 1),
                )
                # Bands of what the viewer sees. The decoder's own coefficients are
                # 4x redundant and can carry energy synthesis discards, which
                # would read as detail the image does not have.
                output_coefficients, _ = system.backbone.image_transform.analysis(output_image)
                zeroed_image = None
                if scenario == "blur_only":
                    zeroed = system.decode(replace(latent, ZI=torch.zeros_like(latent.ZI)), system.motion_for(batch))
                    zeroed_image = zeroed.image.clamp(0, 1)
                _record(stats, bands, clean, noisy, output_image,
                        clean_coefficients, latent.image_coefficients,
                        output_coefficients, zeroed_image)
                if scenario == "blur_only":
                    active = torch.tensor(
                        [any(corruption["image"][key] for key in
                             ("defocus", "motion", "downsample"))
                         for corruption in raw["corruption"]],
                        dtype=torch.bool, device=device_obj,
                    )
                    if active.any():
                        _record(active_stats, active_bands, clean[active], noisy[active],
                                output_image[active], clean_coefficients[active],
                                latent.image_coefficients[active],
                                output_coefficients[active],
                                zeroed_image[active])
                        active_samples += int(active.sum())
                        active_pixels += clean[active].numel()
                seen_samples += clean.shape[0]
                pixels += clean.numel()
        if not seen_samples:
            raise ValueError("Validation loader returned no images")
        if first_sample_ids is None:
            first_sample_ids = scenario_ids
        elif scenario_ids != first_sample_ids:
            raise ValueError("Scenarios did not evaluate the same samples")
        report[scenario] = _finish(stats, bands, seen_samples, pixels)
        if scenario == "blur_only":
            report[scenario]["actual_blur_frames"] = active_samples
            if active_samples:
                report[scenario]["blur_active_only"] = _finish(
                    active_stats, active_bands, active_samples, active_pixels,
                )

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"checkpoint": str(checkpoint), "split": split,
                                "batches": batches if indices is None else None,
                                "spread_samples": len(indices) if indices is not None else None,
                                "scenarios": report}, indent=2), encoding="utf-8")
    print("Image blur audit — cùng các frame valid, fixed realization; chỉ đọc checkpoint")
    print("PSNR ở đây tính từ MSE gộp, không phải trung bình PSNR từng ảnh của evaluate.")
    if indices is not None:
        print(f"Mẫu rải đều trên toàn validation: {len(indices)} / {len(manifest['samples'][split])}")
    for name, result in report.items():
        print(f"\n{name}: {result['samples']} ảnh")
        print(f"  PSNR input → restored: {result['image_psnr_input_db']:.2f} → {result['image_psnr_restored_db']:.2f} dB")
        print(f"  MAE  input → restored: {result['image_mae_input']:.5f} → {result['image_mae_restored']:.5f}")
        print(f"  Lỗi gradient trên cạnh thật: {result['strong_edge_gradient_mae_input']:.5f}"
              f" → {result['strong_edge_gradient_mae_restored']:.5f}"
              f" ({result['strong_edge_gradient_error_reduction_pct']:+.1f}% cải thiện)")
        print("  Độ mạnh cạnh thật clean / input / restored:"
              f" {result['clean_strong_edge_magnitude']:.5f} /"
              f" {result['input_strong_edge_magnitude']:.5f} /"
              f" {result['restored_strong_edge_magnitude']:.5f}")
        print("  Băng  RMS clean  RMS input  RMS out  RMSE input  RMSE out")
        for band, values in result["bands"].items():
            print(f"  {band:<5} {values['rms_clean']:>9.5f} {values['rms_input']:>10.5f}"
                  f" {values['rms_restored']:>8.5f} {values['rmse_input_to_clean']:>11.5f}"
                  f" {values['rmse_restored_to_clean']:>9.5f}")
        if "zero_zi_ablation" in result:
            a = result["zero_zi_ablation"]
            print(f"  Đặt ZI=0: MAE {a['mae_full']:.5f} → {a['mae_zero_zi']:.5f}"
                  f" ({a['mae_increase_pct_when_zeroed']:+.1f}%)")
        if "blur_active_only" in result:
            active = result["blur_active_only"]
            print(f"  Frame có blur thật: {result['actual_blur_frames']}/{result['samples']}")
            print(f"  Chỉ frame có blur: PSNR {active['image_psnr_input_db']:.2f}"
                  f" → {active['image_psnr_restored_db']:.2f} dB"
                  f" | MAE {active['image_mae_input']:.5f} → {active['image_mae_restored']:.5f}")
            print(f"  Chỉ frame có blur: lỗi gradient {active['strong_edge_gradient_mae_input']:.5f}"
                  f" → {active['strong_edge_gradient_mae_restored']:.5f}")
            for band in ("LH", "HL", "HH"):
                values = active["bands"][band]
                print(f"    {band} RMSE input → restored:"
                      f" {values['rmse_input_to_clean']:.5f} →"
                      f" {values['rmse_restored_to_clean']:.5f}")
    print("\nDiễn giải: blur_only cần giảm lỗi gradient trên cạnh thật và giảm RMSE băng LH/HL.")
    print("Nếu RMS out gần clean nhưng RMSE/gradient không giảm, năng lượng tăng sai vị trí.")
    print(f"Đã lưu: {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="valid", choices=("valid", "test"))
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--samples", type=int,
                        help="Spread this many frames across the whole split; overrides --batches")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    audit(args.checkpoint, args.manifest, output=args.output,
          split=args.split, batches=args.batches, samples=args.samples, device=args.device)


if __name__ == "__main__":
    main()
