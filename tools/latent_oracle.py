"""D1 — Latent "hoàn hảo" có làm ảnh nét hơn không? Chỉ đọc checkpoint phase 2, không train.

Trên cùng các frame valid (fixed realization), decoder đã train chạy nhiều lần,
mỗi lần chỉ đổi latent ZI; decoder luôn đọc ẢNH HỎNG như lúc dùng thật:

  ZI thường   ZI của ảnh hỏng — đúng như lúc chạy thật.
  ZI sạch     ZI của chính ẢNH SẠCH qua cùng encoder (cùng IMU, cùng thời gian).
              Đây là latent tốt nhất encoder này cho được; JEPA được train để đưa
              ZI của ảnh hỏng về đúng chỗ này.
  ZI = 0      Mốc tham chiếu: decoder dựa vào latent nhiều hay ít.

Nếu decoder đọc thêm các tầng mịn của encoder (``split_edge_naf_stage_levels``),
có thêm một nhánh "ZI + tầng sạch" thay cả các tầng đó.

Kèm khoảng cách latent: ‖ZI hỏng − ZI sạch‖ / ‖ZI sạch‖ và cosine từng ô, so với
khoảng cách tới ZI sạch của một frame KHÁC trong batch — để biết JEPA đã đưa
latent của ảnh hỏng về gần latent ảnh sạch tới đâu.

Đọc kết quả (cột lỗi gradient trên cạnh thật và RMSE băng LH/HL/HH):
  ZI sạch nét hơn rõ so với ZI thường  -> latent là đòn bẩy, nhưng phase 1 chưa
                                          đưa được ZI ảnh hỏng về ZI ảnh sạch:
                                          sửa phase 1.
  ZI sạch ≈ ZI thường, khoảng cách nhỏ -> JEPA đã làm đúng việc được giao, nhưng
                                          chính latent ảnh sạch 16×16 cũng không
                                          mang vị trí cạnh: cần latent mịn hơn.
  ZI sạch ≈ ZI thường, khoảng cách lớn -> decoder không dùng latent cho đường nét:
                                          độ nét phải đến từ phase 2.
  ZI sạch chỉ hơn ở băng LL            -> latent chỉ mang độ sáng / bố cục.

    python3 tools/latent_oracle.py --checkpoint <phase2 best.pt> --manifest <manifest> \\
        --output outputs/<run>/diagnostics/latent_oracle.json --samples 256
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import Subset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from qjepa.cli import _dataset, _loader, _system_from_phase2, _to_device  # noqa: E402
from qjepa.data import read_manifest  # noqa: E402
from tools.image_blur_audit import (  # noqa: E402
    SCENARIOS, _empty_bands, _finish, _record, _spread_indices,
)

ARM_LABELS = {
    "normal": "ZI thường (ảnh hỏng)",
    "clean": "ZI sạch (latent hoàn hảo)",
    "clean_stages": "ZI + tầng sạch",
    "zero": "ZI = 0 (mốc)",
}


def latent_gap(noisy: torch.Tensor, clean: torch.Tensor) -> dict[str, float]:
    """Sums for ‖noisy − clean‖² / ‖clean‖², per-cell cosine, and the same against another frame.

    ``noisy``/``clean`` are [B, C, *grid]. The "other frame" reference rolls the
    batch by one, so it needs B > 1; with B == 1 it is skipped.
    """
    if noisy.shape != clean.shape:
        raise ValueError("noisy and clean latents must have the same shape")
    flat_noisy = noisy.flatten(2)
    flat_clean = clean.flatten(2)
    values = {
        "squared_error": float((flat_noisy - flat_clean).square().sum()),
        "squared_clean": float(flat_clean.square().sum()),
        "cosine_sum": float(torch.nn.functional.cosine_similarity(flat_noisy, flat_clean, dim=1).sum()),
        "cells": flat_clean.shape[0] * flat_clean.shape[2],
    }
    if noisy.shape[0] > 1:
        other = flat_clean.roll(1, dims=0)
        values["other_squared_error"] = float((flat_noisy - other).square().sum())
        values["other_cosine_sum"] = float(torch.nn.functional.cosine_similarity(flat_noisy, other, dim=1).sum())
        values["other_cells"] = values["cells"]
    return values


def finish_gap(sums: dict[str, float]) -> dict[str, float]:
    result = {
        "relative_distance": math.sqrt(sums["squared_error"] / max(sums["squared_clean"], 1e-12)),
        "cosine": sums["cosine_sum"] / max(sums["cells"], 1),
    }
    if sums.get("other_cells"):
        result["relative_distance_to_other_frame"] = math.sqrt(
            sums["other_squared_error"] / max(sums["squared_clean"], 1e-12))
        result["cosine_to_other_frame"] = sums["other_cosine_sum"] / sums["other_cells"]
    return result


def oracle(checkpoint: str | Path, manifest_path: str | Path, *, output: str | Path,
           split: str = "valid", samples: int = 256, scenarios: tuple[str, ...] = ("blur_only", "full_full"),
           device: str = "cuda") -> dict:
    device_obj = torch.device(device)
    system, config = _system_from_phase2(str(checkpoint), device_obj)
    manifest = read_manifest(manifest_path)
    indices = _spread_indices(len(manifest["samples"][split]), samples)
    config["data"] = dict(config["data"], num_workers=0, pin_memory=False)
    uses_stages = bool(getattr(system.decoders.image, "uses_stages", False))
    arms = ["normal", "clean"] + (["clean_stages"] if uses_stages else []) + ["zero"]
    transform = system.backbone.image_transform
    report: dict[str, dict] = {}
    for scenario in scenarios:
        image_mode, imu_mode = SCENARIOS[scenario]
        dataset = _dataset(config, manifest, split, fixed_realization=True,
                           image_mode=image_mode, imu_mode=imu_mode)
        loader = _loader(config, Subset(dataset, indices), config["phase2"]["batch_size"], train=False)
        stats = {arm: {} for arm in arms}
        bands = {arm: _empty_bands() for arm in arms}
        gap: dict[str, float] = {}
        seen = pixels = 0
        for raw in loader:
            batch = _to_device(raw, device_obj)
            with torch.no_grad():
                latent = system.encode(batch["image_noisy"], batch["imu_noisy_phys"],
                                       batch["image_time"], batch["imu_times"])
                # Same IMU and timestamps: only the image the encoder reads changes.
                latent_clean = system.encode(batch["image_clean"], batch["imu_noisy_phys"],
                                             batch["image_time"], batch["imu_times"])
                variants = {
                    "normal": latent,
                    "clean": replace(latent, ZI=latent_clean.ZI),
                    "zero": replace(latent, ZI=torch.zeros_like(latent.ZI)),
                }
                if uses_stages:
                    variants["clean_stages"] = replace(latent, ZI=latent_clean.ZI,
                                                       image_skips=latent_clean.image_skips)
                clean = batch["image_clean"].clamp(0, 1)
                noisy = batch["image_noisy"].clamp(0, 1)
                clean_coefficients, _ = transform.analysis(clean)
                for arm in arms:
                    restored = system.decode(variants[arm]).image.clamp(0, 1)
                    # Bands of the restored image itself, as the blur audit scores them.
                    restored_coefficients, _ = transform.analysis(restored)
                    _record(stats[arm], bands[arm], clean, noisy, restored, clean_coefficients,
                            latent.image_coefficients, restored_coefficients)
                for key, value in latent_gap(latent.ZI, latent_clean.ZI).items():
                    gap[key] = gap.get(key, 0.0) + value
            seen += clean.shape[0]
            pixels += clean.numel()
        if not seen:
            raise ValueError("Validation loader returned no images")
        report[scenario] = {
            "samples": seen,
            "latent_gap": finish_gap(gap),
            "arms": {arm: _finish(stats[arm], bands[arm], seen, pixels) for arm in arms},
        }

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"checkpoint": str(checkpoint), "split": split, "spread_samples": len(indices),
                                "decoder_uses_stages": uses_stages, "scenarios": report}, indent=2),
                    encoding="utf-8")
    print_report(report)
    print(f"Đã lưu: {path}")
    return report


def _reduction(before: float, after: float) -> float:
    return 100.0 * (1.0 - after / max(before, 1e-12))


def print_report(report: dict) -> None:
    print("D1 — latent hoàn hảo · cùng frame valid, fixed realization · decoder luôn đọc ảnh HỎNG")
    print("LL/LH/HL/HH = RMSE từng băng QWT so với ảnh sạch, lỗi grad cạnh = trên 10% cạnh mạnh nhất"
          " (thấp hơn là tốt); độ mạnh cạnh càng gần ảnh sạch càng tốt.")
    for scenario, result in report.items():
        gap = result["latent_gap"]
        other = ""
        if "relative_distance_to_other_frame" in gap:
            other = (f" | tới ZI sạch của frame KHÁC: {gap['relative_distance_to_other_frame']:.3f}"
                     f" (cos {gap['cosine_to_other_frame']:.3f})")
        print(f"\n{scenario}: {result['samples']} ảnh")
        print(f"  ZI ảnh hỏng cách ZI ảnh sạch: {gap['relative_distance']:.3f} (cos {gap['cosine']:.3f}){other}")
        normal = result["arms"]["normal"]
        header = (f"  {'':<28}{'PSNR':>7}{'MAE':>9}{'LL':>9}{'LH':>9}{'HL':>9}{'HH':>9}"
                  f"{'lỗi grad cạnh':>15}{'độ mạnh cạnh':>14}")
        print(header)
        print(f"  {'ảnh vào':<28}{normal['image_psnr_input_db']:>7.2f}{normal['image_mae_input']:>9.5f}"
              + "".join(f"{normal['bands'][b]['rmse_input_to_clean']:>9.5f}" for b in ("LL", "LH", "HL", "HH"))
              + f"{normal['strong_edge_gradient_mae_input']:>15.5f}{normal['input_strong_edge_magnitude']:>14.5f}")
        for arm, values in result["arms"].items():
            print(f"  {ARM_LABELS[arm]:<28}{values['image_psnr_restored_db']:>7.2f}{values['image_mae_restored']:>9.5f}"
                  + "".join(f"{values['bands'][b]['rmse_restored_to_clean']:>9.5f}" for b in ("LL", "LH", "HL", "HH"))
                  + f"{values['strong_edge_gradient_mae_restored']:>15.5f}"
                  f"{values['restored_strong_edge_magnitude']:>14.5f}")
        print(f"  {'ảnh sạch':<28}{'':>7}{'':>9}{'':>36}{0.0:>15.5f}{normal['clean_strong_edge_magnitude']:>14.5f}")
        best = result["arms"]["clean"]
        print("  Latent hoàn hảo so với ZI thường — giảm thêm bao nhiêu sai số còn lại:")
        print(f"    lỗi gradient cạnh {_reduction(normal['strong_edge_gradient_mae_restored'], best['strong_edge_gradient_mae_restored']):+.1f}%"
              + "".join(f" · {b} {_reduction(normal['bands'][b]['rmse_restored_to_clean'], best['bands'][b]['rmse_restored_to_clean']):+.1f}%"
                        for b in ("LL", "LH", "HL", "HH"))
              + f" · PSNR {best['image_psnr_restored_db'] - normal['image_psnr_restored_db']:+.2f} dB")
    print("\nĐọc: xem docstring đầu file (python3 tools/latent_oracle.py --help).")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="checkpoint PHASE 2")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="file JSON kết quả")
    parser.add_argument("--split", default="valid", choices=("valid", "test"))
    parser.add_argument("--samples", type=int, default=256, help="số frame, rải đều trên cả split")
    parser.add_argument("--scenarios", nargs="+", default=["blur_only", "full_full"], choices=sorted(SCENARIOS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    oracle(args.checkpoint, args.manifest, output=args.output, split=args.split, samples=args.samples,
           scenarios=tuple(args.scenarios), device=args.device)


if __name__ == "__main__":
    main()
