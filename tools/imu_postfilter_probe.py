"""A: what would a FIXED filter on the model's IMU output still gain? No training.

The IMU decoder returns the noisy signal plus a correction. If a plain Gaussian or
median filter applied AFTER the model still lowers the error, the model is leaving
that much on the table -- and the filter can be used at inference as it is. With
the IMU refiner (phase2.imu_refiner_blocks) this also says whether the refiner has
already taken what a fixed filter would. Scored on the checkpoint's own validation
bank (the windows phase-2 validation uses), in physical units, per sensor:

  RMSE          lower = closer to the clean IMU
  rung RMSE     RMSE of the error of the sample-to-sample change (m/s^3, rad/s^2):
                the report's "variation RMSE"
  rung/tong     that change error over the total error: the report's jitter ratio
  rung thua     mean change the output has BEYOND the clean signal's, per sample
                (0 = no extra jitter; the quantity phase2.imu_jitter_weight trains)

    python3 tools/imu_postfilter_probe.py \
        --checkpoint outputs/<run>/phase2/best_joint_validation.pt --manifest manifests/kaggle

Filters run on each 128-sample window with the edges replicated, as the model sees
it. One device, no DataParallel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qjepa.cli import _dataset, _fixed_validation_bank, _loader, _manifest, _system_from_phase2  # noqa: E402
from qjepa.config import resolve_device  # noqa: E402
from qjepa.execution import RestorationForward  # noqa: E402

SENSORS = {"accel": slice(0, 3), "gyro": slice(3, 6)}


def gaussian(signal: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian along time, edges replicated; sigma in samples (10 ms at 100 Hz)."""
    radius = max(1, int(round(3 * sigma)))
    positions = torch.arange(-radius, radius + 1, dtype=signal.dtype, device=signal.device)
    kernel = torch.exp(-0.5 * (positions / sigma) ** 2)
    kernel = (kernel / kernel.sum()).view(1, 1, -1).repeat(signal.shape[1], 1, 1)
    padded = F.pad(signal, (radius, radius), mode="replicate")
    return F.conv1d(padded, kernel, groups=signal.shape[1])


def median(signal: torch.Tensor, width: int) -> torch.Tensor:
    """Running median along time, edges replicated: removes a spike instead of spreading it."""
    radius = width // 2
    padded = F.pad(signal, (radius, radius), mode="replicate")
    return padded.unfold(-1, width, 1).median(dim=-1).values


CANDIDATES = {
    "model": lambda x: x,
    "model + Gauss 0.5": lambda x: gaussian(x, 0.5),
    "model + Gauss 1": lambda x: gaussian(x, 1.0),
    "model + Gauss 1.5": lambda x: gaussian(x, 1.5),
    "model + Gauss 2": lambda x: gaussian(x, 2.0),
    "model + median 3": lambda x: median(x, 3),
    "model + median 5": lambda x: median(x, 5),
    "model + median 3 + Gauss 1": lambda x: gaussian(median(x, 3), 1.0),
}


class _Totals:
    """Running sums per (output, sensor), so the metrics cover the whole bank."""

    def __init__(self) -> None:
        self.sums: dict[tuple[str, str], list[float]] = {}

    def add(self, name: str, output: torch.Tensor, clean: torch.Tensor, dt: torch.Tensor) -> None:
        error = output - clean
        rate_error = torch.diff(error, dim=-1) / dt
        extra = F.relu(torch.diff(output, dim=-1).abs() - torch.diff(clean, dim=-1).abs())
        for sensor, axes in SENSORS.items():
            row = self.sums.setdefault((name, sensor), [0.0] * 6)
            row[0] += float(error[:, axes].square().sum())
            row[1] += error[:, axes].numel()
            row[2] += float(rate_error[:, axes].square().sum())
            row[3] += float((rate_error[:, axes] * dt).square().sum())
            row[4] += rate_error[:, axes].numel()
            row[5] += float(extra[:, axes].sum())

    def metrics(self, name: str, sensor: str) -> dict[str, float]:
        squared, count, rate_squared, change_squared, change_count, extra = self.sums[(name, sensor)]
        rmse = (squared / count) ** 0.5
        return {
            "rmse": rmse,
            "rate_rmse": (rate_squared / change_count) ** 0.5,
            # per-sample change of the error over the error itself: the report's jitter ratio
            "jitter_ratio": (change_squared / change_count) ** 0.5 / max(rmse, 1e-30),
            "excess_jitter": extra / change_count,
        }


@torch.no_grad()
def probe(checkpoint: Path, manifest_path: str | None, device: torch.device, max_batches: int | None) -> dict:
    system, config = _system_from_phase2(str(checkpoint), device)
    forward = RestorationForward(system).eval()
    manifest, _ = _manifest(config, manifest_path)
    batch_size = config["phase2"]["batch_size"]
    dataset = _dataset(config, manifest, "valid", fixed_realization=True)
    _fixed_validation_bank(dataset, config["runtime"]["validation_batches"] * batch_size)
    loader = _loader(config, dataset, batch_size, train=False)
    totals = _Totals()
    windows = 0
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        noisy = batch["imu_noisy_phys"].to(device)
        clean = batch["imu_clean_phys"].to(device)
        times = batch["imu_times"].to(device)
        dt = torch.diff(times, dim=-1)[:, None, :].to(clean.dtype)
        restored = forward(batch["image_noisy"].to(device), noisy, batch["image_time"].to(device), times)
        totals.add("input (chua xu ly)", noisy, clean, dt)
        for name, filtered in CANDIDATES.items():
            totals.add(name, filtered(restored["imu_physical"]), clean, dt)
        windows += noisy.shape[0]
    return {"checkpoint": str(checkpoint), "windows": windows,
            "rows": {name: {sensor: totals.metrics(name, sensor) for sensor in SENSORS}
                     for name in ("input (chua xu ly)", *CANDIDATES)}}


def report(results: dict) -> str:
    lines = [f"Loc co dinh tren dau ra IMU — {results['checkpoint']}",
             f"{results['windows']} cua so validation, don vi vat ly; thap hon la tot o moi cot", ""]
    for sensor in SENSORS:
        lines.append(f"{sensor.upper()}")
        lines.append(f"  {'dau ra':<28}{'RMSE':>10}{'rung RMSE':>12}{'rung/tong':>11}{'rung thua':>12}")
        for name, rows in results["rows"].items():
            row = rows[sensor]
            lines.append(f"  {name:<28}{row['rmse']:>10.4f}{row['rate_rmse']:>12.3f}"
                         f"{row['jitter_ratio']:>11.2f}{row['excess_jitter']:>12.5f}")
        model = results["rows"]["model"][sensor]
        best_name = min(CANDIDATES, key=lambda name: results["rows"][name][sensor]["rmse"])
        best = results["rows"][best_name][sensor]
        if best_name == "model":
            lines.append("  -> Khong bo loc co dinh nao ha duoc RMSE: model da tu lam muot het phan loc de lay.")
        else:
            gain = 100.0 * (1.0 - best["rmse"] / model["rmse"])
            lines.append(f"  -> Tot nhat: {best_name}: RMSE -{gain:.1f}%, rung RMSE "
                         f"{model['rate_rmse']:.3f} -> {best['rate_rmse']:.3f}. Model con bo lo phan nay.")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="phase-2 checkpoint (.pt)")
    parser.add_argument("--manifest", help="manifest folder; default: the one in the checkpoint's config")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, help="default: the whole validation bank")
    parser.add_argument("--json", type=Path, help="also write the numbers here")
    args = parser.parse_args()
    results = probe(args.checkpoint, args.manifest, resolve_device(args.device), args.max_batches)
    print(report(results))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
