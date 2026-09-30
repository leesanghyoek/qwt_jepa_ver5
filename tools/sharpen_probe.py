"""A for the image: how much post-sharpening does a phase-2 checkpoint take? No training.

Every phase-2 loss is a mean error, so where the model is unsure of fine detail it
answers with the average: the restored frames keep about half of the clean frame's
2-4 px detail (p16: 0.465 in place). An unsharp mask on the luminance --
Y + amount * (Y - Gauss_sigma(Y)), the same value added to R, G and B so the colour
does not move -- turns up the detail that is there. It cannot bring back what is not.
This scores a grid of (sigma, amount) on the checkpoint's own validation bank:

  PSNR, SSIM               fall once the boost amplifies more wrong detail than right
  2-4 px in place          grows with ANY boost (it is linear in the output): not a
                           criterion on its own
  2-4 px power             detail energy against the clean frame's: 1.00 = as much as
                           clean; above 1 is over-sharpened
  2 px stripes, roughness  grain, ringing and halos the boost brings

and picks the strongest setting that keeps PSNR within ``--max-psnr-drop`` of the
model and the 2-4 px power at or below the clean frame's. It also saves zoomed
crops (input | model | model + pick | clean) to look at.

    python3 tools/sharpen_probe.py --checkpoint outputs/<run>/phase2/best_joint_validation.pt \
        --manifest manifests/kaggle --panel outputs/<run>/diagnostics/sharpen_panel.png
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
from qjepa.evaluation.metrics import _luma_spectrum, image_metrics  # noqa: E402
from qjepa.execution import RestorationForward  # noqa: E402
from qjepa.models.color_edge import luminance  # noqa: E402

SIGMAS = (0.8, 1.2, 2.0)
AMOUNTS = (0.25, 0.5, 1.0, 1.5, 2.0)
KEYS = ("image_psnr_db", "image_ssim", "image_fine_detail_in_place", "fine_power", "image_edge_in_place",
        "image_stripe_power", "image_excess_roughness")


def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian, edges replicated; sigma in pixels."""
    radius = max(1, int(round(3 * sigma)))
    positions = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    kernel = torch.exp(-0.5 * (positions / sigma) ** 2)
    kernel = kernel / kernel.sum()
    channels = image.shape[1]
    padded = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    padded = F.conv2d(padded, kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1), groups=channels)
    return F.conv2d(padded, kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1), groups=channels)


def sharpen(image: torch.Tensor, sigma: float, amount: float) -> torch.Tensor:
    """Unsharp mask on the luminance; the same offset on R, G, B keeps Cb and Cr."""
    y = luminance(image)
    return (image + amount * (y - gaussian_blur(y, sigma))).clamp(0.0, 1.0)


def fine_power(restored: torch.Tensor, clean: torch.Tensor) -> float:
    """2-4 px detail energy against the clean frame's (same band as image_fine_detail_in_place)."""
    got, want = _luma_spectrum(restored).abs().square().sum(0), _luma_spectrum(clean).abs().square().sum(0)
    fy = torch.fft.fftfreq(got.shape[0], device=got.device).abs()[:, None]
    fx = torch.fft.fftfreq(got.shape[1], device=got.device).abs()[None, :]
    fine = torch.maximum(fy, fx) >= 1 / 4
    return float(got[fine].sum() / want[fine].sum().clamp_min(1e-12))


def _score(restored: torch.Tensor, clean: torch.Tensor) -> dict[str, float]:
    values = image_metrics(restored, clean)
    values["fine_power"] = fine_power(restored.clamp(0, 1), clean.clamp(0, 1))
    return {key: values[key] for key in KEYS}


def pick(rows: dict[str, dict[str, float]], max_psnr_drop: float) -> str:
    """Strongest boost (highest 2-4 px power) within the PSNR budget, not above clean's power."""
    model = rows["model"]
    allowed = [name for name, row in rows.items() if name not in ("input",)
               and row["image_psnr_db"] >= model["image_psnr_db"] - max_psnr_drop
               and row["fine_power"] <= max(1.0, model["fine_power"])]
    return max(allowed, key=lambda name: rows[name]["fine_power"]) if allowed else "model"


def _panel(frames: list[tuple[torch.Tensor, ...]], labels: tuple[str, ...], path: Path, crop: int = 96) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(frames), len(labels), figsize=(3.2 * len(labels), 3.2 * len(frames)))
    axes = axes.reshape(len(frames), len(labels))
    for row, images in enumerate(frames):
        clean = images[-1]
        crop = min(crop, *clean.shape[-2:])
        # the crop with the most clean detail: where sharpness shows
        detail = (luminance(clean[None]) - gaussian_blur(luminance(clean[None]), 1.2)).abs()[0, 0]
        stride = max(1, crop // 4)
        score = F.avg_pool2d(detail[None, None], crop, stride=stride)[0, 0]
        top, left = divmod(int(score.argmax()), score.shape[1])
        top, left = top * stride, left * stride
        for column, (image, label) in enumerate(zip(images, labels)):
            patch = image[:, top:top + crop, left:left + crop].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            axes[row, column].imshow(patch, interpolation="nearest")
            axes[row, column].set_title(label, fontsize=10)
            axes[row, column].axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


@torch.no_grad()
def probe(checkpoint: Path, manifest_path: str | None, device: torch.device, max_batches: int | None,
          max_psnr_drop: float, panel: Path | None, panel_frames: int = 3) -> dict:
    system, config = _system_from_phase2(str(checkpoint), device)
    forward = RestorationForward(system).eval()
    manifest, _ = _manifest(config, manifest_path)
    batch_size = config["phase2"]["batch_size"]
    dataset = _dataset(config, manifest, "valid", fixed_realization=True)
    _fixed_validation_bank(dataset, config["runtime"]["validation_batches"] * batch_size)
    loader = _loader(config, dataset, batch_size, train=False)
    names = ["input", "model"] + [f"sigma {s} · amount {a}" for s in SIGMAS for a in AMOUNTS]
    sums = {name: dict.fromkeys(KEYS, 0.0) for name in names}
    batches, kept = 0, []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        noisy, clean = batch["image_noisy"].to(device), batch["image_clean"].to(device)
        restored = forward(noisy, batch["imu_noisy_phys"].to(device), batch["image_time"].to(device),
                           batch["imu_times"].to(device))["image"].clamp(0, 1)
        outputs = {"input": noisy, "model": restored}
        outputs.update({f"sigma {s} · amount {a}": sharpen(restored, s, a) for s in SIGMAS for a in AMOUNTS})
        for name, output in outputs.items():
            for key, value in _score(output, clean).items():
                sums[name][key] += value
        if len(kept) < panel_frames:
            kept.append((noisy[0], restored[0], clean[0]))
        batches += 1
    rows = {name: {key: value / max(batches, 1) for key, value in row.items()} for name, row in sums.items()}
    choice = pick(rows, max_psnr_drop)
    if panel is not None and kept:
        sigma, amount = (float(part.split()[-1]) for part in choice.split(" · ")) if choice != "model" else (1.2, 0.0)
        frames = [(n, r, sharpen(r[None], sigma, amount)[0], c) for n, r, c in kept]
        _panel(frames, ("ảnh vào", "model", f"model + làm nét ({choice})" if choice != "model" else "model",
                        "ảnh sạch"), panel)
    return {"checkpoint": str(checkpoint), "frames": batches * batch_size, "max_psnr_drop": max_psnr_drop,
            "rows": rows, "pick": choice}


def report(results: dict) -> str:
    lines = [f"Làm nét sau xử lý (unsharp mask trên độ sáng) — {results['checkpoint']}",
             f"{results['frames']} ảnh validation · chọn: mạnh nhất mà PSNR giảm ≤ {results['max_psnr_drop']} dB "
             "và năng lượng 2-4 px không vượt ảnh sạch", "",
             f"{'đầu ra':<26}{'PSNR':>8}{'SSIM':>8}{'2-4px đúng chỗ':>16}{'2-4px năng lượng':>18}"
             f"{'4-16px đúng chỗ':>17}{'sọc 2px':>9}{'gồ ghề':>9}"]
    for name, row in results["rows"].items():
        mark = "  <- chọn" if name == results["pick"] else ""
        lines.append(f"{name:<26}{row['image_psnr_db']:>8.2f}{row['image_ssim']:>8.4f}"
                     f"{row['image_fine_detail_in_place']:>16.3f}{row['fine_power']:>18.3f}"
                     f"{row['image_edge_in_place']:>17.3f}{row['image_stripe_power']:>9.3f}"
                     f"{row['image_excess_roughness']:>9.3f}{mark}")
    model, chosen = results["rows"]["model"], results["rows"][results["pick"]]
    lines.append("")
    if results["pick"] == "model":
        lines.append("-> Không mức nào qua được điều kiện: làm nét cố định chỉ đổi độ nét lấy sai số ở checkpoint này.")
    else:
        lines.append(f"-> {results['pick']}: 2-4 px năng lượng {model['fine_power']:.3f} -> {chosen['fine_power']:.3f}, "
                     f"PSNR {model['image_psnr_db']:.2f} -> {chosen['image_psnr_db']:.2f}, "
                     f"gồ ghề {model['image_excess_roughness']:.3f} -> {chosen['image_excess_roughness']:.3f}. "
                     "Xem ảnh so sánh trước khi dùng.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="phase-2 checkpoint (.pt)")
    parser.add_argument("--manifest", help="manifest folder; default: the one in the checkpoint's config")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, help="default: the whole validation bank")
    parser.add_argument("--max-psnr-drop", type=float, default=0.2)
    parser.add_argument("--panel", type=Path, help="save zoomed crops here (PNG)")
    parser.add_argument("--json", type=Path, help="also write the numbers here")
    args = parser.parse_args()
    results = probe(args.checkpoint, args.manifest, resolve_device(args.device), args.max_batches,
                    args.max_psnr_drop, args.panel)
    print(report(results))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
