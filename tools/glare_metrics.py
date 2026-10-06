"""Do chi so anh cua mot checkpoint phase 2 tren frame co den va loe sang -- in bao cao de dan cho agent.

Cung mot bo frame validation duoc lam nhieu HAI lan voi cung tham so camera (loe sang boc tu luong
ngau nhien rieng): co loe (configs/kaggle_glare.yaml) va khong loe. Moi lan do dau vao va anh khoi
phuc so voi anh sach:

  * chi so chung: PSNR, SSIM, MAE, sai so mau, do bao hoa va tuong phan so voi anh sach, nang luong
    duong net (chu ky 4-16 px), do nham thua;
  * theo vung cua anh sach: vung toi (Y < 0.15) -- model co lam sang cho toi khong; vung sang
    (Y > 0.7); vung quang (noi dau vao co loe sang hon dau vao khong loe > 0.05) -- model go quang
    toi dau; ti le pixel chay trang (Y > 0.98).

Checkpoint train truoc khi co loe thi day la phep thu ngoai phan phoi. Checkpoint va bank validation
co dinh khong doi; seed co dinh nen chay lai ra dung so lieu, so duoc giua cac checkpoint.

    python tools/glare_metrics.py --checkpoint outputs/<run>/phase2/best_joint_validation.pt \\
        --manifest manifests/kaggle --count 64 --image-mode blur_low_light --json report.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qjepa.cli import _dataset, _system_from_phase2  # noqa: E402
from qjepa.data import read_manifest  # noqa: E402
from qjepa.evaluation.metrics import image_metrics  # noqa: E402
from qjepa.models.color_edge import luminance  # noqa: E402
from tools.random_pair_preview import _bright_indices, _eligible_indices, with_glare  # noqa: E402

DARK, BRIGHT, HALO, WHITE = 0.15, 0.7, 0.05, 0.98
COLUMNS = (("glare_input", "vào (lóe)"), ("glare_output", "ra (lóe)"),
           ("plain_input", "vào (k.lóe)"), ("plain_output", "ra (k.lóe)"))


def region_metrics(image: torch.Tensor, clean: torch.Tensor, halo: torch.Tensor,
                   shadow: torch.Tensor | None = None) -> dict[str, float]:
    """Error and brightness of ``image`` by region of ``clean`` ([1,3,H,W] in [0,1]; ``halo``/``shadow`` [H,W] bool).

    ``halo``: where the light corruption made the input brighter; ``shadow``: where it made it darker."""
    image, clean = image.clamp(0, 1), clean.clamp(0, 1)
    y, y_clean = luminance(image)[0, 0], luminance(clean)[0, 0]
    error = (image - clean).abs().mean(1)[0] * 255.0
    values = {"white_fraction": float((y > WHITE).float().mean()),
              "white_fraction_clean": float((y_clean > WHITE).float().mean())}
    regions = [("dark", y_clean < DARK), ("bright", y_clean > BRIGHT), ("halo", halo)]
    if shadow is not None:
        regions.append(("shadow", shadow))
    for name, mask in regions:
        values[f"{name}_fraction"] = float(mask.float().mean())
        if mask.any():
            values[f"{name}_mae255"] = float(error[mask].mean())
            values[f"{name}_brightness"] = float(y[mask].mean() / y_clean[mask].mean().clamp_min(1e-4))
        else:
            values[f"{name}_mae255"] = values[f"{name}_brightness"] = math.nan
    return values


def frame_metrics(image: torch.Tensor, clean: torch.Tensor, halo: torch.Tensor,
                  shadow: torch.Tensor) -> dict[str, float]:
    values = image_metrics(image, clean)
    values["image_mae255"] = values.pop("image_mae") * 255.0
    values.update(region_metrics(image, clean, halo, shadow))
    return values


def _restore(system, sample: dict, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        return system(sample["image_noisy"].unsqueeze(0).to(device),
                      sample["imu_noisy_phys"].T.unsqueeze(0).to(device),
                      sample["image_time"].unsqueeze(0).to(device),
                      sample["imu_times"].unsqueeze(0).to(device)).image.float()


def measure(checkpoint: str | Path, manifest_path: str | Path, *, glare_config: str | Path,
            count: int = 64, split: str = "valid", image_mode: str = "blur_low_light",
            frames: str = "bright", glare_probability: float = 1.0, seed: int = 0,
            device: str = "cuda") -> dict:
    if count < 1:
        raise ValueError("count must be positive")
    device = torch.device(device)
    manifest = read_manifest(manifest_path)
    system, config = _system_from_phase2(str(checkpoint), device)
    system.eval()
    trained_light = float(config["corruption"]["image"].get("light_probability", 0.0))
    trained_illum = float(config["corruption"]["image"].get("illum_probability", 0.0))
    plain = copy.deepcopy(config)
    plain["corruption"]["image"]["light_probability"] = 0.0
    plain["corruption"]["image"]["illum_probability"] = 0.0
    glare = with_glare(config, glare_config, glare_probability)
    datasets = {name: _dataset(cfg, manifest, split, fixed_realization=True, image_mode=image_mode)
                for name, cfg in (("glare", glare), ("plain", plain))}
    rng = np.random.default_rng(seed)
    eligible = _eligible_indices(datasets["glare"], image_mode)
    if frames == "bright":
        chosen = _bright_indices(datasets["glare"], eligible, rng, want=count, scan=max(300, 20 * count))
        if len(chosen) < count:
            print(f"[đo] chỉ {len(chosen)} frame có vùng sáng; thêm frame bất kỳ cho đủ {count}", flush=True)
            rest = [index for index in eligible if index not in set(chosen)]
            chosen += [rest[int(i)] for i in rng.permutation(len(rest))[:count - len(chosen)]]
    else:
        chosen = [eligible[int(i)] for i in rng.permutation(len(eligible))[:count]]
    started = time.perf_counter()
    rows = []
    for position, index in enumerate(chosen):
        with_light, without = datasets["glare"][index], datasets["plain"][index]
        clean = with_light["image_clean"].unsqueeze(0).to(device)
        inputs = {"glare": with_light["image_noisy"].unsqueeze(0).to(device),
                  "plain": without["image_noisy"].unsqueeze(0).to(device)}
        change = (luminance(inputs["glare"]) - luminance(inputs["plain"]))[0, 0]
        halo, shadow = change > HALO, change < -HALO
        row = {"sample_id": with_light["sample_id"], "light": bool(with_light["corruption"]["image"].get("light"))}
        for name, sample in (("glare", with_light), ("plain", without)):
            row[f"{name}_input"] = frame_metrics(inputs[name], clean, halo, shadow)
            row[f"{name}_output"] = frame_metrics(_restore(system, sample, device), clean, halo, shadow)
        rows.append(row)
        if (position + 1) % 16 == 0 or position + 1 == len(chosen):
            print(f"[đo] {position + 1}/{len(chosen)} frame ({time.perf_counter() - started:.0f} s)", flush=True)
    return {"checkpoint": str(checkpoint), "trained_light_probability": trained_light,
            "trained_illum_probability": trained_illum,
            "image_input": config["model"].get("image_input", "rgb"), "split": split,
            "image_mode": image_mode, "frames": frames, "glare_config": str(glare_config),
            "glare_probability": glare_probability, "seed": seed, "rows": rows}


def _mean(rows: list[dict], column: str, key: str) -> float:
    values = np.array([row[column][key] for row in rows], dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else math.nan


def _ratio(rows: list[dict], column: str, key: str) -> float:
    """Ratio of means, as the repo's reports do: 1.00 = as much as the clean frame."""
    return _mean(rows, column, key) / max(_mean(rows, column, f"{key}_clean"), 1e-12)


def format_report(report: dict) -> str:
    rows = report["rows"]
    lines = ["=== ĐO CHỈ SỐ ẢNH (Cell 14e) — dán nguyên khối này cho agent ===",
             f"checkpoint: {report['checkpoint']}",
             f"model train với lóe sáng: {'có' if report['trained_light_probability'] > 0 else 'KHÔNG'} "
             f"(light_probability {report['trained_light_probability']:g}) | ánh sáng không đều: "
             f"{'có' if report.get('trained_illum_probability', 0) > 0 else 'KHÔNG'} | backbone đọc: {report['image_input']}",
             f"{len(rows)} frame {report['split']}, mode {report['image_mode']}, chọn {report['frames']}, "
             f"lóe {report['glare_probability']:.0%} từ {Path(report['glare_config']).name}, seed {report['seed']} "
             f"| frame có lóe: {sum(row['light'] for row in rows)}/{len(rows)}",
             "", f"{'chỉ số (trung bình)':<46}" + "".join(f"{title:>13}" for _, title in COLUMNS)]

    def line(label, values, fmt):
        lines.append(f"{label:<46}" + "".join(f"{format(v, fmt) if math.isfinite(v) else '—':>13}" for v in values))

    def mean_row(label, key, fmt, scale=1.0):
        line(label, [scale * _mean(rows, column, key) for column, _ in COLUMNS], fmt)

    mean_row("PSNR (dB)", "image_psnr_db", ".2f")
    mean_row("SSIM", "image_ssim", ".3f")
    mean_row("MAE (/255)", "image_mae255", ".2f")
    mean_row("sai số màu (color_error)", "image_color_error", ".4f")
    line("độ bão hoà màu / sạch (1 = như sạch)", [_ratio(rows, c, "image_saturation") for c, _ in COLUMNS], ".2f")
    line("độ tương phản / sạch", [_ratio(rows, c, "image_contrast") for c, _ in COLUMNS], ".2f")
    mean_row("năng lượng đường nét / sạch", "image_edge_power", ".2f")
    mean_row("độ nhám thừa (/255)", "image_excess_roughness", ".2f")
    lines.append("--- theo vùng (tối/sáng: trên ảnh sạch; quầng/tối thêm: nơi nhiễu ánh sáng làm sáng/tối thêm > 0,05) ---")
    for region, label in (("dark", f"vùng tối Y<{DARK}"), ("bright", f"vùng sáng Y>{BRIGHT}"), ("halo", "vùng quầng/sáng thêm"),
                          ("shadow", "vùng tối thêm (nhòe tối)")):
        if f"{region}_fraction" not in rows[0]["glare_input"]:
            continue
        share = 100 * _mean(rows, "glare_input", f"{region}_fraction")
        mean_row(f"{label} ({share:.1f}% px): MAE /255", f"{region}_mae255", ".2f")
        mean_row(f"{label}: độ sáng / sạch", f"{region}_brightness", ".2f")
    mean_row(f"pixel cháy trắng Y>{WHITE} (%)", "white_fraction", ".2f", 100.0)
    lines.append(f"  (ảnh sạch: {100 * _mean(rows, 'glare_input', 'white_fraction_clean'):.2f}%)")

    psnr = {column: _mean(rows, column, "image_psnr_db") for column, _ in COLUMNS}
    lines += ["--- tóm tắt ---",
              f"nhiễu ánh sáng làm PSNR đầu vào đổi {psnr['glare_input'] - psnr['plain_input']:+.2f} dB",
              f"model: {psnr['glare_input']:.2f} → {psnr['glare_output']:.2f} dB trên ảnh lóe "
              f"({psnr['glare_output'] - psnr['glare_input']:+.2f}); {psnr['plain_input']:.2f} → "
              f"{psnr['plain_output']:.2f} dB không lóe ({psnr['plain_output'] - psnr['plain_input']:+.2f})",
              "5 frame model kém nhất trên ảnh lóe (PSNR vào → ra | % quầng):"]
    worst = sorted(rows, key=lambda row: row["glare_output"]["image_psnr_db"] - row["glare_input"]["image_psnr_db"])
    for row in worst[:5]:
        lines.append(f"  {row['sample_id'][:46]:<46} {row['glare_input']['image_psnr_db']:6.2f} → "
                     f"{row['glare_output']['image_psnr_db']:6.2f} | {100 * row['glare_input']['halo_fraction']:4.1f}%")
    lines.append("=== hết ===")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--glare-config", default=str(Path(__file__).resolve().parent.parent / "configs/kaggle_glare.yaml"))
    parser.add_argument("--glare-probability", type=float, default=1.0)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--image-mode", choices=("full", "blur_low_light", "low_light_only"), default="blur_low_light")
    parser.add_argument("--frames", choices=("bright", "random"), default="bright",
                        help="bright = frame có đèn/cửa sổ sáng (lóe hiện rõ); random = bất kỳ")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", help="lưu số liệu từng frame")
    args = parser.parse_args()
    report = measure(args.checkpoint, args.manifest, glare_config=args.glare_config, count=args.count,
                     split=args.split, image_mode=args.image_mode, frames=args.frames,
                     glare_probability=args.glare_probability, seed=args.seed, device=args.device)
    text = format_report(report)
    print(text)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        Path(args.json).with_suffix(".txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
