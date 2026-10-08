"""Loi do sang / mau (tan so thap) cua anh vao va anh khoi phuc, theo loai anh, kem moc oracle.

Tren p28, ~95% sai so pixel con lai nam o dai thap tan (LL) cua QWT: do sang va mau o chu ky dai.
Cong cu nay tach loi do theo loai anh, dung tham so nhieu that cua tung frame:
  * "khong doi sang": khong co buoc thieu sang va khong co anh sang khong deu (moi truong trong,
    chi mo, chi nhieu cam bien). Model nen GIU do sang: khoi phuc te hon anh vao la loi sua duoc.
  * "doi sang": con lai.
Tan so thap = anh trung binh khoi 8x8 px (32x32 cho anh 256), RMSE tren sRGB so voi anh sach.
Moc oracle (khong phai model, chi de biet tran):
  * oracle chung: biet dung he so a, b moi kenh cho CA anh (sach ~ a * vao + b, binh phuong toi thieu);
  * oracle vung: nhu tren, rieng cho tung vung 4 x 4.
"sang / sach": do sang trung binh anh khoi phuc chia anh sach; > 1 la lam sang qua, < 1 la con toi.
Model co ban do stop (p32, phase2.split_relight): them "stop doan" = L1 (stop) giua ban do model doan va ban do that
cua buoc nhieu (corruptions.image.brightness_stops), canh "|stop that|" = L1 cua viec doan 0 khap noi.

    python3 tools/brightness_probe.py --checkpoint <phase2 .pt> --manifest <manifest> [--samples 512] [--amp]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qjepa.cli import _dataset, _manifest, _system_from_phase2  # noqa: E402
from qjepa.corruptions.image import brightness_stops  # noqa: E402
from qjepa.data import collate_paired  # noqa: E402
from qjepa.execution import RestorationForward  # noqa: E402

GROUPS = ("khong doi sang", "doi sang", "tat ca")
COLUMNS = ("anh vao", "khoi phuc", "oracle chung", "oracle vung")
STOP_COLUMNS = ("stop doan", "|stop that|")
LUMA = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)


def low(x: torch.Tensor, scale: int = 8) -> torch.Tensor:
    return F.avg_pool2d(x[None], scale)[0]


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).square().mean().sqrt())


def oracle(noisy: torch.Tensor, clean: torch.Tensor, regions: int) -> float:
    """RMSE after the best a * x + b per channel on each of regions x regions areas."""
    _, height, width = noisy.shape
    fitted = torch.empty_like(clean)
    for i in range(regions):
        for j in range(regions):
            area = (slice(None), slice(i * height // regions, (i + 1) * height // regions),
                    slice(j * width // regions, (j + 1) * width // regions))
            x, y = noisy[area].reshape(3, -1), clean[area].reshape(3, -1)
            for c in range(3):
                design = torch.stack((x[c], torch.ones_like(x[c])), dim=1)
                solution = torch.linalg.lstsq(design, y[c][:, None]).solution
                fitted[area][c] = (design @ solution).reshape(fitted[area][c].shape)
    return rmse(fitted.clamp(0, 1), clean)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--split", default="valid", choices=("valid", "test"))
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="fp16 autocast on CUDA, as phase 2 trained")
    parser.add_argument("--output", help="JSON with the table")
    args = parser.parse_args()
    device = torch.device(args.device)
    system, config = _system_from_phase2(args.checkpoint, device)
    system.eval()
    forward = RestorationForward(system)
    forward.amp = bool(args.amp) and device.type == "cuda"
    manifest, _ = _manifest(config, args.manifest)
    dataset = _dataset(config, manifest, args.split, fixed_realization=True)
    indices = np.random.default_rng(0).choice(len(dataset), min(args.samples, len(dataset)), replace=False)
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for start in range(0, len(indices), args.batch):
            samples = [dataset[int(i)] for i in indices[start:start + args.batch]]
            collated = collate_paired(samples)                     # IMU [L, 6] per sample -> [B, 6, L]
            batch = {key: collated[key].to(device) for key in ("image_noisy", "imu_noisy_phys", "image_time",
                                                                "imu_times")}
            outputs = forward(batch["image_noisy"], batch["imu_noisy_phys"], batch["image_time"], batch["imu_times"])
            restored = outputs["image"].float().clamp(0, 1).cpu()
            stops = outputs["image_stops"].float().cpu() if "image_stops" in outputs else None
            for k, sample in enumerate(samples):
                params = sample["corruption"]["image"]
                relit = bool(params.get("low_light")) or bool(params.get("illumination"))
                clean, noisy, out = (low(x) for x in (sample["image_clean"], sample["image_noisy"], restored[k]))
                row = {"anh vao": rmse(noisy, clean), "khoi phuc": rmse(out, clean),
                       "oracle chung": oracle(noisy, clean, 1), "oracle vung": oracle(noisy, clean, 4),
                       "sang / sach": float((out * LUMA).sum(0).mean() / (clean * LUMA).sum(0).mean().clamp_min(1e-3))}
                if stops is not None:
                    truth = torch.from_numpy(brightness_stops(params, *sample["image_noisy"].shape[-2:]))[None, None]
                    truth = F.adaptive_avg_pool2d(truth, stops.shape[-2:])[0]
                    row.update({"stop doan": float((stops[k] - truth).abs().mean()), "|stop that|": float(truth.abs().mean())})
                for group in ("doi sang" if relit else "khong doi sang", "tat ca"):
                    for name, value in row.items():
                        values[group][name].append(value)
    table = {}
    print(f"{len(indices)} anh {args.split}, nhieu day du (fixed realization). RMSE tan so thap (sRGB, khoi 8x8 px), "
          "thap hon la tot.")
    stop_columns = [c for c in STOP_COLUMNS if values["tat ca"].get(c)]
    print(f"{'nhom':<16}{'so anh':>7}" + "".join(f"{c:>14}" for c in COLUMNS) + f"{'te hon vao':>12}{'sang / sach':>13}"
          + "".join(f"{c:>13}" for c in stop_columns))
    for group in GROUPS:
        rows = values[group]
        if not rows:
            continue
        worse = float(np.mean(np.array(rows["khoi phuc"]) > np.array(rows["anh vao"])))
        table[group] = {**{c: float(np.mean(rows[c])) for c in (*COLUMNS, *stop_columns)}, "count": len(rows["anh vao"]),
                        "worse_than_input": worse, "brightness_ratio": float(np.median(rows["sang / sach"]))}
        print(f"{group:<16}{len(rows['anh vao']):>7}" + "".join(f"{table[group][c]:>14.4f}" for c in COLUMNS)
              + f"{100 * worse:>11.0f}%{table[group]['brightness_ratio']:>13.3f}"
              + "".join(f"{table[group][c]:>13.3f}" for c in stop_columns))
    print("\nDoc: 'khong doi sang' te hon vao nhieu -> model lam hong do sang anh dung sang (sua duoc: p30).\n"
          "     'doi sang' khoi phuc gan oracle chung -> da doan dung phoi sang ca anh; xa -> con doan sai.\n"
          "     oracle vung << oracle chung -> phan lon loi la anh sang khong deu theo vung.\n"
          "     stop doan << |stop that| -> model doan dung vung nao bi toi/sang bao nhieu (p32).")
    if args.output:
        Path(args.output).write_text(json.dumps(table, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
