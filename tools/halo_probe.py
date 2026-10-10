"""Model phase 2 go duoc bao nhieu phan loe HALO? Cung mot frame, co loe va khong loe, moi nhieu khac giu nguyen.

Moi frame cua split (mac dinh valid: lop loe cua cac scene HALO danh rieng cho split do, model chua thay luc
train) duoc lam nhieu HAI lan voi cung bo tham so -- mo, hat, phoi sang, ke ca cac buoc moi truong da bo vi co
loe (halo_clear_probability) -- mot lan CO lop loe HALO, mot lan BO lop loe. Model chay tren ca hai ban:
  * "loe lam mat (dB)" = PSNR khong loe - PSNR co loe, cua anh vao va cua anh khoi phuc. Khoi phuc 0 dB = model
    go het thiet hai do loe; bang anh vao = model khong lam gi voi loe.
  * "anh sang loe con lai" = tong (khoi phuc co loe - khoi phuc khong loe) / tong (vao co loe - vao khong loe),
    tren anh sang tuyen tinh: trong vung loe (khoi 8x8 px ma loe them > 0,005) va ca khung. 0% = go het,
    100% = de nguyen, > 100% = lam loe manh them, < 0 = tru qua tay (toi hon ca ban khong loe).
  * "nhe" / "manh": nua it anh sang loe hon / nhieu hon (theo tong anh sang loe them vao anh vao).
Checkpoint KHONG train voi HALO (vd p32) cung do duoc: lop loe lay theo cac khoa halo_* cua
configs/kaggle_halo.yaml, can --halo-root (thu muc co halo_index.csv). So p32 voi p33 cho biet train voi HALO
giup go loe bao nhieu. PSNR tinh tung anh tren sRGB [0, 1] roi lay trung binh.

    python3 tools/halo_probe.py --checkpoint outputs/p33_halo/phase2/best_joint_validation.pt \\
        --manifest manifests/kaggle [--split valid] [--samples 256] [--amp] [--halo-root <thu muc HALO>] \\
        [--output halo_probe.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from qjepa.cli import _dataset, _manifest, _system_from_phase2  # noqa: E402
from qjepa.config import load_config  # noqa: E402
from qjepa.corruptions.halo import HaloFlareBank  # noqa: E402
from qjepa.corruptions.image import LowLightImageCorruptor  # noqa: E402
from qjepa.data import collate_paired  # noqa: E402
from qjepa.execution import RestorationForward  # noqa: E402

REGION = 0.005   # anh sang tuyen tinh ma loe them (trung binh khoi 8x8 px) de tinh la "vung loe"
BLOCK = 8


class WithoutFlare(LowLightImageCorruptor):
    """Cung tham so voi ban co loe (ca buoc moi truong da bo vi loe), chi bo lop loe."""

    def _parameters(self, *args, **kwargs):
        params = super()._parameters(*args, **kwargs)
        if params.get("halo"):
            params = dict(params, halo=False, halo_params=None, halo_dropped=True)
        return params


def linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x.clamp_min(0) + 0.055) / 1.055) ** 2.4)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(10 * torch.log10(1.0 / (a - b).square().mean().clamp_min(1e-10)))


def corruptors(dataset, config: dict, halo_root: str | None):
    """(co loe tren MOI frame, ban sinh doi khong loe), cung bank lop loe va cung seed voi dataset."""
    base = dataset.image_corruptor
    image = config["corruption"]["image"]
    source = image if float(image.get("halo_probability", 0)) > 0 else \
        load_config(REPO / "configs/kaggle_halo.yaml")["corruption"]["image"]
    halo = {key: value for key, value in source.items() if key.startswith("halo_")}
    settings = replace(base.config, **{**halo, "halo_probability": 1.0})
    bank = base.halo_bank
    if bank is None:
        root = halo_root or config["data"].get("halo_root")
        if not root:
            raise SystemExit("Checkpoint nay khong train voi HALO: them --halo-root <thu muc co halo_index.csv>")
        bank = HaloFlareBank(root, revision=settings.halo_revision, effects=settings.halo_effects,
                             holdout_fraction=settings.halo_holdout_fraction)
    return (LowLightImageCorruptor(settings, base.master_seed, halo_bank=bank),
            WithoutFlare(settings, base.master_seed, halo_bank=bank))


def summarize(rows: list[dict]) -> dict:
    def ratio(numerator: str, denominator: str) -> float:
        total = sum(row[denominator] for row in rows)
        return sum(row[numerator] for row in rows) / total if total > 0 else float("nan")

    mean = {key: float(np.mean([row[key] for row in rows])) for key in ("in_with", "in_without", "out_with",
                                                                         "out_without")}
    lost_in, lost_out = mean["in_without"] - mean["in_with"], mean["out_without"] - mean["out_with"]
    return {"count": len(rows), "psnr": mean, "flare_cost_db": {"input": lost_in, "restored": lost_out},
            "flare_cost_kept": lost_out / lost_in if lost_in > 0 else float("nan"),
            "flare_light_left_region": ratio("flare_out_region", "flare_in_region"),
            "flare_light_left_frame": ratio("flare_out", "flare_in"),
            "region_fraction": float(np.mean([row["region_fraction"] for row in rows]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="checkpoint phase 2")
    parser.add_argument("--manifest")
    parser.add_argument("--split", default="valid", choices=("valid", "test"))
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="fp16 autocast tren CUDA, nhu phase 2 da train")
    parser.add_argument("--halo-root", help="thu muc co halo_index.csv (mac dinh: data.halo_root cua checkpoint)")
    parser.add_argument("--output", help="JSON ket qua")
    args = parser.parse_args()
    device = torch.device(args.device)
    system, config = _system_from_phase2(args.checkpoint, device)
    if args.halo_root:
        config["data"]["halo_root"] = str(args.halo_root)
    forward = RestorationForward(system)
    forward.amp = bool(args.amp) and device.type == "cuda"
    manifest, _ = _manifest(config, args.manifest)
    dataset = _dataset(config, manifest, args.split, fixed_realization=True)
    with_flare, without_flare = corruptors(dataset, config, args.halo_root)
    trained_with_halo = float(config["corruption"]["image"].get("halo_probability", 0)) > 0
    indices = np.random.default_rng(0).choice(len(dataset), min(args.samples, len(dataset)), replace=False)
    rows: list[dict] = []
    with torch.no_grad():
        for start in range(0, len(indices), args.batch):
            chosen = [int(i) for i in indices[start:start + args.batch]]
            renders = []
            for corruptor in (with_flare, without_flare):
                dataset.image_corruptor = corruptor
                samples = [dataset[i] for i in chosen]
                collated = collate_paired(samples)
                batch = {key: collated[key].to(device) for key in ("image_noisy", "imu_noisy_phys", "image_time",
                                                                    "imu_times")}
                outputs = forward(batch["image_noisy"], batch["imu_noisy_phys"], batch["image_time"],
                                  batch["imu_times"])
                renders.append((samples, collated, outputs["image"].float().clamp(0, 1).cpu()))
            (samples, collated, out_with), (_, twin, out_without) = renders
            clean, in_with, in_without = (x["image_" + key].float() for x, key in
                                          ((collated, "clean"), (collated, "noisy"), (twin, "noisy")))
            for k, sample in enumerate(samples):
                halo = sample["corruption"]["image"]["halo_params"]
                added = (linear(in_with[k]) - linear(in_without[k])).mean(0)          # [H, W] anh sang loe them
                left = (linear(out_with[k]) - linear(out_without[k])).mean(0)
                region = F.avg_pool2d(added[None, None], BLOCK)[0, 0] > REGION
                mask = region.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
                rows.append({"uid": halo["uid"], "gain": halo["gain"],
                             "in_with": psnr(in_with[k], clean[k]), "in_without": psnr(in_without[k], clean[k]),
                             "out_with": psnr(out_with[k], clean[k]), "out_without": psnr(out_without[k], clean[k]),
                             "flare_in_region": float(added[mask].sum()), "flare_out_region": float(left[mask].sum()),
                             "flare_in": float(added.sum()), "flare_out": float(left.sum()),
                             "region_fraction": float(mask.float().mean())})
    median = float(np.median([row["flare_in"] for row in rows]))
    groups = {"tat ca": rows, "nhe": [r for r in rows if r["flare_in"] <= median],
              "manh": [r for r in rows if r["flare_in"] > median]}
    table = {name: summarize(group) for name, group in groups.items() if group}
    print(f"{len(rows)} anh {args.split}, moi anh co loe HALO (scene HALO cua split {args.split}); ban sinh doi chi bo lop "
          f"loe. Checkpoint {'CO' if trained_with_halo else 'KHONG'} train voi HALO.")
    print(f"\n{'nhom':<8}{'so anh':>7}{'PSNR vao':>18}{'PSNR khoi phuc':>22}{'loe lam mat (dB)':>22}"
          f"{'anh sang loe con lai':>26}")
    print(f"{'':<8}{'':>7}{'co loe / khong':>18}{'co loe / khong':>22}{'vao -> khoi phuc':>22}"
          f"{'vung loe / ca khung':>26}")
    for name, row in table.items():
        psnrs = row["psnr"]
        print(f"{name:<8}{row['count']:>7}{psnrs['in_with']:>10.2f} / {psnrs['in_without']:<5.2f}"
              f"{psnrs['out_with']:>14.2f} / {psnrs['out_without']:<5.2f}"
              f"{row['flare_cost_db']['input']:>12.2f} -> {row['flare_cost_db']['restored']:<6.2f}"
              f"{100 * row['flare_light_left_region']:>16.0f}% / {100 * row['flare_light_left_frame']:.0f}%")
    print("\nDoc: 'anh sang loe con lai' la so do chinh: gan 0% -> model go duoc loe; gan 100% -> de nguyen loe, can mot\n"
          "     buoc go loe rieng. 'loe lam mat' cua khoi phuc nho ma anh sang loe con lai cao -> loi khac (do sang, chi\n"
          "     tiet) lan at loe trong PSNR, loe van con. 'manh' con lai nhieu hon 'nhe' -> model chi go duoc loe nhat.")
    if args.output:
        Path(args.output).write_text(json.dumps({"split": args.split, "trained_with_halo": trained_with_halo,
                                                 "groups": table}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
