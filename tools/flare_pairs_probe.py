"""Model phase 2 lam gi voi anh loe THAT cua FlareImage? PSNR vao -> ra tren cac cap giu lai (valid / test).

Moi cap FlareImage cua split (data.flare_pairs_*: chia theo nhom canh, model khong thay luc train) di qua dung
duong mau FlareImage luc train (PairedCameraImuDataset._flare_item): o cat image_size cua ban thu nho, anh train/ vao,
gt/ la dich, IMU muon cua mot mau TartanAir cua cung split. Hai che do:
  * "chi loe" (image_mode clean): anh vao la anh chup that, khong them nhieu nao -- model go duoc loe toi dau;
  * "nhu luc train" (image_mode full): them mo va hat nhu mau FlareImage luc train (flare_pairs_clear_probability).
Bang theo tung nguon (flare_removal, halo, flare7k_real, flare7k_synthetic). PSNR tinh tung anh tren sRGB [0, 1]
roi lay trung binh; "ra tot hon vao" = % cap ma PSNR ra > vao.

Checkpoint KHONG train voi FlareImage (vd p35) cung do duoc: cac khoa flare_pairs_* lay tu
configs/kaggle_flareimage.yaml (cung phep chia), can --flare-pairs-root. So p35 voi p36 cho biet train voi FlareImage
giup bao nhieu tren anh loe that.

    python3 tools/flare_pairs_probe.py --checkpoint outputs/p36_flareimage/phase2/best_joint_validation.pt \\
        --manifest manifests/kaggle [--split valid] [--amp] [--flare-pairs-root <thu muc FlareImage>] \\
        [--output flare_pairs_probe.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from qjepa.cli import _dataset, _manifest, _system_from_phase2, flare_pair_bank  # noqa: E402
from qjepa.config import FLARE_PAIR_KEYS, load_config  # noqa: E402
from qjepa.data import collate_paired  # noqa: E402
from qjepa.execution import RestorationForward  # noqa: E402

MODES = {"chi loe": "clean", "nhu luc train": "full"}


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(10 * torch.log10(1.0 / (a - b).square().mean().clamp_min(1e-10)))


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    before = np.array([row["input"] for row in rows])
    after = np.array([row["restored"] for row in rows])
    return {"count": len(rows), "input_psnr_db": float(before.mean()), "restored_psnr_db": float(after.mean()),
            "gain_db": float((after - before).mean()), "better_fraction": float((after > before).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="checkpoint phase 2")
    parser.add_argument("--manifest")
    parser.add_argument("--split", default="valid", choices=("valid", "test"))
    parser.add_argument("--samples", type=int, default=0, help="toi da bao nhieu cap (0 = moi cap cua split)")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="fp16 autocast tren CUDA, nhu phase 2 da train")
    parser.add_argument("--flare-pairs-root", help="thu muc FlareImage (mac dinh: data.flare_pairs_root cua checkpoint)")
    parser.add_argument("--output", help="JSON ket qua")
    args = parser.parse_args()
    device = torch.device(args.device)
    system, config = _system_from_phase2(args.checkpoint, device)
    trained_with_pairs = float(config["data"].get("flare_pairs_fraction", 0)) > 0
    if not trained_with_pairs:
        recipe = load_config(REPO / "configs/kaggle_flareimage.yaml")["data"]
        config["data"].update({key: recipe[key] for key in FLARE_PAIR_KEYS})
        if not args.flare_pairs_root:
            parser.error("checkpoint nay khong train voi FlareImage: can --flare-pairs-root")
    if args.flare_pairs_root:
        config["data"]["flare_pairs_root"] = str(args.flare_pairs_root)
    forward = RestorationForward(system)
    forward.amp = bool(args.amp) and device.type == "cuda"
    manifest, _ = _manifest(config, args.manifest)
    bank = flare_pair_bank(config)
    dataset = _dataset(config, manifest, args.split, fixed_realization=True, flare_pairs=bank)
    dataset.flare_pairs_split = args.split
    count = len(bank.pairs(args.split)) if args.samples <= 0 else min(args.samples, len(bank.pairs(args.split)))
    first = len(dataset.samples)               # draw d < so cap: moi cap dung mot lan (hoan vi cua chu ky 0)
    report = {"checkpoint": str(args.checkpoint), "split": args.split, "trained_with_flare_pairs": trained_with_pairs,
              "bank": bank.describe(), "modes": {}}
    with torch.no_grad():
        for label, mode in MODES.items():
            dataset.image_mode = mode
            rows = []
            for start in range(0, count, args.batch):
                samples = [dataset[first + draw] for draw in range(start, min(count, start + args.batch))]
                collated = collate_paired(samples)
                batch = {key: collated[key].to(device) for key in ("image_noisy", "imu_noisy_phys", "image_time",
                                                                    "imu_times")}
                restored = forward(batch["image_noisy"], batch["imu_noisy_phys"], batch["image_time"],
                                   batch["imu_times"])["image"].float().clamp(0, 1).cpu()
                clean, noisy = collated["image_clean"].float(), collated["image_noisy"].float()
                for k, sample in enumerate(samples):
                    pair = sample["corruption"]["image"]["flare_pair"]
                    rows.append({"name": pair["name"], "source": pair["source"],
                                 "input": psnr(noisy[k], clean[k]), "restored": psnr(restored[k], clean[k])})
            sources = sorted({row["source"] for row in rows})
            report["modes"][label] = {"image_mode": mode, "all": summarize(rows),
                                      "by_source": {s: summarize([r for r in rows if r["source"] == s]) for s in sources},
                                      "pairs": rows}
    print(f"FlareImage {args.split}: {count} cap | train voi FlareImage: {trained_with_pairs} | "
          f"{bank.describe()['root']}")
    for label, table in report["modes"].items():
        print(f"\n{label} (image_mode {table['image_mode']})")
        print(f"  {'nguon':<20}{'so cap':>7}{'PSNR vao':>10}{'PSNR ra':>10}{'loi (dB)':>10}{'ra tot hon vao':>16}")
        for name, row in [("tat ca", table["all"]), *table["by_source"].items()]:
            if row["count"]:
                print(f"  {name:<20}{row['count']:>7}{row['input_psnr_db']:>10.2f}{row['restored_psnr_db']:>10.2f}"
                      f"{row['gain_db']:>+10.2f}{100 * row['better_fraction']:>15.0f}%")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
