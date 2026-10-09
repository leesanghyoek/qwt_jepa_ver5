"""Xem truoc LOE SANG HALO tren anh TartanAir (mac dinh configs/kaggle_halo.yaml, run p33_halo).

Lop loe that cua HALO (bong ma phan xa giua cac mat thau kinh) duoc cong vao anh sach tren anh sang
tuyen tinh, roi qua chuoi nhieu cu cua repo. Cac buoc lay thang tu code train (qjepa/corruptions/halo.py
va LowLightImageCorruptor), nen hinh la dung thu model se thay:

  Sach | Lop loe HALO (x4) | Sach + loe | Nhieu train, KHONG HALO | Nhieu train, CO HALO

Hai cot cuoi cung mot bo tham so nhieu (HALO boc tu luong ngau nhien rieng), chi khac lop loe. Trong train
halo_probability = 0,5; o day ep 1,0 de hang nao cung co. --stats N do tren N anh ngau nhien: loe lam
bao nhieu % pixel sang hon 8/255 va 32/255 (truoc chuoi nhieu), de chon halo_gain.

    python3 tools/halo_flare_preview.py --halo-root ~/Datasets/halo-reflective-1280
    python3 tools/halo_flare_preview.py --halo-root <thu muc co halo_index.csv> --count 6 --seed 7 --stats 200
    python3 tools/halo_flare_preview.py --halo-root ... --gain 1.0 3.0     # thu khoang halo_gain khac

Lay dataset Kaggle ve may: kaggle datasets download <chu>/halo-reflective-1280 -p ~/Datasets/halo-reflective-1280
--unzip (moi <scene>.tar thanh thu muc). Chay duoc bang python3 cua he thong (khong can torch). Hinh luu o
outputs/halo_flare/ (gitignore).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from light_corruption_preview import REPO, _config_chain, bring_to_front, frames, load_many, tonemap  # noqa: E402

from qjepa.corruptions.halo import HaloFlareBank, apply_halo  # noqa: E402
from qjepa.corruptions.image import LowLightImageCorruptionConfig, LowLightImageCorruptor  # noqa: E402
from qjepa.corruptions.light import linear_to_srgb  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def corruptors(config_name: str, halo_root: Path, gain: list[float] | None):
    """(co HALO ep 1,0, khong HALO, bank) dung nhu luc train voi configs/<config_name>."""
    config = _config_chain(REPO / "configs" / config_name)
    image = LowLightImageCorruptionConfig(**config["corruption"]["image"])
    if image.halo_probability <= 0:
        raise SystemExit(f"configs/{config_name} khong bat HALO (halo_probability = 0)")
    if gain is not None:
        image = replace(image, halo_gain=tuple(gain))
    bank = HaloFlareBank(halo_root, revision=image.halo_revision, effects=image.halo_effects,
                         holdout_fraction=image.halo_holdout_fraction)
    seed = config["data"]["corruption_seed"]
    with_halo = LowLightImageCorruptor(replace(image, halo_probability=1.0, clean_probability=0.0), seed, halo_bank=bank)
    without = LowLightImageCorruptor(replace(image, halo_probability=0.0, clean_probability=0.0), seed)
    return with_halo, without, bank


def brightening(clean: np.ndarray, flared: np.ndarray) -> np.ndarray:
    """Loe lam do sang (sRGB, kenh max) tang bao nhieu o moi pixel, cat o 1 nhu cam bien."""
    return (np.clip(flared, 0, 1) - clean).max(axis=-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path.home() / "Datasets/tartanair-v2-jepa")
    parser.add_argument("--halo-root", type=Path, default=Path.home() / "Datasets/halo-reflective-1280",
                        help="thu muc chua halo_index.csv")
    parser.add_argument("--split", default="valid", choices=("train", "valid", "test"))
    parser.add_argument("--count", type=int, default=4, help="so anh ve ra")
    parser.add_argument("--size", type=int, default=256, help="canh anh (256 nhu luc train)")
    parser.add_argument("--gain", type=float, nargs=2, default=None, help="thu halo_gain khac config")
    parser.add_argument("--stats", type=int, default=0, help="do do sang them tren N anh ngau nhien")
    parser.add_argument("--seed", type=int, default=None,
                        help="bo ngau nhien; mac dinh moi lan chay mot seed moi (in ra de chay lai dung hinh do)")
    parser.add_argument("--config", default="kaggle_halo.yaml", help="config nhieu dang train (trong configs/)")
    parser.add_argument("--output", type=Path, default=REPO / "outputs/halo_flare/preview.png")
    parser.add_argument("--no-show", action="store_true", help="chi luu PNG, khong mo cua so")
    args = parser.parse_args([] if "ipykernel" in sys.modules else None)
    if args.seed is None:
        args.seed = int(np.random.default_rng().integers(0, 100_000))
    print(f"seed {args.seed} (chay lai dung hinh nay: --seed {args.seed})", flush=True)

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with_halo, without, bank = corruptors(args.config, args.halo_root, args.gain)
    gain = with_halo.config.halo_gain
    info = bank.describe()
    print(f"HALO: {info['samples']} lop loe (train/valid/test), halo_gain {list(gain)}")
    paths = frames(args.root, args.split)
    rng = np.random.default_rng(args.seed)

    if args.stats:
        chosen = [paths[i] for i in rng.choice(len(paths), size=min(args.stats, len(paths)), replace=False)]
        over8, over32, mean = [], [], []
        for row, clean in enumerate(load_many(chosen, args.size)):
            params = with_halo._parameters(args.split, 0, f"stats{args.seed}-{row}", 0.1 * row, "full")
            lift = brightening(clean, apply_halo(clean, bank, params["halo_params"]))
            over8.append(float((lift > 8 / 255).mean()))
            over32.append(float((lift > 32 / 255).mean()))
            mean.append(float(lift.mean()))
        for name, values in (("% pixel sang hon 8/255", over8), ("% pixel sang hon 32/255", over32)):
            p = np.percentile(np.asarray(values) * 100, [10, 50, 90])
            print(f"{name}: p10 {p[0]:.1f}  trung vi {p[1]:.1f}  p90 {p[2]:.1f}  ({len(values)} anh)")
        print(f"do sang them trung binh (x255): trung vi {np.median(mean) * 255:.1f}, p90 {np.percentile(mean, 90) * 255:.1f}")

    columns = ["Sạch", "Lớp lóe HALO (×4)", "Sạch + lóe", "Nhiễu train,\nKHÔNG HALO", f"Nhiễu train,\nCÓ HALO ({args.config})"]
    cell = min(16.0 / len(columns), 9.0 / args.count)
    figure, axes = plt.subplots(args.count, len(columns), figsize=(cell * len(columns), cell * args.count + 1.0),
                                squeeze=False, layout="constrained")
    for axis in axes.flat:
        axis.set_xticks([]), axis.set_yticks([])
    for col, title in enumerate(columns):
        axes[0, col].set_title(title, fontsize=9)
    chosen = [paths[i] for i in rng.choice(len(paths), size=args.count, replace=False)]
    for row, (path, clean) in enumerate(zip(chosen, load_many(chosen, args.size))):
        corrupt = dict(split=args.split, realization=0, trajectory=f"preview{args.seed}-{row}",
                       timestamp=0.1 * row, frame_index=row, mode="full")
        noisy, params = with_halo(clean, **corrupt)
        plain = without(clean, **corrupt)[0]
        halo = params["halo_params"]
        layer = bank.layer(halo["uid"], clean.shape[0], clean.shape[1])
        layer = layer[::-1] if halo["flip_y"] else layer
        layer = layer[:, ::-1] if halo["flip_x"] else layer
        flared = apply_halo(clean, bank, halo)
        lift = brightening(clean, flared)
        shown = [clean, linear_to_srgb(4 * halo["gain"] * layer),
                 tonemap(flared), plain, noisy]
        for col, image in enumerate(shown):
            axes[row, col].imshow(np.clip(image, 0, 1))
        axes[row, 0].set_ylabel(path.parts[-5] if len(path.parts) >= 5 else path.stem, fontsize=8)
        axes[row, 2].set_xlabel(f"gain {halo['gain']:.2f} · {(lift > 8 / 255).mean() * 100:.0f}% px > 8/255", fontsize=7)
        axes[row, 1].set_xlabel(halo["uid"], fontsize=6)
        print(f"[{row}] {path.name}: {halo['uid']} gain {halo['gain']:.2f}, "
              f"{(lift > 8 / 255).mean() * 100:.1f}% pixel sang hon 8/255")
    figure.suptitle(f"seed {args.seed} · split {args.split} · halo_gain {list(gain)}", fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    print("Da luu:", args.output)
    if args.no_show:
        return 0
    backend = matplotlib.get_backend().lower()
    if any(name in backend for name in ("qt", "tk", "gtk", "wx", "macosx")) or "inline" in backend:
        bring_to_front(figure)
        plt.show()
    else:
        subprocess.Popen(["xdg-open", str(args.output)])
    return 0


if __name__ == "__main__":
    code = main()
    if "ipykernel" not in sys.modules:
        raise SystemExit(code)
