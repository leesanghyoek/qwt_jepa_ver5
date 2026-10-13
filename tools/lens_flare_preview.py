"""Xem loe ong kinh (corruption.image.lens_flare_*) tren anh TartanAir: khong train, khong can checkpoint.

Moi hang mot frame: sach | DICH (sach + nguon sang cua lop loe) | chi loe (loe ong kinh cong vao anh sach, khong buoc
nao khac) | VAO that (du chuoi nhieu cua config: loe, mo, hat...). Moi frame deu co loe (lens_flare_probability 1 cho
rieng cong cu nay); con lai dung dung config. In bang: nguon (Flare7K / Flare-R), co loi quang hoc khong, PSNR so voi
dich, va thoi gian dung lop loe (ms/anh, CPU) de uoc tinh tai them cho DataLoader.

    python tools/lens_flare_preview.py --config configs/kaggle_lensflare.yaml --lens-flare-root <thu muc Flare7K++> \
        --images <thu muc TartanAir> --count 8 --output outputs/lens_flare_preview
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from qjepa.config import build_corruptors, load_config  # noqa: E402
from qjepa.corruptions.lens_flare import lens_flare_layers  # noqa: E402
from qjepa.corruptions.light import linear_to_srgb, srgb_to_linear  # noqa: E402
from qjepa.data.dataset import load_rgb  # noqa: E402


def find_frames(root: Path) -> list[Path]:
    found = []
    for directory, children, files in os.walk(root, followlinks=True):
        children.sort()
        if Path(directory).name == "image_lcam_front":
            found.extend(Path(directory) / name for name in sorted(files) if name.lower().endswith((".png", ".jpg")))
            children.clear()
    return found


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(10 * np.log10(1.0 / max(float(np.mean((a - b) ** 2)), 1e-12)))


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/kaggle_lensflare.yaml")
    parser.add_argument("--lens-flare-root", required=True)
    parser.add_argument("--images", required=True, help="thu muc co cac image_lcam_front (TartanAir)")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--split", default="train", choices=("train", "valid", "test"), help="lop loe cua split nao")
    parser.add_argument("--seed", type=int, default=0, help="doi seed de xem frame va lop loe khac")
    parser.add_argument("--output", default="outputs/lens_flare_preview")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config["data"]["lens_flare_root"] = args.lens_flare_root
    if float(config["corruption"]["image"].get("lens_flare_probability", 0.0)) <= 0:
        raise SystemExit(f"{args.config} khong bat loe ong kinh (corruption.image.lens_flare_probability = 0)")
    corruptor, _ = build_corruptors(config)
    corruptor.config = replace(corruptor.config, lens_flare_probability=1.0, clean_probability=0.0)
    frames = find_frames(Path(args.images))
    if not frames:
        raise SystemExit(f"khong co anh image_lcam_front nao trong {args.images}")
    rng = np.random.default_rng(args.seed)
    picked = [frames[int(k)] for k in rng.choice(len(frames), size=min(args.count, len(frames)), replace=False)]
    size = tuple(config["data"]["image_size"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, panels = [], []
    print(f"{'frame':<44} {'nguon':<8} {'quang hoc':>9} {'vao/dich':>9} {'chi loe/dich':>13} {'ms loe':>7}")
    for index, path in enumerate(picked):
        clean = load_rgb(path, size)
        noisy, params = corruptor(clean, split=args.split, realization=args.seed, trajectory=str(path), timestamp=0.0,
                                  frame_index=index, mode="full")
        target = corruptor.target(clean, params)
        drawn = params["lens_flare_params"]
        started = time.perf_counter()
        flare, _ = lens_flare_layers(corruptor.lens_bank, drawn, *clean.shape[:2])
        elapsed = 1000 * (time.perf_counter() - started)
        flare_only = np.clip(linear_to_srgb(srgb_to_linear(clean) + flare), 0, 1)
        row = {"frame": str(path), "source": drawn["source"], "name": drawn["name"], "optics": drawn["optics"] is not None,
               "psnr_input_db": psnr(noisy, target), "psnr_flare_only_db": psnr(flare_only, target),
               "light_kept_mean": float(np.mean(target - clean)), "ms": elapsed}
        rows.append(row)
        label = "/".join(path.parts[-5:-2])
        print(f"{label:<44} {row['source']:<8} {'co' if row['optics'] else 'khong':>9} {row['psnr_input_db']:9.2f} "
              f"{row['psnr_flare_only_db']:13.2f} {elapsed:7.1f}")
        panels.append(np.concatenate([clean, target, flare_only, noisy], axis=1))
    from PIL import Image
    sheet = np.concatenate(panels, axis=0)
    Image.fromarray(np.uint8(np.clip(sheet, 0, 1) * 255 + 0.5)).save(output / "lens_flare_preview.png")
    report = {"columns": ["sach", "dich (giu nguon sang)", "chi loe", "vao (du nhieu)"], "rows": rows,
              "mean_ms": float(np.mean([row["ms"] for row in rows]))}
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"cot: sach | dich (giu nguon sang) | chi loe | vao (du nhieu) -> {output / 'lens_flare_preview.png'}")
    print(f"dung lop loe trung binh {report['mean_ms']:.1f} ms/anh")
    return report


if __name__ == "__main__":
    main()
