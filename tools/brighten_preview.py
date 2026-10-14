"""Xem thu phase2.target_brighten (p43): anh DICH toi duoc lam sang the nao (khong train).

Hai cach:
  * --images: anh bat ky tren may (file hoac thu muc) -> moi anh: goc | da lam sang, kem he so g va do sang trung binh;
  * --manifest (tren Kaggle, sau Cell 5): anh that cua dataset nhu phase 2 doc -> moi mau: VAO (nhieu) | DICH goc | DICH
    da lam sang; va thong ke tren --stats anh sach: bao nhieu % bi lam sang, phan bo g.
Tham so lam sang doc tu phase2.target_brighten cua --config (mac dinh configs/kaggle_brightgt.yaml); --below, --target,
--max-gain de thu gia tri khac ma khong sua config.

    python3 tools/brighten_preview.py --images anh_cua_toi/ --output brighten.png
    python3 tools/brighten_preview.py --manifest manifests/kaggle --split train --count 12 --stats 400 --output brighten.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from qjepa.config import load_config  # noqa: E402
from qjepa.corruptions.light import LUMA_709, brighten_dark  # noqa: E402

SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def settings(args) -> dict:
    config = load_config(args.config)
    chosen = dict(config["phase2"].get("target_brighten") or {"below": 0.30, "target": 0.40, "max_gain": 3.0})
    for key, value in (("below", args.below), ("target", args.target), ("max_gain", args.max_gain)):
        if value is not None:
            chosen[key] = float(value)
    return chosen


def luma(image: np.ndarray) -> float:
    return float((np.asarray(image, dtype=np.float32) @ LUMA_709.astype(np.float32)).mean())


def image_files(entries: list[str]) -> list[Path]:
    files = []
    for entry in map(Path, entries):
        files += sorted(p for p in entry.rglob("*") if p.suffix.lower() in SUFFIXES) if entry.is_dir() else [entry]
    return files


def load(path: Path, longest: int = 512) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((longest, longest))
        return np.asarray(image, dtype=np.float32) / 255.0


def draw(rows: list[list[tuple[np.ndarray, str]]], output: Path, title: str) -> None:
    columns = max(len(row) for row in rows)
    figure, axes = plt.subplots(len(rows), columns, figsize=(3.2 * columns, 3.2 * len(rows)), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for r, row in enumerate(rows):
        for c, (image, label) in enumerate(row):
            axes[r, c].imshow(np.clip(image, 0, 1))
            axes[r, c].set_title(label, fontsize=9)
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=110)
    plt.close(figure)
    print("Đã lưu", output)


def from_images(args, chosen: dict) -> None:
    rows, gains = [], []
    print(f"{'ảnh':40s} {'sáng vào':>9s} {'g':>6s} {'sáng ra':>8s}")
    for path in image_files(args.images):
        image = load(path)
        bright, gain = brighten_dark(image, **chosen)
        gains.append(gain)
        print(f"{path.name[:40]:40s} {luma(image):9.3f} {gain:6.2f} {luma(bright):8.3f}")
        rows.append([(image, f"{path.name[:28]}\ngốc · sáng {luma(image):.2f}"),
                     (bright, f"làm sáng ×{gain:.2f} · sáng {luma(bright):.2f}" if gain > 1 else "giữ nguyên")])
    if not rows:
        raise SystemExit("Không thấy ảnh nào trong --images")
    print(f"\n{sum(g > 1 for g in gains)}/{len(gains)} ảnh được làm sáng")
    draw(rows, Path(args.output), f"phase2.target_brighten {chosen}")


def from_dataset(args, chosen: dict) -> None:
    from qjepa.cli import _dataset, _manifest
    config = load_config(args.config)
    manifest, _ = _manifest(config, args.manifest)
    # Anh dich GOC (target_brighten=False), lam sang o day de ve canh nhau; anh vao nhu phase 2 doc.
    dataset = _dataset(config, manifest, args.split, fixed_realization=True, target_brighten=False)
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(dataset.samples), size=min(args.count, len(dataset.samples)), replace=False)
    rows = []
    print(f"{'mẫu':48s} {'sáng đích':>10s} {'g':>6s}")
    for index in picks:
        item = dataset[int(index)]
        target = item["image_clean"].permute(1, 2, 0).numpy()
        bright, gain = brighten_dark(target, **chosen)
        print(f"{item['sample_id'][-48:]:48s} {luma(target):10.3f} {gain:6.2f}")
        rows.append([(item["image_noisy"].permute(1, 2, 0).numpy(), "vào (nhiễu)"),
                     (target, f"đích gốc · sáng {luma(target):.2f}"),
                     (bright, f"đích mới ×{gain:.2f}" if gain > 1 else "đích giữ nguyên")])
    if args.stats:
        # Chi anh sach (nhanh): ty le anh dich bi lam sang tren split.
        chosen_stats = rng.choice(len(dataset.samples), size=min(args.stats, len(dataset.samples)), replace=False)
        gains = np.array([brighten_dark(dataset._clean_frame(dataset.samples[int(k)]), **chosen)[1]
                          for k in chosen_stats])
        lifted = gains[gains > 1]
        print(f"\nThống kê {len(gains)} ảnh đích {args.split}: {100 * len(lifted) / len(gains):.1f}% được làm sáng"
              + (f"; g trung vị {np.median(lifted):.2f}, p90 {np.percentile(lifted, 90):.2f}, "
                 f"chạm trần ×{chosen['max_gain']:g}: {100 * np.mean(lifted >= chosen['max_gain'] - 1e-6):.0f}%"
                 if len(lifted) else ""))
    draw(rows, Path(args.output), f"{args.split}: vào | đích gốc | đích làm sáng — {chosen}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO / "configs/kaggle_brightgt.yaml"))
    parser.add_argument("--images", nargs="+", help="file hoac thu muc anh tren may")
    parser.add_argument("--manifest", help="thu muc manifest (Kaggle: manifests/kaggle)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--stats", type=int, default=400, help="so anh sach de dem ty le bi lam sang (0 = bo qua)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--below", type=float)
    parser.add_argument("--target", type=float)
    parser.add_argument("--max-gain", type=float)
    parser.add_argument("--output", default="brighten_preview.png")
    args = parser.parse_args(argv)
    chosen = settings(args)
    print("Tham số làm sáng:", chosen)
    if args.images:
        from_images(args, chosen)
    elif args.manifest:
        from_dataset(args, chosen)
    else:
        parser.error("cần --images hoặc --manifest")


if __name__ == "__main__":
    main()
