"""Anh loe THAT cua FlareImage qua model phase 2: vao (co loe) -> ra -> anh sach, tung cap, kem bang so (khong train).

Cac cap lay tu mot split cua FlareImage (mac dinh test: FlarePairBank chia theo nhom canh, khong cap nao cung canh voi
train/valid), rai deu qua cac nguon (flare_removal, halo, flare7k_real, flare7k_synthetic), hoac dung ten --names.
Moi cap duoc thu nho nhu luc train (canh ngan data.flare_pairs_short_side, cache), roi:
  * --frame full (mac dinh): CA khung, moi canh cat bot (o giua) ve boi so cua 16. Model chay duoc tren co bat ky boi
    so cua 16 -- vi tri sin-cos cua ViT tinh theo luoi token, cac dau toan cuc doc thong ke va pooling thich ung --
    nhung luc train no chi thay o cat data.image_size;
  * --frame crop: o cat data.image_size o giua khung, dung co luc train.
--mode chi_loe (mac dinh): anh chup that vao thang model, khong them nhieu. --mode nhu_luc_train: them mo va hat nhu mau
FlareImage luc train (PairedCameraImuDataset._item, "clear" theo data.flare_pairs_clear_probability).
FlareImage khong co IMU: moi cap muon cua so IMU cua mot mau TartanAir test, co dinh theo ten cap.

--bank visdrone (p46): cac cap "Lens flare on VisDrone, 640x640" (data.scene_pairs_*, anh drone that cong mot lop loe
Wu/FlareX) thay cho FlareImage. Bo nay chi chia train / val (index.csv); valid = anh VisDrone test-dev + lop loe nhom
valid, model chua train tren cap nao (giai doan loe co dung valid de chon best.pt). --imu khong: cua so IMU bang 0, nhu
25% mau p46 khong co IMU (data.imu_absent_probability) -- dung anh khong kem IMU; --imu muon (mac dinh): muon IMU.
--count 0: moi cap cua split. --panels N: chi ve hinh N cap, rai deu tu cap te nhat toi tot nhat (theo PSNR ra - vao),
sau khi cham het; mac dinh moi cap mot hinh. Cuoi bang in thoi gian moi cap (model / ve hinh / con lai).

In bang tung cap va theo nguon: PSNR va SSIM vao -> ra, va PSNR rieng VUNG LOE -- pixel ma anh co loe lech anh sach
hon 0,05 (trung binh 3 kenh, tren cap anh truoc khi them nhieu, nhu validation), noi model phai go loe -- va PSNR
NGOAI vung loe, noi model phai de nguyen (p40 lam hong o day, trung vi -4 dB). Ve moi cap mot hinh: vao | ra | sach |
|vao - sach| | |ra - sach|.

    python3 tools/flare_image_test.py --checkpoint outputs/p38_expnorm/phase2/best_joint_validation.pt \\
        --manifest manifests/kaggle --output outputs/p38_expnorm/flare_image_test/<gio> [--count 8] \\
        [--split test] [--source flare_removal] [--names 000395.png,000431.png] [--mode chi_loe] [--frame full] [--amp]
    python3 tools/flare_image_test.py --bank visdrone --split valid --count 0 [--imu khong] ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from qjepa.cli import _dataset, _manifest, _system_from_phase2, flare_pair_bank  # noqa: E402
from qjepa.corruptions.rng import derive_seed  # noqa: E402
from qjepa.data.flare_pairs import FLARE_PAIR_SPLITS  # noqa: E402
from qjepa.data.scene_pairs import SCENE_PAIR_SPLITS, ScenePairBank  # noqa: E402
from qjepa.evaluation.metrics import image_metrics  # noqa: E402
from qjepa.execution import RestorationForward  # noqa: E402
from tools.small_test import with_flare_pairs  # noqa: E402

MODES = {"chi_loe": "clean", "nhu_luc_train": "full"}
FLARE_THRESHOLD = 0.05
# VisDrone: mac dinh test neu bank co (p49 data.scene_pairs_holdout_fraction), khong thi valid (xem duoi).
BANKS = {"flareimage": ("FlareImage", FLARE_PAIR_SPLITS, "test"),
         "visdrone": ("VisDrone", (*SCENE_PAIR_SPLITS, "test"), None)}
SCENE_SHORT_SIDE = 384      # p46 data.scene_pairs_short_side, cho checkpoint khong train voi cap canh


def pick_pairs(bank, split: str, count: int, source: str | None = None, names: list[str] | None = None):
    """Cac cap se thu: dung ``names`` neu co; khong thi ``count`` cap rai deu qua cac nguon (theo thu tu ten), hoac
    moi cap (cua ``source`` neu co) khi ``count`` = 0."""
    pairs = list(bank.pairs(split))
    if names:
        by_name = {pair.name: pair for pair in pairs}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(f"{split} khong co cap {missing}: chi cac cap cua split nay (model chua thay)")
        return [by_name[name] for name in names]
    if source:
        pairs = [pair for pair in pairs if pair.source == source]
        if not pairs:
            raise ValueError(f"{split} khong co cap nao cua nguon {source!r}")
    if count == 0:
        return pairs
    sources = sorted({pair.source for pair in pairs})
    chosen = []
    for position, name in enumerate(sources):
        members = [pair for pair in pairs if pair.source == name]
        want = min(len(members), count // len(sources) + (position < count % len(sources)))
        if want:
            chosen += [members[int(k)] for k in np.linspace(0, len(members) - 1, want).round()]
    return chosen


def frame_of(array: np.ndarray, frame: str, size: tuple[int, int]) -> np.ndarray:
    """``full``: ca khung, moi canh cat (o giua) ve boi so cua 16. ``crop``: o cat ``size`` o giua."""
    height, width = array.shape[:2]
    if frame == "crop":
        if height < size[0] or width < size[1]:
            raise ValueError(f"anh {width}x{height} nho hon o cat {size[1]}x{size[0]}: dung --frame full")
        want = size
    else:
        want = (height // 16 * 16, width // 16 * 16)
        if min(want) < 16:
            raise ValueError(f"anh {width}x{height} qua nho")
    top, left = (height - want[0]) // 2, (width - want[1]) // 2
    return np.ascontiguousarray(array[top:top + want[0], left:left + want[1]])


def psnr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    error = (a.float().clamp(0, 1) - b.float().clamp(0, 1)).square()
    if mask is not None:
        if not bool(mask.any()):
            return float("nan")
        error = error[:, mask]
    return float(-10 * torch.log10(error.mean().clamp_min(1e-12)))


def panel(path: Path, noisy: torch.Tensor, restored: torch.Tensor, clean: torch.Tensor, row: dict) -> None:
    images = [noisy, restored, clean]
    differences = [(noisy - clean).abs().mean(0), (restored - clean).abs().mean(0)]
    top = float(max(d.max() for d in differences).clamp_min(1e-3))
    figure, axes = plt.subplots(1, 5, figsize=(5 * 3.6, 3.6 * clean.shape[1] / clean.shape[2] + 0.8))
    titles = (f"vào (có lóe) {row['psnr_in']:.2f} dB", f"ra {row['psnr_out']:.2f} dB ({row['psnr_out'] - row['psnr_in']:+.2f})",
              "sạch (đích)", "|vào − sạch|", "|ra − sạch|")
    for axis, title, picture in zip(axes, titles, images + differences):
        if picture.ndim == 3:
            axis.imshow(picture.permute(1, 2, 0).clamp(0, 1).numpy())
        else:
            axis.imshow(picture.numpy(), cmap="magma", vmin=0, vmax=top)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    flare = (f" | vùng lóe {row['flare_fraction']:.0%}: {row['flare_psnr_in']:.2f} → {row['flare_psnr_out']:.2f} dB"
             if np.isfinite(row["flare_psnr_in"]) else "")
    figure.suptitle(f"{row['source']} {row['name']} ({row['width']}x{row['height']}){flare}", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=100)
    plt.close(figure)


def summarize(rows):
    keys = ("psnr_in", "psnr_out", "ssim_in", "ssim_out", "flare_psnr_in", "flare_psnr_out", "rest_psnr_in",
            "rest_psnr_out")
    summary = {"count": len(rows)}
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=float)
        summary[key] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    summary["better_fraction"] = float(np.mean([row["psnr_out"] > row["psnr_in"] for row in rows]))
    return summary


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="checkpoint phase 2")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="thu muc moi: hinh tung cap va report.json")
    parser.add_argument("--bank", default="flareimage", choices=tuple(BANKS))
    parser.add_argument("--split", choices=sorted(set(FLARE_PAIR_SPLITS) | set(SCENE_PAIR_SPLITS)),
                        help="mac dinh: test (FlareImage; VisDrone khi checkpoint chia lai test, p49) / valid (VisDrone cu)")
    parser.add_argument("--count", type=int, default=8, help="so cap, rai deu qua cac nguon; 0 = moi cap cua split")
    parser.add_argument("--source", help="chi mot nguon (FlareImage: flare_removal, halo, flare7k_real, "
                                         "flare7k_synthetic; VisDrone: glint, flarex, captured, simulated)")
    parser.add_argument("--names", help="ten cap, cach nhau dau phay (thay cho --count/--source)")
    parser.add_argument("--mode", default="chi_loe", choices=tuple(MODES))
    parser.add_argument("--frame", default="full", choices=("full", "crop"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="fp16 autocast tren CUDA, nhu phase 2 da train")
    parser.add_argument("--flare-pairs-root", help="thu muc FlareImage (mac dinh: data.flare_pairs_root cua checkpoint)")
    parser.add_argument("--scene-pairs-root", help="thu muc VisDrone co index.csv, gt/, input/ "
                                                   "(mac dinh: data.scene_pairs_root cua checkpoint)")
    parser.add_argument("--panels", type=int, default=-1,
                        help="so cap ve hinh, rai deu tu te nhat toi tot nhat; -1 (mac dinh) = moi cap")
    parser.add_argument("--imu", default="muon", choices=("muon", "khong"),
                        help="muon: IMU cua mot mau TartanAir test | khong: cua so IMU bang 0 (anh khong kem IMU)")
    args = parser.parse_args(argv)
    label, splits, default_split = BANKS[args.bank]
    args.split = args.split or default_split
    if args.split is not None and args.split not in splits:
        parser.error(f"{label} chi co split {list(splits)}")
    if args.count < 0:
        parser.error("--count >= 0")
    if args.split == "train":
        print("CANH BAO: split train -- model da thay cac cap nay luc train.")
    device = torch.device(args.device)
    system, config = _system_from_phase2(args.checkpoint, device)
    if args.bank == "flareimage":
        config = with_flare_pairs(config, args.flare_pairs_root, True)
        bank = flare_pair_bank(config)
    else:
        data = config["data"]
        root = args.scene_pairs_root or data.get("scene_pairs_root")
        if not root:
            raise ValueError("checkpoint nay khong train voi VisDrone: can --scene-pairs-root")
        bank = ScenePairBank(root, short_side=int(data.get("scene_pairs_short_side", SCENE_SHORT_SIDE)),
                             holdout_fraction=data.get("scene_pairs_holdout_fraction"))
        args.split = args.split or ("test" if "test" in bank.splits else "valid")
        if args.split not in bank.splits:
            parser.error(f"VisDrone cua checkpoint nay chi co split {list(bank.splits)}")
    manifest, _ = _manifest(config, args.manifest)
    # IMU muon tu TartanAir test; --imu khong: moi mau khong co IMU (cua so bang 0, nhu luc train p46).
    hosts = _dataset(config, manifest, "test", fixed_realization=True,
                     flare_pairs=bank if args.bank == "flareimage" else None,
                     imu_absent_probability=1.0 if args.imu == "khong" else 0.0)
    names = [name.strip() for name in args.names.split(",") if name.strip()] if args.names else None
    pairs = pick_pairs(bank, args.split, args.count, args.source, names)
    forward = RestorationForward(system)
    forward.amp = bool(args.amp) and device.type == "cuda"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    size = tuple(int(side) for side in config["data"]["image_size"])
    # Cap VisDrone luon "clear" (bo cac buoc moi truong), nhu validation p46 (dataset._scene_item).
    clear_probability = (float(config["data"].get("flare_pairs_clear_probability", 1.0)) if args.bank == "flareimage"
                         else 1.0)
    rows = []
    pictures = []                  # --panels N: anh uint8 (vao, ra, sach) tung cap, ve sau khi xep hang
    seconds = {"model": 0.0, "hinh": 0.0}
    started = time.perf_counter()
    print(f"{label} {args.split}: {len(pairs)} cặp | chế độ {args.mode} | khung {args.frame} | IMU {args.imu}"
          f" | {bank.root}")
    print(f"  {'nguồn':18s} {'cặp':11s} {'cỡ':>9s} {'PSNR vào→ra':>16s} {'SSIM vào→ra':>14s} {'vùng lóe: % | vào→ra':>26s} {'ngoài lóe vào→ra':>17s}")
    for number, pair in enumerate(pairs):
        gt, flared = bank.load(pair)
        gt, flared = (frame_of(array, args.frame, size).astype(np.float32) / 255.0 for array in (gt, flared))
        rng = np.random.default_rng(derive_seed(config["data"]["corruption_seed"], "flare_image_test", pair.name))
        host = hosts.samples[int(rng.integers(len(hosts.samples)))]
        clear = bool(rng.random() < clear_probability)
        mask = torch.from_numpy(np.abs(flared - gt).mean(-1) > FLARE_THRESHOLD)   # loe: anh co loe lech anh sach
        sample_id = f"{args.bank}/{pair.name}"
        item = hosts._item(host, sample_id, f"{args.bank}/{pair.source}", gt, flared, MODES[args.mode], "full",
                           trajectory=sample_id, timestamp=0.0, frame_index=0,
                           flare_pair={"name": pair.name, "source": pair.source, "clear": clear},
                           imu_donor=host.sample_id)
        noisy, clean = item["image_noisy"][None], item["image_clean"][None]
        tick = time.perf_counter()
        with torch.no_grad():
            out = forward(noisy.to(device), item["imu_noisy_phys"].T[None].to(device),
                          item["image_time"][None].to(device), item["imu_times"][None].to(device))
        restored = out["image"].float().clamp(0, 1).cpu()
        seconds["model"] += time.perf_counter() - tick
        before, after = image_metrics(noisy, clean), image_metrics(restored, clean)
        row = {"name": pair.name, "source": pair.source, "width": int(clean.shape[-1]), "height": int(clean.shape[-2]),
               "imu_donor": host.sample_id, "psnr_in": before["image_psnr_db"], "psnr_out": after["image_psnr_db"],
               "ssim_in": before["image_ssim"], "ssim_out": after["image_ssim"],
               "flare_fraction": float(mask.float().mean()), "flare_psnr_in": psnr(noisy[0], clean[0], mask),
               "flare_psnr_out": psnr(restored[0], clean[0], mask),
               # Ngoai vung loe: noi model phai de nguyen (p40 lam hong o day 4-13 dB).
               "rest_psnr_in": psnr(noisy[0], clean[0], ~mask), "rest_psnr_out": psnr(restored[0], clean[0], ~mask),
               "panel": f"pair_{number:02d}.png" if args.panels < 0 else None}
        rows.append(row)
        tick = time.perf_counter()
        if args.panels < 0:
            panel(output / row["panel"], noisy[0], restored[0], clean[0], row)
        else:
            pictures.append(tuple((image[0].clamp(0, 1) * 255).round().to(torch.uint8)
                                  for image in (noisy, restored, clean)))
        seconds["hinh"] += time.perf_counter() - tick
        flare = (f"{100 * row['flare_fraction']:5.1f}% | {row['flare_psnr_in']:5.2f} → {row['flare_psnr_out']:5.2f}"
                 if np.isfinite(row["flare_psnr_in"]) else "—")
        rest = (f"{row['rest_psnr_in']:6.2f} → {row['rest_psnr_out']:6.2f}" if np.isfinite(row["rest_psnr_in"])
                else "—")
        print(f"  {pair.source:18s} {pair.name:11s} {str(row['width']) + 'x' + str(row['height']):>9s}"
              f" {row['psnr_in']:6.2f} → {row['psnr_out']:6.2f} {row['ssim_in']:6.3f} → {row['ssim_out']:.3f}"
              f" {flare:>26s} {rest:>17s}", flush=True)
    if args.panels > 0 and rows:
        tick = time.perf_counter()
        ranked = sorted(range(len(rows)), key=lambda k: rows[k]["psnr_out"] - rows[k]["psnr_in"])
        for k in sorted({ranked[int(i)] for i in np.linspace(0, len(rows) - 1, min(args.panels, len(rows))).round()}):
            rows[k]["panel"] = f"pair_{k:02d}.png"
            panel(output / rows[k]["panel"], *(image.float() / 255 for image in pictures[k]), rows[k])
        seconds["hinh"] += time.perf_counter() - tick
    total = time.perf_counter() - started
    count = max(1, len(rows))
    print(f"\n  Thời gian: {total:.0f} s, {total / count:.2f} s/cặp (model {seconds['model'] / count:.2f}"
          f" | vẽ hình {seconds['hinh'] / count:.2f} | đọc ảnh + nhiễu + chấm {(total - sum(seconds.values())) / count:.2f})")
    sources = sorted({row["source"] for row in rows})
    summary = {"tat ca": summarize(rows), **{source: summarize([r for r in rows if r["source"] == source])
                                            for source in sources}}
    print(f"\n  {'nhóm':18s} {'số':>4s} {'PSNR vào→ra':>16s} {'ra > vào':>9s} {'vùng lóe vào→ra':>18s}"
          f" {'ngoài lóe vào→ra':>18s}")
    for name, values in summary.items():
        print(f"  {name:18s} {values['count']:>4d} {values['psnr_in']:6.2f} → {values['psnr_out']:6.2f}"
              f" {100 * values['better_fraction']:8.0f}% {values['flare_psnr_in']:7.2f} → {values['flare_psnr_out']:6.2f}"
              f" {values['rest_psnr_in']:8.2f} → {values['rest_psnr_out']:6.2f}")
    report = {"checkpoint": str(args.checkpoint), "bank": args.bank, "split": args.split, "imu": args.imu,
              "mode": args.mode, "frame": args.frame,
              "rows": rows, "summary": summary}
    (output / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã lưu {output}: {sum(row['panel'] is not None for row in rows)} hình pair_XX.png, report.json")
    return report


if __name__ == "__main__":
    main()
