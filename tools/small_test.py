"""Tap test NHO rieng va mot batch train: model phase 2 lam gi voi anh va IMU (khong train).

Tap test nho (``--selection``, mot file CSV) la phan model khong thay luc train lan luc chon checkpoint:
  * ``--tartanair`` anh TartanAir cua split test, rai deu tren moi trajectory test (frame o giua cac doan bang nhau
    theo thoi gian cua tung trajectory; it anh hon so trajectory thi lay cac trajectory cach deu nhau);
  * ``--flare`` cap FlareImage cua split test (FlarePairBank chia theo nhom canh: khong cap nao cung canh voi
    train/valid). Anh train/ (co loe) qua nhieu nhu luc train la dau vao, gt/ la dich, IMU muon cua mot mau test.
Validation trong luc train chi doc TartanAir valid, tools/flare_pairs_probe.py doc FlareImage valid: tap nay khong
trung ca hai, va tool kiem lai dieu do moi lan chay. Lua chon chi phu thuoc manifest, metadata FlareImage va hai con
so; nhieu la realization validation (co dinh). File da co thi dung dung cac mau trong file (kiem tung sample_id): moi
checkpoint duoc cham tren cung mot tap.

Khong co ``--checkpoint`` (can ``--config``): chi chon tap va ghi file -- notebook lam viec nay TRUOC khi train.
Co ``--checkpoint``: them
  * batch train: dung cac microbatch CLI dua vao model o update ``--train-update`` (cung sampler, cung nhieu, cung o
    FlareImage; phase 2: gradient_accumulation microbatch = mot update; phase 1: batch thu --train-update). Moi
    microbatch phai co dung so cap FlareImage cua config;
  * model phase 2 chay tren batch train va tap test nho. In bang tung mau cua batch train, bang tap test theo nhom
    (TartanAir theo loai anh, FlareImage theo nguon): PSNR / SSIM / duong net 4-16 px vao -> ra va RMSE IMU (gia toc,
    con quay) vao -> ra. Ve train_batch.png, train_imu.png, test_images.png, test_imu.png; ghi report.json.

    python3 tools/small_test.py --config configs/kaggle.yaml --manifest manifests/kaggle \\
        --selection outputs/p37_flareblur/small_test/small_test.csv [--tartanair 48] [--flare 48]
    python3 tools/small_test.py --checkpoint outputs/p37_flareblur/phase2/best_joint_validation.pt \\
        --manifest manifests/kaggle --selection outputs/p37_flareblur/small_test/small_test.csv \\
        --output outputs/p37_flareblur/small_test/<tag> [--train-phase phase2] [--train-update 0] [--show 8] [--amp]
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from qjepa.cli import (_dataset, _manifest, _system_from_phase2, _train_dataset,  # noqa: E402
                       _training_batch_stream, flare_pair_bank)
from qjepa.config import FLARE_PAIR_KEYS, load_config  # noqa: E402
from qjepa.corruptions.image import frame_kind  # noqa: E402
from qjepa.data import collate_paired  # noqa: E402
from qjepa.evaluation.metrics import image_metrics, imu_metrics  # noqa: E402
from qjepa.execution import RestorationForward  # noqa: E402

SELECTION_FIELDS = ("kind", "index", "sample_id", "group", "name")
CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")
KIND_LABELS = {"plain": "tốt", "relit": "đổi sáng", "flare": "lóe"}
OPTICAL_MODES = ("full", "blur_only", "blur_low_light")


def spread(samples, count: int) -> list[int]:
    """``count`` chi so cua ``samples`` rai deu: moi trajectory mot phan (con du chia cho cac trajectory cach deu),
    hoac ``count`` trajectory cach deu khi it hon; trong trajectory, frame o giua cac doan bang nhau theo thoi gian."""
    if count <= 0:
        return []
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[sample.trajectory_key].append(index)
    keys = sorted(groups)
    if count < len(keys):
        quota = {keys[int(k)]: 1 for k in np.linspace(0, len(keys) - 1, count).round()}
    else:
        base, extra = divmod(count, len(keys))
        bonus = {int(k) for k in np.linspace(0, len(keys) - 1, extra).round()} if extra else set()
        quota = {key: base + (position in bonus) for position, key in enumerate(keys)}
    chosen = []
    for key, want in quota.items():
        frames = sorted(groups[key], key=lambda index: samples[index].image_index)
        want = min(want, len(frames))
        chosen += [frames[int((k + 0.5) * len(frames) / want)] for k in range(want)]
    return sorted(chosen)


def flare_sample_id(pair_name: str, realization: int, draw: int) -> str:
    return f"flareimage/{pair_name}#{realization}:{draw}"       # = PairedCameraImuDataset._flare_item


def select(dataset, tartanair: int, flare: int) -> list[dict[str, object]]:
    """Cac dong cua tap test nho: TartanAir rai deu, roi ``flare`` cap FlareImage khac nhau cua split (draw 0, 1, ...
    cua chu ky 0 la mot hoan vi cac cap)."""
    rows = [{"kind": "tartanair", "index": index, "sample_id": dataset.samples[index].sample_id,
             "group": dataset.samples[index].trajectory_key, "name": ""}
            for index in spread(dataset.samples, tartanair)]
    if flare > 0:
        if dataset.flare_bank is None:
            raise ValueError("--flare > 0 can cac cap FlareImage: data.flare_pairs_* hoac --flare-pairs-root")
        for draw in range(min(flare, len(dataset.flare_bank.pairs(dataset.flare_pairs_split)))):
            pair = dataset.flare_pair_of_draw(draw)
            rows.append({"kind": "flareimage", "index": draw,
                         "sample_id": flare_sample_id(pair.name, dataset.realization, draw),
                         "group": pair.source, "name": pair.name})
    return rows


def verify(dataset, rows: list[dict[str, object]]) -> None:
    """Moi dong cua file phai chi dung mau do trong dataset nay (cung manifest, cung FlareImage, cung realization)."""
    for row in rows:
        index = int(row["index"])
        if row["kind"] == "tartanair":
            found = dataset.samples[index].sample_id if index < len(dataset.samples) else None
        elif row["kind"] == "flareimage":
            if dataset.flare_bank is None:
                raise ValueError("Tap test nho co cap FlareImage: can data.flare_pairs_* hoac --flare-pairs-root")
            found = flare_sample_id(dataset.flare_pair_of_draw(index).name, dataset.realization, index)
        else:
            raise ValueError(f"kind la tartanair hoac flareimage, khong phai {row['kind']!r}")
        if found != row["sample_id"]:
            raise ValueError(f"Tap test nho: dong {row['sample_id']} khong con khop ({found}); manifest, FlareImage "
                             "hoac realization da doi -- chon file moi")


def held_out_overlap(dataset, rows, manifest) -> dict[str, list[str]]:
    """Mau nao cua tap test nho cung chuyen dong (TartanAir) hay cung cap (FlareImage) voi train/valid: phai rong."""
    seen = {(sample.environment, sample.trajectory_id) for split in ("train", "valid")
            for sample in manifest["samples"][split]}
    tartan = sorted({dataset.samples[int(row["index"])].trajectory_key for row in rows if row["kind"] == "tartanair"
                     if (dataset.samples[int(row["index"])].environment,
                         dataset.samples[int(row["index"])].trajectory_id) in seen})
    flare = []
    if dataset.flare_bank is not None:
        used = {pair.name for split in ("train", "valid") for pair in dataset.flare_bank.pairs(split)}
        flare = sorted(str(row["name"]) for row in rows if row["kind"] == "flareimage" and row["name"] in used)
    return {"tartanair": tartan, "flareimage": flare}


def read_selection(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(SELECTION_FIELDS) - set(rows[0]):
        raise ValueError(f"{path}: can cac cot {SELECTION_FIELDS}")
    return [{**row, "index": int(row["index"])} for row in rows]


def write_selection(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, SELECTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def describe_selection(dataset, rows) -> str:
    tartan = [dataset.samples[int(row["index"])] for row in rows if row["kind"] == "tartanair"]
    flare = Counter(str(row["group"]) for row in rows if row["kind"] == "flareimage")
    return (f"{len(tartan)} ảnh TartanAir ({len({s.trajectory_key for s in tartan})} trajectory, "
            f"{len({s.environment for s in tartan})} môi trường) + {sum(flare.values())} cặp FlareImage "
            f"({', '.join(f'{source} {count}' for source, count in sorted(flare.items())) or 'không có'})")


def blur_text(image: dict[str, object]) -> str:
    if image.get("clean"):
        return "không nhiễu"
    if image.get("mode") not in OPTICAL_MODES:
        return "không mờ"
    parts = []
    if image.get("defocus"):
        parts.append(f"lệch tiêu cự σ{float(image['defocus_sigma']):.2f}")
    if image.get("motion"):
        parts.append(f"chuyển động {int(image['motion_length'])}px")
    if image.get("downsample"):
        parts.append(f"thu nhỏ ×{float(image['downsample_scale']):.2f}")
    return ", ".join(parts) or "không mờ"


def restore(forward, batch, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(anh khoi phuc [B,3,H,W] trong [0,1], IMU khoi phuc [B,6,T] don vi vat ly), tren CPU."""
    with torch.no_grad():
        out = forward(batch["image_noisy"].to(device), batch["imu_noisy_phys"].to(device),
                      batch["image_time"].to(device), batch["imu_times"].to(device))
    return out["image"].float().clamp(0, 1).cpu(), out["imu_physical"].float().cpu()


def score(batch, image, imu) -> list[dict[str, object]]:
    """Mot dong moi mau: anh vao/ra so voi anh sach, IMU vao/ra so voi IMU sach."""
    rows = []
    for k, sample_id in enumerate(batch["sample_id"]):
        clean = batch["image_clean"][k:k + 1].float()
        before = image_metrics(batch["image_noisy"][k:k + 1].float(), clean)
        after = image_metrics(image[k:k + 1], clean)
        imu_clean = batch["imu_clean_phys"][k:k + 1].float()
        imu_before = imu_metrics(batch["imu_noisy_phys"][k:k + 1].float(), imu_clean)
        imu_after = imu_metrics(imu[k:k + 1], imu_clean)
        corruption = batch["corruption"][k]
        pair = corruption["image"].get("flare_pair")
        rows.append({
            "sample_id": sample_id, "kind": pair_kind(pair),
            "source": pair["source"] if pair else batch["trajectory_key"][k],
            "frame_kind": frame_kind(corruption["image"]), "blur": blur_text(corruption["image"]),
            "imu_donor": corruption.get("imu_donor"),
            "psnr_in": before["image_psnr_db"], "psnr_out": after["image_psnr_db"],
            "ssim_in": before["image_ssim"], "ssim_out": after["image_ssim"],
            "edge_in": before["image_edge_power"], "edge_out": after["image_edge_power"],
            "accel_in": imu_before["accel_rmse"], "accel_out": imu_after["accel_rmse"],
            "gyro_in": imu_before["gyro_rmse"], "gyro_out": imu_after["gyro_rmse"],
        })
    return rows


METRICS = ("psnr_in", "psnr_out", "ssim_in", "ssim_out", "edge_in", "edge_out", "accel_in", "accel_out",
           "gyro_in", "gyro_out")


def summarize(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {"count": 0}
    values = {key: np.array([float(row[key]) for row in rows]) for key in METRICS}
    return {"count": len(rows), **{key: float(value.mean()) for key, value in values.items()},
            "gain_db": float((values["psnr_out"] - values["psnr_in"]).mean()),
            "better_fraction": float((values["psnr_out"] > values["psnr_in"]).mean())}


def pair_kind(pair) -> str:
    """Loai mau tu thong tin cap cua no: FlareImage, cap drone (scene/..., p46+), cap anh toi (lowlight/..., p47),
    khong co cap: khung TartanAir (hay cap drone sach, dich = dau vao)."""
    if not pair:
        return "tartanair"
    source = str(pair.get("source", ""))
    return "scene" if source.startswith("scene/") else "lowlight" if source.startswith("lowlight/") else "flareimage"


def groups(rows) -> list[tuple[str, list[dict[str, object]]]]:
    tartan = [row for row in rows if row["kind"] == "tartanair"]
    flare = [row for row in rows if row["kind"] == "flareimage"]
    found = [("tất cả", rows), ("TartanAir", tartan)]
    found += [(f"  TartanAir · {label}", [row for row in tartan if row["frame_kind"] == kind])
              for kind, label in KIND_LABELS.items()]
    found += [("FlareImage", flare)]
    found += [(f"  FlareImage · {source}", [row for row in flare if row["source"] == source])
              for source in sorted({row["source"] for row in flare})]
    found += [("Cặp drone", [row for row in rows if row["kind"] == "scene"]),
              ("Cặp ảnh tối", [row for row in rows if row["kind"] == "lowlight"])]
    return [(name, members) for name, members in found if members]


def short(row) -> str:
    if row["kind"] == "flareimage":
        return f"FlareImage {row['source']}"
    if row["kind"] in ("scene", "lowlight"):
        return str(row["source"])
    environment, _, trajectory = str(row["source"]).partition("/")
    return f"{environment}/{trajectory.rpartition('/')[2]}"


def print_rows(rows, offset: int = 0) -> None:
    print(f"  {'#':>3} {'mẫu':30s} {'loại':9s} {'mờ':34s} {'PSNR vào→ra (dB)':>18s} {'SSIM vào→ra':>14s} "
          f"{'gia tốc vào→ra':>17s} {'con quay vào→ra':>17s}")
    for k, row in enumerate(rows):
        print(f"  {k + offset:>3} {short(row)[:30]:30s} {KIND_LABELS[row['frame_kind']]:9s} {row['blur'][:34]:34s} "
              f"{row['psnr_in']:8.2f} → {row['psnr_out']:5.2f} {row['ssim_in']:6.3f} → {row['ssim_out']:.3f} "
              f"{row['accel_in']:7.3f} → {row['accel_out']:.3f} {row['gyro_in']:7.4f} → {row['gyro_out']:.4f}")


def print_summary(rows) -> dict[str, dict[str, float]]:
    table = {}
    print(f"  {'nhóm':24s} {'số':>4s} {'PSNR vào':>9s} {'PSNR ra':>8s} {'lời dB':>7s} {'ra>vào':>7s} "
          f"{'SSIM vào→ra':>14s} {'nét /sạch vào→ra':>17s} {'gia tốc vào→ra':>16s} {'con quay vào→ra':>17s}")
    for name, members in groups(rows):
        row = table[name.strip()] = summarize(members)
        print(f"  {name:24s} {row['count']:>4d} {row['psnr_in']:>9.2f} {row['psnr_out']:>8.2f} {row['gain_db']:>+7.2f}"
              f" {100 * row['better_fraction']:>6.0f}% {row['ssim_in']:>6.3f} → {row['ssim_out']:.3f}"
              f" {row['edge_in']:>8.3f} → {row['edge_out']:.3f} {row['accel_in']:>7.3f} → {row['accel_out']:.3f}"
              f" {row['gyro_in']:>8.4f} → {row['gyro_out']:.4f}")
    return table


def pick(rows, count: int) -> list[int]:
    """Toi da ``count`` vi tri de ve: mot nua FlareImage, mot nua TartanAir (cach deu trong moi loai), theo thu tu."""
    kinds = {kind: [k for k, row in enumerate(rows) if row["kind"] == kind] for kind in ("flareimage", "tartanair")}
    want_flare = min(len(kinds["flareimage"]), max(count // 2, count - len(kinds["tartanair"])))
    chosen = []
    for kind, want in (("flareimage", want_flare), ("tartanair", min(count - want_flare, len(kinds["tartanair"])))):
        if want:
            chosen += [kinds[kind][int(k)] for k in np.linspace(0, len(kinds[kind]) - 1, want).round()]
    return sorted(set(chosen))


def plot_images(path: Path, title: str, entries) -> None:
    """entries: (nhan, sach, vao, ra, dong) -- hai mau moi hang: sach | dau vao | khoi phuc; do = FlareImage."""
    per_row = 2
    lines = math.ceil(len(entries) / per_row)
    figure, axes = plt.subplots(lines, 3 * per_row, figsize=(2.8 * 3 * per_row, 3.1 * lines), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for n, (label, clean, noisy, restored, row) in enumerate(entries):
        line, column = divmod(n, per_row)
        colour = "crimson" if row["kind"] == "flareimage" else "black"
        panels = ((f"{label}\nsạch", clean), (f"vào {row['psnr_in']:.1f} dB\n{row['blur'][:28]}", noisy),
                  (f"ra {row['psnr_out']:.1f} dB ({row['psnr_out'] - row['psnr_in']:+.1f})", restored))
        for offset, (text, tensor) in enumerate(panels):
            axis = axes[line, 3 * column + offset]
            axis.imshow(tensor.permute(1, 2, 0).clamp(0, 1).numpy())
            axis.set_title(text, fontsize=8, color=colour)
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def plot_imu(path: Path, title: str, entries) -> None:
    """entries: (nhan, thoi diem [T], sach [6,T], nhieu [6,T], khoi phuc [6,T], dong) -- moi mau mot hang 6 kenh."""
    figure, axes = plt.subplots(len(entries), 6, figsize=(24, 2.7 * len(entries)), squeeze=False)
    for line, (label, times, clean, noisy, restored, row) in enumerate(entries):
        seconds = times - times[0]
        for channel, name in enumerate(CHANNELS):
            axis = axes[line, channel]
            axis.plot(seconds, noisy[channel], color="#e87531", linewidth=1.8, alpha=0.55, label="nhiễu (vào)")
            axis.plot(seconds, clean[channel], color="#333333", linewidth=1.0, label="sạch")
            axis.plot(seconds, restored[channel], color="#2476c4", linewidth=1.0, label="khôi phục")
            axis.set_title(f"{name} ({'m/s²' if channel < 3 else 'rad/s'})", fontsize=8)
            axis.grid(alpha=0.2)
        donor = f"\nIMU mượn {str(row['imu_donor'])[:28]}" if row["imu_donor"] else ""
        axes[line, 0].set_ylabel(f"{label}{donor}\ngia tốc {row['accel_in']:.3f}→{row['accel_out']:.3f}\n"
                                 f"con quay {row['gyro_in']:.4f}→{row['gyro_out']:.4f}", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("giây")
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=100)
    plt.close(figure)


def imu_picks(rows, shown: list[int]) -> list[int]:
    """Hai mau de ve IMU: TartanAir dau tien va FlareImage dau tien trong cac mau da ve (thieu loai nao thi lay hai
    mau cua loai con lai)."""
    first = [next((k for k in shown if rows[k]["kind"] == kind), None) for kind in ("tartanair", "flareimage")]
    chosen = [k for k in first if k is not None]
    return chosen + [k for k in shown if k not in chosen][:max(0, 2 - len(chosen))]


def with_flare_pairs(config: dict, root: str | None, needed: bool) -> dict:
    """``config`` (ban sao) co cac khoa data.flare_pairs_*: cua chinh no, hoac cua kaggle_flareimage.yaml (cung phep
    chia) khi checkpoint khong train voi FlareImage; ``root`` thay thu muc FlareImage."""
    config = copy.deepcopy(config)
    if needed and float(config["data"].get("flare_pairs_fraction", 0)) <= 0:
        if not root:
            raise ValueError("config nay khong co FlareImage: can --flare-pairs-root (hoac --flare 0)")
        recipe = load_config(REPO / "configs/kaggle_flareimage.yaml")["data"]
        config["data"].update({key: recipe[key] for key in FLARE_PAIR_KEYS})
    if root:
        config["data"]["flare_pairs_root"] = str(root)
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="checkpoint phase 2; khong co thi chi chon tap test nho (can --config)")
    parser.add_argument("--config",
                        help="config da resolve (configs/kaggle.yaml cua notebook), khi khong co --checkpoint")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selection", required=True,
                        help="CSV tap test nho: co roi thi dung dung cac mau trong do, chua co thi chon va ghi")
    parser.add_argument("--split", default="test", choices=("valid", "test"))
    parser.add_argument("--tartanair", type=int, default=48, help="so anh TartanAir khi chon tap moi")
    parser.add_argument("--flare", type=int, default=48, help="so cap FlareImage khi chon tap moi (0 = khong co)")
    parser.add_argument("--output", help="thu muc hinh va report.json (can --checkpoint)")
    parser.add_argument("--train-phase", default="phase2", choices=("phase1", "phase2"))
    parser.add_argument("--train-update", type=int, default=0, help="update thu may (0 = update dau tien)")
    parser.add_argument("--show", type=int, default=8, help="so mau ve anh cho moi phan")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="fp16 autocast tren CUDA, nhu phase 2 da train")
    parser.add_argument("--flare-pairs-root", help="thu muc FlareImage (mac dinh: data.flare_pairs_root cua config)")
    args = parser.parse_args(argv)
    if args.checkpoint is None and args.config is None:
        parser.error("can --checkpoint (cham) hoac --config (chi chon tap test nho)")
    if args.checkpoint is not None and args.output is None:
        parser.error("--checkpoint can --output")
    if args.train_update < 0 or args.show < 1 or args.batch < 1:
        parser.error("--train-update >= 0, --show >= 1, --batch >= 1")
    device = torch.device(args.device)
    system = None
    if args.checkpoint is not None:
        system, config = _system_from_phase2(args.checkpoint, device)
    else:
        config = load_config(args.config)
    selection = Path(args.selection)
    existing = read_selection(selection) if selection.is_file() else None
    needs_pairs = any(row["kind"] == "flareimage" for row in existing) if existing is not None else args.flare > 0
    test_config = with_flare_pairs(config, args.flare_pairs_root, needs_pairs)
    manifest, _ = _manifest(test_config, args.manifest)
    bank = flare_pair_bank(test_config) if needs_pairs else None
    dataset = _dataset(test_config, manifest, args.split, fixed_realization=True, flare_pairs=bank)
    dataset.flare_pairs_split = args.split
    if existing is not None:
        verify(dataset, existing)
        rows = existing
        print(f"Tập test nhỏ có sẵn: {selection}")
    else:
        rows = select(dataset, args.tartanair, args.flare)
        if not rows:
            parser.error("tap test nho rong: --tartanair hoac --flare phai > 0")
        write_selection(selection, rows)
        print(f"Tập test nhỏ mới: {selection}")
    print(f"  split {args.split}, nhiễu realization {dataset.realization}: {describe_selection(dataset, rows)}")
    overlap = held_out_overlap(dataset, rows, manifest)
    if overlap["tartanair"] or overlap["flareimage"]:
        raise ValueError(f"Tap test nho trung train/valid: {overlap}")
    print("  không trùng train/valid: 0 trajectory TartanAir chung, 0 cặp FlareImage chung")
    if system is None:
        return

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    forward = RestorationForward(system)
    forward.amp = bool(args.amp) and device.type == "cuda"
    report = {"checkpoint": str(args.checkpoint), "selection": str(selection), "split": args.split,
              "realization": dataset.realization}

    # --- Batch train: dung cac microbatch CLI dua vao model (cung sampler, cung nhieu, cung o FlareImage) ---
    phase = args.train_phase
    train_config = copy.deepcopy(config)
    train_config["data"]["num_workers"] = 0
    if args.flare_pairs_root and float(train_config["data"].get("flare_pairs_fraction", 0)) > 0:
        train_config["data"]["flare_pairs_root"] = str(args.flare_pairs_root)
    batch_size = int(train_config[phase]["batch_size"])
    microbatches = int(train_config["phase2"]["gradient_accumulation"]) if phase == "phase2" else 1
    extra = {"scenarios": train_config["phase2"].get("train_scenarios")} if phase == "phase2" else {}
    train_dataset = _train_dataset(train_config, manifest, phase, **extra)
    expected = train_dataset.flare_pairs_per_batch(batch_size)
    stream = _training_batch_stream(train_config, train_dataset, batch_size,
                                    start_microbatch=args.train_update * microbatches, namespace=phase)
    train_rows, train_tensors = [], []
    for number in range(microbatches):
        batch = next(stream)
        # Chi cap FlareImage: cap drone co loe va cap anh toi cung mang flare_pair (p49 tren Kaggle: 3 thay vi 2).
        found = sum(pair_kind(corruption["image"].get("flare_pair")) == "flareimage" for corruption in batch["corruption"])
        if found != expected:
            raise AssertionError(f"microbatch {number}: {found} mau FlareImage, config doi {expected}")
        image, imu = restore(forward, batch, device)
        train_rows += score(batch, image, imu)
        for k in range(len(batch["sample_id"])):
            train_tensors.append((batch["image_clean"][k], batch["image_noisy"][k], image[k],
                                  batch["imu_times"][k].numpy(), batch["imu_clean_phys"][k].numpy(),
                                  batch["imu_noisy_phys"][k].numpy(), imu[k].numpy()))
    flare_total = sum(row["kind"] == "flareimage" for row in train_rows)
    print(f"\nBATCH TRAIN {phase}, update {args.train_update}: {microbatches} microbatch × {batch_size} = "
          f"{len(train_rows)} mẫu, {flare_total} FlareImage ({expected} mỗi microbatch) -- model đã train trên chúng")
    print_rows(train_rows)
    report["train"] = {"phase": phase, "update": args.train_update, "microbatches": microbatches,
                       "batch_size": batch_size, "flare_per_microbatch": expected, "rows": train_rows,
                       "summary": {name.strip(): summarize(members) for name, members in groups(train_rows)}}
    shown = pick(train_rows, args.show)
    plot_images(output / "train_batch.png",
                f"Batch train {phase} update {args.train_update}: sạch | vào | ra (đỏ = FlareImage), "
                f"{len(shown)}/{len(train_rows)} mẫu",
                [(f"#{k} {short(train_rows[k])}", *train_tensors[k][:3], train_rows[k]) for k in shown])
    plot_imu(output / "train_imu.png", f"IMU batch train {phase} update {args.train_update}: sạch / nhiễu / khôi phục",
             [(f"#{k} {short(train_rows[k])}", *train_tensors[k][3:], train_rows[k])
              for k in imu_picks(train_rows, shown)])

    # --- Tap test nho ---
    test_rows, test_tensors = [], {}
    shown = pick(rows, args.show)                     # chi theo loai mau: giu tensor cua cac mau se ve
    for start in range(0, len(rows), args.batch):
        chunk = rows[start:start + args.batch]
        items = [dataset[int(row["index"]) + (len(dataset.samples) if row["kind"] == "flareimage" else 0)]
                 for row in chunk]
        if [item["sample_id"] for item in items] != [row["sample_id"] for row in chunk]:
            raise AssertionError("dataset tra mau khac file tap test nho")
        batch = collate_paired(items)
        image, imu = restore(forward, batch, device)
        test_rows += score(batch, image, imu)
        for k in range(len(items)):
            if start + k in shown:
                test_tensors[start + k] = (batch["image_clean"][k], batch["image_noisy"][k], image[k],
                                           batch["imu_times"][k].numpy(), batch["imu_clean_phys"][k].numpy(),
                                           batch["imu_noisy_phys"][k].numpy(), imu[k].numpy())
    print(f"\nTẬP TEST NHỎ ({args.split}, model chưa thấy): {describe_selection(dataset, rows)}")
    report["test"] = {"rows": test_rows, "summary": print_summary(test_rows)}
    print("\n  các mẫu được vẽ:")
    print_rows([test_rows[k] for k in shown])
    plot_images(output / "test_images.png",
                f"Tập test nhỏ ({args.split}): sạch | vào | ra (đỏ = FlareImage), {len(shown)}/{len(test_rows)} mẫu",
                [(f"{short(test_rows[k])}", *test_tensors[k][:3], test_rows[k]) for k in shown])
    plot_imu(output / "test_imu.png", f"IMU tập test nhỏ ({args.split}): sạch / nhiễu / khôi phục",
             [(f"{short(test_rows[k])}", *test_tensors[k][3:], test_rows[k]) for k in imu_picks(test_rows, shown)])
    (output / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã lưu {output}: train_batch.png, train_imu.png, test_images.png, test_imu.png, report.json")


if __name__ == "__main__":
    main()
