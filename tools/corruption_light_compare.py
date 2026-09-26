"""So sánh nhiễu A (p5–p11 đã train) với nhiễu B (recipe train từ p12), trên cùng một frame.

Mỗi frame được làm hỏng hai lần:

    A — nhiễu cũ:  NOISE_A bên dưới, đúng như p5–p11 đã train
    B — nhiễu nhẹ: corruption.image của configs/pipeline_v3.yaml — ít mờ hơn A, ít hạt nhiễu
        hơn rõ rệt, tối nhẹ hơn một chút; và mỗi frame một biến thể: phần lớn vừa tối vừa
        nhiễu, ~10% CHỈ NHIỄU (không tối), ~10% CHỈ TỐI (không hạt)

Hai ảnh dùng CHUNG một lượt bốc thông số: cùng frame nào bị defocus / motion blur /
thu nhỏ / JPEG, cùng hướng vệt mờ. Mỗi giá trị đã bốc trong khoảng của A được ánh xạ
tuyến tính sang khoảng của B (vị trí tương đối giữ nguyên), nên hai ảnh chỉ khác nhau
ở ĐỘ MẠNH. Hạt nhiễu có cùng kiểu nhưng không trùng từng điểm: số photon khác thì phép
bốc Poisson tiêu số ngẫu nhiên khác. Xác suất (frame nào bị mờ) giữ nguyên; muốn đổi
thì đổi trong YAML khi train.

Hàng dưới phóng to vùng nhiều chi tiết nhất của ảnh sạch để nhìn rõ độ mờ và hạt.
"Hạt nhiễu σ" là độ lệch chuẩn của phần cảm biến thêm vào — ảnh cuối trừ chính ảnh đó
khi chưa có nhiễu cảm biến (cùng lượt bốc) — theo đơn vị /255.

    python3 tools/corruption_light_compare.py
    python3 tools/corruption_light_compare.py --samples 4 --seed 7 --brighten 2

Không import torch; numpy, scipy, Pillow, matplotlib là đủ.
"""

from __future__ import annotations

import argparse
import csv
import math
import secrets
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Imported before pyplot on purpose: this module picks the backend, and
# matplotlib.use() only takes effect before pyplot is loaded.
from imu_blur_preview import CAN_SHOW, GRID, INK, INK_SOFT, SURFACE, load_rgb, show_or_close  # noqa: F401

import matplotlib.pyplot as plt
from PIL import Image

from qjepa.corruptions.image import LowLightImageCorruptionConfig, LowLightImageCorruptor
from qjepa.corruptions.rng import generator

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = Path("/home/buidinhkhoi/Datasets/tartanair-v2-jepa")

# Nhiễu A — nhiễu p5–p11 đã train, trước khi recipe chuyển sang B (26/09/2026).
# Nhiễu B đọc từ corruption.image của --config; các khoá không có ở đây giống nhau ở A và B.
NOISE_A: dict[str, tuple[float, float]] = {
    "defocus_sigma_px": (0.30, 1.45),
    "motion_length_px": (3, 9),
    "downsample_scale": (0.72, 0.96),
    "photon_count": (550.0, 4000.0),
    "read_noise_std": (0.7 / 255, 3.2 / 255),
    "row_noise_std": (0.0, 1.0 / 255),
    "hot_pixel_probability": (0.0, 1.5e-4),
    "quantization_bits": (6, 8),
    "exposure_gain": (0.21, 0.63),
    "tone_gamma": (0.46, 0.79),
    "jpeg_quality": (35, 80),
}
# A không có biến thể: frame nào cũng vừa tối vừa nhiễu.
VARIANTS = ("noise_only_probability", "low_light_only_probability")
# Tên trong bảng thông số đã bốc, và cách bốc: tuyến tính, log (photon), hay số nguyên.
PARAMETER = {
    "defocus_sigma_px": ("defocus_sigma", "linear"),
    "motion_length_px": ("motion_length", "integer"),
    "downsample_scale": ("downsample_scale", "linear"),
    "photon_count": ("photon_count", "log"),
    "read_noise_std": ("read_noise_std", "linear"),
    "row_noise_std": ("row_noise_std", "linear"),
    "hot_pixel_probability": ("hot_pixel_probability", "linear"),
    "quantization_bits": ("quantization_bits", "integer"),
    "exposure_gain": ("exposure_gain", "linear"),
    "tone_gamma": ("tone_gamma", "linear"),
    "jpeg_quality": ("jpeg_quality", "integer"),
}


def remap(value: float, source: tuple[float, float], target: tuple[float, float], kind: str) -> float:
    """Giữ vị trí tương đối của ``value`` trong ``source``, đặt nó vào ``target``."""
    low, high = source
    if kind == "log":
        low, high, value = math.log(low), math.log(high), math.log(value)
    fraction = 0.5 if high == low else (value - low) / (high - low)
    new_low, new_high = target
    if kind == "log":
        return math.exp(math.log(new_low) + fraction * (math.log(new_high) - math.log(new_low)))
    mapped = new_low + fraction * (new_high - new_low)
    return int(round(mapped)) if kind == "integer" else mapped


class RemappedCorruptor(LowLightImageCorruptor):
    """Bốc thông số đúng như A, rồi đưa từng giá trị về khoảng của B.

    Đổi thẳng khoảng trong config thì không so được từng cặp: ``rng.integers`` với
    khoảng khác có thể tiêu số lần bốc khác, làm lệch mọi thông số bốc sau nó. Ở đây
    dòng bốc ngẫu nhiên giữ nguyên, nên A và B là cùng một "camera", chỉ nhẹ tay hơn.
    """

    def __init__(self, config: LowLightImageCorruptionConfig, master_seed: int,
                 target: LowLightImageCorruptionConfig):
        super().__init__(config, master_seed)
        self.target = target

    def _parameters(self, split, realization, trajectory, timestamp, mode):
        params = super()._parameters(split, realization, trajectory, timestamp, mode)
        for key, (name, kind) in PARAMETER.items():
            params[name] = remap(params[name], getattr(self.config, key), getattr(self.target, key), kind)
        # Biến thể của B, từ chính số ngẫu nhiên đã bốc cho A (A luôn "cả hai").
        noise_only, low_light_only = (getattr(self.target, key) for key in VARIANTS)
        variant = params["variant_draw"]
        params["low_light"] = not variant < noise_only
        params["sensor_noise"] = not noise_only <= variant < noise_only + low_light_only
        return params


def psnr(reference: np.ndarray, other: np.ndarray) -> float:
    error = float(np.mean((reference.astype(np.float64) - other.astype(np.float64)) ** 2))
    return 10.0 * np.log10(1.0 / max(error, 1e-12))


def edge_power(image: np.ndarray) -> float:
    """Năng lượng phổ độ sáng ở chu kỳ 4–16 px — nơi có cạnh và texture."""
    luma = image.astype(np.float64) @ (0.299, 0.587, 0.114)
    height, width = luma.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    power = np.abs(np.fft.fft2((luma - luma.mean()) * window)) ** 2
    band = np.maximum(np.abs(np.fft.fftfreq(height))[:, None], np.abs(np.fft.fftfreq(width))[None, :])
    return float(power[(band >= 1 / 16) & (band < 1 / 4)].sum())


def grain(noisy: np.ndarray, before_sensor: np.ndarray) -> float:
    """σ của phần cảm biến thêm vào (photon, đọc, hàng, hot pixel, lượng tử, JPEG), /255."""
    return float((noisy.astype(np.float64) - before_sensor.astype(np.float64)).std() * 255.0)


def busiest_crop(clean: np.ndarray, size: int) -> tuple[int, int]:
    """Góc trên-trái của ô ``size`` x ``size`` nhiều cạnh nhất: chỗ độ mờ lộ rõ nhất."""
    luma = clean.astype(np.float64) @ (0.299, 0.587, 0.114)
    gradient = np.abs(np.diff(luma, axis=0))[:, :-1] + np.abs(np.diff(luma, axis=1))[:-1, :]
    integral = np.pad(gradient, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best, corner = -1.0, (0, 0)
    for top in range(0, gradient.shape[0] - size + 1, 8):
        for left in range(0, gradient.shape[1] - size + 1, 8):
            total = (integral[top + size, left + size] - integral[top, left + size]
                     - integral[top + size, left] + integral[top, left])
            if total > best:
                best, corner = total, (top, left)
    return corner


def describe(params: dict) -> str:
    blur = []
    if params["defocus"]:
        blur.append(f"defocus σ {params['defocus_sigma']:.2f} px")
    if params["motion"]:
        blur.append(f"motion {params['motion_length']} px")
    if params["downsample"]:
        blur.append(f"thu nhỏ ×{params['downsample_scale']:.2f}")
    line1 = " · ".join(blur) if blur else "không mờ quang học"
    dark = f"sáng ×{params['exposure_gain']:.2f} · gamma {params['tone_gamma']:.2f}"
    grainy = (f"{params['photon_count']:.0f} photon · {params['quantization_bits']} bit"
              + (f" · JPEG q{params['jpeg_quality']}" if params["jpeg"] else ""))
    if not params.get("low_light", True):
        line2 = f"CHỈ NHIỄU, không tối · {grainy}"
    elif not params.get("sensor_noise", True):
        line2 = f"CHỈ TỐI, không hạt · {dark}"
    else:
        line2 = f"{dark} · {grainy}"
    return f"{line1}\n{line2}"


def show(axis, image, title, caption, brighten):
    axis.imshow(np.clip(image * brighten, 0, 1), interpolation="nearest")
    axis.set_xticks([]); axis.set_yticks([])
    for side in axis.spines.values():
        side.set_color(GRID)
    if title:
        axis.set_title(title, color=INK, fontsize=11, loc="left", pad=6)
    if caption:
        axis.set_xlabel(caption, color=INK_SOFT, fontsize=8.5, linespacing=1.5)


def build_figure(sample: dict, brighten: float, crop: int):
    clean = sample["clean"]
    reference = edge_power(clean)
    top, left = busiest_crop(clean, crop)
    region = (slice(top, top + crop), slice(left, left + crop))
    figure = plt.figure(figsize=(15.5, 10.4), facecolor=SURFACE)
    grid = figure.add_gridspec(2, 3, hspace=0.32, wspace=0.08, left=0.03, right=0.985, top=0.9, bottom=0.07)
    lit = f" (hiển thị sáng ×{brighten:g})" if brighten != 1 else ""
    show(figure.add_subplot(grid[0, 0]), clean, "ẢNH SẠCH", "tham chiếu", 1.0)
    show(figure.add_subplot(grid[1, 0]), clean[region], "", f"phóng to {crop}×{crop} px", 1.0)
    for column, key, title in ((1, "A", "A — NHIỄU CŨ (p5–p11)"), (2, "B", "B — NHIỄU NHẸ (train từ p12)")):
        image, params = sample[key], sample[key + "_params"]
        caption = (describe(params) + f"\nPSNR {psnr(clean, image):.2f} dB · đường nét 4–16 px "
                   f"{edge_power(image) / reference:.2f}× ảnh sạch")
        if sample.get(key + "_grain") is not None:
            caption += f" · hạt nhiễu σ {sample[key + '_grain']:.1f}/255"
        show(figure.add_subplot(grid[0, column]), image, title + lit, caption, brighten)
        show(figure.add_subplot(grid[1, column]), image[region], "", f"phóng to {crop}×{crop} px", brighten)
    figure.suptitle(f"{sample['name']}  ·  seed {sample['seed']}\n"
                    "cùng một lượt bốc ngẫu nhiên: A và B chỉ khác nhau ở độ mạnh",
                    color=INK, fontsize=12.5, x=0.03, ha="left", y=0.985)
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", default="valid", choices=("train", "valid", "test"))
    parser.add_argument("--config", type=Path, default=REPO / "configs/pipeline_v3.yaml")
    parser.add_argument("--out", type=Path, default=REPO / "outputs/corruption_light_compare")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None,
                        help="mặc định: ngẫu nhiên mỗi lần chạy; đặt để lặp lại đúng mẫu cũ")
    parser.add_argument("--mode", default="full",
                        choices=("full", "blur_only", "blur_low_light", "low_light_only", "sensor_noise_only"),
                        help="full = đúng như lúc train")
    parser.add_argument("--brighten", type=float, default=1.0,
                        help="chỉ làm sáng khi HIỂN THỊ A và B (ảnh lưu vẫn giữ nguyên)")
    parser.add_argument("--crop", type=int, default=96)
    parser.add_argument("--no-show", action="store_true", help="chỉ ghi file, không mở cửa sổ")
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    # clean_probability 0: lúc train 2% frame được để sạch; ở đây frame nào cũng phải bị làm hỏng.
    noise_b = replace(LowLightImageCorruptionConfig(**raw["corruption"]["image"]), clean_probability=0.0)
    if noise_b.motion_from_imu:
        raise SystemExit("Config dùng motion_from_imu: true; tool này so motion blur bốc ngẫu nhiên.")
    noise_a = replace(noise_b, **NOISE_A, **{key: 0.0 for key in VARIANTS})
    seed_master = raw["data"]["corruption_seed"]
    image_size = tuple(raw["data"].get("image_size", (256, 256)))
    # B không bốc lại: nó lấy đúng lượt bốc của A rồi đưa về khoảng của B, để hai ảnh so từng cặp.
    corruptors = {"A": LowLightImageCorruptor(noise_a, master_seed=seed_master),
                  "B": RemappedCorruptor(noise_a, seed_master, noise_b)}

    manifest = args.data_root / args.split / "manifest.csv"
    if not manifest.is_file():
        raise SystemExit(f"Không thấy {manifest}")
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seed = args.seed if args.seed is not None else secrets.randbelow(1_000_000)
    rng = generator(seed, "corruption_light_compare")
    picks = rng.choice(len(rows), size=min(args.samples, len(rows)), replace=False)

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} frame trong {args.split} · seed {seed} (chạy lại đúng mẫu này bằng --seed {seed})\n")
    figures = []
    for number, index in enumerate(picks):
        row = rows[int(index)]
        clean = load_rgb(args.data_root / args.split / row["image_path"], image_size)
        frame = int(Path(row["image_path"]).name.split("_")[0])
        trajectory = f"{row['environment']}/{row['difficulty']}/{row['trajectory']}"
        # Cùng khoá ngẫu nhiên cho cả hai: thời điểm lấy theo 10 Hz của TartanAir V2.
        shared = dict(split=args.split, realization=0, trajectory=trajectory,
                      timestamp=frame * 0.1, frame_index=frame, mode=args.mode)
        name = row["sample_id"]
        sample = {"name": name, "seed": seed, "clean": clean}
        for key, corruptor in corruptors.items():
            sample[key], sample[key + "_params"] = corruptor(clean, **shared)
            if args.mode == "full":
                # Cùng lượt bốc, dừng trước nhiễu cảm biến: hiệu hai ảnh là đúng phần hạt nhiễu.
                params = sample[key + "_params"]
                before_mode = "blur_low_light" if params["low_light"] else "blur_only"
                before, _ = corruptor(clean, **dict(shared, mode=before_mode))
                sample[key + "_grain"] = grain(sample[key], before)
        stem = args.out / f"{number:02d}_{name}"
        for suffix, image in (("clean", clean), ("A_train", sample["A"]), ("B_light", sample["B"])):
            Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255 + 0.5)).save(f"{stem}_{suffix}.png")
        figure = build_figure(sample, args.brighten, args.crop)
        figure.savefig(f"{stem}_compare.png", dpi=110, facecolor=SURFACE)
        figures.append(figure)
        reference = edge_power(clean)
        print(name)
        for key, label in (("A", "A cũ   "), ("B", "B nhẹ  ")):
            noise = f" · hạt nhiễu σ {sample[key + '_grain']:4.1f}/255" if key + "_grain" in sample else ""
            print(f"   {label}: PSNR {psnr(clean, sample[key]):5.2f} dB · đường nét 4–16 px "
                  f"{edge_power(sample[key]) / reference:4.2f}×{noise} · "
                  f"{describe(sample[key + '_params']).replace(chr(10), ' · ')}")
        print(f"   -> {stem}_compare.png\n")

    print(f"B = corruption.image trong {args.config} — recipe train từ p12.")
    show_or_close(figures, args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
