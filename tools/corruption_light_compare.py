"""So sánh nhiễu TRAIN (A) với nhiễu NHẸ (B), trên cùng một frame.

Mỗi frame được làm hỏng hai lần:

    A — nhiễu train: đúng corruption.image của configs/pipeline_v3.yaml, cùng phân phối
        model thấy lúc train
    B — nhiễu nhẹ:   bộ thông số NOISE_B bên dưới — ít mờ hơn A, ít hạt nhiễu hơn rõ rệt,
        phần còn lại nhẹ hơn một chút

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

Không import torch; numpy, scipy, Pillow, matplotlib là đủ. Cuối cùng script in đoạn
YAML của B để dán vào corruption.image nếu muốn train với nó.
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

# Nhiễu B: khoá = khoá trong corruption.image, giá trị = khoảng mới.
# Sửa ở đây để thử mức khác; các khoá không có ở đây giữ nguyên như A.
NOISE_B: dict[str, tuple[float, float]] = {
    # Độ mờ — nhẹ hơn A, nhưng nhỉnh hơn bản đầu của B (0.30–0.80 px · 2–5 px · ×0.85–0.98).
    "defocus_sigma_px": (0.30, 0.95),     # A 0.30–1.45 px
    "motion_length_px": (2, 6),           # A 3–9 px
    "downsample_scale": (0.82, 0.97),     # A 0.72–0.96 (1.0 = không thu nhỏ)
    # Hạt nhiễu — giảm rõ. Nhiễu photon có σ = sqrt(độ sáng / số photon): gấp ~4 lần
    # số photon thì hạt còn ~1/2. Nhiễu hàng là sọc ngang mảnh, hot pixel là chấm lẻ.
    "photon_count": (2500.0, 15000.0),    # A 550–4000
    "read_noise_std": (0.3 / 255, 1.2 / 255),   # A 0.7–3.2 / 255
    "row_noise_std": (0.0, 0.3 / 255),    # A 0–1 / 255
    "hot_pixel_probability": (0.0, 3e-5), # A 0–1.5e-4
    "quantization_bits": (8, 8),          # A 6–8 bit (6 bit trên ảnh tối thành bậc thang)
    # Phần còn lại — nhẹ hơn một chút.
    "exposure_gain": (0.26, 0.68),        # A 0.21–0.63 (cao hơn = sáng hơn)
    "tone_gamma": (0.52, 0.82),           # A 0.46–0.79 (gần 1 hơn = ít nhạt màu hơn)
    "jpeg_quality": (50, 85),             # A 35–80
}
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
                 ranges: dict[str, tuple[float, float]]):
        super().__init__(config, master_seed)
        self.ranges = ranges

    def _parameters(self, split, realization, trajectory, timestamp, mode):
        params = super()._parameters(split, realization, trajectory, timestamp, mode)
        for key, target in self.ranges.items():
            name, kind = PARAMETER[key]
            params[name] = remap(params[name], getattr(self.config, key), target, kind)
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
    line2 = (f"sáng ×{params['exposure_gain']:.2f} · gamma {params['tone_gamma']:.2f} · "
             f"{params['photon_count']:.0f} photon · {params['quantization_bits']} bit"
             + (f" · JPEG q{params['jpeg_quality']}" if params["jpeg"] else ""))
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
    for column, key, title in ((1, "A", "A — NHIỄU TRAIN"), (2, "B", "B — NHIỄU NHẸ")):
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


def noise_b_yaml(base: LowLightImageCorruptionConfig) -> str:
    lines = ["corruption:", "  image:"]
    for key, target in NOISE_B.items():
        value = [round(v, 6) if isinstance(v, float) else v for v in target]
        lines.append(f"    {key}: {value}    # A: {list(getattr(base, key))}")
    return "\n".join(lines)


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
    base = replace(LowLightImageCorruptionConfig(**raw["corruption"]["image"]), clean_probability=0.0)
    if base.motion_from_imu:
        raise SystemExit("Config dùng motion_from_imu: true; tool này so motion blur bốc ngẫu nhiên.")
    for key, target in NOISE_B.items():
        if key not in PARAMETER:
            raise SystemExit(f"NOISE_B có khoá lạ: {key}")
        if target[0] > target[1]:
            raise SystemExit(f"NOISE_B[{key}] phải là [thấp, cao]")
    seed_master = raw["data"]["corruption_seed"]
    image_size = tuple(raw["data"].get("image_size", (256, 256)))
    corruptors = {"A": LowLightImageCorruptor(base, master_seed=seed_master),
                  "B": RemappedCorruptor(base, seed_master, NOISE_B)}

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
                before, _ = corruptor(clean, **dict(shared, mode="blur_low_light"))
                sample[key + "_grain"] = grain(sample[key], before)
        stem = args.out / f"{number:02d}_{name}"
        for suffix, image in (("clean", clean), ("A_train", sample["A"]), ("B_light", sample["B"])):
            Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255 + 0.5)).save(f"{stem}_{suffix}.png")
        figure = build_figure(sample, args.brighten, args.crop)
        figure.savefig(f"{stem}_compare.png", dpi=110, facecolor=SURFACE)
        figures.append(figure)
        reference = edge_power(clean)
        print(name)
        for key, label in (("A", "A train"), ("B", "B nhẹ  ")):
            noise = f" · hạt nhiễu σ {sample[key + '_grain']:4.1f}/255" if key + "_grain" in sample else ""
            print(f"   {label}: PSNR {psnr(clean, sample[key]):5.2f} dB · đường nét 4–16 px "
                  f"{edge_power(sample[key]) / reference:4.2f}×{noise} · "
                  f"{describe(sample[key + '_params']).replace(chr(10), ' · ')}")
        print(f"   -> {stem}_compare.png\n")

    print("Nhiễu B (dán vào configs nếu muốn train với nó; xác suất giữ nguyên như A):")
    print(noise_b_yaml(base))
    show_or_close(figures, args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
