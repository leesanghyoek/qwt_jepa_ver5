"""So sánh ảnh nhiễu lúc TRAIN với ảnh nhiễu theo thông số NHẸ HƠN, trên cùng một frame.

Mỗi frame được làm hỏng hai lần:

    ② theo đúng corruption.image của configs/pipeline_v3.yaml — cùng phân phối model thấy lúc train
    ③ theo bộ thông số LIGHTER bên dưới — ít mờ hơn rõ rệt, phần còn lại nhẹ hơn một chút

Hai ảnh dùng CHUNG một lượt bốc thông số: cùng frame nào bị defocus / motion blur /
thu nhỏ / JPEG, cùng hướng vệt mờ. Mỗi giá trị đã bốc trong khoảng lúc train được ánh
xạ tuyến tính sang khoảng nhẹ hơn (vị trí tương đối giữ nguyên), nên hai ảnh chỉ khác
nhau ở ĐỘ MẠNH. Hạt nhiễu có cùng thống kê nhưng không trùng từng điểm: số photon khác
thì phép bốc Poisson tiêu số ngẫu nhiên khác. Xác suất (frame nào bị mờ) giữ nguyên;
muốn đổi thì đổi trong YAML khi train.

Hàng dưới phóng to vùng nhiều chi tiết nhất của ảnh sạch để nhìn rõ độ mờ.

    python3 tools/corruption_light_compare.py
    python3 tools/corruption_light_compare.py --samples 4 --seed 7 --brighten 2

Không import torch; numpy, scipy, Pillow, matplotlib là đủ. Cuối cùng script in đoạn
YAML của bộ nhẹ hơn để dán vào corruption.image nếu muốn train với nó.
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

# Bộ nhẹ hơn: khoá = khoá trong corruption.image, giá trị = khoảng mới.
# Sửa ở đây để thử mức khác; các khoá không có ở đây giữ nguyên như lúc train.
LIGHTER: dict[str, tuple[float, float]] = {
    # Độ mờ — giảm rõ.
    "defocus_sigma_px": (0.30, 0.80),     # train 0.30–1.45 px
    "motion_length_px": (2, 5),           # train 3–9 px
    "downsample_scale": (0.85, 0.98),     # train 0.72–0.96 (1.0 = không thu nhỏ)
    # Phần còn lại — nhẹ hơn một chút.
    "exposure_gain": (0.26, 0.68),        # train 0.21–0.63 (cao hơn = sáng hơn)
    "tone_gamma": (0.52, 0.82),           # train 0.46–0.79 (gần 1 hơn = ít nhạt màu hơn)
    "photon_count": (800.0, 5000.0),      # train 550–4000 (nhiều photon = ít hạt hơn)
    "read_noise_std": (0.6 / 255, 2.6 / 255),   # train 0.7–3.2 / 255
    "quantization_bits": (7, 8),          # train 6–8 bit
    "jpeg_quality": (50, 85),             # train 35–80
}
# Tên trong bảng thông số đã bốc, và cách bốc: tuyến tính, log (photon), hay số nguyên.
PARAMETER = {
    "defocus_sigma_px": ("defocus_sigma", "linear"),
    "motion_length_px": ("motion_length", "integer"),
    "downsample_scale": ("downsample_scale", "linear"),
    "exposure_gain": ("exposure_gain", "linear"),
    "tone_gamma": ("tone_gamma", "linear"),
    "photon_count": ("photon_count", "log"),
    "read_noise_std": ("read_noise_std", "linear"),
    "quantization_bits": ("quantization_bits", "integer"),
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


class LighterCorruptor(LowLightImageCorruptor):
    """Bốc thông số đúng như lúc train, rồi đưa từng giá trị về khoảng nhẹ hơn.

    Đổi thẳng khoảng trong config thì không so được từng cặp: ``rng.integers`` với
    khoảng khác có thể tiêu số lần bốc khác, làm lệch mọi thông số bốc sau nó. Ở đây
    dòng bốc ngẫu nhiên giữ nguyên, nên ② và ③ là cùng một "camera", chỉ nhẹ tay hơn.
    """

    def __init__(self, config: LowLightImageCorruptionConfig, master_seed: int,
                 lighter: dict[str, tuple[float, float]]):
        super().__init__(config, master_seed)
        self.lighter = lighter

    def _parameters(self, split, realization, trajectory, timestamp, mode):
        params = super()._parameters(split, realization, trajectory, timestamp, mode)
        for key, target in self.lighter.items():
            name, kind = PARAMETER[key]
            params[name] = remap(params[name], getattr(self.config, key), target, kind)
        return params


def psnr(reference: np.ndarray, other: np.ndarray) -> float:
    error = float(np.mean((reference.astype(np.float64) - other.astype(np.float64)) ** 2))
    return 10.0 * np.log10(1.0 / max(error, 1e-12))


def edge_power(image: np.ndarray) -> np.ndarray:
    """Năng lượng phổ độ sáng ở chu kỳ 4–16 px — nơi có cạnh và texture."""
    luma = image.astype(np.float64) @ (0.299, 0.587, 0.114)
    height, width = luma.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    power = np.abs(np.fft.fft2((luma - luma.mean()) * window)) ** 2
    band = np.maximum(np.abs(np.fft.fftfreq(height))[:, None], np.abs(np.fft.fftfreq(width))[None, :])
    return power[(band >= 1 / 16) & (band < 1 / 4)].sum()


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
    clean, train, light = sample["clean"], sample["train"], sample["light"]
    reference = edge_power(clean)
    top, left = busiest_crop(clean, crop)
    region = (slice(top, top + crop), slice(left, left + crop))
    figure = plt.figure(figsize=(15.5, 10.2), facecolor=SURFACE)
    grid = figure.add_gridspec(2, 3, hspace=0.30, wspace=0.08, left=0.03, right=0.985, top=0.9, bottom=0.07)
    lit = f" (hiển thị sáng ×{brighten:g})" if brighten != 1 else ""
    columns = (
        ("① ẢNH SẠCH", clean, 1.0, "tham chiếu"),
        ("② NHIỄU LÚC TRAIN" + lit, train, brighten, describe(sample["train_params"])),
        ("③ NHIỄU NHẸ HƠN" + lit, light, brighten, describe(sample["light_params"])),
    )
    for index, (title, image, gain, caption) in enumerate(columns):
        if index:
            caption += (f"\nPSNR {psnr(clean, image):.2f} dB · đường nét 4–16 px "
                        f"{edge_power(image) / reference:.2f}× ảnh sạch")
        show(figure.add_subplot(grid[0, index]), image, title, caption, gain)
        show(figure.add_subplot(grid[1, index]), image[region], "", f"phóng to {crop}×{crop} px", gain)
    figure.suptitle(f"{sample['name']}  ·  seed {sample['seed']}\n"
                    "cùng một lượt bốc ngẫu nhiên: ② và ③ chỉ khác nhau ở độ mạnh",
                    color=INK, fontsize=12.5, x=0.03, ha="left", y=0.985)
    return figure


def lighter_yaml(base: LowLightImageCorruptionConfig) -> str:
    values = {key: [round(v, 6) if isinstance(v, float) else v for v in target]
              for key, target in LIGHTER.items()}
    lines = ["corruption:", "  image:"]
    for key, value in values.items():
        lines.append(f"    {key}: {value}    # train: {list(getattr(base, key))}")
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
                        help="chỉ làm sáng khi HIỂN THỊ ② và ③ (ảnh lưu vẫn đúng như lúc train)")
    parser.add_argument("--crop", type=int, default=96)
    parser.add_argument("--no-show", action="store_true", help="chỉ ghi file, không mở cửa sổ")
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    # clean_probability 0: lúc train 2% frame được để sạch; ở đây frame nào cũng phải bị làm hỏng.
    base = replace(LowLightImageCorruptionConfig(**raw["corruption"]["image"]), clean_probability=0.0)
    if base.motion_from_imu:
        raise SystemExit("Config dùng motion_from_imu: true; tool này so motion blur bốc ngẫu nhiên.")
    for key, target in LIGHTER.items():
        if key not in PARAMETER:
            raise SystemExit(f"LIGHTER có khoá lạ: {key}")
        if target[0] > target[1]:
            raise SystemExit(f"LIGHTER[{key}] phải là [thấp, cao]")
    seed_master = raw["data"]["corruption_seed"]
    image_size = tuple(raw["data"].get("image_size", (256, 256)))
    train_corruptor = LowLightImageCorruptor(base, master_seed=seed_master)
    light_corruptor = LighterCorruptor(base, seed_master, LIGHTER)

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
        train, train_params = train_corruptor(clean, **shared)
        light, light_params = light_corruptor(clean, **shared)
        name = row["sample_id"]
        stem = args.out / f"{number:02d}_{name}"
        for suffix, image in (("clean", clean), ("train", train), ("light", light)):
            Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255 + 0.5)).save(f"{stem}_{suffix}.png")
        figure = build_figure({"name": name, "seed": seed, "clean": clean, "train": train,
                               "light": light, "train_params": train_params,
                               "light_params": light_params}, args.brighten, args.crop)
        figure.savefig(f"{stem}_compare.png", dpi=110, facecolor=SURFACE)
        figures.append(figure)
        reference = edge_power(clean)
        print(name)
        for label, image, params in (("② train  ", train, train_params), ("③ nhẹ hơn", light, light_params)):
            print(f"   {label}: PSNR {psnr(clean, image):5.2f} dB · đường nét 4–16 px "
                  f"{edge_power(image) / reference:4.2f}× · {describe(params).replace(chr(10), ' · ')}")
        print(f"   -> {stem}_compare.png\n")

    print("Bộ thông số nhẹ hơn (dán vào configs nếu muốn train với nó; xác suất giữ nguyên):")
    print(lighter_yaml(base))
    show_or_close(figures, args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
