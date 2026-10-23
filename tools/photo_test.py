"""Anh chup that bat ky + cua so IMU bat ky qua checkpoint phase 2 (khong train, khong can dataset).

Anh (--size): "256" = o vuong 256 o giua (nhu `python -m qjepa infer`, dung co luc train); "full" = ca khung, canh ngan
thu ve --short-side (384 nhu FlareImage / VisDrone luc train), moi canh cat o giua ve boi so cua 16 (model chay duoc
tren co bat ky boi so cua 16); "ca_hai" (mac dinh). Anh AVIF can Pillow co AVIF.

IMU (--imu, mot gia tri cho moi anh, hoac dung so gia tri bang so anh, theo thu tu):
  * khong (mac dinh)        cua so bang 0: "khong co IMU", nhu 25% mau p46 (data.imu_absent_probability);
  * <file.csv>              cua so [L,6] (ax,ay,az,gx,gy,gz) hoac [L,7] (timestamp,...), L = data.imu_window (128,
                            100 Hz quanh luc chup); dong dau co the la ten cot -- cung dinh dang `qjepa infer --imu`;
  * ax,ay,az,gx,gy,gz       6 so, cua so khong doi (vd 0,0,9.81,0,0,0: dung yen);
  * ngau_nhien              dung yen + nhieu (gia toc sigma --noise-accel, con quay sigma --noise-gyro), seed theo anh.
Don vi nhu IMU TartanAir luc train: gia toc m/s^2 (co trong luc), con quay rad/s.

Duong dan (doi so vi tri): anh, hoac THU MUC -- moi anh trong do (jpg, jpeg, png, avif, webp, bmp, tif; khong vao thu
muc con), theo ten. Khong --checkpoint: tim best_joint_validation.pt trong cac thu muc da dua, roi ~/Downloads. Khong
--output: <thu muc dau tien>/ket_qua_test (hoac canh anh). Chay bang python khong co torch (vd /usr/bin/python3) thi
script tu chay lai bang ~/venv-torch/bin/python neu co.

Moi anh ghi: <anh>_<co>_so_sanh.png (vao | sau lop loe | ra | |ra - vao|), _vao.png, _ra.png; <anh>_imu.csv (IMU vao
va IMU model khoi phuc, cung timestamp) va <anh>_imu.png (6 kenh vao / ra). In: do sang vao -> ra, |ra - vao| trung
binh, phan lop loe da doi, cong tone (gan 0 = model cho la khong can doi sang), IMU ra lech IMU vao bao nhieu.

    python3 tools/photo_test.py ~/Downloads                     # moi anh trong Downloads, khong IMU
    python3 tools/photo_test.py --checkpoint ~/Downloads/best_joint_validation.pt --output ~/Downloads/ket_qua_p46 \\
        ~/Downloads/xe.jpg ~/Downloads/images.jpeg --imu 0,0,9.81,0,0,0 ngau_nhien [--size full] [--device cuda]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    # Python he thong khong co torch (README: dung venv ~/venv-torch): chay lai bang venv do.
    # So theo sys.prefix: python cua venv la symlink toi chinh /usr/bin/python3.
    VENV = Path.home() / "venv-torch/bin/python"
    if VENV.is_file() and Path(sys.prefix).resolve() != VENV.parent.parent.resolve():
        print(f"{sys.executable} khong co torch: chay lai bang {VENV}", flush=True)
        os.execv(str(VENV), [str(VENV), *sys.argv])
    raise
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from qjepa.cli import _read_imu_csv, _system_from_phase2  # noqa: E402
from qjepa.corruptions.rng import derive_seed  # noqa: E402
from qjepa.data.dataset import load_rgb  # noqa: E402
from qjepa.execution import RestorationForward  # noqa: E402


def full_frame(path: Path, short_side: int) -> np.ndarray:
    """Ca khung [H,W,3] float: canh ngan thu ve ``short_side`` (khong phong to), moi canh cat o giua ve boi so cua 16."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        scale = short_side / min(image.size)
        if scale < 1:
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
        width, height = image.width // 16 * 16, image.height // 16 * 16
        if min(width, height) < 16:
            raise ValueError(f"{path}: anh qua nho")
        left, top = (image.width - width) // 2, (image.height - height) // 2
        return np.asarray(image.crop((left, top, left + width, top + height)), dtype=np.float32) / 255.0


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".avif", ".webp", ".bmp", ".tif", ".tiff"}
CHECKPOINT_NAME = "best_joint_validation.pt"


def collect_images(paths: list[str]) -> list[Path]:
    """Anh tu cac duong dan: file anh giu nguyen, thu muc -> moi anh trong do (khong vao thu muc con), theo ten."""
    images = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            images += sorted(child for child in path.iterdir()
                             if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file():
            images.append(path)
        else:
            raise FileNotFoundError(f"khong thay {path}")
    if not images:
        raise FileNotFoundError(f"khong co anh nao trong {paths} (duoi {sorted(IMAGE_SUFFIXES)})")
    return images


def find_checkpoint(paths: list[str]) -> Path:
    """best_joint_validation.pt trong cac thu muc da dua (hoac thu muc cua anh), roi ~/Downloads."""
    folders = [Path(raw).expanduser() if Path(raw).expanduser().is_dir() else Path(raw).expanduser().parent
               for raw in paths] + [Path.home() / "Downloads"]
    for folder in folders:
        if (folder / CHECKPOINT_NAME).is_file():
            return folder / CHECKPOINT_NAME
    raise FileNotFoundError(f"khong thay {CHECKPOINT_NAME} trong {[str(f) for f in folders]}: them --checkpoint")


IMU_CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")
SIZES = {"256": ("256",), "full": ("full",), "ca_hai": ("256", "full")}


def imu_window(spec: str, image: Path, length: int, dt: float, noise: tuple[float, float]):
    """(cua so [L,6] float32, timestamp [L] float64, mo ta) tu mot gia tri --imu."""
    times = np.arange(length, dtype=np.float64) * dt
    if spec == "khong":
        return np.zeros((length, 6), np.float32), times, "không IMU (cửa sổ 0)"
    if spec == "ngau_nhien":
        rng = np.random.default_rng(derive_seed(0, "photo_test_imu", image.name))
        window = np.zeros((length, 6), np.float32)
        window[:, 2] = 9.81
        window[:, :3] += rng.normal(0, noise[0], (length, 3))
        window[:, 3:] += rng.normal(0, noise[1], (length, 3))
        return window.astype(np.float32), times, f"ngẫu nhiên (σ {noise[0]} m/s², {noise[1]} rad/s)"
    if Path(spec).expanduser().is_file():
        window, stamps = _read_imu_csv(str(Path(spec).expanduser()), length)
        return window, (stamps if stamps is not None else times), f"file {Path(spec).name}"
    try:
        values = [float(value) for value in spec.split(",")]
    except ValueError:
        values = []
    if len(values) != 6:
        raise ValueError(f"--imu {spec!r}: can 'khong', 'ngau_nhien', mot file CSV co, hoac 6 so ax,ay,az,gx,gy,gz")
    return np.tile(np.asarray(values, np.float32), (length, 1)), times, f"không đổi {spec}"


def imu_plot(path: Path, title: str, times: np.ndarray, given: np.ndarray, restored: np.ndarray) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 5.5), sharex=True)
    for k, axis in enumerate(axes.flat):
        axis.plot(times, given[:, k], label="vào", lw=1)
        axis.plot(times, restored[:, k], label="ra", lw=1)
        axis.set_title(f"{IMU_CHANNELS[k]} ({'m/s²' if k < 3 else 'rad/s'})", fontsize=9)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=90)
    plt.close(figure)


def panel(path: Path, title: str, frame: np.ndarray, deflared: np.ndarray | None, restored: np.ndarray) -> None:
    change = np.abs(restored - frame).mean(-1)
    pictures = [(frame, "vào"), (deflared if deflared is not None else frame, "sau lớp lóe"), (restored, "ra"),
                (change, f"|ra − vào| (TB {change.mean():.3f})")]
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.2 * frame.shape[0] / frame.shape[1] + 0.7))
    for axis, (picture, name) in zip(axes, pictures):
        if picture.ndim == 3:
            axis.imshow(np.clip(picture, 0, 1))
        else:
            axis.imshow(picture, cmap="magma", vmin=0, vmax=max(0.05, float(picture.max())))
        axis.set_title(name, fontsize=10)
        axis.axis("off")
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=90)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="anh hoac thu muc anh (jpg, png, avif, ...)")
    parser.add_argument("--checkpoint", help=f"checkpoint phase 2 (mac dinh: tim {CHECKPOINT_NAME})")
    parser.add_argument("--output", help="thu muc ket qua (mac dinh: <thu muc dau tien>/ket_qua_test)")
    parser.add_argument("--imu", nargs="+", default=["khong"],
                        help="khong | <file.csv> | ax,ay,az,gx,gy,gz | ngau_nhien; 1 gia tri cho moi anh hoac 1 cho tung anh")
    parser.add_argument("--imu-dt", type=float, default=0.01, help="buoc thoi gian IMU khi file khong co timestamp (s)")
    parser.add_argument("--noise-accel", type=float, default=0.5, help="ngau_nhien: sigma gia toc (m/s^2)")
    parser.add_argument("--noise-gyro", type=float, default=0.1, help="ngau_nhien: sigma con quay (rad/s)")
    parser.add_argument("--size", default="ca_hai", choices=tuple(SIZES))
    parser.add_argument("--short-side", type=int, default=384, help="canh ngan cua ban ca khung")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    images = collect_images(args.paths)
    if len(args.imu) not in (1, len(images)):
        parser.error(f"--imu: 1 gia tri hoac {len(images)} (bang so anh), dang co {len(args.imu)}")
    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else find_checkpoint(args.paths)
    first = Path(args.paths[0]).expanduser()
    output = Path(args.output).expanduser() if args.output else (first if first.is_dir() else first.parent) / "ket_qua_test"
    device = torch.device(args.device)
    system, config = _system_from_phase2(str(checkpoint), device)
    forward = RestorationForward(system)
    forward.amp = device.type == "cuda" and config["phase2"]["precision"] == "amp_fp16"    # fp16 nhu luc train
    length = int(config["data"]["imu_window"])
    size = tuple(int(side) for side in config["data"]["image_size"])
    output.mkdir(parents=True, exist_ok=True)
    print(f"{len(images)} ảnh | checkpoint {checkpoint} | thiết bị {device} | {'fp16' if forward.amp else 'fp32'}"
          f" | cửa sổ IMU {length} mẫu")
    for number, path in enumerate(images):
        spec = args.imu[number if len(args.imu) > 1 else 0]
        given, stamps, described = imu_window(spec, path, length, args.imu_dt, (args.noise_accel, args.noise_gyro))
        imu = torch.from_numpy(given.T.copy())[None].to(device)
        times = torch.from_numpy(stamps)[None].to(device)
        image_time = torch.tensor([0.5 * (stamps[0] + stamps[-1])], dtype=torch.float64, device=device)
        print(f"\n{path.name} | IMU: {described}")
        print(f"  {'cỡ':>5s} {'khung':>9s} {'sáng vào→ra':>14s} {'|ra−vào|':>9s} {'lớp lóe đổi':>12s} {'cổng tone':>9s}")
        frames = {"256": lambda: load_rgb(path, size), "full": lambda: full_frame(path, args.short_side)}
        for name in SIZES[args.size]:
            frame = frames[name]()
            x = torch.from_numpy(frame.transpose(2, 0, 1).copy())[None].to(device)
            with torch.no_grad():
                out = forward(x, imu, image_time, times)
            as_image = lambda key: (out[key][0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() if key in out else None)
            restored, deflared = as_image("image"), as_image("image_deflared")
            gate = float(torch.sigmoid(out["image_tone_logit"]).item()) if "image_tone_logit" in out else float("nan")
            stem = f"{path.stem[:24]}_{name}"
            Image.fromarray(np.uint8(np.round(restored * 255))).save(output / f"{stem}_ra.png")
            Image.fromarray(np.uint8(np.round(frame * 255))).save(output / f"{stem}_vao.png")
            flare = float(np.abs(deflared - frame).mean()) if deflared is not None else float("nan")
            label = f"{frame.shape[1]}x{frame.shape[0]}"
            panel(output / f"{stem}_so_sanh.png", f"{path.name} | {name} {label} | cổng tone {gate:.2f}",
                  frame, deflared, restored)
            print(f"  {name:>5s} {label:>9s} {frame.mean():6.3f} → {restored.mean():.3f}"
                  f" {np.abs(restored - frame).mean():9.4f} {flare:12.4f} {gate:9.2f}", flush=True)
        # IMU khoi phuc (cua lan chay cuoi; nhanh IMU khong phu thuoc co anh nhieu).
        imu_out = out["imu_physical"][0].transpose(0, 1).cpu().numpy()
        np.savetxt(output / f"{path.stem[:24]}_imu.csv", np.column_stack((stamps, given, imu_out)), delimiter=",",
                   header="timestamp," + ",".join(f"{c}_vao" for c in IMU_CHANNELS) + ","
                          + ",".join(f"{c}_ra" for c in IMU_CHANNELS), comments="")
        imu_plot(output / f"{path.stem[:24]}_imu.png", f"{path.name} | IMU: {described}", stamps, given, imu_out)
        change = np.abs(imu_out - given)
        print(f"  IMU ra − vào: gia tốc TB {change[:, :3].mean():.3f} m/s², con quay TB {change[:, 3:].mean():.4f} rad/s")
    if device.type == "cuda":
        print(f"GPU tối đa {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    print(f"Đã lưu {output}: <ảnh>_<256|full>_so_sanh.png (vào | sau lớp lóe | ra | |ra − vào|), _vao.png, _ra.png,"
          f" <ảnh>_imu.csv / _imu.png (IMU vào và ra)")


if __name__ == "__main__":
    main()
