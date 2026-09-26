"""Show that the image blur really is the IMU's motion, frame by frame.

No model, no checkpoint, no manifest -- only the raw trajectories and the
corruption pipeline. The point is to make the coupling visible: each figure puts
the clean pair and the corrupted pair side by side, and then shows the gyro
segment that produced that exact blur kernel.

Read the third column first. The blur kernel is the path the principal point
traced while the shutter was open, and that path is the integral of the gyro
trace drawn beneath it. If the kernel is long and diagonal, the gyro says the
camera was yawing and pitching together during those milliseconds.

    python3 tools/imu_blur_preview.py --data-root <root> --samples 4

Pass ``--random-blur`` to render the same frames with the legacy independent
draw. Put the two side by side and the difference is the whole argument: with
the random draw the kernel has no relationship to the trace below it.

Torch is not imported, so this runs in a plain numpy/scipy/matplotlib
environment.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib


def select_backend() -> bool:
    """Pick an on-screen backend when one exists; return whether it can show.

    Forcing Agg unconditionally means the figures only ever reach a file, which
    is useless when the point is to look at them. Forcing a GUI backend instead
    would break every headless run -- Kaggle, CI, a shell over ssh -- so the
    choice is made from the environment, and Agg remains the fallback.
    """
    import matplotlib

    if os.environ.get("MPLBACKEND"):
        return matplotlib.get_backend().lower() not in {"agg", "pdf", "ps", "svg", "template"}
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        matplotlib.use("Agg")
        return False
    import importlib

    for candidate in ("QtAgg", "TkAgg", "GTK3Agg"):
        try:
            # use() alone does not load the backend, so a missing Qt only surfaced
            # at the first figure. Importing its module fails here instead.
            importlib.import_module(f"matplotlib.backends.backend_{candidate.lower()}")
            matplotlib.use(candidate)
            return True
        except Exception:
            continue
    matplotlib.use("Agg")
    return False


CAN_SHOW = select_backend()

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qjepa.corruptions.image import LowLightImageCorruptionConfig, LowLightImageCorruptor
from qjepa.corruptions.imu import ImuCorruptionConfig, TrajectoryImuCorruptor
from qjepa.corruptions.motion import exposure_path
from qjepa.corruptions.rng import generator

# Categorical slots 1-3, validated for the light surface (CVD worst pair dE 9.2).
# Identity is the axis, so clean and corrupted share a hue and differ in weight;
# repainting the corrupted trace would say the axis had changed.
AXIS_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8985"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"


def load_rgb(path: Path, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Match qjepa.data.dataset.load_rgb without importing torch."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        target_height, target_width = size
        if image.size != (target_width, target_height):
            scale = max(target_width / image.width, target_height / image.height)
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.Resampling.LANCZOS,
            )
            left = (image.width - target_width) // 2
            top = (image.height - target_height) // 2
            image = image.crop((left, top, left + target_width, top + target_height))
        return np.asarray(image, dtype=np.float32) / 255.0


def _array(directory: Path, name: str) -> np.ndarray:
    npy, txt = directory / f"{name}.npy", directory / f"{name}.txt"
    if npy.is_file():
        return np.load(npy)
    if txt.is_file():
        return np.loadtxt(txt, dtype=np.float64)
    raise FileNotFoundError(f"Missing {name}.npy/.txt in {directory}")


def find_trajectories(root: Path, limit: int | None = None) -> list[Path]:
    """Any directory holding both imu/ and image_lcam_front/, at any depth.

    Deliberately layout agnostic: this dataset nests an extra train/valid/test
    level that the raw TartanAir tree does not have, and its trajectory
    directories are symlinks into the original tree. ``Path.glob("**/...")``
    refuses to descend into symlinked directories, so it finds nothing here --
    ``os.walk(followlinks=True)`` is what actually reaches them. Once a
    trajectory is recognised its subtree is pruned, which also bounds the walk
    if a link ever points back upward.
    """
    found: list[Path] = []
    for directory, children, _ in os.walk(root, followlinks=True):
        current = Path(directory)
        if "imu" in children and "image_lcam_front" in children:
            if (current / "imu" / "gyro.npy").is_file() or (current / "imu" / "gyro.txt").is_file():
                found.append(current)
                children.clear()
                if limit is not None and len(found) >= limit:
                    return found
                continue
        children.sort()
    return sorted(found)


DATA_ROOT_ENV = "QJEPA_DATA_ROOT"


def data_root_candidates() -> list[Path]:
    """Where a TartanAir tree usually sits, most specific first."""
    home = Path.home()
    candidates = [
        home / "Datasets/tartanair-v2-jepa",
        home / "Datasets/tartanair-v2",
        Path("/kaggle/input/tartanairkhoi/tartanair-v2"),
    ]
    for parent in (home / "Datasets", Path("/kaggle/input")):
        if parent.is_dir():
            candidates.extend(sorted(child for child in parent.iterdir() if child.is_dir()))
    return candidates


def resolve_data_root(explicit: Path | None) -> Path:
    """Fall back to discovery so the tool is runnable with no arguments.

    Running a file straight from an editor passes no arguments, and a required
    flag turns that into an error message instead of a picture. The repository
    cannot carry one machine's path as a default, so look for the tree instead.
    """
    if explicit is not None:
        if not find_trajectories(explicit, limit=1):
            raise SystemExit(f"Không thấy trajectory nào dưới {explicit}")
        return explicit
    from os import environ

    if environ.get(DATA_ROOT_ENV):
        root = Path(environ[DATA_ROOT_ENV])
        if find_trajectories(root, limit=1):
            print(f"Dataset từ ${DATA_ROOT_ENV}: {root}")
            return root
        raise SystemExit(f"${DATA_ROOT_ENV}={root} nhưng không có trajectory nào ở đó")
    for candidate in data_root_candidates():
        if candidate.is_dir() and find_trajectories(candidate, limit=1):
            print(f"Tự dò ra dataset: {candidate}"
                  f"   (đặt --data-root hoặc ${DATA_ROOT_ENV} để chỉ định khác)")
            return candidate
    raise SystemExit(
        "Không tự dò được dataset. Truyền --data-root <đường dẫn>, "
        f"hoặc đặt biến môi trường {DATA_ROOT_ENV}."
    )


def choose_frame(corruptor, imu, imu_time, cam_time, usable, name,
                 window_length, minimum_span, rng):
    """Prefer a frame the camera was actually moving through.

    Picking uniformly gives mostly near-still frames, where the blur is a 3x3
    kernel and the figure demonstrates nothing. The span is computed from the
    gyro exactly as the corruptor will, including that frame's own exposure, so
    the selection cannot disagree with what is later rendered.
    """
    spans = np.zeros(usable)
    for frame in range(1, usable - 1):
        start, end = pick_window(imu_time, float(cam_time[frame]), window_length)
        parameters = corruptor._parameters("preview", 0, name, float(cam_time[frame]), "blur_only")
        u, v, _ = exposure_path(
            imu[start:end, 3:6], imu_time[start:end], float(cam_time[frame]),
            float(parameters["exposure_seconds"]),
            focal_length_px=corruptor.config.focal_length_px,
            angular_gain=corruptor.config.angular_gain,
            samples=corruptor.config.motion_path_samples,
        )
        spans[frame] = float(np.hypot(u.max() - u.min(), v.max() - v.min()))
    qualifying = np.flatnonzero(spans >= minimum_span)
    if len(qualifying):
        frame = int(rng.choice(qualifying))
    else:
        frame = int(np.argmax(spans))
    return frame, float(spans[frame]), len(qualifying)


def load_trajectory(path: Path):
    imu = path / "imu"
    accel = np.asarray(_array(imu, "acc"), dtype=np.float64)
    gyro = np.asarray(_array(imu, "gyro"), dtype=np.float64)
    imu_time = np.asarray(_array(imu, "imu_time"), dtype=np.float64).reshape(-1)
    cam_time = np.asarray(_array(imu, "cam_time"), dtype=np.float64).reshape(-1)
    images = sorted((path / "image_lcam_front").glob("*.png"))
    return np.concatenate((accel, gyro), axis=1), imu_time, cam_time, images


def pick_window(imu_time: np.ndarray, centre: float, length: int) -> tuple[int, int]:
    """A window of `length` rows surrounding the capture instant."""
    nearest = int(np.argmin(np.abs(imu_time - centre)))
    start = nearest - length // 2
    start = max(0, min(start, len(imu_time) - length))
    return start, start + length


def _style(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=INK_MUTED, labelsize=8, length=3)


def draw_imu(axis, time, values, labels, *, ghost=None, title, ylabel, limits=None,
             capture=None):
    """Three axes as three hues; the clean ghost sits behind when given.

    Direct labels sit at the right edge in ink, never in the series colour -- the
    aqua slot warns on contrast against this surface, so identity must not rest
    on the line colour alone.
    """
    _style(axis)
    if ghost is not None:
        for index in range(3):
            axis.plot(time, ghost[:, index], color=AXIS_COLORS[index],
                      linewidth=2.6, alpha=0.22, zorder=2, solid_capstyle="round")
    for index, label in enumerate(labels):
        axis.plot(time, values[:, index], color=AXIS_COLORS[index], linewidth=1.6,
                  zorder=3, solid_capstyle="round", label=label)
        axis.annotate(label, xy=(time[-1], values[-1, index]),
                      xytext=(4, 0), textcoords="offset points",
                      color=INK_SOFT, fontsize=8, va="center")
    axis.set_title(title, color=INK, fontsize=10, loc="left", pad=6)
    axis.set_ylabel(ylabel, color=INK_SOFT, fontsize=8)
    axis.set_xlabel("giây", color=INK_SOFT, fontsize=8)
    if capture is not None:
        # The exposure is ~25 ms inside a 1.27 s window, far too thin to shade;
        # a rule at the capture instant is what actually connects this panel to
        # the kernel above it.
        axis.axvline(capture, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
                     zorder=1)
        axis.annotate("thời điểm chụp", xy=(capture, axis.get_ylim()[1]),
                      xytext=(3, -10), textcoords="offset points",
                      color=INK_MUTED, fontsize=7.5, va="top")
    if limits is not None:
        axis.set_ylim(*limits)
    axis.margins(x=0.02)


def draw_image(axis, image, title, subtitle=""):
    axis.imshow(np.clip(image, 0, 1))
    axis.set_xticks([]); axis.set_yticks([])
    for side in axis.spines.values():
        side.set_color(GRID)
    axis.set_title(title, color=INK, fontsize=11, loc="left", pad=6)
    if subtitle:
        axis.set_xlabel(subtitle, color=INK_SOFT, fontsize=8)


def draw_kernel(axis, kernel, report):
    _style(axis)
    axis.grid(False)
    # Magnitude -> one hue, light to dark. Never a rainbow.
    axis.imshow(kernel, cmap="Blues", interpolation="nearest")
    axis.set_xticks([]); axis.set_yticks([])
    if "path_span_px" in report:
        title = "Kernel blur (từ gyro)"
        caption = (
            f"quét {report['path_span_px']:.2f} px · phơi sáng "
            f"{report['exposure_seconds'] * 1000:.1f} ms\n"
            f"roll {np.degrees(report['roll_radians']):+.2f}° (không nằm trong kernel)"
        )
    else:
        # Control arm: the report has no path because no path was integrated.
        title = "Kernel blur (BỐC NGẪU NHIÊN)"
        caption = (
            f"dài {report.get('motion_length', 0)} px · góc "
            f"{np.degrees(report.get('motion_angle', 0.0)):.0f}°\n"
            "không đọc gyro — gyro bên dưới không giải thích được kernel này"
        )
    axis.set_title(title, color=INK, fontsize=10, loc="left", pad=6)
    axis.set_xlabel(caption, color=INK_SOFT, fontsize=8)


def draw_path(axis, u, v):
    _style(axis)
    axis.plot(u, v, color=AXIS_COLORS[0], linewidth=2.0, zorder=3,
              solid_capstyle="round")
    axis.scatter([u[0]], [v[0]], s=34, color=AXIS_COLORS[0], zorder=4,
                 edgecolor=SURFACE, linewidth=1.2)
    axis.scatter([u[len(u) // 2]], [v[len(v) // 2]], s=44, color=AXIS_COLORS[1],
                 zorder=5, edgecolor=SURFACE, linewidth=1.2)
    axis.annotate("mở màn trập", xy=(u[0], v[0]), xytext=(5, 5),
                  textcoords="offset points", color=INK_SOFT, fontsize=8)
    axis.annotate("thời điểm chụp", xy=(u[len(u) // 2], v[len(v) // 2]),
                  xytext=(5, -12), textcoords="offset points", color=INK_SOFT, fontsize=8)
    axis.set_title("Đường dịch chuyển trong lúc phơi sáng", color=INK,
                   fontsize=10, loc="left", pad=6)
    axis.set_xlabel("u — ngang (px), từ yaw = gyro_z", color=INK_SOFT, fontsize=8)
    axis.set_ylabel("v — dọc (px), từ pitch = gyro_y", color=INK_SOFT, fontsize=8)
    axis.set_aspect("equal", adjustable="datalim")
    axis.margins(0.22)
    axis.invert_yaxis()


def build_figure(sample, destination: Path) -> None:
    figure = plt.figure(figsize=(15.5, 8.4), facecolor=SURFACE)
    grid = figure.add_gridspec(2, 3, height_ratios=[1.0, 0.92],
                               hspace=0.34, wspace=0.24,
                               left=0.05, right=0.965, top=0.88, bottom=0.09)

    draw_image(figure.add_subplot(grid[0, 0]), sample["image_clean"],
               "CẶP SẠCH — ảnh", "tham chiếu, chỉ tồn tại lúc train")
    draw_image(figure.add_subplot(grid[0, 1]), sample["image_noisy"],
               "CẶP MỜ NHIỄU — ảnh",
               f"PSNR {sample['psnr']:.2f} dB so với ảnh sạch")

    axis_kernel = figure.add_subplot(grid[0, 2])
    draw_kernel(axis_kernel, sample["kernel"], sample["report"])

    time = sample["time"] - sample["time"][0]
    capture = float(sample["capture_time"] - sample["time"][0])
    columns = slice(3, 6) if sample["channel"] == "gyro" else slice(0, 3)
    labels = ("gx", "gy", "gz") if sample["channel"] == "gyro" else ("ax", "ay", "az")
    unit = "rad/s" if sample["channel"] == "gyro" else "m/s²"
    what = "vận tốc góc" if sample["channel"] == "gyro" else "gia tốc"
    clean_values = sample["imu_clean"][:, columns]
    noisy_values = sample["imu_noisy"][:, columns]
    limits = (min(clean_values.min(), noisy_values.min()) - 0.1 * abs(clean_values.min() or 1),
              max(clean_values.max(), noisy_values.max()) + 0.1 * abs(clean_values.max() or 1))
    axis_clean = figure.add_subplot(grid[1, 0])
    draw_imu(axis_clean, time, clean_values, labels, capture=capture,
             title=f"CẶP SẠCH — {what}", ylabel=unit, limits=limits)
    axis_noisy = figure.add_subplot(grid[1, 1])
    draw_imu(axis_noisy, time, noisy_values, labels, ghost=clean_values, capture=capture,
             title=f"CẶP MỜ NHIỄU — {what} (mờ = bản sạch)", ylabel=unit, limits=limits)
    axis_clean.legend(loc="upper left", fontsize=8, frameon=False,
                      labelcolor=INK_SOFT, ncol=3, columnspacing=1.1)

    draw_path(figure.add_subplot(grid[1, 2]), sample["u"], sample["v"])

    source = "gyro SẠCH tích phân trên thời gian phơi sáng" if sample["from_imu"] \
        else "BỐC NGẪU NHIÊN — không liên quan gì tới IMU (nhánh control)"
    figure.suptitle(
        f"{sample['name']}  ·  frame {sample['frame']}\n"
        f"blur = {source}",
        color=INK, fontsize=12.5, x=0.05, ha="left", y=0.975,
    )
    figure.savefig(destination, dpi=125, facecolor=SURFACE)
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="mặc định: tự dò trong ~/Datasets và /kaggle/input")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parent.parent / "configs/pipeline_v3.yaml")
    # Relative to the repository, not the shell's working directory: run from an
    # editor or another folder and a relative default silently scatters output
    # wherever that happened to be.
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent.parent / "outputs/imu_blur_preview")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--imu-window", type=int, default=128)
    parser.add_argument("--mode", default="blur_only",
                        choices=("full", "blur_only", "low_light_only", "sensor_noise_only"))
    parser.add_argument("--random-blur", action="store_true",
                        help="control arm: legacy independent draw, ignores the IMU")
    parser.add_argument("--imu-channel", default="gyro", choices=("gyro", "accel"),
                        help="gyro is what produces the blur; accel is the familiar trace")
    parser.add_argument("--min-span-px", type=float, default=2.0,
                        help="prefer frames whose gyro sweeps at least this far")
    parser.add_argument("--summary", type=int, default=0, metavar="N",
                        help="also plot gyro-predicted vs applied blur over N frames, "
                             "for both arms — one frame proves nothing, this does")
    parser.add_argument("--no-show", action="store_true",
                        help="chỉ ghi file, không mở cửa sổ")
    parser.add_argument("--keep-optical-blur", action="store_true",
                        help="also apply defocus and downsampling (hides what the IMU did)")
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    image_config = LowLightImageCorruptionConfig(**raw["corruption"]["image"])
    if args.random_blur:
        image_config = replace(image_config, motion_from_imu=False)
    # A clean frame would show nothing, and this tool exists to show something.
    image_config = replace(image_config, clean_probability=0.0)
    if not args.keep_optical_blur:
        # Defocus and downsampling blur too, and they owe nothing to the IMU. Leave
        # them on and the reader cannot tell which softening the gyro explains --
        # which is the one claim this tool exists to support.
        image_config = replace(image_config, defocus_probability=0.0,
                               downsample_probability=0.0)
    image_corruptor = LowLightImageCorruptor(image_config, master_seed=raw["data"]["corruption_seed"])
    imu_corruptor = TrajectoryImuCorruptor(
        ImuCorruptionConfig(**raw["corruption"]["imu"]), master_seed=raw["data"]["corruption_seed"]
    )

    args.data_root = resolve_data_root(args.data_root)
    trajectories = find_trajectories(args.data_root)
    print(f"{len(trajectories)} trajectory dưới {args.data_root}")

    args.out.mkdir(parents=True, exist_ok=True)
    rng = generator(args.seed, "imu_blur_preview")
    chosen = rng.choice(len(trajectories), size=min(args.samples, len(trajectories)),
                        replace=len(trajectories) < args.samples)

    figures = []
    print(f"\n{'frame':<34} {'quét px':>8} {'phơi sáng ms':>13} {'|ω| rad/s':>10} {'PSNR dB':>9}")
    for number, index in enumerate(chosen):
        path = trajectories[int(index)]
        imu, imu_time, cam_time, images = load_trajectory(path)
        name = "/".join(path.parts[-3:])
        usable = min(len(cam_time), len(images))
        frame, _, available = choose_frame(
            image_corruptor, imu, imu_time, cam_time, usable, name,
            args.imu_window, args.min_span_px, rng,
        )
        start, end = pick_window(imu_time, float(cam_time[frame]), args.imu_window)

        image_clean = load_rgb(images[frame])
        window = imu[start:end]
        window_time = imu_time[start:end]

        image_noisy, report = image_corruptor(
            image_clean, split="preview", realization=0, trajectory=name,
            timestamp=float(cam_time[frame]), frame_index=frame, mode=args.mode,
            gyro=window[:, 3:6], imu_times=window_time,
        )
        imu_noisy, _ = imu_corruptor.window(
            imu, imu_time, start, end, split="preview", realization=0,
            trajectory=name, mode="full",
        )
        kernel = image_corruptor_kernel(image_corruptor, report, window, window_time,
                                        float(cam_time[frame]))
        u, v, _ = exposure_path(
            window[:, 3:6], window_time, float(cam_time[frame]),
            float(report["exposure_seconds"]),
            focal_length_px=image_config.focal_length_px,
            angular_gain=image_config.angular_gain,
            samples=image_config.motion_path_samples,
        )
        error = np.mean((image_clean.astype(np.float64) - image_noisy.astype(np.float64)) ** 2)
        psnr = 10.0 * np.log10(1.0 / max(error, 1e-12))
        peak = float(np.abs(window[:, 3:6]).max())

        destination = args.out / f"{number:02d}_{name.replace('/', '_')}_f{frame:06d}.png"
        figures.append(build_figure({
            "name": name, "frame": frame,
            "image_clean": image_clean, "image_noisy": image_noisy,
            "imu_clean": window.astype(np.float64), "imu_noisy": imu_noisy.astype(np.float64),
            "time": window_time, "kernel": kernel, "report": report,
            "u": u, "v": v, "psnr": psnr, "from_imu": not args.random_blur,
            "channel": args.imu_channel, "capture_time": float(cam_time[frame]),
        }, destination))
        print(f"{name + ' f' + str(frame):<34} {report.get('path_span_px', float('nan')):>8.2f} "
              f"{report['exposure_seconds'] * 1000:>13.1f} {peak:>10.2f} {psnr:>9.2f}"
              f"   ({available} / {usable} frame đạt ngưỡng)")

    if args.summary:
        summarise(args, image_config, raw, trajectories, rng)
    print(f"\nĐã lưu {len(chosen)} hình vào {args.out}")
    show_or_close(figures, args.no_show)
    return 0


def show_or_close(figures, suppressed: bool) -> None:
    if suppressed or not CAN_SHOW:
        if not suppressed and not CAN_SHOW:
            print("Không có màn hình để hiển thị; chỉ ghi file.")
        for figure in figures:
            plt.close(figure)
        return
    print("Đang mở cửa sổ — đóng hết cửa sổ để kết thúc.")
    plt.show()


def _blur_extent(kernel: np.ndarray) -> float:
    """Length of the kernel's principal axis, weighted by mass."""
    if kernel.shape[0] < 2:
        return 0.0
    rows, columns = np.nonzero(kernel > kernel.max() * 1e-3)
    weights = kernel[rows, columns]
    points = np.stack([columns - np.average(columns, weights=weights),
                       rows - np.average(rows, weights=weights)])
    covariance = (points * weights) @ points.T / weights.sum()
    eigenvalues = np.linalg.eigvalsh(covariance)
    return float(2.0 * np.sqrt(max(eigenvalues[-1], 0.0)) * np.sqrt(3.0))


def summarise(args, image_config, raw, trajectories, rng) -> None:
    """Gyro prediction against the blur that was actually applied.

    A single frame can look convincing by accident -- a random angle lands near
    the true one often enough. What cannot happen by accident is agreement across
    hundreds of frames, so that is what this measures, for both arms at once.
    """
    from dataclasses import replace as _replace

    arms = {
        "blur từ IMU": _replace(image_config, motion_from_imu=True),
        "blur ngẫu nhiên": _replace(image_config, motion_from_imu=False),
    }
    predicted, applied = {}, {}
    picks = [(trajectories[int(i)], None) for i in
             rng.choice(len(trajectories), size=min(len(trajectories), 12), replace=False)]
    for label, config in arms.items():
        corruptor = LowLightImageCorruptor(config, master_seed=raw["data"]["corruption_seed"])
        xs, ys = [], []
        for path, _ in picks:
            imu, imu_time, cam_time, images = load_trajectory(path)
            name = "/".join(path.parts[-3:])
            usable = min(len(cam_time), len(images))
            step = max(1, usable // max(1, args.summary // len(picks)))
            for frame in range(1, usable - 1, step):
                start, end = pick_window(imu_time, float(cam_time[frame]), args.imu_window)
                parameters = corruptor._parameters("preview", 0, name,
                                                   float(cam_time[frame]), "blur_only")
                u, v, _ = exposure_path(
                    imu[start:end, 3:6], imu_time[start:end], float(cam_time[frame]),
                    float(parameters["exposure_seconds"]),
                    focal_length_px=config.focal_length_px,
                    angular_gain=config.angular_gain, samples=config.motion_path_samples,
                )
                xs.append(float(np.hypot(u.max() - u.min(), v.max() - v.min())))
                kernel = image_corruptor_kernel(
                    corruptor, parameters, imu[start:end], imu_time[start:end],
                    float(cam_time[frame]),
                )
                ys.append(_blur_extent(kernel))
        predicted[label], applied[label] = np.array(xs), np.array(ys)

    figure, axis = plt.subplots(figsize=(7.4, 6.0), facecolor=SURFACE)
    _style(axis)
    for index, label in enumerate(arms):
        x, y = predicted[label], applied[label]
        correlation = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")
        axis.scatter(x, y, s=16, color=AXIS_COLORS[index], alpha=0.55, zorder=3,
                     edgecolor=SURFACE, linewidth=0.5,
                     label=f"{label} — r = {correlation:+.3f}")
    limit = max(predicted[l].max() for l in arms) * 1.05
    axis.plot([0, limit], [0, limit], color=INK_MUTED, linewidth=1.2,
              linestyle=(0, (4, 3)), zorder=2)
    axis.annotate("y = x", xy=(limit * 0.82, limit * 0.86), color=INK_MUTED, fontsize=8)
    axis.set_xlabel("gyro dự đoán — quét trong lúc phơi sáng (px)", color=INK_SOFT, fontsize=9)
    axis.set_ylabel("blur thực sự áp lên ảnh (px)", color=INK_SOFT, fontsize=9)
    axis.set_title(f"Gyro có giải thích được blur không? · {len(predicted['blur từ IMU'])} frame",
                   color=INK, fontsize=12, loc="left", pad=10)
    axis.legend(loc="upper left", fontsize=9, frameon=False, labelcolor=INK_SOFT)
    figure.tight_layout()
    destination = args.out / "summary_gyro_vs_blur.png"
    figure.savefig(destination, dpi=130, facecolor=SURFACE)
    for label in arms:
        x, y = predicted[label], applied[label]
        print(f"  {label:<18} r = {np.corrcoef(x, y)[0, 1]:+.3f}  (n={len(x)})")
    print(f"  -> {destination}")


def image_corruptor_kernel(corruptor, report, window, window_time, centre):
    """Rebuild the kernel that produced this frame, for display only."""
    from qjepa.corruptions.motion import imu_blur_kernel

    if not corruptor.config.motion_from_imu:
        from qjepa.corruptions.image import _motion_kernel
        if not report.get("motion"):
            return np.array([[1.0]])
        return _motion_kernel(int(report["motion_length"]), float(report["motion_angle"]))
    kernel, _ = imu_blur_kernel(
        window[:, 3:6], window_time, centre, float(report["exposure_seconds"]),
        focal_length_px=corruptor.config.focal_length_px,
        angular_gain=corruptor.config.angular_gain,
        samples=corruptor.config.motion_path_samples,
        max_radius_px=corruptor.config.motion_max_radius_px,
    )
    return kernel


if __name__ == "__main__":
    raise SystemExit(main())
