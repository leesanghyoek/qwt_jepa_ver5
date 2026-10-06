"""Show fresh paired camera/IMU restorations from an existing checkpoint.

This is a qualitative preview. The fixed validation audit remains unchanged so
its metrics can be compared across checkpoints and runs.

``--glare-config configs/kaggle_glare.yaml`` tests any checkpoint -- one trained
before the glare existed too -- on frames with lamps and glare: the light_* keys of
that config go into the checkpoint's corruption for this preview only, and the
panels favour frames that have bright areas (lamps, windows) for the glare to come
from. A model that never saw glare in training is being tested out of distribution.
"""

from __future__ import annotations

import argparse
import copy
import json
import secrets
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qjepa.cli import _dataset, _system_from_phase2
from qjepa.config import FOG_KEYS, ILLUM_KEYS, LIGHT_KEYS, load_config
from qjepa.data import read_manifest
from qjepa.data.dataset import load_rgb
from qjepa.evaluation.metrics import image_metrics, imu_metrics


CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")


def choose_indices(
    sample_ids: list[str], eligible: list[int], count: int,
    previous_ids: set[str], rng: np.random.Generator,
) -> list[int]:
    """Draw without replacement and never repeat an ID from the previous run."""
    if count < 1 or not eligible:
        raise ValueError("Need a positive count and at least one eligible sample")
    fresh = [index for index in eligible if sample_ids[index] not in previous_ids]
    if not fresh:
        raise ValueError("No new paired samples remain; use a larger split or fewer panels")
    return rng.choice(fresh, size=min(count, len(fresh)), replace=False).tolist()


def _eligible_indices(dataset, image_mode: str) -> list[int]:
    """Frames where some optical degradation actually fired, so a panel shows one."""
    if image_mode not in ("blur_only", "blur_low_light"):
        return list(range(len(dataset)))
    eligible = []
    for index, sample in enumerate(dataset.samples):
        parameters = dataset.image_corruptor._parameters(
            sample.split, dataset.realization, sample.trajectory_key,
            sample.image_time, image_mode,
        )
        if parameters["defocus"] or parameters["motion"] or parameters["downsample"]:
            eligible.append(index)
    return eligible


def with_glare(config: dict, glare_config: str | Path, probability: float | None = 1.0) -> dict:
    """The checkpoint's recipe plus the light_* (and illum_*, uneven light) keys of ``glare_config`` (a copy)."""
    image = load_config(glare_config)["corruption"]["image"]
    light = {key: value for key, value in image.items() if key in LIGHT_KEYS or key in ILLUM_KEYS or key in FOG_KEYS}
    if not any(key in LIGHT_KEYS for key in light):
        raise ValueError(f"{glare_config} has no corruption.image.light_* keys")
    config = copy.deepcopy(config)
    config["corruption"]["image"].update(light)
    if probability is not None:
        if not 0 < probability <= 1:
            raise ValueError("glare probability must be in (0, 1]")
        config["corruption"]["image"]["light_probability"] = float(probability)
        for switch in ("illum_probability", "fog_probability"):
            if float(image.get(switch, 0.0)) > 0:
                config["corruption"]["image"][switch] = float(probability)
    return config


def _bright_indices(dataset, eligible: list[int], rng: np.random.Generator, want: int,
                    scan: int = 300, fraction: float = 0.002) -> list[int]:
    """Up to ``want`` eligible frames with bright areas (>= ``fraction`` of pixels near white)."""
    found = []
    order = rng.permutation(len(eligible))[:scan]
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    for position in order:
        index = eligible[int(position)]
        clean = load_rgb(dataset.samples[index].image_path, dataset.image_size)
        if float(((clean @ weights) > 0.9).mean()) >= fraction:
            found.append(index)
            if len(found) >= want:
                break
    return found


def _plot_pair(path: Path, sample: dict, restored, image_input: dict,
               image_output: dict, imu_input: dict, imu_output: dict,
               realization: int) -> None:
    fig = plt.figure(figsize=(16, 10), layout="constrained")
    grid = fig.add_gridspec(3, 3, height_ratios=(2.3, 1, 1))
    images = (
        ("Ảnh sạch", sample["image_clean"]),
        ("Ảnh hư hại", sample["image_noisy"]),
        ("Ảnh khôi phục", restored.image[0]),
    )
    for column, (title, tensor) in enumerate(images):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(tensor.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy())
        ax.set_title(title)
        ax.axis("off")

    times = sample["imu_times"].numpy()
    times = times - times[0]
    traces = (
        ("Sạch", sample["imu_clean_phys"].numpy(), "#333333"),
        ("Hư hại", sample["imu_noisy_phys"].numpy(), "#e87531"),
        ("Khôi phục", restored.imu_physical[0].detach().cpu().T.numpy(), "#2476c4"),
    )
    for channel, label in enumerate(CHANNELS):
        ax = fig.add_subplot(grid[1 + channel // 3, channel % 3])
        for name, values, colour in traces:
            ax.plot(times, values[:, channel], label=name, color=colour, linewidth=1.1)
        ax.set_title(f"IMU {label}")
        ax.set_xlabel("Giây")
        ax.set_ylabel("m/s²" if channel < 3 else "rad/s")
        ax.grid(alpha=0.2)
        if channel == 0:
            ax.legend(loc="best", fontsize=8)

    image_flags = sample["corruption"]["image"]
    blur = "+".join(name for name in ("defocus", "motion", "downsample") if image_flags[name]) or "none"
    fig.suptitle(
        f"{sample['sample_id']} | realization={realization} | blur={blur}\n"
        f"Ảnh PSNR {image_input['image_psnr_db']:.2f} → {image_output['image_psnr_db']:.2f} dB"
        f"  |  IMU accel RMSE {imu_input['accel_rmse']:.3g} → {imu_output['accel_rmse']:.3g}"
        f"  |  gyro {imu_input['gyro_rmse']:.3g} → {imu_output['gyro_rmse']:.3g}",
        fontsize=11,
    )
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def preview(
    checkpoint: str | Path, manifest_path: str | Path, *, output: str | Path,
    state: str | Path, count: int = 4, split: str = "valid",
    image_mode: str = "blur_only", imu_mode: str = "full",
    device: str = "cuda", seed: int | None = None, light_scale: float = 1.0,
    glare_config: str | Path | None = None, glare_probability: float = 1.0,
) -> dict:
    if count < 1:
        raise ValueError("count must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    state = Path(state)
    manifest = read_manifest(manifest_path)
    system, config = _system_from_phase2(str(checkpoint), torch.device(device))
    if light_scale != 1.0:
        # Brightening only the preview, never the checkpoint's own recipe: the
        # panels are for looking at, and metrics stay comparable only while the
        # corruption matches what the run was measured on.
        if light_scale <= 0:
            raise ValueError("light_scale must be positive")
        low, high = config["corruption"]["image"]["exposure_gain"]
        config["corruption"]["image"]["exposure_gain"] = [
            min(1.0, low * light_scale), min(1.0, high * light_scale)
        ]
        print(f"[preview] exposure_gain {low:.3f}..{high:.3f} -> "
              f"{config['corruption']['image']['exposure_gain'][0]:.3f}.."
              f"{config['corruption']['image']['exposure_gain'][1]:.3f} "
              f"(chỉ ảnh hưởng panel này, không đổi checkpoint)")
    if glare_config is not None:
        config = with_glare(config, glare_config, glare_probability)
        if image_mode == "blur_only":
            print("[preview] blur_only không có lóe sáng (chỉ full / blur_low_light / low_light_only)")
        print(f"[preview] thêm đèn và lóe sáng từ {glare_config} vào nhiễu, {glare_probability:.0%} ảnh "
              f"(chỉ panel này, không đổi checkpoint)")
    previous = json.loads(state.read_text()) if state.is_file() else {}
    identity = (manifest["meta"]["manifest_hash"], split, image_mode, imu_mode, bool(glare_config))
    previous_ids = set(previous.get("sample_ids", [])) if tuple(previous.get("identity", ())) == identity else set()
    explicit_seed = seed is not None
    seed = int(seed) if explicit_seed else secrets.randbits(63)
    rng = np.random.default_rng(seed)
    realization = int(rng.integers(1, 2**31))
    if not explicit_seed and realization == previous.get("realization"):
        realization = 1 + realization % (2**31 - 1)
    dataset = _dataset(config, manifest, split, fixed_realization=True,
                       image_mode=image_mode, imu_mode=imu_mode)
    dataset.set_realization(realization)
    eligible = _eligible_indices(dataset, image_mode)
    if glare_config is not None:
        bright = _bright_indices(dataset, eligible, rng, want=4 * count)
        print(f"[preview] {len(bright)} frame có vùng sáng (đèn, cửa sổ) để lóe")
        eligible = bright if len(bright) >= count else eligible
    indices = choose_indices(
        [sample.sample_id for sample in dataset.samples], eligible, count,
        set() if explicit_seed else previous_ids, rng,
    )
    output.mkdir(parents=True)
    items = []
    for position, index in enumerate(indices):
        sample = dataset[index]
        with torch.no_grad():
            restored = system(
                sample["image_noisy"].unsqueeze(0).to(device),
                sample["imu_noisy_phys"].T.unsqueeze(0).to(device),
                sample["image_time"].unsqueeze(0).to(device),
                sample["imu_times"].unsqueeze(0).to(device),
            )
            image_clean = sample["image_clean"].unsqueeze(0).to(device)
            imu_clean = sample["imu_clean_phys"].T.unsqueeze(0).to(device)
            imu_noisy = sample["imu_noisy_phys"].T.unsqueeze(0).to(device)
            image_input = image_metrics(sample["image_noisy"].unsqueeze(0).to(device), image_clean)
            image_output = image_metrics(restored.image, image_clean)
            imu_input = imu_metrics(imu_noisy, imu_clean)
            imu_output = imu_metrics(restored.imu_physical, imu_clean)
        panel_name = f"pair_{position:02d}.png"
        _plot_pair(output / panel_name, sample, restored, image_input,
                   image_output, imu_input, imu_output, realization)
        items.append({
            "sample_id": sample["sample_id"], "index": index, "panel": panel_name,
            "image_corruption": sample["corruption"]["image"],
            "imu_corruption": sample["corruption"]["imu"],
            "image_input_metrics": image_input, "image_restored_metrics": image_output,
            "imu_input_metrics": imu_input, "imu_restored_metrics": imu_output,
        })
        print(f"{position + 1}/{len(indices)} {sample['sample_id']} → {output / panel_name}", flush=True)

    report = {
        "checkpoint": str(checkpoint), "manifest_hash": identity[0],
        "split": split, "image_mode": image_mode, "imu_mode": imu_mode,
        "light_scale": light_scale,
        "glare_config": None if glare_config is None else str(glare_config),
        "glare_probability": glare_probability if glare_config is not None else None,
        "seed": seed, "realization": realization, "items": items,
    }
    (output / "preview.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"identity": identity, "sample_ids":
                                  [item["sample_id"] for item in items],
                                  "realization": realization}, indent=2), encoding="utf-8")
    print(f"Đã lưu {len(items)} cặp ảnh + IMU | seed={seed} | realization={realization}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--image-mode",
                        choices=("full", "blur_only", "blur_low_light"),
                        default="blur_only")
    parser.add_argument("--light-scale", type=float, default=1.0,
                        help="nhân vào exposure_gain: >1 sáng hơn, 1.0 giữ nguyên recipe train")
    parser.add_argument("--imu-mode", choices=("full", "clean"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--glare-config",
                        help="config có corruption.image.light_* (vd. configs/kaggle_glare.yaml): thêm đèn và "
                             "lóe sáng vào nhiễu của panel, kể cả với checkpoint train trước khi có lóe")
    parser.add_argument("--glare-probability", type=float, default=1.0,
                        help="tỉ lệ ảnh có lóe khi dùng --glare-config (1.0 = mọi ảnh)")
    args = parser.parse_args()
    preview(args.checkpoint, args.manifest, output=args.output, state=args.state,
            count=args.count, split=args.split, image_mode=args.image_mode,
            imu_mode=args.imu_mode, device=args.device, seed=args.seed,
            light_scale=args.light_scale, glare_config=args.glare_config,
            glare_probability=args.glare_probability)


if __name__ == "__main__":
    main()
