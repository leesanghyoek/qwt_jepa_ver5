"""TartanAir camera/IMU discovery and strict timeline validation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_DIRECTORY = "image_lcam_front"
CAMERA = "lcam_front"
SPLIT_NAMES = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}


def _load_array(directory: Path, name: str) -> np.ndarray:
    npy = directory / f"{name}.npy"
    txt = directory / f"{name}.txt"
    if npy.is_file():
        return np.load(npy)
    if txt.is_file():
        return np.loadtxt(txt, dtype=np.float64)
    raise FileNotFoundError(f"Missing {name}.npy/.txt in {directory}")


@dataclass(frozen=True)
class Trajectory:
    path: Path
    environment: str
    difficulty: str
    trajectory_id: str
    split_hint: str | None = None

    @property
    def key(self) -> str:
        return f"{self.environment}/{self.difficulty}/{self.trajectory_id}"

    @property
    def motion_key(self) -> str:
        return f"{self.environment}/{self.trajectory_id}"

    def image_paths(self) -> list[Path]:
        directory = self.path / IMAGE_DIRECTORY
        paths = list(directory.glob(f"*_{CAMERA}.png"))
        if not paths:
            # tools/build_tartanair640.py stores the 640x640 frames as lossless WebP.
            paths = [*directory.glob("*.png"), *directory.glob("*.jpg"), *directory.glob("*.jpeg"),
                     *directory.glob("*.webp")]
        return sorted(set(paths))

    def load_imu(self) -> tuple[np.ndarray, np.ndarray]:
        directory = self.path / "imu"
        accel = np.asarray(_load_array(directory, "acc"), dtype=np.float64)
        gyro = np.asarray(_load_array(directory, "gyro"), dtype=np.float64)
        timestamps = np.asarray(_load_array(directory, "imu_time"), dtype=np.float64).reshape(-1)
        if accel.ndim != 2 or accel.shape[1] != 3 or gyro.shape != accel.shape:
            raise ValueError(f"{self.key}: acc and gyro must both be [N,3]")
        if timestamps.shape != (len(accel),):
            raise ValueError(f"{self.key}: IMU timestamp count does not match measurements")
        imu = np.concatenate((accel, gyro), axis=1)
        if not np.isfinite(imu).all() or not np.isfinite(timestamps).all():
            raise ValueError(f"{self.key}: IMU contains NaN/Inf")
        if len(timestamps) < 2 or (np.diff(timestamps) <= 0).any():
            raise ValueError(f"{self.key}: IMU timestamps are not strictly increasing")
        return imu, timestamps

    def load_camera_times(self) -> np.ndarray:
        timestamps = np.asarray(_load_array(self.path / "imu", "cam_time"), dtype=np.float64).reshape(-1)
        if not np.isfinite(timestamps).all() or len(timestamps) > 1 and (np.diff(timestamps) <= 0).any():
            raise ValueError(f"{self.key}: camera timestamps are invalid")
        return timestamps


def discover_trajectories(root: str | Path) -> list[Trajectory]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    found: list[Trajectory] = []
    visited: set[Path] = set()
    for directory, child_names, _ in os.walk(root, followlinks=True):
        path = Path(directory)
        resolved = path.resolve()
        if resolved in visited:
            child_names[:] = []
            continue
        visited.add(resolved)
        if path.name != "imu" or not ((path / "imu_time.npy").is_file() or (path / "imu_time.txt").is_file()):
            continue
        trajectory_path = path.parent
        if not (trajectory_path / IMAGE_DIRECTORY).is_dir():
            continue
        relative = trajectory_path.relative_to(root).parts
        if len(relative) < 3:
            continue
        trajectory_id, difficulty, environment = relative[-1], relative[-2], relative[-3]
        hint = next((SPLIT_NAMES[p] for p in relative if p in SPLIT_NAMES), None)
        found.append(Trajectory(trajectory_path, environment, difficulty, trajectory_id, hint))
    if not found:
        raise FileNotFoundError(
            f"No TartanAir trajectories found under {root}; expected Pxxx/imu and "
            f"Pxxx/{IMAGE_DIRECTORY}"
        )
    return sorted(found, key=lambda item: item.key)


def link_trajectories(roots: list[str | Path], target: str | Path) -> Path:
    """One dataset root over several: ``target/<env>/<difficulty>/<Pxxx>`` links to each trajectory.

    The 640x640 dataset comes as three Kaggle datasets, and Kaggle unpacks an archive
    into a folder named after it, so a shard may hold its environments a level or two
    deeper (shard 0: ``AbandonedCable/tartanair640/AbandonedCable/Data_easy/...``).
    Discovery reads the last three levels of a trajectory's path, so the depth does not
    matter; the same trajectory found twice is an error.
    """
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    found: dict[str, Path] = {}
    for root in roots:
        for trajectory in discover_trajectories(root):
            if trajectory.key in found:
                raise ValueError(f"{trajectory.key} is in both {found[trajectory.key]} and {trajectory.path}")
            found[trajectory.key] = trajectory.path
            link = target / trajectory.environment / trajectory.difficulty / trajectory.trajectory_id
            if link.is_symlink():
                if link.resolve() != trajectory.path.resolve():
                    raise ValueError(f"{link} already points at {link.resolve()}")
                continue
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(trajectory.path.resolve(), target_is_directory=True)
    return target


def find_shard_roots(base: str | Path = "/kaggle/input", depth: int = 4) -> dict[int, Path]:
    """{shard: dataset root} of the 640x640 TartanAir shards mounted under ``base``.

    A shard's root is the folder holding ``build_meta/shard<N>.json`` (written by
    ``tools/build_tartanair640.py``). Every shard of the plan must be there exactly
    once and from the same plan: the per-environment split is only stable over all
    of them. A shard whose archives Kaggle did not unpack is refused by name.
    """
    base = Path(base)
    found: dict[int, Path] = {}
    plans: dict[int, tuple[str, int]] = {}
    for level in range(depth):
        for report_path in sorted(base.glob("/".join(["*"] * level + ["build_meta", "shard*.json"]))):
            root = report_path.parent.parent
            report = json.loads(report_path.read_text())
            plan = json.loads((root / "build_meta" / "plan.json").read_text())
            shard = int(report["shard"])
            if shard in found:
                raise ValueError(f"shard {shard} is mounted twice: {found[shard]} and {root}")
            archives = [path for path in root.glob("tartanair640/*") if path.suffix in (".tar", ".zip")]
            if archives:
                raise ValueError(f"{root} holds {len(archives)} archives that Kaggle did not unpack "
                                 f"({archives[0].name}, ...); trajectories must be folders")
            found[shard] = root
            plans[shard] = (report["fingerprint"], int(plan["num_shards"]))
    if not found:
        raise FileNotFoundError(f"no build_meta/shard*.json within {depth} levels of {base}")
    if len(set(plans.values())) != 1:
        raise ValueError(f"the shards come from different plans: {plans}")
    expected = range(next(iter(plans.values()))[1])
    if sorted(found) != list(expected):
        raise ValueError(f"shards {sorted(set(expected) - set(found))} are not mounted (found {sorted(found)})")
    return dict(sorted(found.items()))


def audit_trajectory(trajectory: Trajectory, window: int = 128) -> dict[str, object]:
    imu, imu_times = trajectory.load_imu()
    camera_times = trajectory.load_camera_times()
    images = trajectory.image_paths()
    dt = np.diff(imu_times)
    camera_dt = np.diff(camera_times)
    median_dt = float(np.median(dt))
    return {
        "trajectory": trajectory.key,
        "imu_rows": len(imu),
        "images": len(images),
        "camera_timestamps": len(camera_times),
        "images_match_camera_timestamps": len(images) == len(camera_times),
        "imu_rate_hz": 1.0 / median_dt,
        "camera_rate_hz": 1.0 / float(np.median(camera_dt)) if len(camera_dt) else None,
        "window": window,
        "window_duration_s": (window - 1) * median_dt,
        "enough_imu": len(imu) >= window,
        "max_relative_dt_deviation": float(np.max(np.abs(dt - median_dt)) / median_dt),
    }

