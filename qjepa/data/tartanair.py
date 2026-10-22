"""TartanAir camera/IMU discovery and strict timeline validation."""

from __future__ import annotations

import fnmatch
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

    def _image_entries(self) -> list[tuple[str, bool]]:
        """(name, is a symlink) of the frames, sorted: ``*_lcam_front.png``, else any PNG/JPEG/WebP.

        One listing of the folder, where five globs listed it up to five times: on Kaggle's
        /kaggle/input every listing waits on network storage. Names match as pathlib's glob
        matched them (case-sensitive, hidden files too)."""
        try:
            with os.scandir(self.path / IMAGE_DIRECTORY) as listing:
                entries = {entry.name: entry.is_symlink() for entry in listing}
        except (FileNotFoundError, NotADirectoryError):
            entries = {}                                 # as glob: no folder, no frames
        names = [name for name in entries if fnmatch.fnmatchcase(name, f"*_{CAMERA}.png")]
        if not names:
            # tools/build_tartanair640.py stores the 640x640 frames as lossless WebP.
            names = [name for name in entries
                     if any(fnmatch.fnmatchcase(name, f"*.{suffix}") for suffix in ("png", "jpg", "jpeg", "webp"))]
        return [(name, entries[name]) for name in sorted(names)]

    def image_paths(self) -> list[Path]:
        directory = self.path / IMAGE_DIRECTORY
        return [directory / name for name, _ in self._image_entries()]

    def resolved_image_paths(self) -> list[Path]:
        """``[path.resolve() for path in image_paths()]``, resolving the folder once.

        A frame that is not a symlink resolves to the resolved folder plus its name, so only
        symlinked frames are resolved one by one. Per frame, resolve() cost ~20 stat calls
        through the linked dataset root -- about 2.4 million on /kaggle/input's network storage
        for the 122k frames of the 640 shards."""
        directory = self.path / IMAGE_DIRECTORY
        resolved = directory.resolve()
        return [(directory / name).resolve() if symlink else resolved / name
                for name, symlink in self._image_entries()]

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
        # A frame folder holds no trajectory: listing its frames only waits on storage.
        child_names[:] = [name for name in child_names if name != IMAGE_DIRECTORY]
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


COPIED_FOLDERS = ("imu", IMAGE_DIRECTORY)


def copy_trajectories(roots: list[str | Path], target: str | Path, *, workers: int = 128,
                      report_every: float = 30.0, log=print) -> Path:
    """link_trajectories, but each trajectory's imu/ and image_lcam_front/ are copied onto ``target``'s disk.

    p46 on Kaggle: /kaggle/input answered every file after ~1 s (one 640 PNG 1.15 s, one trajectory's IMU 9.8 s),
    so phase 1 waited ~6 s per update for its data. The wait is per file, not bandwidth: 8 threads read 7x as
    fast, 32 threads 28x. So the files are read here by ``workers`` threads, once. A link left by an earlier
    link_trajectories into ``target`` becomes a folder; files already copied are kept, so a rerun copies only
    what is missing. A file appears under its name only once complete (written aside, then renamed).
    The manifest resolves frame paths, so one built on the copy names the local files.
    """
    import shutil
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    target = Path(target)
    staging = target / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    found: dict[str, Trajectory] = {}
    for root in roots:
        for trajectory in discover_trajectories(root):
            if trajectory.key in found:
                raise ValueError(f"{trajectory.key} is in both {found[trajectory.key].path} and {trajectory.path}")
            found[trajectory.key] = trajectory

    def listing(trajectory: Trajectory) -> tuple[Path, list[tuple[Path, Path]]]:
        destination = target / trajectory.environment / trajectory.difficulty / trajectory.trajectory_id
        files = []
        for folder in COPIED_FOLDERS:
            with os.scandir(trajectory.path / folder) as entries:
                files += [(Path(entry.path), destination / folder / entry.name) for entry in entries if entry.is_file()]
        return destination, files

    started = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        listed = list(pool.map(listing, found.values()))
    jobs = []
    for destination, files in listed:
        if destination.is_symlink():
            destination.unlink()                        # the link, not what it points at
        for folder in COPIED_FOLDERS:
            (destination / folder).mkdir(parents=True, exist_ok=True)
        jobs += [(source, copy) for source, copy in files if not copy.is_file()]
    total = sum(len(files) for _, files in listed)
    log(f"{len(found)} trajectory, {total} file: {total - len(jobs)} đã có, chép {len(jobs)} "
        f"({workers} luồng; liệt kê {time.perf_counter() - started:.0f} s)", flush=True)
    counter = iter(range(len(jobs)))
    lock = threading.Lock()

    def copy(job: tuple[Path, Path]) -> int:
        source, destination = job
        with lock:
            partial = staging / f"{next(counter)}.part"
        shutil.copyfile(source, partial)
        size = partial.stat().st_size
        os.replace(partial, destination)
        return size

    copied, done, last = 0, 0, time.perf_counter()
    started = last
    with ThreadPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(copy, job) for job in jobs]):
            copied += future.result()
            done += 1
            now = time.perf_counter()
            if now - last >= report_every or done == len(jobs):
                rate = done / max(now - started, 1e-9)
                log(f"  chép {done}/{len(jobs)} file · {copied / 2**30:.1f} GiB · {rate:.0f} file/s · "
                    f"{copied / max(now - started, 1e-9) / 2**20:.0f} MiB/s · còn ~{(len(jobs) - done) / rate / 60:.0f} phút",
                    flush=True)
                last = now
    shutil.rmtree(staging, ignore_errors=True)
    return target


def find_shard_roots(
    base: str | Path = "/kaggle/input", depth: int = 4, shards: tuple[int, ...] | list[int] | None = None
) -> dict[int, Path]:
    """{shard: dataset root} of the 640x640 TartanAir shards mounted under ``base``.

    A shard's root is the folder holding ``build_meta/shard<N>.json`` (written by
    ``tools/build_tartanair640.py``). Each shard must be there once and all from the
    same plan. ``shards=None`` wants every shard of the plan; a list takes just those
    (others mounted are left out). With ``split_rule: per_environment`` a subset
    keeps every trajectory in the split it has over all shards: each scene lies in
    one shard, and the one that spans two (ArchVizTinyHouse Day/Night) has the same
    Pxxx in both. Checked on the real plan: shards 0+2 move none of 766 trajectories.
    The hash rule draws over the whole pool, so there a subset does move them. A
    shard whose archives Kaggle did not unpack is refused by name.
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
    expected = sorted(shards) if shards is not None else list(range(next(iter(plans.values()))[1]))
    if not set(expected) <= set(found):
        raise ValueError(f"shards {sorted(set(expected) - set(found))} are not mounted (found {sorted(found)})")
    return {shard: found[shard] for shard in expected}


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

