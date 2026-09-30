"""Trajectory-level split, timestamp pairing, and train-only IMU statistics."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .tartanair import CAMERA, Trajectory, discover_trajectories


@dataclass(frozen=True)
class PairedSample:
    sample_id: str
    split: str
    environment: str
    difficulty: str
    trajectory_id: str
    trajectory_path: str
    image_path: str
    image_index: int
    image_time: float
    imu_start: int
    imu_end: int
    imu_start_time: float
    imu_end_time: float
    centre_error_s: float
    camera: str = CAMERA

    @property
    def trajectory_key(self) -> str:
        return f"{self.environment}/{self.difficulty}/{self.trajectory_id}"


FIELDS = tuple(PairedSample.__dataclass_fields__)
SPLIT_RULES = ("hash", "per_environment")
# TartanAir V2 renders two environments twice, by day and by night, along the very
# same camera paths: same difficulty, same Pxxx, identical poses. A scan of all 1122
# V2 trajectories finds no other pair. "per_environment" keeps each such Pxxx in one
# split, so a path trained on by day is not tested on by night.
SHARED_PATH_SCENES = {
    "ArchVizTinyHouseDay": "ArchVizTinyHouse",
    "ArchVizTinyHouseNight": "ArchVizTinyHouse",
    "OldBrickHouseDay": "OldBrickHouse",
    "OldBrickHouseNight": "OldBrickHouse",
}


def _per_environment_splits(
    trajectories: list[Trajectory], ratios: tuple[float, float, float], seed: int
) -> dict[str, str]:
    """Valid and test drawn from every environment rather than from the pool.

    With 74 environments, a 10% hash draw over the pool leaves about half of them out
    of test. Here each environment gives max(1, round(ratio * n)) of its n Pxxx groups
    to test when n >= 2, and the same to valid when that still leaves three groups to
    train (n >= 5 at 10%); the rest trains. Groups are Pxxx within an environment
    (easy and hard together), or within a SHARED_PATH_SCENES scene.
    """
    scenes: dict[str, dict[str, list[Trajectory]]] = {}
    for trajectory in trajectories:
        scene = SHARED_PATH_SCENES.get(trajectory.environment, trajectory.environment)
        scenes.setdefault(scene, {}).setdefault(f"{scene}/{trajectory.trajectory_id}", []).append(trajectory)
    assignment: dict[str, str] = {}
    for groups in scenes.values():
        keys = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}|{key}".encode()).hexdigest())
        count = len(keys)
        test = max(1, round(ratios[2] * count)) if ratios[2] > 0 and count >= 2 else 0
        valid = max(1, round(ratios[1] * count)) if ratios[1] > 0 else 0
        if count - test - valid < 3:
            valid = 0
        for index, key in enumerate(keys):
            split = "test" if index < test else "valid" if index < test + valid else "train"
            for trajectory in groups[key]:
                assignment[trajectory.key] = split
    return assignment


def assign_splits(
    trajectories: list[Trajectory],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 73128,
    rule: str = "hash",
) -> dict[str, str]:
    if len(ratios) != 3 or any(value < 0 for value in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must be three non-negative values summing to one")
    if rule not in SPLIT_RULES:
        raise ValueError(f"Unknown split rule {rule!r}; choose one of {SPLIT_RULES}")
    hints = {trajectory.motion_key: trajectory.split_hint for trajectory in trajectories}
    if any(trajectory.split_hint is not None for trajectory in trajectories):
        if not all(trajectory.split_hint is not None for trajectory in trajectories):
            hinted = [item for item in trajectories if item.split_hint is not None]
            plain = [item for item in trajectories if item.split_hint is None]
            raise ValueError(
                "Mixed explicit and missing split directories; use a consistent dataset root. "
                f"{len(hinted)} trajectory(ies) sit under a train/valid/test directory, for example "
                f"{hinted[0].path} (split={hinted[0].split_hint!r}); "
                f"{len(plain)} do not, for example {plain[0].path}. "
                "This usually means --data-root points one level too high and is picking up a second "
                "copy of the data. Point it at the single directory whose children are the environments."
            )
        for trajectory in trajectories:
            if hints[trajectory.motion_key] != trajectory.split_hint:
                raise ValueError(f"Conflicting split hints for {trajectory.motion_key}")
        return {trajectory.key: str(trajectory.split_hint) for trajectory in trajectories}
    if rule == "per_environment":
        return _per_environment_splits(trajectories, ratios, seed)

    groups: dict[str, list[Trajectory]] = {}
    for trajectory in trajectories:
        groups.setdefault(trajectory.motion_key, []).append(trajectory)
    keys = sorted(
        groups,
        key=lambda key: hashlib.sha256(f"{seed}|{key}".encode()).hexdigest(),
    )
    count = len(keys)
    if count == 1:
        validation_count = test_count = 0
    elif count == 2:
        validation_count, test_count = 0, 1
    else:
        validation_count = max(1, round(ratios[1] * count))
        test_count = max(1, round(ratios[2] * count))
        while validation_count + test_count >= count:
            if validation_count >= test_count and validation_count > 1:
                validation_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                break
    train_count = count - validation_count - test_count
    assignment: dict[str, str] = {}
    for index, key in enumerate(keys):
        split = "train" if index < train_count else "valid" if index < train_count + validation_count else "test"
        for trajectory in groups[key]:
            assignment[trajectory.key] = split
    return assignment


def pair_trajectory(
    trajectory: Trajectory,
    split: str,
    window: int = 128,
    max_relative_dt_deviation: float = 0.02,
    max_centre_error_in_dt: float = 1.0,
) -> tuple[list[PairedSample], dict[str, int | str]]:
    _, imu_times = trajectory.load_imu()
    camera_times = trajectory.load_camera_times()
    images = trajectory.image_paths()
    if len(images) != len(camera_times):
        raise ValueError(f"{trajectory.key}: {len(images)} images != {len(camera_times)} camera times")
    counters: dict[str, int | str] = {
        "trajectory": trajectory.key,
        "accepted": 0,
        "rejected_outside": 0,
        "rejected_centre": 0,
        "rejected_nonuniform": 0,
    }
    if len(imu_times) < window:
        counters["rejected_outside"] = len(images)
        return [], counters
    centres = 0.5 * (imu_times[: len(imu_times) - window + 1] + imu_times[window - 1 :])
    median_dt = float(np.median(np.diff(imu_times)))
    samples: list[PairedSample] = []
    for image_index, (image_path, image_time) in enumerate(zip(images, camera_times)):
        position = int(np.searchsorted(centres, image_time))
        candidates = [item for item in (position - 1, position) if 0 <= item < len(centres)]
        if not candidates:
            counters["rejected_outside"] = int(counters["rejected_outside"]) + 1
            continue
        start = min(candidates, key=lambda item: abs(float(centres[item] - image_time)))
        end = start + window
        error = abs(float(centres[start] - image_time))
        if error > max_centre_error_in_dt * median_dt:
            counters["rejected_centre"] = int(counters["rejected_centre"]) + 1
            continue
        local_dt = np.diff(imu_times[start:end])
        if np.max(np.abs(local_dt - median_dt)) / median_dt > max_relative_dt_deviation:
            counters["rejected_nonuniform"] = int(counters["rejected_nonuniform"]) + 1
            continue
        samples.append(
            PairedSample(
                sample_id=f"{trajectory.environment}__{trajectory.difficulty}__{trajectory.trajectory_id}__{image_index:06d}",
                split=split,
                environment=trajectory.environment,
                difficulty=trajectory.difficulty,
                trajectory_id=trajectory.trajectory_id,
                trajectory_path=str(trajectory.path.resolve()),
                image_path=str(image_path.resolve()),
                image_index=image_index,
                image_time=float(image_time),
                imu_start=start,
                imu_end=end,
                imu_start_time=float(imu_times[start]),
                imu_end_time=float(imu_times[end - 1]),
                centre_error_s=error,
            )
        )
    counters["accepted"] = len(samples)
    return samples, counters


def compute_train_normalization(
    trajectories: list[Trajectory], assignments: dict[str, str]
) -> dict[str, object]:
    total = np.zeros(6, dtype=np.float64)
    total_square = np.zeros(6, dtype=np.float64)
    count = 0
    used_stream_hashes: set[str] = set()
    used_trajectories: list[str] = []
    for trajectory in trajectories:
        if assignments[trajectory.key] != "train":
            continue
        imu, timestamps = trajectory.load_imu()
        identity = hashlib.sha256(imu.tobytes() + timestamps.tobytes()).hexdigest()
        if identity in used_stream_hashes:
            continue
        used_stream_hashes.add(identity)
        total += imu.sum(axis=0)
        total_square += np.square(imu).sum(axis=0)
        count += len(imu)
        used_trajectories.append(trajectory.key)
    if count == 0:
        raise ValueError("No clean training IMU rows are available for normalization")
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 0.0)
    return {
        "source": "unique_clean_train_trajectory_rows",
        "rows": count,
        "trajectories": used_trajectories,
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "channels": ["ax", "ay", "az", "gx", "gy", "gz"],
        "units": ["m/s^2"] * 3 + ["rad/s"] * 3,
    }


def manifest_hash(samples: dict[str, list[PairedSample]], normalization: dict | None = None) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(normalization, sort_keys=True, separators=(",", ":")).encode())
    for split in ("train", "valid", "test"):
        digest.update(split.encode())
        for sample in samples[split]:
            digest.update(json.dumps(asdict(sample), sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def build_manifest(
    root: str | Path,
    window: int = 128,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 73128,
    split_rule: str = "hash",
) -> dict[str, object]:
    trajectories = discover_trajectories(root)
    assignments = assign_splits(trajectories, ratios, seed, split_rule)
    samples: dict[str, list[PairedSample]] = {"train": [], "valid": [], "test": []}
    audit = []
    for trajectory in trajectories:
        paired, stats = pair_trajectory(trajectory, assignments[trajectory.key], window)
        samples[assignments[trajectory.key]].extend(paired)
        audit.append(stats)
    meta = {
        "manifest_schema": 2,
        "root": str(Path(root).expanduser().resolve()),
        "window": window,
        "seed": seed,
        "split_unit": "environment/trajectory_id",
        "split_rule": split_rule,
        "samples_per_split": {key: len(value) for key, value in samples.items()},
        "trajectories_per_split": {
            split: sorted(item.key for item in trajectories if assignments[item.key] == split)
            for split in samples
        },
        "normalization": compute_train_normalization(trajectories, assignments),
        "pairing_audit": audit,
    }
    meta["manifest_hash"] = manifest_hash(samples, meta["normalization"])
    return {"samples": samples, "meta": meta}


def write_manifest(manifest: dict[str, object], output: str | Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    samples = manifest["samples"]
    assert isinstance(samples, dict)
    for split, rows in samples.items():
        with (output / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
    (output / "meta.json").write_text(
        json.dumps(manifest["meta"], indent=2, ensure_ascii=False), encoding="utf-8"
    )


def read_manifest(directory: str | Path) -> dict[str, object]:
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    if meta.get("manifest_schema") != 2:
        raise ValueError("Manifest schema changed; rebuild with build-manifest and start a new matching run")
    samples: dict[str, list[PairedSample]] = {}
    types = {
        "image_index": int,
        "image_time": float,
        "imu_start": int,
        "imu_end": int,
        "imu_start_time": float,
        "imu_end_time": float,
        "centre_error_s": float,
    }
    for split in ("train", "valid", "test"):
        rows: list[PairedSample] = []
        path = directory / f"{split}.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                for raw in csv.DictReader(handle):
                    rows.append(PairedSample(**{key: types.get(key, str)(value) for key, value in raw.items()}))
        samples[split] = rows
    if manifest_hash(samples, meta["normalization"]) != meta.get("manifest_hash"):
        raise ValueError("Manifest CSV contents do not match meta.json hash")
    return {"samples": samples, "meta": meta}
