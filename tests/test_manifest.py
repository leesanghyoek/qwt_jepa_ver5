from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from qjepa.data.manifest import assign_splits, build_manifest, read_manifest, write_manifest
from qjepa.data.tartanair import Trajectory


def _trajectory(root: Path, environment: str):
    path = root / environment / "Data_easy" / "P000"
    imu_dir = path / "imu"
    image_dir = path / "image_lcam_front"
    imu_dir.mkdir(parents=True)
    image_dir.mkdir()
    times = np.arange(64, dtype=np.float64) * 0.01
    imu = np.stack([np.sin(times * (axis + 1)) for axis in range(6)], axis=-1)
    np.save(imu_dir / "imu_time.npy", times)
    np.save(imu_dir / "acc.npy", imu[:, :3])
    np.save(imu_dir / "gyro.npy", imu[:, 3:])
    starts = (0, 16, 32)
    camera_times = np.array([0.5 * (times[start] + times[start + 15]) for start in starts])
    np.save(imu_dir / "cam_time.npy", camera_times)
    for index in range(len(camera_times)):
        Image.new("RGB", (32, 32), (20 * index, 30, 40)).save(
            image_dir / f"{index:06d}_lcam_front.png"
        )


def test_manifest_splits_trajectories_before_windows_and_roundtrips(tmp_path):
    root = tmp_path / "dataset"
    for environment in ("env_a", "env_b", "env_c"):
        _trajectory(root, environment)
    manifest = build_manifest(root, window=16, seed=4)
    meta = manifest["meta"]
    assert sum(meta["samples_per_split"].values()) == 9
    split_trajectories = [set(meta["trajectories_per_split"][split]) for split in ("train", "valid", "test")]
    assert split_trajectories[0].isdisjoint(split_trajectories[1])
    assert split_trajectories[0].isdisjoint(split_trajectories[2])
    assert meta["normalization"]["rows"] == 64
    output = tmp_path / "manifest"
    write_manifest(manifest, output)
    loaded = read_manifest(output)
    assert loaded["meta"]["manifest_hash"] == meta["manifest_hash"]


def test_manifest_detects_timestamp_and_normalization_edits(tmp_path):
    root = tmp_path / "dataset"
    for environment in ("env_a", "env_b", "env_c"):
        _trajectory(root, environment)
    manifest = build_manifest(root, window=16)
    original = manifest["samples"]["train"][0]
    manifest["samples"]["train"][0] = replace(original, image_time=original.image_time + 0.01)
    write_manifest(manifest, tmp_path / "manifest")
    with pytest.raises(ValueError, match="hash"):
        read_manifest(tmp_path / "manifest")
    manifest["samples"]["train"][0] = original
    manifest["meta"]["normalization"]["mean"][0] += 1
    write_manifest(manifest, tmp_path / "manifest")
    with pytest.raises(ValueError, match="hash"):
        read_manifest(tmp_path / "manifest")


def test_partial_explicit_splits_are_rejected(tmp_path):
    trajectories = [Trajectory(tmp_path, "A", "Data_easy", "P000", "train"),
                    Trajectory(tmp_path, "B", "Data_easy", "P000", None)]
    with pytest.raises(ValueError, match="Mixed explicit"):
        assign_splits(trajectories)


def test_threads_build_the_very_manifest_one_thread_does(tmp_path):
    """build_manifest reads trajectories on a thread pool (Kaggle's /kaggle/input waits on
    network storage for every file); the manifest and its hash must not depend on it."""
    from qjepa.data.manifest import build_manifest
    from test_kaggle_workflow import _write_dataset
    _write_dataset(tmp_path)
    one, many = (build_manifest(tmp_path, workers=workers) for workers in (1, 8))
    assert one["meta"] == many["meta"] and one["samples"] == many["samples"]
