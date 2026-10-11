"""build_manifest touches storage less, for the very same manifest.

On Kaggle every stat, listing and file read under /kaggle/input waits on network storage, and the manifest of the
640 shards (1122 trajectories, 122k frames) took 28 to 60+ minutes. Measured on the local TartanAir set through a
linked root (30.5k frames): stat calls 603,823 -> 24,044 and readlink 30,823 -> 320, meta.json and the three CSVs
byte for byte the same.

Pins: the manifest (samples, meta, hash) is the one per-frame resolve() and a separate normalization read give,
through a linked root and with a symlinked frame; resolving costs the same however many frames a trajectory has; each
frame folder is listed once and never by discovery; each trajectory's IMU is read once; the frames listed are those
the five globs listed (the camera's PNGs first, else any PNG/JPEG/WebP; case-sensitive; hidden files too; none for a
missing folder).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from qjepa.data import manifest as manifest_module
from qjepa.data.manifest import assign_splits, build_manifest, compute_train_normalization
from qjepa.data.tartanair import IMAGE_DIRECTORY, Trajectory, discover_trajectories, link_trajectories

ENVIRONMENTS = ("env_a", "env_b", "env_c", "env_d", "env_e")


def _dataset(root: Path, frames: int) -> Path:
    """Five trajectories of ``frames`` frames, each centred on a 16-row IMU window."""
    for environment in ENVIRONMENTS:
        path = root / environment / "Data_easy" / "P000"
        (path / "imu").mkdir(parents=True)
        (path / IMAGE_DIRECTORY).mkdir()
        times = np.arange(16 * frames + 32, dtype=np.float64) * 0.01
        imu = np.stack([np.sin(times * (axis + 1)) + len(environment) for axis in range(6)], axis=-1)
        np.save(path / "imu/imu_time.npy", times)
        np.save(path / "imu/acc.npy", imu[:, :3])
        np.save(path / "imu/gyro.npy", imu[:, 3:])
        np.save(path / "imu/cam_time.npy", np.array([0.5 * (times[16 * k] + times[16 * k + 15])
                                                     for k in range(frames)]))
        for index in range(frames):
            Image.new("RGB", (8, 8), (index % 255, 30, 40)).save(path / IMAGE_DIRECTORY / f"{index:06d}_lcam_front.png")
    return root


def _per_frame_resolve(self):
    return [path.resolve() for path in self.image_paths()]


def test_the_manifest_is_the_per_frame_resolve_one_through_a_linked_root_and_a_symlinked_frame(tmp_path, monkeypatch):
    source = _dataset(tmp_path / "source", frames=6)
    # A frame that is itself a symlink resolves to its target, as resolve() did.
    frame = source / "env_b/Data_easy/P000" / IMAGE_DIRECTORY / "000003_lcam_front.png"
    elsewhere = tmp_path / "elsewhere.png"
    frame.rename(elsewhere)
    frame.symlink_to(elsewhere)
    linked = link_trajectories([source], tmp_path / "linked")
    fast = build_manifest(linked, window=16, seed=4, workers=3)
    # The reference: one resolve() per frame, and the normalization from its own read of the train IMU.
    trajectories = discover_trajectories(linked)
    normalization = compute_train_normalization(trajectories, assign_splits(trajectories, (0.8, 0.1, 0.1), 4, "hash"))
    monkeypatch.setattr(Trajectory, "resolved_image_paths", _per_frame_resolve)
    monkeypatch.setattr(manifest_module, "_train_normalization", lambda keys, summaries: normalization)
    reference = build_manifest(linked, window=16, seed=4, workers=3)
    assert fast["samples"] == reference["samples"] and fast["meta"] == reference["meta"]
    paths = {sample.image_path for split in fast["samples"].values() for sample in split}
    assert str(elsewhere) in paths and all(path.startswith(str(source)) or path == str(elsewhere) for path in paths)


def _counted(monkeypatch, root: Path) -> dict[str, int]:
    calls = {"lstat": 0, "frame_listings": 0, "loads": 0}
    lstat, scandir, load = os.lstat, os.scandir, np.load

    def counting_lstat(*args, **kwargs):
        calls["lstat"] += 1
        return lstat(*args, **kwargs)

    def counting_scandir(path=".", *args, **kwargs):
        calls["frame_listings"] += Path(path).name == IMAGE_DIRECTORY
        return scandir(path, *args, **kwargs)

    def counting_load(*args, **kwargs):
        calls["loads"] += 1
        return load(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "lstat", counting_lstat)
        patch.setattr(os, "scandir", counting_scandir)
        patch.setattr(np, "load", counting_load)
        build_manifest(root, window=16, seed=4, workers=1)
    return calls


def test_resolving_costs_the_same_for_any_number_of_frames_and_each_trajectory_is_read_once(tmp_path, monkeypatch):
    small = link_trajectories([_dataset(tmp_path / "a" / "source", frames=5)], tmp_path / "a" / "linked")
    large = link_trajectories([_dataset(tmp_path / "b" / "source", frames=40)], tmp_path / "b" / "linked")
    few, many = _counted(monkeypatch, small), _counted(monkeypatch, large)
    assert few == many                               # 5 or 40 frames a trajectory: the same storage calls
    assert many["frame_listings"] == len(ENVIRONMENTS)   # one listing per frame folder, none by discovery
    assert many["loads"] == 4 * len(ENVIRONMENTS)    # acc, gyro, imu_time, cam_time: the IMU once, train or not
    monkeypatch.setattr(Trajectory, "resolved_image_paths", _per_frame_resolve)
    per_frame = _counted(monkeypatch, large)
    assert per_frame["lstat"] > many["lstat"] + 40 * len(ENVIRONMENTS)     # what one resolve() per frame cost


def _five_globs(directory: Path) -> list[Path]:
    """Trajectory.image_paths before it listed the folder once."""
    paths = list(directory.glob("*_lcam_front.png"))
    if not paths:
        paths = [*directory.glob("*.png"), *directory.glob("*.jpg"), *directory.glob("*.jpeg"),
                 *directory.glob("*.webp")]
    return sorted(set(paths))


def test_one_listing_finds_the_frames_the_five_globs_found(tmp_path):
    camera = tmp_path / "camera" / "Data_easy" / "P000"
    other = tmp_path / "other" / "Data_easy" / "P000"
    for path, names in ((camera, ("000002_lcam_front.png", "000001_lcam_front.png", ".h_lcam_front.png",
                                  "000003_lcam_front.PNG", "x.webp", "notes.txt")),
                        (other, ("b.webp", "a.png", "c.jpg", "d.jpeg", "E.PNG", ".f.png", "g.txt", "h.webp.bak"))):
        (path / IMAGE_DIRECTORY).mkdir(parents=True)
        for name in names:
            (path / IMAGE_DIRECTORY / name).write_bytes(b"x")
        (path / IMAGE_DIRECTORY / "folder.png").mkdir()
    for path in (camera, other):
        trajectory = Trajectory(path, path.parts[-3], "Data_easy", "P000")
        assert trajectory.image_paths() == _five_globs(path / IMAGE_DIRECTORY)
        assert trajectory.resolved_image_paths() == [item.resolve() for item in _five_globs(path / IMAGE_DIRECTORY)]
    assert [item.name for item in Trajectory(camera, "camera", "Data_easy", "P000").image_paths()] == [
        ".h_lcam_front.png", "000001_lcam_front.png", "000002_lcam_front.png"]
    assert Trajectory(tmp_path / "missing", "m", "Data_easy", "P000").image_paths() == []
