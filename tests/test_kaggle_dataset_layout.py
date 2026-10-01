"""Regression tests for the real Kaggle TartanAir V2 layout (see kaggle_dataset.md).

The dataset ships 166 trajectories as <env>/<Data_easy|Data_hard>/<Pxxx>/ with
.npy IMU at 100 Hz, a 10 Hz camera and 256x256 RGB frames. These tests pin the
geometry that the audit measured, so a loader change cannot silently stop
matching the dataset. Images are small here because framing does not depend on
pixel size; the timing is the part under test.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from qjepa.data.manifest import assign_splits, build_manifest
from qjepa.data.tartanair import discover_trajectories

IMU_HZ, CAMERA_HZ, WINDOW = 100.0, 10.0, 128
# A 128-sample window at 100 Hz spans 1.27 s and must bracket the frame, so the
# first and last ~6.35 frames of every trajectory can never be centred.
EXPECTED_REJECTED_CENTRE = 13


def _trajectory(root: Path, environment: str, difficulty: str, trajectory_id: str, frames: int) -> None:
    path = root / environment / difficulty / trajectory_id
    (path / "image_lcam_front").mkdir(parents=True)
    (path / "imu").mkdir()
    rows = int(frames * IMU_HZ / CAMERA_HZ)
    imu_time = np.arange(rows, dtype=np.float64) / IMU_HZ
    cam_time = np.arange(frames, dtype=np.float64) / CAMERA_HZ
    accel = np.stack([np.sin(imu_time * (axis + 1)) for axis in range(3)], axis=-1)
    accel[:, 2] -= 9.81  # TartanAir acc.npy keeps gravity; acc_nograv.npy does not.
    gyro = np.stack([np.cos(imu_time * (axis + 1)) for axis in range(3)], axis=-1) * 0.3
    np.save(path / "imu" / "acc.npy", accel)
    np.save(path / "imu" / "gyro.npy", gyro)
    np.save(path / "imu" / "imu_time.npy", imu_time)
    np.save(path / "imu" / "cam_time.npy", cam_time)
    # Files TartanAir V2 also ships; the loader must ignore them.
    np.save(path / "imu" / "acc_nograv.npy", accel + np.array([0.0, 0.0, 9.81]))
    np.save(path / "imu" / "vel_body.npy", gyro)
    for index in range(frames):
        Image.new("RGB", (32, 32), (index % 256, 30, 40)).save(
            path / "image_lcam_front" / f"{index:06d}_lcam_front.png"
        )


def _dataset(root: Path, frames: int = 40) -> Path:
    for environment in ("AmericanDiner", "Office"):
        for difficulty in ("Data_easy", "Data_hard"):
            for index in range(2):
                _trajectory(root, environment, difficulty, f"P{index:03d}", frames)
    return root


def test_discovery_parses_environment_difficulty_and_ignores_extra_imu_files(tmp_path):
    trajectories = discover_trajectories(_dataset(tmp_path / "tartanair-v2"))
    assert len(trajectories) == 8
    assert {item.difficulty for item in trajectories} == {"Data_easy", "Data_hard"}
    assert {item.environment for item in trajectories} == {"AmericanDiner", "Office"}
    # Data_easy/Data_hard of one motion share a motion_key, so they cannot split apart.
    assert len({item.motion_key for item in trajectories}) == 4
    assert all(item.split_hint is None for item in trajectories)
    imu, times = trajectories[0].load_imu()
    assert imu.shape == (400, 6) and times.shape == (400,)
    assert imu[:, 2].mean() < -9.0, "acc.npy must keep gravity"


def test_window_framing_matches_the_audited_100hz_10hz_geometry(tmp_path):
    manifest = build_manifest(_dataset(tmp_path / "tartanair-v2"), window=WINDOW)
    audit = manifest["meta"]["pairing_audit"]
    assert len(audit) == 8
    for entry in audit:
        assert entry["rejected_centre"] == EXPECTED_REJECTED_CENTRE
        assert entry["rejected_nonuniform"] == 0 and entry["rejected_outside"] == 0
        assert entry["accepted"] == 40 - EXPECTED_REJECTED_CENTRE
    # Losing a fixed 13 frames per trajectory is a shrinking cost on longer ones.
    longer = build_manifest(_dataset(tmp_path / "longer", frames=80), window=WINDOW)
    assert all(entry["rejected_centre"] == EXPECTED_REJECTED_CENTRE
               for entry in longer["meta"]["pairing_audit"])


def test_easy_and_hard_of_one_motion_stay_in_the_same_split(tmp_path):
    meta = build_manifest(_dataset(tmp_path / "tartanair-v2"), window=WINDOW)["meta"]
    placement: dict[str, set[str]] = {}
    for split, keys in meta["trajectories_per_split"].items():
        for key in keys:
            environment, _, trajectory_id = key.split("/")
            placement.setdefault(f"{environment}/{trajectory_id}", set()).add(split)
    assert placement, "no trajectories were assigned"
    assert all(len(splits) == 1 for splits in placement.values())
    assert meta["normalization"]["mean"][2] < -9.0
    assert sum(meta["samples_per_split"].values()) == 8 * (40 - EXPECTED_REJECTED_CENTRE)


def test_root_one_level_too_high_names_the_offending_directories(tmp_path):
    """The Kaggle mount holds tartanair-v2 beside tartanair-v2-jepa/train/."""
    _dataset(tmp_path / "tartanair-v2")
    _trajectory(tmp_path / "tartanair-v2-jepa" / "train", "Office", "Data_easy", "P000", 40)
    with pytest.raises(ValueError) as error:
        assign_splits(discover_trajectories(tmp_path))
    message = str(error.value)
    assert "Mixed explicit" in message and "tartanair-v2-jepa" in message
    assert "points one level too high" in message
    # Pointing at the right root works.
    assert len(discover_trajectories(tmp_path / "tartanair-v2")) == 8


def test_link_trajectories_merges_shards_whatever_their_nesting(tmp_path):
    from qjepa.data.tartanair import link_trajectories

    # Shard 0 as Kaggle unpacked its tar files: <env>/tartanair640/<env>/Data_easy/P000.
    shard0 = tmp_path / "input" / "tartanair640-s0" / "tartanair640_archives"
    _trajectory(shard0 / "AmericanDiner" / "tartanair640", "AmericanDiner", "Data_easy", "P000", 40)
    _trajectory(shard0 / "AmericanDiner" / "tartanair640", "AmericanDiner", "Data_hard", "P000", 40)
    # Shard 1 as Kaggle unpacks the builder's zips: tartanair640/<env>/Data_easy/P000.
    shard1 = tmp_path / "input" / "tartanair640-s1" / "tartanair640"
    _trajectory(shard1, "Office", "Data_easy", "P000", 40)
    _trajectory(shard1, "Office", "Data_easy", "P001", 40)

    merged = link_trajectories([shard0, shard1], tmp_path / "merged")
    keys = sorted(item.key for item in discover_trajectories(merged))
    assert keys == ["AmericanDiner/Data_easy/P000", "AmericanDiner/Data_hard/P000",
                    "Office/Data_easy/P000", "Office/Data_easy/P001"]
    manifest = build_manifest(merged, window=WINDOW)
    paths = [sample.image_path for split in manifest["samples"].values() for sample in split]
    assert paths and all(str(tmp_path / "input") in path for path in paths)  # resolved to the real files
    assert link_trajectories([shard0, shard1], tmp_path / "merged") == tmp_path / "merged"  # rerun is harmless
    with pytest.raises(ValueError, match="is in both"):
        link_trajectories([shard1, shard1], tmp_path / "twice")


def test_per_environment_split_draws_valid_and_test_from_every_environment():
    from qjepa.data.manifest import SHARED_PATH_SCENES
    from qjepa.data.tartanair import Trajectory

    sizes = {"Big": 9, "Five": 5, "Four": 4, "Two": 2, "One": 1,
             "ArchVizTinyHouseDay": 7, "ArchVizTinyHouseNight": 7}
    trajectories = [
        Trajectory(Path(f"/data/{env}/{difficulty}/P{index:03d}"), env, difficulty, f"P{index:03d}")
        for env, count in sizes.items() for index in range(count) for difficulty in ("Data_easy", "Data_hard")
    ]
    split = assign_splits(trajectories, rule="per_environment")
    groups = {}
    for trajectory in trajectories:
        scene = SHARED_PATH_SCENES.get(trajectory.environment, trajectory.environment)
        groups.setdefault(scene, {}).setdefault(trajectory.trajectory_id, set()).add(split[trajectory.key])
    # Easy and hard of a Pxxx, and a Pxxx by day and by night, never part ways.
    assert all(len(splits) == 1 for scene in groups.values() for splits in scene.values())
    count = {scene: {name: sum(next(iter(s)) == name for s in ids.values()) for name in ("train", "valid", "test")}
             for scene, ids in groups.items()}
    assert count["Big"] == {"train": 7, "valid": 1, "test": 1}
    assert count["Five"] == {"train": 3, "valid": 1, "test": 1}
    assert count["Four"] == {"train": 3, "valid": 0, "test": 1}
    assert count["Two"] == {"train": 1, "valid": 0, "test": 1}
    assert count["One"] == {"train": 1, "valid": 0, "test": 0}
    assert count["ArchVizTinyHouse"] == {"train": 5, "valid": 1, "test": 1}
    # The default stays the hash rule.
    assert assign_splits(trajectories) == assign_splits(trajectories, rule="hash")
    with pytest.raises(ValueError, match="Unknown split rule"):
        assign_splits(trajectories, rule="random")


def test_find_shard_roots_wants_every_shard_of_one_plan_once(tmp_path):
    import json

    from qjepa.data.tartanair import find_shard_roots

    def shard(root: Path, number: int, fingerprint: str = "6707c5738eaa") -> Path:
        (root / "build_meta").mkdir(parents=True)
        (root / "build_meta" / f"shard{number}.json").write_text(json.dumps({"shard": number, "fingerprint": fingerprint}))
        (root / "build_meta" / "plan.json").write_text(json.dumps({"num_shards": 3}))
        return root

    base = tmp_path / "input"
    roots = {0: shard(base / "tartanairshard0", 0), 1: shard(base / "tartanair640-s1", 1),
             2: shard(base / "datasets" / "owner" / "tartanair640-s2", 2)}
    assert find_shard_roots(base) == roots
    shard(base / "copy-of-s1", 1)
    with pytest.raises(ValueError, match="shard 1 is mounted twice"):
        find_shard_roots(base)
    shutil.rmtree(base / "copy-of-s1")
    shutil.rmtree(roots[2])
    with pytest.raises(ValueError, match=r"shards \[2\] are not mounted"):
        find_shard_roots(base)
    # A trial on the shards there are: only those asked for, still all from one plan.
    assert find_shard_roots(base, shards=(0, 1)) == {0: roots[0], 1: roots[1]}
    assert find_shard_roots(base, shards=[1]) == {1: roots[1]}
    with pytest.raises(ValueError, match=r"shards \[2\] are not mounted"):
        find_shard_roots(base, shards=(0, 2))
    shard(base / "other-plan", 2, fingerprint="000000000000")
    with pytest.raises(ValueError, match="different plans"):
        find_shard_roots(base)
    shutil.rmtree(base / "other-plan")
    packed = shard(base / "tartanair640-s2", 2)
    (packed / "tartanair640").mkdir()
    (packed / "tartanair640" / "Office.tar").write_bytes(b"")
    with pytest.raises(ValueError, match="did not unpack"):
        find_shard_roots(base)
