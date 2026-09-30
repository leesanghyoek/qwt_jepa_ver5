"""The 640x640 shard builder, run on fake TartanAir zips on disk (no network).

The part under test is what the loader will see: kept frames paired with their own
camera times, the full IMU, and a barometer that follows the true height.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from qjepa.data.manifest import build_manifest
from qjepa.data.tartanair import discover_trajectories

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import build_tartanair640 as builder  # noqa: E402

SIZE = (32, 32)
FRAMES = 40  # 4 s at 10 Hz: a 128-sample window fits around the middle frames only


def _png(index: int) -> bytes:
    rng = np.random.default_rng(index)
    buffer = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)).save(buffer, "PNG")
    return buffer.getvalue()


def _source(root: Path, environments=("AmericanDiner", "Office"), trajectories=("P000", "P001"),
            imu_files: dict[str, bytes] | None = None) -> Path:
    """``imu_files``: zip member name -> bytes, replacing a generated file or adding one."""
    imu_files = imu_files or {}
    for environment in environments:
        for difficulty in ("Data_easy", "Data_hard"):
            folder = root / environment / difficulty
            folder.mkdir(parents=True)
            with zipfile.ZipFile(folder / "image_lcam_front.zip", "w", zipfile.ZIP_DEFLATED) as images, \
                    zipfile.ZipFile(folder / "imu.zip", "w", zipfile.ZIP_DEFLATED) as imu:
                for trajectory in trajectories:
                    prefix = f"{environment}/{difficulty}/{trajectory}"
                    for index in range(FRAMES):
                        images.writestr(f"{prefix}/image_lcam_front/{index:06d}_lcam_front.png", _png(index))
                    pose = np.column_stack([np.arange(FRAMES), np.zeros((FRAMES, 6))])
                    text = io.BytesIO()
                    np.savetxt(text, pose)
                    images.writestr(f"{prefix}/pose_lcam_front.txt", text.getvalue())
                    rows = FRAMES * 10
                    imu_time = 100.0 + np.arange(rows) / 100.0
                    arrays = {
                        "imu_time": imu_time,
                        "cam_time": 100.0 + np.arange(FRAMES) / 10.0,
                        "acc": np.tile([0.0, 0.0, -9.81], (rows, 1)),
                        "gyro": np.zeros((rows, 3)),
                        # NED: climbs 2 m (z goes from 0 to -2)
                        "pos_global": np.column_stack([np.zeros(rows), np.zeros(rows), -np.linspace(0, 2, rows)]),
                        "vel_body": np.zeros((rows, 3)),
                    }
                    for name, array in arrays.items():
                        buffer = io.BytesIO()
                        np.save(buffer, array)
                        member = f"{prefix}/imu/{name}.npy"
                        imu.writestr(member, imu_files.get(member, buffer.getvalue()))
                        imu.writestr(f"{prefix}/imu/{name}.txt", "ignored")
                    imu.writestr(f"{prefix}/imu/parameter.yaml", "img_fps: 10\nimu_fps: 100\n")
                    for member, data in imu_files.items():
                        if member.startswith(f"{prefix}/imu/") and Path(member).stem not in arrays:
                            imu.writestr(member, data)
    return root


def _build(source: Path, out: Path, *, budget_frames: int, fmt: str = "webp", num_shards: int = 1, shard: int = 0):
    catalog, missing = builder.list_catalog(str(source), ["AmericanDiner", "Office"], threads=2)
    assert not missing
    mean = np.mean([entry.mean_png_bytes for entry in catalog.values()])
    plan = builder.make_plan(catalog, budget_bytes=budget_frames * mean, ratio=1.0)
    assignment = builder.assign_shards(builder.environment_bytes(catalog, plan, 1.0), num_shards)
    report = builder.build_shard(catalog, plan, assignment, shard=shard, source=str(source), out_root=out,
                                 fmt=fmt, workers=2, frame_size=SIZE, log=lambda *_: None)
    return catalog, plan, report


def test_remote_zip_reads_stored_and_deflated_members(tmp_path):
    path = tmp_path / "mixed.zip"
    payload = {"stored.bin": bytes(range(256)) * 40, "deflated.txt": b"tartanair " * 500}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("stored.bin", payload["stored.bin"], compress_type=zipfile.ZIP_STORED)
        archive.writestr("deflated.txt", payload["deflated.txt"], compress_type=zipfile.ZIP_DEFLATED)
    remote = builder.RemoteZip(str(path))
    members = {member.name: member for member in remote.members()}
    assert {name: remote.member(member) for name, member in members.items()} == payload


def test_plan_shares_frames_equally_across_environments_and_skips_the_ends(tmp_path):
    catalog, _ = builder.list_catalog(str(_source(tmp_path / "src")), ["AmericanDiner", "Office"], threads=2)
    mean = np.mean([entry.mean_png_bytes for entry in catalog.values()])
    plan = builder.make_plan(catalog, budget_bytes=60 * mean, ratio=1.0)
    per_environment = {}
    for key, positions in plan.items():
        per_environment[key.split("/")[0]] = per_environment.get(key.split("/")[0], 0) + len(positions)
        assert positions == sorted(set(positions))
        assert builder.DEFAULT_MARGIN <= positions[0] and positions[-1] < FRAMES - builder.DEFAULT_MARGIN
    assert abs(per_environment["AmericanDiner"] - per_environment["Office"]) <= 1
    assert 56 <= sum(per_environment.values()) <= 60
    # Same inputs, same plan: every shard has to agree on it.
    assert builder.make_plan(catalog, budget_bytes=60 * mean, ratio=1.0) == plan


def test_water_filling_gives_a_small_environment_everything_it_has():
    member = builder.Member("f", 0, 1000, 1000, 0, 0)

    def entry(environment, frames):
        return builder.TrajectoryEntry(environment, "Data_easy", "P000", [(index, member) for index in range(frames)])

    catalog = {"Small/Data_easy/P000": entry("Small", 30), "Big/Data_easy/P000": entry("Big", 1000)}
    plan = builder.make_plan(catalog, budget_bytes=300 * 1000, ratio=1.0, margin=7)
    assert len(plan["Small/Data_easy/P000"]) == 30 - 14
    assert len(plan["Small/Data_easy/P000"]) + len(plan["Big/Data_easy/P000"]) == 300


def test_shards_split_environments_and_balance_bytes():
    # Water-filling makes most environments the same size; a few small ones fall short.
    rng = np.random.default_rng(0)
    sizes = {f"E{index}": 0.7 if index < 60 else float(rng.uniform(0.1, 0.7)) for index in range(74)}
    assignment = builder.assign_shards(sizes, 3)
    assert set(assignment) == set(sizes) and set(assignment.values()) == {0, 1, 2}
    loads = [sum(size for name, size in sizes.items() if assignment[name] == shard) for shard in range(3)]
    assert max(loads) - min(loads) < max(sizes.values())


def test_shard_output_is_what_the_loader_discovers_and_pairs(tmp_path):
    source = _source(tmp_path / "src")
    out = tmp_path / "tartanair640"
    catalog, plan, report = _build(source, out, budget_frames=48)
    assert not report["skipped"] and report["trajectories"] == 8

    trajectories = discover_trajectories(out)
    assert len(trajectories) == 8
    for trajectory in trajectories:
        kept = np.load(trajectory.path / "frames.npy")
        assert [path.name for path in trajectory.image_paths()] == [f"{index:06d}_lcam_front.webp" for index in kept]
        with Image.open(trajectory.image_paths()[0]) as image:  # lossless: the very same pixels
            expected = Image.open(io.BytesIO(_png(int(kept[0])))).convert("RGB")
            assert np.array_equal(np.asarray(image.convert("RGB")), np.asarray(expected))
        imu, imu_times = trajectory.load_imu()
        assert len(imu) == FRAMES * 10  # the full IMU, not only around kept frames
        np.testing.assert_allclose(trajectory.load_camera_times(), 100.0 + kept / 10.0)
        assert not (trajectory.path / "imu" / "acc.txt").exists()
        pose = np.loadtxt(trajectory.path / "pose_lcam_front.txt", ndmin=2)
        np.testing.assert_array_equal(pose[:, 0], kept)
        pressure = np.load(trajectory.path / "imu" / "baro.npy")
        assert pressure.shape == imu_times.shape
        # Up 2 m is about 24 Pa less, at any site altitude up to 1.5 km.
        assert 20.0 < pressure[0] - pressure[-1] < 26.0

    manifest = build_manifest(out)
    samples = sum(manifest["meta"]["samples_per_split"].values())
    assert samples == sum(len(positions) for positions in plan.values())  # no frame wasted on the ends


def test_an_interrupted_trajectory_is_invisible_and_the_rerun_finishes_it(tmp_path):
    source = _source(tmp_path / "src", environments=("AmericanDiner",), trajectories=("P000",))
    out = tmp_path / "tartanair640"
    catalog, missing = builder.list_catalog(str(source), ["AmericanDiner"], ["Data_easy"], threads=1)
    plan = builder.make_plan(catalog, budget_bytes=1e12, ratio=1.0)
    key = "AmericanDiner/Data_easy/P000"
    # A lost session: a few frames written, IMU never renamed into place.
    target = out / key
    (target / "image_lcam_front").mkdir(parents=True)
    first = catalog[key].frames[plan[key][0]]
    (target / "image_lcam_front" / f"{first[0]:06d}_lcam_front.webp").write_bytes(
        builder.encode_frame(_png(first[0]), "webp", SIZE))
    (target / "imu.part").mkdir()
    with pytest.raises(FileNotFoundError):
        discover_trajectories(out)

    report = builder.build_shard(catalog, plan, {"AmericanDiner": 0}, shard=0, source=str(source), out_root=out,
                                 fmt="webp", workers=2, frame_size=SIZE, log=lambda *_: None)
    assert report["frames"] == len(plan[key]) and not (target / "imu.part").exists()
    assert len(discover_trajectories(out)[0].image_paths()) == len(plan[key])
    again = builder.build_shard(catalog, plan, {"AmericanDiner": 0}, shard=0, source=str(source), out_root=out,
                                fmt="webp", workers=2, frame_size=SIZE, log=lambda *_: None)
    assert again["reports"] == report["reports"]


def test_a_frame_of_the_wrong_size_is_refused():
    with pytest.raises(ValueError, match="expected 640x640"):
        builder.encode_frame(_png(0), "webp")


def test_a_leftover_hidden_download_next_to_the_real_file_is_ignored(tmp_path):
    # CarWelding/Data_hard/imu.zip really ships one: all zeros, not an .npy at all.
    leftover = "AmericanDiner/Data_hard/P001/imu/.azDownload-11e5a608-ori_global.npy"
    source = _source(tmp_path / "src", imu_files={leftover: bytes(4096)})
    out = tmp_path / "tartanair640"
    _, _, report = _build(source, out, budget_frames=48)
    assert not report["skipped"] and report["trajectories"] == 8
    assert not list(out.rglob(".azDownload*"))


def test_one_broken_trajectory_is_skipped_and_the_rest_are_built(tmp_path):
    broken = "Office/Data_easy/P000"
    source = _source(tmp_path / "src", imu_files={f"{broken}/imu/imu_time.npy": b"not an npy file"})
    out = tmp_path / "tartanair640"
    _, _, report = _build(source, out, budget_frames=48)
    assert [item["trajectory"] for item in report["skipped"]] == [broken]
    assert "ValueError" in report["skipped"][0]["skipped"]
    assert report["trajectories"] == 7
    assert broken not in {item.key for item in discover_trajectories(out)}


def _kaggle_unpack(archive_dir: Path, into: Path) -> None:
    """What Kaggle does when a dataset is made from the output: <name>.zip -> folder <name>/."""
    for archive in sorted(archive_dir.glob("*.zip")):
        with zipfile.ZipFile(archive) as opened:
            opened.extractall(into / archive.stem)


def test_packing_leaves_one_zip_per_environment_that_kaggle_unpacks_to_the_same_dataset(tmp_path):
    source = _source(tmp_path / "src")
    out = tmp_path / "tartanair640"
    _build(source, out, budget_frames=48)
    before = build_manifest(out)["meta"]["samples_per_split"]
    packed = builder.pack_environments(out, log=lambda *_: None)
    assert sorted(path.name for path in out.iterdir()) == ["AmericanDiner.zip", "Office.zip"]
    assert all(item["files"] > 0 for item in packed)
    assert builder.pack_environments(out, log=lambda *_: None) == []  # nothing left to do
    restored = tmp_path / "dataset" / "tartanair640"
    _kaggle_unpack(out, restored)
    # No level repeated: tartanair640/<env>/Data_easy/P000
    assert sorted(path.relative_to(restored).as_posix() for path in restored.glob("*/*/P000")) == [
        "AmericanDiner/Data_easy/P000", "AmericanDiner/Data_hard/P000", "Office/Data_easy/P000", "Office/Data_hard/P000"]
    assert build_manifest(restored)["meta"]["samples_per_split"] == before
