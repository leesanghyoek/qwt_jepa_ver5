"""evaluate --scenarios, --amp and its progress lines -- a test split scored faster, on two GPUs at once.

p26's --protocol ran over half an hour on one GPU and printed nothing until it ended. Pins:
--scenarios scores a subset of the protocol, in the protocol's order, with exactly the numbers the
whole protocol gives those scenarios (so two processes, one per GPU, together give the protocol);
it needs --protocol and known names; each scenario announces itself; --amp (CUDA only) moves the
metrics by float noise and is recorded in evaluation_config.json. --every N keeps every N-th frame of
each trajectory in time order and its last one -- every trajectory stays, frames close in time are thinned -- so a test
split scores ~N times faster; the IMU windows still overlap and cover what the whole split covers; it is
recorded, and below 1 refused.
"""

from __future__ import annotations

import json

import pytest
import torch
import yaml

from qjepa.cli import PROTOCOL_SCENARIOS, _dataset, _every_nth_frame, _manifest, main
from qjepa.config import load_config, serializable_config
from test_kaggle_workflow import _write_dataset


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A smoke run through both phases: (config path, manifest, phase-2 checkpoint)."""
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("evaluate_options")
    _write_dataset(root / "dataset")
    path = root / "smoke.yaml"
    path.write_text(yaml.safe_dump(serializable_config(load_config("configs/smoke.yaml"))))
    common = ["--config", str(path), "--manifest", str(root / "manifest"), "--output", str(root / "run")]
    main(["build-manifest", "--config", str(path), "--data-root", str(root / "dataset"),
          "--output", str(root / "manifest")])
    main(["train-phase1", *common])
    main(["train-phase2", *common, "--backbone-checkpoint", str(root / "run/phase1/last.pt")])
    return root / "manifest", root / "run/phase2/best_joint_validation.pt", root


def _evaluate(trained, name, *extra, device="cpu"):
    manifest, checkpoint, root = trained
    output = root / name
    main(["evaluate", "--checkpoint", str(checkpoint), "--manifest", str(manifest), "--device", device,
          "--gpus", "1", "--split", "valid", "--panels", "0", "--output", str(output), *extra])
    return json.loads((output / "metrics.json").read_text()), json.loads((output / "evaluation_config.json").read_text())


def test_a_subset_gives_the_protocols_own_numbers_in_its_order(trained, capsys):
    whole, _ = _evaluate(trained, "whole", "--protocol")
    assert list(whole) == list(PROTOCOL_SCENARIOS)
    capsys.readouterr()
    part, config = _evaluate(trained, "part", "--protocol", "--scenarios", "blur_only, clean_clean")
    assert list(part) == ["clean_clean", "blur_only"] == list(config["scenarios"])
    for name, values in part.items():
        for key, value in values.items():
            assert value == pytest.approx(whole[name][key], rel=1e-6, abs=1e-9), (name, key)
    printed = capsys.readouterr().out
    assert "[1/2] clean_clean" in printed and "[2/2] blur_only" in printed


def test_every_keeps_each_trajectory_thinned_in_time_order(trained):
    manifest_path, checkpoint, root = trained
    config = load_config(root / "smoke.yaml") if (root / "smoke.yaml").exists() else load_config("configs/smoke.yaml")
    manifest, _ = _manifest(config, str(manifest_path))
    dataset = _dataset(config, manifest, "valid", fixed_realization=True)
    groups = {}
    for sample in dataset.samples:
        groups.setdefault(sample.trajectory_key, []).append(sample)
    _every_nth_frame(dataset, 2)
    kept = {}
    for sample in dataset.samples:
        kept.setdefault(sample.trajectory_key, []).append(sample)
    assert set(kept) == set(groups)
    for key, rows in groups.items():
        ordered = sorted(rows, key=lambda row: row.image_time)
        expected = ordered[::2] + ([] if (len(ordered) - 1) % 2 == 0 else [ordered[-1]])   # and the last
        assert [row.sample_id for row in kept[key]] == [row.sample_id for row in expected]


def test_every_scores_fewer_frames_and_the_imu_still_covers_the_split(trained):
    whole, _ = _evaluate(trained, "whole_requested")
    thinned, config = _evaluate(trained, "every2", "--every", "2")
    assert config["every"] == 2
    a, b = whole["requested"], thinned["requested"]
    assert b["image_count"] < a["image_count"]
    assert b["imu_covered_unique_rows"] == a["imu_covered_unique_rows"]


@pytest.mark.parametrize("extra, message", [
    (("--scenarios", "blur_only"), "add --protocol"),
    (("--protocol", "--scenarios", "blur_only,fog_only"), "fog_only"),
    (("--every", "0"), "--every"),
])
def test_bad_scenarios_are_refused(trained, capsys, extra, message):
    with pytest.raises(SystemExit):
        _evaluate(trained, "refused", *extra)
    assert message in capsys.readouterr().err


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 autocast is measured on CUDA")
def test_amp_moves_the_metrics_by_float_noise_only(trained):
    plain, plain_config = _evaluate(trained, "fp32", device="cuda")
    half, half_config = _evaluate(trained, "fp16", "--amp", device="cuda")
    assert plain_config["amp"] is False and half_config["amp"] is True
    a, b = plain["requested"], half["requested"]
    assert a["image_count"] == b["image_count"]
    assert abs(a["image_psnr_db"] - b["image_psnr_db"]) < 0.05
    assert abs(a["image_ssim"] - b["image_ssim"]) < 1e-3
    assert a["baseline_image_psnr_db"] == pytest.approx(b["baseline_image_psnr_db"])


def test_brightness_probe_splits_frames_by_whether_their_light_was_touched(trained):
    """tools/brightness_probe.py: low-frequency error of input and restored frames by group, with oracle floors."""
    import math
    import subprocess
    import sys
    manifest, checkpoint, root = trained
    output = root / "brightness.json"
    result = subprocess.run([sys.executable, "tools/brightness_probe.py", "--checkpoint", str(checkpoint),
                             "--manifest", str(manifest), "--samples", "4", "--batch", "2", "--device", "cpu",
                             "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    table = json.loads(output.read_text())
    assert "tat ca" in table and table["tat ca"]["count"] == 4
    assert sum(table[group]["count"] for group in ("khong doi sang", "doi sang") if group in table) == 4
    for row in table.values():
        assert all(math.isfinite(row[key]) for key in ("anh vao", "khoi phuc", "oracle chung", "oracle vung"))
        assert 0.0 <= row["worse_than_input"] <= 1.0 and row["brightness_ratio"] > 0
        # The per-region fit has more freedom than the whole-frame one: never worse.
        assert row["oracle vung"] <= row["oracle chung"] + 1e-6
    assert "te hon vao" in result.stdout

