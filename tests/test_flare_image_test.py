"""tools/flare_image_test.py: FlareImage's real flared photos through a phase-2 checkpoint, pair by pair.

Pins: the pairs come only from the asked split, spread over the sources, or by name (a name outside the split is
refused); a full frame keeps the shrunk pair cut to multiples of 16 around its centre and a crop is the training crop
size; the model runs on whole frames of any such size; "chi loe" feeds the photo itself and "nhu luc train" adds the
training corruption; each pair gets a panel and finite numbers, the flare region is where input and target differ.
--bank visdrone (p46) reads the "Lens flare on VisDrone" pairs' valid split (it has no test), --count 0 every pair of
it, --imu khong the same frames without IMU, --panels N draws only N pairs from the worst to the best.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from qjepa.data.flare_pairs import FlarePairBank
from test_flare_pairs import SHORT_SIDE, _train_tiny
from test_scene_pairs import _write_scene_pairs
from tools.flare_image_test import frame_of, main, pick_pairs


def test_frames_are_cut_to_multiples_of_16_around_the_centre_or_cropped_to_the_training_size():
    array = np.arange(50 * 70 * 3).reshape(50, 70, 3)
    full = frame_of(array, "full", (32, 32))
    assert full.shape == (48, 64, 3) and full[0, 0, 0] == array[1, 3, 0]
    crop = frame_of(array, "crop", (32, 32))
    assert crop.shape == (32, 32, 3) and crop[0, 0, 0] == array[9, 19, 0]
    with pytest.raises(ValueError, match="--frame full"):
        frame_of(array, "crop", (64, 64))


def test_a_checkpoint_restores_whole_held_out_frames_pair_by_pair(tmp_path):
    torch.set_num_threads(1)
    last, manifest = _train_tiny(tmp_path, "flare", True)
    bank = FlarePairBank(tmp_path / "flare_dataset", short_side=SHORT_SIDE, holdout_fraction=0.1,
                         cache_dir=tmp_path / "cache")
    test_names = {pair.name for pair in bank.pairs("test")}
    picked = pick_pairs(bank, "test", 4)
    assert len(picked) == 4 and len({pair.source for pair in picked}) == 4      # one per source
    assert {pair.name for pair in picked} <= test_names
    reports = {}
    for mode in ("chi_loe", "nhu_luc_train"):
        output = tmp_path / mode
        reports[mode] = main(["--checkpoint", str(last), "--manifest", str(manifest), "--output", str(output),
                              "--count", "4", "--mode", mode, "--device", "cpu"])
        rows = reports[mode]["rows"]
        assert [row["name"] for row in rows] == [pair.name for pair in picked]
        for row in rows:
            assert row["width"] % 16 == 0 and row["height"] % 16 == 0
            assert np.isfinite(row["psnr_in"]) and np.isfinite(row["psnr_out"]) and 0 <= row["flare_fraction"] <= 1
            assert (output / row["panel"]).stat().st_size > 1000
            # Inside and outside the flare split the frame's squared error by their pixel shares.
            if 0 < row["flare_fraction"] < 1:
                mse = lambda db: 10 ** (-db / 10)
                for side in ("in", "out"):
                    whole = (row["flare_fraction"] * mse(row[f"flare_psnr_{side}"])
                             + (1 - row["flare_fraction"]) * mse(row[f"rest_psnr_{side}"]))
                    assert whole == pytest.approx(mse(row[f"psnr_{side}"]), rel=1e-3)
        assert json.loads((output / "report.json").read_text())["summary"]["tat ca"]["count"] == 4
    # "chi loe" feeds the photo itself; the training corruption changes the input.
    assert any(a["psnr_in"] != b["psnr_in"] for a, b in zip(reports["chi_loe"]["rows"],
                                                            reports["nhu_luc_train"]["rows"]))
    named = main(["--checkpoint", str(last), "--manifest", str(manifest), "--output", str(tmp_path / "named"),
                  "--names", picked[1].name, "--device", "cpu"])
    assert [row["name"] for row in named["rows"]] == [picked[1].name]
    train_name = bank.pairs("train")[0].name
    with pytest.raises(ValueError, match="chi cac cap cua split"):
        main(["--checkpoint", str(last), "--manifest", str(manifest), "--output", str(tmp_path / "refused"),
              "--names", train_name, "--device", "cpu"])


def test_visdrone_valid_pairs_all_of_them_with_and_without_imu(tmp_path):
    torch.set_num_threads(1)
    last, manifest = _train_tiny(tmp_path, "flare", True)          # trained without scene pairs: needs the root
    root = _write_scene_pairs(tmp_path / "visdrone", train=6, valid=5, size=(48, 64))
    common = ["--checkpoint", str(last), "--manifest", str(manifest), "--bank", "visdrone",
              "--scene-pairs-root", str(root), "--count", "0", "--device", "cpu"]
    reports = {imu: main([*common, "--output", str(tmp_path / imu), "--imu", imu]) for imu in ("muon", "khong")}
    for imu, report in reports.items():
        assert report["bank"] == "visdrone" and report["split"] == "valid" and report["imu"] == imu
        rows = report["rows"]
        assert [row["name"] for row in rows] == [f"{index:05d}.png" for index in range(6, 11)]   # every valid pair
        for row in rows:
            assert (row["width"], row["height"]) == (64, 48)
            # chi loe: the input is the flared photo, the flare region its 10 x 10 px blob.
            assert row["flare_fraction"] == pytest.approx(100 / (48 * 64))
            assert np.isfinite(row["psnr_out"]) and np.isfinite(row["flare_psnr_out"])
            assert (tmp_path / imu / row["panel"]).stat().st_size > 1000
    with_imu, without = reports["muon"]["rows"], reports["khong"]["rows"]
    assert [row["psnr_in"] for row in with_imu] == [row["psnr_in"] for row in without]
    assert any(a["psnr_out"] != b["psnr_out"] for a, b in zip(with_imu, without))      # the IMU window changed
    with pytest.raises(SystemExit):
        main([*common, "--output", str(tmp_path / "test"), "--split", "test"])        # VisDrone has no test split
    # --panels 2: every pair scored, only the worst and the best (by PSNR gain) drawn.
    few = main([*common, "--output", str(tmp_path / "few"), "--panels", "2"])["rows"]
    assert [row["psnr_out"] for row in few] == [row["psnr_out"] for row in with_imu]
    gains = [row["psnr_out"] - row["psnr_in"] for row in few]
    drawn = [k for k, row in enumerate(few) if row["panel"]]
    assert drawn == sorted({gains.index(min(gains)), gains.index(max(gains))})
    assert sorted(path.name for path in (tmp_path / "few").glob("pair_*.png")) == [few[k]["panel"] for k in drawn]
