"""tools/flare_image_test.py: FlareImage's real flared photos through a phase-2 checkpoint, pair by pair.

Pins: the pairs come only from the asked split, spread over the sources, or by name (a name outside the split is
refused); a full frame keeps the shrunk pair cut to multiples of 16 around its centre and a crop is the training crop
size; the model runs on whole frames of any such size; "chi loe" feeds the photo itself and "nhu luc train" adds the
training corruption; each pair gets a panel and finite numbers, the flare region is where input and target differ.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from qjepa.data.flare_pairs import FlarePairBank
from test_flare_pairs import SHORT_SIDE, _train_tiny
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
