"""tools/small_test.py: a small held-out test set written to its own file, and the model on a training batch and on it.

Pins: the TartanAir part spreads over every test trajectory (or over evenly spaced ones when it is smaller), frames
centred in equal stretches of each, deterministically; the plot picks are half FlareImage, half TartanAir; without a
checkpoint the tool only writes the selection file (TartanAir test frames and FlareImage test pairs, none from
train or valid), names each sample by the id the dataset gives it and reads the same file back; with a phase-2
checkpoint it scores the CLI's own training microbatches -- each holding the FlareImage pairs the config asks for --
and every sample of the file, the same numbers on a second run, and refuses a file that no longer matches.
"""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from qjepa.cli import _dataset, _manifest, flare_pair_bank
from qjepa.data.flare_pairs import FlarePairBank
from test_flare_pairs import SHORT_SIDE, _smoke, _train_tiny
from tools.small_test import main, pick, spread


def _samples(lengths):
    return [SimpleNamespace(trajectory_key=f"Env/Data_easy/P{t:03d}", image_index=frame)
            for t, length in enumerate(lengths) for frame in reversed(range(length))]   # out of time order on purpose


def test_spread_covers_every_trajectory_with_centred_frames_and_is_deterministic():
    samples = _samples([20, 7, 30, 12, 9])
    chosen = spread(samples, 12)
    assert chosen == spread(samples, 12) and len(chosen) == len(set(chosen)) == 12
    per_trajectory = {}
    for index in chosen:
        per_trajectory.setdefault(samples[index].trajectory_key, []).append(samples[index].image_index)
    assert sorted(len(frames) for frames in per_trajectory.values()) == [2, 2, 2, 3, 3]   # every trajectory
    assert sorted(per_trajectory["Env/Data_easy/P002"]) == [7, 22]                      # centres of 2 halves of 30
    few = spread(samples, 3)                                                             # fewer than trajectories
    assert sorted(samples[index].trajectory_key[-4:] for index in few) == ["P000", "P002", "P004"]
    assert spread(samples, 0) == []
    assert spread(_samples([2]), 5) == [0, 1]                                            # no more than it has
    rows = [{"kind": "flareimage"}] * 3 + [{"kind": "tartanair"}] * 9
    shown = pick(rows, 6)
    assert sum(rows[k]["kind"] == "flareimage" for k in shown) == 3 and len(shown) == 6
    assert len(pick([{"kind": "tartanair"}] * 9, 4)) == 4


def test_the_selection_file_holds_unseen_samples_and_the_tool_scores_a_training_batch_and_it(tmp_path):
    torch.set_num_threads(1)
    last, manifest_dir = _train_tiny(tmp_path, "flare", True)
    config_path = tmp_path / "flare.yaml"
    selection = tmp_path / "small_test.csv"
    main(["--config", str(config_path), "--manifest", str(manifest_dir), "--selection", str(selection),
          "--tartanair", "5", "--flare", "4"])
    with selection.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    tartan = [row for row in rows if row["kind"] == "tartanair"]
    flare = [row for row in rows if row["kind"] == "flareimage"]
    assert len(flare) == 4 and len({row["name"] for row in flare}) == 4
    config, manifest, _, _ = _smoke(tmp_path)
    tests = manifest["samples"]["test"]
    assert len(tartan) == min(5, len(tests))
    assert {row["sample_id"] for row in tartan} <= {sample.sample_id for sample in tests}
    seen = {(s.environment, s.trajectory_id) for split in ("train", "valid") for s in manifest["samples"][split]}
    assert all((tests[int(row["index"])].environment, tests[int(row["index"])].trajectory_id) not in seen
               for row in tartan)
    bank = FlarePairBank(tmp_path / "flare_dataset", short_side=SHORT_SIDE, holdout_fraction=0.1,
                         cache_dir=tmp_path / "cache")
    assert {row["name"] for row in flare} <= {pair.name for pair in bank.pairs("test")}
    # The ids are those the dataset gives the samples it reads.
    dataset = _dataset(config, manifest, "test", fixed_realization=True, flare_pairs=flare_pair_bank(config))
    dataset.flare_pairs_split = "test"
    for row in rows:
        index = int(row["index"]) + (len(dataset.samples) if row["kind"] == "flareimage" else 0)
        item = dataset[index]
        assert item["sample_id"] == row["sample_id"]
        if row["kind"] == "flareimage":
            assert item["corruption"]["image"]["flare_pair"]["name"] == row["name"]

    reports = []
    for run in ("a", "b"):
        output = tmp_path / f"view_{run}"
        main(["--checkpoint", str(last), "--manifest", str(manifest_dir), "--selection", str(selection),
              "--output", str(output), "--device", "cpu", "--show", "4", "--batch", "3"])
        for name in ("train_batch.png", "train_imu.png", "test_images.png", "test_imu.png"):
            assert (output / name).stat().st_size > 1000
        reports.append(json.loads((output / "report.json").read_text()))
    report = reports[0]
    assert selection.read_text().count("\n") == len(rows) + 1                       # the file was read, not rewritten
    train = report["train"]
    payload = torch.load(last, map_location="cpu", weights_only=False)
    phase2 = payload["config"]["phase2"]
    assert train["phase"] == "phase2" and train["microbatches"] == phase2["gradient_accumulation"]
    assert len(train["rows"]) == phase2["batch_size"] * phase2["gradient_accumulation"]
    assert sum(row["kind"] == "flareimage" for row in train["rows"]) == train["flare_per_microbatch"] * \
        train["microbatches"] and train["flare_per_microbatch"] >= 1
    assert [row["sample_id"] for row in report["test"]["rows"]] == [row["sample_id"] for row in rows]
    for row in train["rows"] + report["test"]["rows"]:
        assert all(np.isfinite(row[key]) for key in ("psnr_in", "psnr_out", "accel_in", "accel_out", "gyro_out"))
    assert report["test"]["summary"]["FlareImage"]["count"] == 4
    assert report["test"]["summary"]["tất cả"]["count"] == len(rows)
    assert reports[1]["test"] == report["test"] and reports[1]["train"] == report["train"]
    # Notebook Cell 12c shuffles the file's rows each run: the same set and numbers, other samples drawn.
    import random
    shuffled_rows = list(rows)
    random.Random(3).shuffle(shuffled_rows)
    shuffled = tmp_path / "shuffled.csv"
    with shuffled.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(shuffled_rows)
    main(["--checkpoint", str(last), "--manifest", str(manifest_dir), "--selection", str(shuffled),
          "--output", str(tmp_path / "view_shuffled"), "--device", "cpu", "--show", "4", "--batch", "3"])
    again = json.loads((tmp_path / "view_shuffled/report.json").read_text())
    for group, values in report["test"]["summary"].items():
        assert again["test"]["summary"][group] == pytest.approx(values, rel=1e-4, abs=1e-6), group
    drawn = lambda order: {order[k]["sample_id"] for k in pick(order, 4)}
    assert drawn(shuffled_rows) != drawn(rows)
    # A file that no longer matches the data is refused, not silently re-chosen.
    stale = tmp_path / "stale.csv"
    stale.write_text(selection.read_text().replace(rows[0]["sample_id"], "Env__Data_easy__P999__000000"))
    with pytest.raises(ValueError, match="khong con khop"):
        main(["--config", str(config_path), "--manifest", str(manifest_dir), "--selection", str(stale)])
