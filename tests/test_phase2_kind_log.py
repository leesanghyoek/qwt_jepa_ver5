"""Phase 2 says which kind of frame it is learning, and scores the flare while it trains.

Pins: corruptions.image.log_kind splits frames into nhe / mo / toi / toi_mo / loe from the corruption's own draws (a
brightened target counts as "toi"); Phase2Trainer.step reports, per kind present in the update, its frame count and
mean L1 of input and of output against the target, the counts adding up to the update's frames; cli._KindMean pools
them frame-weighted; every phase-2 validation of a run with FlareImage scores each valid pair once, the same crops
each time, inside and outside the raw flare, and logs it; a model that hands the input back scores out = in.
"""

from __future__ import annotations

import json

import pytest
import torch
import yaml

from qjepa.cli import _KindMean, _dataset, _train_dataset, flare_pair_bank
from qjepa.corruptions.image import LOG_KINDS, log_kind
from qjepa.evaluation.flare_pairs import FLARE_THRESHOLD, evaluate_flare_pairs, flare_validation_dataset
from test_flare_pairs import _smoke
from test_sharp_cli import _run_until_done

BASE = {"mode": "full", "clean": False, "low_light": False, "sensor_noise": True, "defocus": False, "motion": False}


def test_log_kind_reads_the_corruptions_own_draws():
    cases = (({}, "nhe"), ({"defocus": True}, "mo"), ({"motion": True}, "mo"), ({"low_light": True}, "toi"),
             ({"low_light": True, "defocus": True}, "toi_mo"), ({"fog": True}, "toi"), ({"target_gain": 2.0}, "toi"),
             ({"target_gain": 1.5, "motion": True}, "toi_mo"), ({"flare_pair": {"name": "x"}, "low_light": True}, "loe"),
             ({"lens_flare": True, "defocus": True}, "loe"), ({"clean": True, "defocus": True}, "nhe"),
             ({"mode": "sensor_noise_only", "defocus": True}, "nhe"))
    for change, kind in cases:
        assert log_kind({**BASE, **change}) == kind, change
    assert set(LOG_KINDS) == {kind for _, kind in cases}


def test_kind_mean_pools_frames_not_updates():
    pooled = _KindMean(window=2)
    pooled.add({"kind_loe_count": 1.0, "kind_loe_l1_in": 0.4, "kind_loe_l1_out": 0.2})
    pooled.add({"kind_loe_count": 3.0, "kind_loe_l1_in": 0.0, "kind_loe_l1_out": 0.2, "kind_mo_count": 2.0,
                "kind_mo_l1_in": 0.1, "kind_mo_l1_out": 0.12})
    totals = pooled.totals()
    assert list(totals) == ["mo", "loe"]                         # LOG_KINDS order
    assert totals["loe"] == pytest.approx((4.0, 0.1, 0.2))      # (0.4 + 0) / 4 frames, not (0.4 + 0) / 2 updates
    assert "loe 0.1000->0.2000 (+100%, n=4)" in pooled.text() and "mo 0.1000->0.1200 (+20%, n=2)" in pooled.text()
    pooled.add({})
    pooled.add({})                                               # the window forgot both
    assert pooled.totals() == {} and pooled.text() == ""


class _Identity(torch.nn.Module):
    def forward(self, image, imu, image_time, imu_times):
        return {"image": image}


def test_flare_validation_scores_each_valid_pair_inside_and_outside_its_raw_flare(tmp_path):
    config, manifest, _, _ = _smoke(tmp_path)
    bank = flare_pair_bank(config)
    dataset = flare_validation_dataset(_dataset(config, manifest, "valid", fixed_realization=True, flare_pairs=bank,
                                                flare_layer=True))
    pairs = bank.pairs("valid")
    assert dataset.flare_pairs_split == "valid" and len(pairs) > 0
    item = dataset._flare_item(0)
    assert item["corruption"]["image"]["flare_pair"]["name"] in {pair.name for pair in pairs}
    mask = item["image_flare_target"].abs().mean(0) > FLARE_THRESHOLD
    assert 0 < float(mask.float().mean()) < 1                     # the raw flare, not the blur and grain on top
    first = evaluate_flare_pairs(dataset, _Identity(), torch.device("cpu"), batch_size=3)
    assert first == evaluate_flare_pairs(dataset, _Identity(), torch.device("cpu"), batch_size=5)  # same crops
    assert first["count"] == len(pairs)
    for part in ("psnr", "flare", "rest"):
        assert first[f"{part}_out"] == pytest.approx(first[f"{part}_in"])
    assert first["worse_fraction"] == 0.0 and first["rest_worse_fraction"] == 0.0
    assert first["flare_in"] < first["rest_in"]                   # the flare is where the input is wrong


def test_phase2_logs_the_kinds_of_every_update_and_the_flareimage_validation(tmp_path, capsys):
    torch.set_num_threads(1)
    config, loaded, _, manifest = _smoke(tmp_path, phase1_batch=4, phase2_batch=4)
    config["phase2"].update(gradient_accumulation=2, max_successful_updates=2)
    pairs_per_update = 2 * _train_dataset(config, loaded, "phase2").flare_pairs_per_batch(4)
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "run"
    common = ["--config", str(path), "--manifest", str(manifest), "--output", str(output)]
    _run_until_done(["train-phase1", *common], output / "phase1/last.pt")
    _run_until_done(["train-phase2", *common, "--backbone-checkpoint", str(output / "phase1/last.pt")],
                    output / "phase2/last.pt")
    records = [json.loads(line) for line in (output / "phase2/train.jsonl").read_text().splitlines()]
    updates = [record for record in records if "loss" in record and "successful_updates" in record]
    assert len(updates) == 2
    for record in updates:
        counts = {kind: record.get(f"kind_{kind}_count", 0) for kind in LOG_KINDS}
        assert sum(counts.values()) == 8                          # 4 frames x 2 microbatches
        assert counts["loe"] == pairs_per_update > 0              # the FlareImage slots of both microbatches
        for kind, count in counts.items():
            if count:
                assert record[f"kind_{kind}_l1_in"] >= 0 and record[f"kind_{kind}_l1_out"] >= 0
    validations = [record for record in records if "flare_pairs_validation_count" in record]
    assert validations and all(record["flare_pairs_validation_count"] == len(flare_pair_bank(config).pairs("valid"))
                               for record in validations)
    shown = capsys.readouterr().out
    assert "theo loai anh (TB" in shown and "loe " in shown and "FlareImage valid n=" in shown
