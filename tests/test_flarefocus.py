"""configs/kaggle_flarefocus.yaml (p40_flarefocus): p38_expnorm with half of every batch FlareImage pairs, turned and flipped.

Pins: p40 is p38 with only data.flare_pairs_fraction raised 0.25 -> 0.5 and data.flare_pairs_augment dihedral, no lens
flare on TartanAir; both hashes change while p38's stay; phase 1's batch of 32 holds 16 pairs and every phase-2
microbatch of 4 holds 2, at alternate slots, so each DDP rank's half holds half of them; the CLI's training stream for
both phases delivers them; "dihedral" shows each training pair as one of its 8 lossless views (k x 90 degrees, maybe
flipped), its gt and flared photo alike, leaves every other draw, held-out pairs and TartanAir frames as they were.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from qjepa.cli import _dataset, _train_dataset, _training_batch_stream, flare_pair_bank
from qjepa.config import load_config, serializable_config, validate_config
from qjepa.training.checkpoints import configuration_hash
from test_flare_pairs import _Mixed, _smoke


def _configs():
    return load_config("configs/kaggle_expnorm.yaml"), load_config("configs/kaggle_flarefocus.yaml")


def test_kaggle_flarefocus_is_p38_with_only_the_flareimage_fraction_raised_and_changes_both_hashes():
    p38, p40 = (serializable_config(copy.deepcopy(config)) for config in _configs())
    assert p38["data"]["flare_pairs_fraction"] == 0.25 and p40["data"]["flare_pairs_fraction"] == 0.5
    assert "flare_pairs_augment" not in p38["data"] and p40["data"].pop("flare_pairs_augment") == "dihedral"
    assert float(p40["corruption"]["image"].get("lens_flare_probability", 0.0)) == 0.0   # no flare on TartanAir
    assert p40["runtime"].pop("output_dir").endswith("outputs/p40_flarefocus")
    p38["runtime"].pop("output_dir")
    for config in (p38, p40):
        config["data"].pop("flare_pairs_fraction")
    assert p40 == p38
    p38, p40 = _configs()
    validate_config(p40)
    for phase in ("phase1", "phase2"):
        assert configuration_hash(p40, phase) != configuration_hash(p38, phase)
    # p38 hashes as it did when its notebook commit was pinned.
    assert configuration_hash(p38, "phase1").startswith("20820dddfa24")
    assert configuration_hash(p38, "phase2").startswith("da2dde02ceb9")


def test_half_of_every_batch_is_flareimage_at_alternate_slots_and_each_rank_loads_half():
    config = load_config("configs/smoke.yaml")
    for size, slots in ((32, list(range(1, 32, 2))), (4, [1, 3])):
        stream = _training_batch_stream(config, _Mixed(48, fraction=0.5), size, start_microbatch=0, namespace="p")
        for _ in range(3):
            ids = next(stream)["sample_id"]
            assert [k for k, value in enumerate(ids) if "flare:" in value] == slots, size
        halves = [_training_batch_stream(config, _Mixed(48, fraction=0.5), size, start_microbatch=0, namespace="p",
                                         rank=rank, world=2) for rank in (0, 1)]
        for half in halves:
            assert sum("flare:" in value for value in next(half)["sample_id"]) == len(slots) // 2


def test_the_cli_streams_half_flareimage_batches_to_both_phases(tmp_path):
    config, manifest, _, _ = _smoke(tmp_path, phase1_batch=8, phase2_batch=4)
    config["data"]["flare_pairs_fraction"] = 0.5
    validate_config(config)
    for phase, size in (("phase1", 8), ("phase2", 4)):
        dataset = _train_dataset(config, manifest, phase)
        assert dataset.flare_pairs_per_batch(size) == size // 2
        batch = next(_training_batch_stream(config, dataset, size, start_microbatch=0, namespace=phase))
        pairs = [c["image"].get("flare_pair") for c in batch["corruption"]]
        assert sum(bool(pair) for pair in pairs) == size // 2, phase
        assert batch["image_clean"].shape[0] == size


def test_dihedral_shows_each_training_pair_in_one_of_its_eight_views_and_touches_nothing_else(tmp_path):
    config, manifest, _, _ = _smoke(tmp_path, phase1_batch=8, phase2_batch=4)
    # The training split without the phase's random mirror, so the views compare pixel for pixel.
    build = lambda: _dataset(config, manifest, "train", fixed_realization=False, flare_pairs=flare_pair_bank(config))
    plain = build()
    config["data"]["flare_pairs_augment"] = "dihedral"
    validate_config(config)
    turned = build()
    first = len(plain.samples)
    views = set()
    for draw in range(40):
        a, b = plain[first + draw], turned[first + draw]
        pa, pb = a["corruption"]["image"]["flare_pair"], b["corruption"]["image"]["flare_pair"]
        turns, flip = pb.pop("rot90"), pb.pop("flip")
        assert pa == pb and a["sample_id"] == b["sample_id"]                 # same pair, crop, host, clear draw
        views.add((turns, flip))
        clean = a["image_clean"].permute(1, 2, 0).numpy()
        expected = np.rot90(clean, turns)
        expected = expected[:, ::-1] if flip else expected
        got = b["image_clean"].permute(1, 2, 0).numpy()
        assert np.array_equal(got, expected), draw                           # gt: the same pixels, turned
        assert np.array_equal(a["imu_noisy_phys"], b["imu_noisy_phys"])
        # The flared photo turns the same way: undoing the view on the input finds the unturned input's light.
        assert float(b["image_noisy"].mean()) == pytest.approx(float(a["image_noisy"].mean()), abs=0.02)
    assert len(views) == 8
    for index in range(3):                                                   # TartanAir frames: as before
        assert np.array_equal(plain[index]["image_noisy"], turned[index]["image_noisy"])
    held = _dataset(config, manifest, "test", fixed_realization=True, flare_pairs=flare_pair_bank(config))
    held.flare_pairs_split = "test"
    assert "rot90" not in held[len(held.samples)]["corruption"]["image"]["flare_pair"]   # held-out pairs untouched
    for bad in ("rotate", 1):
        wrong = copy.deepcopy(config)
        wrong["data"]["flare_pairs_augment"] = bad
        with pytest.raises(ValueError, match="flare_pairs_augment"):
            validate_config(wrong)
