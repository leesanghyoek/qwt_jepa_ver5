"""configs/kaggle_flarefocus.yaml (p40_flarefocus): p38_expnorm with half of every batch FlareImage pairs.

Pins: p40 is p38 with only data.flare_pairs_fraction raised 0.25 -> 0.5 and no lens flare on TartanAir; both hashes
change while p38's stay; phase 1's batch of 32 holds 16 pairs and every phase-2 microbatch of 4 holds 2, at
alternate slots, so each DDP rank's half holds half of them; the CLI's training stream for both phases delivers them.
"""

from __future__ import annotations

import copy

from qjepa.cli import _train_dataset, _training_batch_stream
from qjepa.config import load_config, serializable_config, validate_config
from qjepa.training.checkpoints import configuration_hash
from test_flare_pairs import _Mixed, _smoke


def _configs():
    return load_config("configs/kaggle_expnorm.yaml"), load_config("configs/kaggle_flarefocus.yaml")


def test_kaggle_flarefocus_is_p38_with_only_the_flareimage_fraction_raised_and_changes_both_hashes():
    p38, p40 = (serializable_config(copy.deepcopy(config)) for config in _configs())
    assert p38["data"]["flare_pairs_fraction"] == 0.25 and p40["data"]["flare_pairs_fraction"] == 0.5
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
