"""p49_glint (configs/kaggle_glint.yaml): p48_recon for the user's drone frames, without the LOL pairs.

The user asked (27/10/2026) to train on drone-glint-flare-640 and the earlier data only, without lowlight-pairs. Pins:
p49 is p48 with only the LOL fraction off, the scene-pair share taking LOL's slots, the flare stage reading more scene
pairs and the output directory changed; it validates, builds no low-light bank, and its batches still hold FlareImage,
scene pairs and enough TartanAir samples; every hash differs from p48's, so nothing of p47/p48 is reused by mistake.
"""

from __future__ import annotations

from qjepa.cli import lowlight_pair_bank
from qjepa.config import load_config, serializable_config, validate_config
from qjepa.training.checkpoints import configuration_hash


def test_p49_is_p48_without_lol_and_with_more_drone_pairs():
    p48 = serializable_config(load_config("configs/kaggle_recon.yaml"))
    p49 = serializable_config(load_config("configs/kaggle_glint.yaml"))
    assert p49["runtime"]["output_dir"].endswith("/p49_glint")
    assert p49["data"]["lowlight_pairs_fraction"] == 0.0
    assert p49["data"]["scene_pairs_fraction"] == 0.375
    assert p49["data"]["scene_pairs_root"].endswith("drone-glint-flare-640")
    assert p49["flare"]["kinds"] == {"composite": 0.35, "pair": 0.15, "plain": 0.15, "scene": 0.35}
    for key in ("lowlight_pairs_fraction", "scene_pairs_fraction", "scene_pairs_root"):
        p49["data"][key] = p48["data"][key]
    p49["flare"]["kinds"] = p48["flare"]["kinds"]
    p49["runtime"]["output_dir"] = p48["runtime"]["output_dir"]
    assert p49 == p48


def test_p49_validates_needs_no_lol_and_keeps_tartanair_in_every_batch():
    config = load_config("configs/kaggle_glint.yaml")
    validate_config(config)
    assert lowlight_pair_bank(config) is None
    data = config["data"]
    for phase, scene, flare in (("phase1", 12, 8), ("phase2", 3, 2)):
        size = config[phase]["batch_size"]
        assert round(data["scene_pairs_fraction"] * size) == scene
        assert round(data["flare_pairs_fraction"] * size) == flare
        assert size - scene - flare >= data["minimum_trajectories_per_batch"]


def test_p49_hashes_differ_from_p48():
    old, new = load_config("configs/kaggle_recon.yaml"), load_config("configs/kaggle_glint.yaml")
    for phase in ("phase1", "phase2", "flare"):
        assert configuration_hash(old, phase) != configuration_hash(new, phase), phase
