"""p49_glint (configs/kaggle_glint.yaml): p48_recon for the user's drone frames, without the LOL pairs, split 90 / 5 / 5.

The user asked (27/10/2026) to train on drone-glint-flare-640 and the earlier data only, without lowlight-pairs, and
then to keep most data for training with only 10% to check the model. Pins:
- p49 is p48 with only the LOL fraction off, the scene-pair share taking LOL's slots, the flare stage reading more scene
  pairs, the four split keys and the output directory changed; it validates, builds no low-light bank, and its batches
  still hold FlareImage, scene pairs and enough TartanAir samples; every hash differs from p48's;
- data.split_ratios gives TartanAir's motion groups that train / valid / test share and the manifest records it; a
  manifest split otherwise is refused, and a manifest from before the key reads as 0.8 / 0.1 / 0.1;
- data.scene_pairs_holdout_fraction splits the scene pairs train / valid / test: held-out pairs come only from the
  dataset's val pairs, whole background groups at a time (a VisDrone sequence, a Flickr photo), about the fraction of
  each origin for test and for valid; no train pair shares a flare layer with a held-out pair, and every train pair of
  index.csv stays in train (a val group using a train pair's flare layer is never held out); a dataset whose val pairs
  all share the train pairs' flare layers is refused; without the key the bank is the index.csv split as before;
- bad values of both keys are refused.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from qjepa import cli
from qjepa.cli import lowlight_pair_bank, scene_pair_bank
from qjepa.config import load_config, serializable_config, validate_config
from qjepa.data.manifest import DEFAULT_SPLIT_RATIOS, assign_splits
from qjepa.data.scene_pairs import ScenePairBank, scene_group, scene_origin
from qjepa.data.tartanair import Trajectory
from qjepa.training.checkpoints import configuration_hash

SPLIT_KEYS = ("split_ratios", "flare_pairs_holdout_fraction", "scene_pairs_holdout_fraction")


def test_p49_is_p48_without_lol_with_more_drone_pairs_and_a_90_5_5_split():
    p48 = serializable_config(load_config("configs/kaggle_recon.yaml"))
    p49 = serializable_config(load_config("configs/kaggle_glint.yaml"))
    assert p49["runtime"]["output_dir"].endswith("/p49_glint")
    assert p49["data"]["lowlight_pairs_fraction"] == 0.0
    assert p49["data"]["scene_pairs_fraction"] == 0.375
    assert p49["data"]["scene_pairs_root"].endswith("drone-glint-flare-640")
    assert p49["data"]["split_ratios"] == [0.9, 0.05, 0.05]
    assert p49["data"]["flare_pairs_holdout_fraction"] == 0.05
    assert p49["data"]["scene_pairs_holdout_fraction"] == 0.05
    assert p49["flare"]["lens"]["lens_flare_holdout_fraction"] == 0.05
    assert p49["flare"]["kinds"] == {"composite": 0.35, "pair": 0.15, "plain": 0.15, "scene": 0.35}
    for key in ("lowlight_pairs_fraction", "scene_pairs_fraction", "scene_pairs_root", "flare_pairs_holdout_fraction"):
        p49["data"][key] = p48["data"][key]
    for key in ("split_ratios", "scene_pairs_holdout_fraction"):
        assert key not in p48["data"]
        del p49["data"][key]
    p49["flare"]["kinds"] = p48["flare"]["kinds"]
    p49["flare"]["lens"]["lens_flare_holdout_fraction"] = p48["flare"]["lens"]["lens_flare_holdout_fraction"]
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


def _trajectories(count: int) -> list[Trajectory]:
    return [Trajectory(Path(f"/x/E{k % 7}/Data_easy/P{k:03d}"), f"E{k % 7}", "Data_easy", f"P{k:03d}")
            for k in range(count)]


def test_split_ratios_give_tartanair_90_5_5_and_the_default_is_as_before():
    trajectories = _trajectories(200)
    split = assign_splits(trajectories, (0.9, 0.05, 0.05), seed=1)
    counts = {name: sum(value == name for value in split.values()) for name in ("train", "valid", "test")}
    assert counts == {"train": 180, "valid": 10, "test": 10}
    assert assign_splits(trajectories, seed=1) == assign_splits(trajectories, DEFAULT_SPLIT_RATIOS, seed=1)
    old = {name: sum(value == name for value in assign_splits(trajectories, seed=1).values())
           for name in ("train", "valid", "test")}
    assert old == {"train": 160, "valid": 20, "test": 20}


def test_the_cli_refuses_a_manifest_split_otherwise(monkeypatch):
    config = load_config("configs/kaggle_glint.yaml")
    plain = load_config("configs/kaggle_recon.yaml")
    old_manifest = {"meta": {}, "samples": {}}                                  # before data.split_ratios existed
    new_manifest = {"meta": {"split_ratios": [0.9, 0.05, 0.05]}, "samples": {}}
    for manifest, good, bad in ((old_manifest, plain, config), (new_manifest, config, plain)):
        monkeypatch.setattr(cli, "read_manifest", lambda location, manifest=manifest: manifest)
        assert cli._manifest(good, "/m")[0] is manifest
        with pytest.raises(ValueError, match="split_ratios"):
            cli._manifest(bad, "/m")


def _scene_root(tmp_path: Path) -> Path:
    """index.csv like wu-visdrone-composite: Flickr photos (every 10th val), VisDrone train and test-dev sequences;
    train and val pairs use disjoint flare layers (as the builder does). Val pairs mostly have their own layer, but
    every fifth frame of a test-dev sequence shares one with the same frame of the other test-dev sequences."""
    root = tmp_path / "pairs"
    (root / "gt").mkdir(parents=True)
    (root / "input").mkdir()
    rows = []
    for k in range(400):                                       # 400 Flickr pairs, 200 photos, val = photo % 10 == 0
        photo = k // 2
        val = photo % 10 == 0
        rows.append({"split": "val" if val else "train", "scene": f"Flickr24K__img_{photo}.jpg",
                     "scene_split": "flickr24k", "flare": f"glint/aperture{k % 11}_v.png" if val else f"glint/a{k}.png"})
    for sequence in range(12):                                 # 8 train sequences, 4 test-dev, 25 frames each
        for frame in range(25):
            split = "val" if sequence >= 8 else "train"
            flare = (f"captured/{sequence * 25 + frame:06d}t.png" if split == "train" else
                     f"captured/shared{frame}.png" if frame % 5 == 0 else f"captured/{sequence * 25 + frame:06d}v.png")
            rows.append({"split": split, "scene": f"{sequence:07d}_00000_d_{frame:07d}.jpg",
                         "scene_split": "test-dev" if split == "val" else "train", "flare": flare})
    with (root / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "split", "scene", "scene_split", "flare", "seed"])
        writer.writeheader()
        for index, row in enumerate(rows):
            name = f"{index:05d}.png"
            writer.writerow({"file": name, **row, "seed": index})
            (root / "gt" / name).touch()
            (root / "input" / name).touch()
    return root


def test_scene_pairs_holdout_splits_by_group_from_val_only(tmp_path):
    root = _scene_root(tmp_path)
    with (root / "index.csv").open(newline="") as handle:
        rows = {row["file"]: row for row in csv.DictReader(handle)}
    plain = ScenePairBank(root, short_side=256)
    assert set(plain.splits) == {"train", "valid"}                      # no key: the index.csv split, as before
    assert {pair.name for pair in plain.pairs("valid")} == {f for f, r in rows.items() if r["split"] == "val"}

    bank = ScenePairBank(root, short_side=256, holdout_fraction=0.05)
    names = {split: {pair.name for pair in bank.pairs(split)} for split in ("train", "valid", "test")}
    held = names["valid"] | names["test"]
    assert all(rows[name]["split"] == "val" for name in held)          # held-out only from the dataset's val pairs
    groups = {split: {scene_group(rows[name]) for name in members} for split, members in names.items()}
    assert groups["train"].isdisjoint(held_groups := groups["valid"] | groups["test"])
    assert groups["valid"].isdisjoint(groups["test"])
    for origin in ("flickr", "drone"):
        total = sum(scene_origin(row) == origin for row in rows.values())
        for split in ("valid", "test"):
            share = sum(scene_origin(rows[name]) == origin for name in names[split]) / total
            assert 0.05 <= share < 0.15, (origin, split, share)       # whole groups: at least 5%, one group over
    held_flares = {rows[name]["flare"] for name in held}
    assert not any(rows[name]["flare"] in held_flares for name in names["train"])
    kept = set().union(*names.values())
    assert bank.dropped == len(rows) - len(kept)
    assert names["train"] >= {pair.name for pair in plain.pairs("train")}    # the dataset's train pairs all stay
    assert len(names["train"]) > len(plain.pairs("train"))              # val pairs left over go back to train
    assert held_groups and bank.describe()["holdout_fraction"] == 0.05


def test_held_out_groups_never_use_a_train_flare_layer(tmp_path):
    root = _scene_root(tmp_path)
    with (root / "index.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    # Two test-dev sequences now share their flare layers with train pairs: they can no longer be held out.
    train_flares = [row["flare"] for row in rows if row["split"] == "train"]
    for index, row in enumerate(rows):
        if row["split"] == "val" and row["scene"].startswith(("0000008", "0000009")):
            row["flare"] = train_flares[index % len(train_flares)]
    with (root / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    bank = ScenePairBank(root, short_side=256, holdout_fraction=0.05)
    held = {pair.scene.split("_")[0] for split in ("valid", "test") for pair in bank.pairs(split)
            if not pair.scene.startswith("Flickr")}
    assert held and held.isdisjoint({"0000008", "0000009"})
    assert {pair.name for pair in bank.pairs("train")} >= {row["file"] for row in rows if row["split"] == "train"}
    for row in rows:                                              # every val pair shares a train layer: refused
        if row["split"] == "val":
            row["flare"] = train_flares[0]
    with (root / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="khong co cap nao"):
        ScenePairBank(root, short_side=256, holdout_fraction=0.05)


def test_scene_pair_bank_reads_the_key_and_bad_values_are_refused(tmp_path):
    root = _scene_root(tmp_path)
    config = load_config("configs/kaggle_glint.yaml")
    config["data"]["scene_pairs_root"] = str(root)
    assert set(scene_pair_bank(config).splits) == {"train", "valid", "test"}
    for key, value in (("scene_pairs_holdout_fraction", 0.3), ("scene_pairs_holdout_fraction", 0),
                       ("split_ratios", [0.9, 0.1, 0.0]), ("split_ratios", [0.9, 0.05]),
                       ("split_ratios", [0.8, 0.1, 0.05])):
        bad = load_config("configs/kaggle_glint.yaml")
        bad["data"][key] = value
        with pytest.raises(ValueError, match=key):
            validate_config(bad)
