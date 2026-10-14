"""p42: the flare stage (train-flare) -- the flare layer trained alone on Flare7K++ flares composited afresh for every
sample, then loaded and frozen in phase 2.

p36-p41 taught flare with ~840 FlareImage pairs on repeat; every run took only ~1/4-1/3 of the flare off, even on the
pairs it trained on. The flare stage draws a new flare for every sample, as Flare7K++ trains.

Pins: a composite is the background plus the flare in linear light, its target the background plus the light source
(blurred alike), and the layer's target the input before grain minus that target; a pair's layer is its flared crop
minus its clean crop; a plain frame is phase 2's own corrupted input with a zero layer; every sample is a pure function
of its index, and the three kinds come in their configured shares; train-flare trains, validates on composites and on
held-out FlareImage pairs, and an interrupted run resumed from its checkpoint ends with the very weights of one that
ran straight through; phase 2 loads the stage's layer and leaves it untouched (out of the optimizer, its hash in the
checkpoint) and refuses a missing or unfinished stage; p42 is p41 plus the flare section and its split_flare_* keys,
with p40's phase-1 hash, p40's and p41's hashes unchanged; bad settings fail.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
import yaml

from qjepa.cli import _dataset, flare_pair_bank, main
from qjepa.config import load_config, serializable_config, validate_config
from qjepa.corruptions.lens_flare import LensFlareBank
from qjepa.corruptions.light import linear_to_srgb, srgb_to_linear
from qjepa.data.flare_stage import FlareStageDataset
from qjepa.training.checkpoints import configuration_hash, load_checkpoint, state_dict_hash
from test_flare_pairs import _smoke
from test_ijepa import _run_until_done
from test_lens_flare import _write_flare7k

P42 = load_config("configs/kaggle_flarestage.yaml")


def _stage(tmp_path, **change):
    """Smoke recipe (32x32 frames) with FlareImage pairs, a tiny Flare7K++ folder and a tiny flare section."""
    config, manifest, _, manifest_dir = _smoke(tmp_path, phase1_batch=4, phase2_batch=4)
    root = tmp_path / "f7k"
    if not root.exists():
        _write_flare7k(root)
    config["data"]["lens_flare_root"] = str(root)
    stage = copy.deepcopy(P42["flare"])
    stage.update(widths=[8, 8, 8], blocks=1, batch_size=4, max_successful_updates=4, warmup_updates=1,
                 validation_every_updates=2, validation_composites=6, checkpoint_every_updates=2,
                 precision="fp32")
    stage.update(change)
    config["flare"] = stage
    validate_config(config)
    return config, manifest, manifest_dir


def _dataset_of(config, manifest, split="train", length=40, **section):
    stage = dict(config["flare"], **section)
    bank = LensFlareBank(config["data"]["lens_flare_root"], holdout_fraction=0.1)
    tartan = _dataset(config, manifest, split, fixed_realization=split != "train")
    return FlareStageDataset(tartan, lens_bank=bank, flare_pairs=flare_pair_bank(config), section=stage, split=split,
                             length=length, seed=int(stage["seed"]))


def test_each_kind_builds_its_input_target_and_layer(tmp_path):
    config, manifest, _ = _stage(tmp_path)
    # Without grain the input is exactly the target plus the layer, for every kind.
    clean = _dataset_of(config, manifest, grain_probability=0.0)
    seen = set()
    for index in range(40):
        item = clean[index]
        kind = item["info"]["kind"]
        seen.add(kind)
        assert torch.allclose(item["image_noisy"], item["image_target"] + item["flare_target"], atol=1e-6)
        if kind == "plain":
            assert float(item["flare_target"].abs().max()) == 0.0
            assert torch.equal(item["image_noisy"], item["image_target"])
        else:
            assert float(item["flare_target"].abs().max()) > 0.01, (index, kind)
    assert seen == {"composite", "pair", "plain"}
    # A composite: the background plus the flare's light, the target the background plus the source alone.
    from qjepa.corruptions.lens_flare import draw_lens_parameters, lens_flare_layers
    sharp = _dataset_of(config, manifest, grain_probability=0.0, blur_probability=0.0, background_pair_fraction=0.0,
                        kinds={"composite": 1.0, "pair": 0.0, "plain": 0.0})
    item = sharp[3]
    rng = sharp._rng(3)
    rng.choice(3, p=sharp.weights)
    sharp._camera(rng)
    background, frame = sharp._background(rng)
    assert frame == item["info"]["frame"]
    params = draw_lens_parameters(rng, sharp.lens, sharp.lens_bank, "train")
    flare, light = lens_flare_layers(sharp.lens_bank, params, 32, 32)
    expected_in = np.clip(linear_to_srgb(srgb_to_linear(background) + flare), 0, 1)
    expected_target = np.clip(linear_to_srgb(srgb_to_linear(background) + light), 0, 1)
    assert np.allclose(item["image_noisy"].permute(1, 2, 0).numpy(), expected_in, atol=1e-6)
    assert np.allclose(item["image_target"].permute(1, 2, 0).numpy(), expected_target, atol=1e-6)
    # With grain only the input moves: the layer's target is the frame before grain minus the target.
    grainy = _dataset_of(config, manifest, grain_probability=1.0)
    item = grainy[0]
    if item["info"]["kind"] != "plain":
        assert not torch.allclose(item["image_noisy"], item["image_target"] + item["flare_target"], atol=1e-3)


def test_samples_are_a_function_of_their_index_and_kinds_follow_their_shares(tmp_path):
    config, manifest, _ = _stage(tmp_path)
    data = _dataset_of(config, manifest, length=600)
    first, again = data[7], data[7]
    for key in ("image_noisy", "image_target", "flare_target"):
        assert torch.equal(first[key], again[key])
    assert not torch.equal(data[8]["image_noisy"], first["image_noisy"])
    kinds = [data.kind_of(index) for index in range(600)]
    shares = {kind: kinds.count(kind) / 600 for kind in ("composite", "pair", "plain")}
    assert shares["composite"] == pytest.approx(0.6, abs=0.06)
    assert shares["pair"] == pytest.approx(0.2, abs=0.05) and shares["plain"] == pytest.approx(0.2, abs=0.05)
    # Validation composites draw only the held-out flare layers, and pairs only the held-out FlareImage pairs.
    valid = _dataset_of(config, manifest, split="valid", length=30)
    held_pairs = {pair.name for pair in valid.flare_pairs.pairs("valid")}
    for k in range(30):
        info = valid[k]["info"]
        if info["kind"] == "composite":
            assert info["layer"] in valid.lens_bank.splits["valid"][info["source"]]
        elif info["kind"] == "pair":
            assert info["pair"] in held_pairs


def _train_flare(config, tmp_path, name):
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(serializable_config(copy.deepcopy(config))))
    output = tmp_path / name
    return path, output


def test_train_flare_trains_validates_and_a_resumed_run_ends_where_a_straight_one_does(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config, _, manifest_dir = _stage(tmp_path)
    path, straight = _train_flare(config, tmp_path, "straight")
    main(["train-flare", "--config", str(path), "--manifest", str(manifest_dir), "--output", str(straight)])
    final = load_checkpoint(straight / "flare/last.pt")
    assert final["metadata"]["phase"] == "flare_stage" and final["successful_updates"] == 4
    assert final["metadata"]["configuration_hash"] == configuration_hash(config, "flare")
    rows = [json.loads(line) for line in (straight / "flare/train.jsonl").read_text().splitlines()]
    updates = [row for row in rows if "loss" in row]
    checks = [row["validation"] for row in rows if "validation" in row]
    assert [row["successful_updates"] for row in updates] == [1, 2, 3, 4] and len(checks) == 2
    assert checks[-1]["real_pairs"] > 0 and np.isfinite(checks[-1]["real_flare_in"])
    assert np.isfinite(checks[-1]["composite_psnr_out"]) and 0 < checks[-1]["composite_flare_left"]
    # Interrupted after update 2 (its checkpoint), then resumed: the same weights as the straight run.
    from qjepa.training import flare_stage
    original = flare_stage.FlareStageTrainer.step

    def stop_after_two(self, batch):
        if self.successful_updates == 2:
            raise RuntimeError("interrupted")
        return original(self, batch)

    path, broken = _train_flare(config, tmp_path, "broken")
    monkeypatch.setattr(flare_stage.FlareStageTrainer, "step", stop_after_two)
    with pytest.raises(SystemExit):
        main(["train-flare", "--config", str(path), "--manifest", str(manifest_dir), "--output", str(broken)])
    assert load_checkpoint(broken / "flare/last.pt")["successful_updates"] == 2
    monkeypatch.setattr(flare_stage.FlareStageTrainer, "step", original)
    main(["train-flare", "--config", str(path), "--manifest", str(manifest_dir), "--output", str(broken),
          "--resume", str(broken / "flare/last.pt")])
    resumed = load_checkpoint(broken / "flare/last.pt")
    for key, value in final["model"].items():
        assert torch.equal(value, resumed["model"][key]), key
    logged = [json.loads(line)["successful_updates"] for line in (broken / "flare/train.jsonl").read_text().splitlines()
              if "loss" in json.loads(line)]
    assert logged == [1, 2, 3, 4]


def test_phase2_loads_the_stage_layer_and_never_moves_it(tmp_path):
    torch.set_num_threads(1)
    config, _, manifest_dir = _stage(tmp_path)
    config["phase2"].update(split_flare_layer=True, split_flare_widths=[8, 8, 8], split_flare_blocks=1,
                            split_flare_weight=0.0, split_flare_latent=False, split_flare_pretrained=True,
                            split_flare_freeze=True)
    validate_config(config)
    path, run = _train_flare(config, tmp_path, "run")
    common = ["--config", str(path), "--manifest", str(manifest_dir), "--output", str(run)]
    _run_until_done(["train-phase1", *common], run / "phase1/last.pt")
    phase2 = ["train-phase2", *common, "--backbone-checkpoint", str(run / "phase1/last.pt")]
    with pytest.raises(SystemExit):                                   # no flare stage yet
        main(phase2)
    main(["train-flare", *common])
    stage = load_checkpoint(run / "flare/last.pt")
    _run_until_done(phase2, run / "phase2/last.pt")
    final = load_checkpoint(run / "phase2/last.pt")
    assert final["successful_updates"] == 3
    for key, value in stage["model"].items():
        assert torch.equal(final["system"][f"decoders.image.flare.{key}"], value), key
    assert final["metadata"]["frozen_flare_hash"] == stage["metadata"]["layer_hash"]
    # An unfinished stage is refused.
    longer = copy.deepcopy(config)
    longer["flare"]["max_successful_updates"] = 6
    path.write_text(yaml.safe_dump(serializable_config(longer)))
    import shutil
    shutil.rmtree(run / "phase2")
    with pytest.raises(SystemExit):
        main(phase2)


def test_kaggle_flarestage_is_p41_plus_the_stage_with_p40s_phase1_hash():
    p40, p41 = load_config("configs/kaggle_flarefocus.yaml"), load_config("configs/kaggle_flarelayer.yaml")
    validate_config(P42)
    assert configuration_hash(P42, "phase1") == configuration_hash(p40, "phase1")
    assert configuration_hash(p41, "phase1").startswith("d4d03229be4b")
    assert configuration_hash(p41, "phase2").startswith("9d7ab2fdf850")          # p41 as trained
    assert configuration_hash(p40, "phase2").startswith("55cab3f2f5ce")
    assert configuration_hash(P42, "phase2") != configuration_hash(p41, "phase2")
    a, b = (serializable_config(copy.deepcopy(config)) for config in (p41, P42))
    assert b["runtime"].pop("output_dir").endswith("outputs/p42_flarestage")
    a["runtime"].pop("output_dir")
    assert b["data"].pop("lens_flare_root")
    b.pop("flare")
    for key, value in {"split_flare_latent": False, "split_flare_pretrained": True, "split_flare_freeze": True,
                       "split_flare_widths": [32, 64, 96, 128, 192], "split_flare_blocks": 2,
                       "split_flare_weight": 0.0}.items():
        assert b["phase2"].pop(key) == value
        a["phase2"].pop(key, None)
    assert a == b
    # The stage's recipe is part of phase 2's hash; a path is not.
    moved = copy.deepcopy(P42)
    moved["data"]["lens_flare_root"] = "/elsewhere"
    assert configuration_hash(moved, "phase2") == configuration_hash(P42, "phase2")
    faster = copy.deepcopy(P42)
    faster["flare"]["learning_rate"] = 1e-3
    assert configuration_hash(faster, "phase2") != configuration_hash(P42, "phase2")


@pytest.mark.parametrize("change, message", [
    (lambda c: c["flare"].pop("seed"), "missing: seed"),
    (lambda c: c["flare"].update(kinds={"composite": 1.0}), "flare.kinds"),
    (lambda c: c["flare"].update(precision="bf16"), "precision"),
    (lambda c: c["flare"]["lens"].update(lens_flare_probability_typo=1), "lens_"),
    (lambda c: c["data"].pop("lens_flare_root"), "lens_flare_root"),
    (lambda c: c["phase2"].update(split_flare_latent=True), "split_flare_latent: false"),
    (lambda c: c["phase2"].update(split_flare_blocks=1), "must equal flare"),
    (lambda c: c.pop("flare"), "add the flare section"),
    (lambda c: c["phase2"].update(split_flare_pretrained=False), "set split_flare_pretrained"),
])
def test_bad_settings_fail(change, message):
    config = copy.deepcopy(P42)
    change(config)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_the_image_only_layer_runs_in_fp32_under_autocast_and_train_flare_stops_on_repeated_nan(tmp_path, monkeypatch):
    # p42 on Kaggle: under fp16 the layer overflowed after ~2 300 updates and every later batch was NaN.
    from qjepa.training.flare_stage import build_flare_layer
    layer = build_flare_layer({"widths": [8, 8, 8], "blocks": 1})
    image = torch.rand(2, 3, 32, 32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out, flare = layer(None, image)
    assert flare.dtype == torch.float32 and out.dtype == image.dtype
    torch.set_num_threads(1)
    config, _, manifest_dir = _stage(tmp_path, max_successful_updates=8)
    path, run = _train_flare(config, tmp_path, "nan")
    from qjepa import cli
    from qjepa.training import flare_stage
    monkeypatch.setattr(cli, "NONFINITE_LIMIT", 3)
    monkeypatch.setattr(flare_stage.FlareStageTrainer, "_forward",
                        lambda self, image: torch.full_like(image, float("nan")))
    with pytest.raises(SystemExit):
        main(["train-flare", "--config", str(path), "--manifest", str(manifest_dir), "--output", str(run)])
    rows = [json.loads(line) for line in (run / "flare/train.jsonl").read_text().splitlines()]
    assert [row["nonfinite"] for row in rows if "loss" in row] == [True, True, True]
