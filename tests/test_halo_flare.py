"""corruption.image.halo_*: HALO's real lens flare added to TartanAir frames (qjepa/corruptions/halo.py).

HALO has no IMU, so its frames never become samples: its flare-only layer is a corruption of a
TartanAir frame, whose IMU stays its own, and the target stays the clean frame.

Pins: a config without the keys (or with halo_probability 0) renders and reports exactly what it
did before, and turning HALO on leaves every other draw where it was; the layer is cropped about the
image centre (the optical axis the ghosts line up on), resized in linear light with the total light
kept, decoded once per process (the cache changes no value), and added in linear light -- with every other stage off, the noisy frame is exactly
srgb(linear(clean) + gain * layer), flips included; HALO's scenes are split whole into train / valid /
test and each split draws only its own layers; the stage runs in full and blur_low_light (env_clear
frames too) but not blur_only / low_light_only / sensor_noise_only / clean, fires on about
halo_probability of the frames, is deterministic, segment-stable and JSON-serialisable; with
halo_clear_probability a flared frame skips the environment (no darkness, uneven light, lamps, fog: zero
stop map) while its camera blur, sensor noise and flare stay as drawn, and frames without the flare --
or in the named blur_low_light scenario -- render exactly as with the switch off; the bank
refuses a folder from another build, missing layers, or no folder at all; configs/kaggle_halo is
p32_relight plus the halo keys and data.halo_root, changes both hashes, keeps data.halo_root out of
them, and must spell every key out; the recipe trains both phases through the CLI with HALO on, and
tools/halo_probe.py scores how much of the flare a phase-2 model leaves (trained with HALO or, given
--halo-root, without it).
"""

from __future__ import annotations

import copy
import csv
import json

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from qjepa.cli import main
from qjepa.config import HALO_KEYS, build_corruptors, load_config, serializable_config, validate_config
from qjepa.corruptions.halo import HaloFlareBank, apply_halo, split_scenes
from qjepa.corruptions.image import LowLightImageCorruptionConfig, LowLightImageCorruptor, brightness_stops
from qjepa.corruptions.light import linear_to_srgb, srgb_to_linear
from qjepa.training.checkpoints import configuration_hash, load_checkpoint
from test_kaggle_workflow import _write_dataset
from test_sharp_cli import _run_until_done, _sharp_smoke

HALO = {key: value for key, value in load_config("configs/kaggle_halo.yaml")["corruption"]["image"].items()
        if key.startswith("halo_")}
REVISION = HALO["halo_revision"]


def _write_halo(root, scenes=8, per_scene=3, size=(72, 128), revision=REVISION, effects=("Reflective",)):
    """A HALO folder as Kaggle unpacks halo-reflective-1280: index, build info, <scene>/<uid>.separate.png.

    Each layer: dark, a bright square at the centre and a bright stripe on the left edge (outside the
    centre square), so a centre crop keeps the square and drops the stripe."""
    rows = []
    height, width = size
    for s in range(scenes):
        scene = f"Scene{s:03d}"
        (root / scene).mkdir(parents=True)
        for k in range(per_scene):
            uid = f"{scene}_Reflective{k:03d}_camera01_{s * per_scene + k:04d}"
            pixels = np.full((height, width, 3), 10, dtype=np.uint8)
            pixels[height // 2 - 6:height // 2 + 6, width // 2 - 6:width // 2 + 6] = (200, 120 + 10 * k, 40)
            pixels[:, :8] = 250
            Image.fromarray(pixels).save(root / scene / f"{uid}.separate.png")
            rows.append({"uid": uid, "scene": scene, "effect_type": "Reflective", "effect_id": f"Reflective{k:03d}",
                         "sample_id": uid.rsplit("_", 1)[0], "final_idx": s * per_scene + k, "orig_idx": 0,
                         "shard": "shard-00000.tar", "render_dir": "x"})
    with (root / "halo_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "halo_build.json").write_text(json.dumps({"revision": revision, "effects": list(effects), "max_side": width}))
    return root


def _bank(root):
    return HaloFlareBank(root, revision=REVISION, effects=HALO["halo_effects"],
                         holdout_fraction=HALO["halo_holdout_fraction"])


def _frame(size=64):
    rng = np.random.default_rng(1)
    return rng.uniform(0.05, 0.6, (size, size, 3))


def _corruptor(bank=None, **overrides):
    values = dict(HALO, clean_probability=0.0)
    values.update(overrides)
    return LowLightImageCorruptor(LowLightImageCorruptionConfig(**values), 73128,
                                  halo_bank=bank if values["halo_probability"] > 0 else None)


def _call(corruptor, image, index=0, mode="full", split="train", trajectory="t"):
    return corruptor(image, split=split, realization=0, trajectory=trajectory, timestamp=0.1 * index,
                     frame_index=index, mode=mode)


# Every other stage off: no optics, no low light (clear environment), no sensor (low_light_only variant).
STILL = dict(defocus_probability=0.0, motion_probability=0.0, downsample_probability=0.0,
             env_clear_probability=1.0, low_light_only_probability=1.0)


def test_old_configs_render_and_report_exactly_as_before(tmp_path):
    bank = _bank(_write_halo(tmp_path / "halo"))
    assert LowLightImageCorruptionConfig().halo_probability == 0.0
    old = LowLightImageCorruptor(LowLightImageCorruptionConfig(), 73128)
    zero = _corruptor(halo_probability=0.0, clean_probability=LowLightImageCorruptionConfig().clean_probability)
    on = _corruptor(bank, halo_probability=1.0, halo_clear_probability=0.0,
                    clean_probability=LowLightImageCorruptionConfig().clean_probability)
    image = _frame()
    for index in range(20):
        (a, pa), (b, pb), (c, pc) = (_call(x, image, index) for x in (old, zero, on))
        assert "halo" not in pa and pa == pb and np.array_equal(a, b)
        # HALO draws from its own stream: every other draw stays put.
        assert {key: pc[key] for key in pa} == pa
        if not pc["clean"]:
            assert pc["halo"] and not np.array_equal(a, c)


def test_the_layer_is_the_centre_crop_resized_in_linear_light_with_the_light_kept(tmp_path):
    bank = _bank(_write_halo(tmp_path / "halo"))
    uid = bank.splits["train"][0]
    layer = bank.layer(uid, 36, 36)
    assert layer.shape == (36, 36, 3) and layer.dtype == np.float32
    # The bright stripe on the source's left edge lies outside the centre square: gone.
    background = float(srgb_to_linear(np.full(3, 10 / 255, dtype=np.float32))[0])
    assert layer[:, :3].max() == pytest.approx(background, rel=1e-3)
    assert layer[18, 18, 0] > 0.5                                  # the centre square stays
    # BOX in linear light: the mean light of the 72 x 72 centre crop is kept.
    with Image.open(bank.paths[uid]) as image:
        crop = np.asarray(image.convert("RGB"))[:, 28:100].astype(np.float32) / 255
    assert layer.mean() == pytest.approx(float(srgb_to_linear(crop).mean()), rel=2e-3)
    # A 4:3 target crops the 16:9 source about its centre too; a 16:9 target keeps the whole frame.
    assert bank.layer(uid, 36, 48)[:, :2].max() == pytest.approx(background, rel=1e-3)
    assert bank.layer(uid, 36, 64)[:, :2].max() > 0.5


def test_a_layer_is_decoded_once_per_process_and_the_cache_changes_no_value(tmp_path, monkeypatch):
    import qjepa.corruptions.halo as halo_module
    root = _write_halo(tmp_path / "halo")
    bank, fresh = _bank(root), _bank(root)
    opened = []
    real_open = halo_module.Image.open
    monkeypatch.setattr(halo_module.Image, "open", lambda path, *a, **k: opened.append(path) or real_open(path, *a, **k))
    uid = bank.splits["train"][0]
    first = bank.layer(uid, 40, 40)
    for _ in range(5):
        again = bank.layer(uid, 40, 40)
        assert np.array_equal(again, first) and again is not first   # a new array each call: callers may write
    assert len(opened) == 1                                            # the PNG is read once per process
    assert np.array_equal(fresh.layer(uid, 40, 40), first)            # cache or not, the same layer
    bank.layer(uid, 40, 64)
    assert len(opened) == 3                                            # another size is another entry


def test_with_every_other_stage_off_the_frame_is_exactly_the_flare_added_in_linear_light(tmp_path):
    bank = _bank(_write_halo(tmp_path / "halo"))
    corruptor = _corruptor(bank, halo_probability=1.0, **STILL)
    image = _frame()
    flips = set()
    for index in range(16):
        noisy, params = _call(corruptor, image, index)
        assert params["halo"] and params["low_light"] is False and params["sensor_noise"] is False
        halo = params["halo_params"]
        layer = bank.layer(halo["uid"], 64, 64)
        layer = layer[::-1] if halo["flip_y"] else layer
        layer = layer[:, ::-1] if halo["flip_x"] else layer
        expected = np.clip(linear_to_srgb(srgb_to_linear(image) + np.float32(halo["gain"]) * layer), 0, 1)
        assert np.allclose(noisy, expected, atol=1e-5), index
        assert np.allclose(noisy, np.clip(apply_halo(image, bank, halo), 0, 1), atol=1e-6)
        low, high = HALO["halo_gain"]
        assert low <= halo["gain"] <= high
        flips.add((halo["flip_x"], halo["flip_y"]))
    assert len(flips) >= 3                                          # both flips are drawn


def test_halo_scenes_are_split_whole_and_each_split_draws_only_its_own_layers(tmp_path):
    root = _write_halo(tmp_path / "halo", scenes=10)
    bank = _bank(root)
    scenes = {split: {bank.paths[uid].parent.name for uid in uids} for split, uids in bank.splits.items()}
    assert all(scenes.values())
    assert not (scenes["train"] & scenes["valid"]) and not (scenes["train"] & scenes["test"]) \
        and not (scenes["valid"] & scenes["test"])
    assert sum(len(uids) for uids in bank.splits.values()) == 30
    assert bank.splits == _bank(root).splits                        # deterministic
    assert split_scenes({"a": 5, "b": 5, "c": 5, "d": 5}, 0.25) == split_scenes({"d": 5, "c": 5, "b": 5, "a": 5}, 0.25)
    corruptor = _corruptor(bank, halo_probability=1.0)
    for split in ("train", "valid", "test"):
        picked = {corruptor._parameters(split, 0, f"t{i}", 0.1 * i, "full")["halo_params"]["uid"] for i in range(60)}
        assert picked <= set(bank.splits[split]), split


def test_the_stage_runs_in_the_lens_and_lighting_modes_on_about_halo_probability_of_frames(tmp_path):
    bank = _bank(_write_halo(tmp_path / "halo"))
    always = _corruptor(bank, halo_probability=1.0)
    plain = _corruptor(halo_probability=0.0)
    image = _frame()
    for mode in ("full", "blur_low_light"):
        noisy, params = _call(always, image, 3, mode=mode)
        assert params["halo"] and not np.array_equal(noisy, _call(plain, image, 3, mode=mode)[0]), mode
    for mode in ("blur_only", "low_light_only", "sensor_noise_only", "clean"):
        noisy, params = _call(always, image, 3, mode=mode)
        assert params["halo"] is False and params["halo_params"] is None
        assert np.array_equal(noisy, _call(plain, image, 3, mode=mode)[0]), mode
    # The lens, not the environment: clear-environment frames keep it; clean frames never have it.
    cleared = _corruptor(bank, halo_probability=1.0, env_clear_probability=1.0)
    assert all(_call(cleared, image, i)[1]["halo"] for i in range(8))
    clean = _corruptor(bank, halo_probability=1.0, clean_probability=1.0)
    assert not any(_call(clean, image, i)[1]["halo"] for i in range(8))
    half = _corruptor(bank)
    fired = [half._parameters("train", 0, f"t{i % 7}", 0.1 * i, "full")["halo"] for i in range(600)]
    assert abs(np.mean(fired) - HALO["halo_probability"]) < 0.06


def test_the_stage_is_deterministic_segment_stable_and_serialisable(tmp_path):
    bank = _bank(_write_halo(tmp_path / "halo"))
    corruptor = _corruptor(bank, halo_probability=1.0)
    image = _frame()
    (a, pa), (b, pb) = _call(corruptor, image, 3), _call(corruptor, image, 3)
    assert np.array_equal(a, b) and pa == pb and pa["halo"]
    json.dumps(pa)
    json.dumps(corruptor.metadata())
    same = corruptor._parameters("train", 0, "t", 0.301, "full")["halo_params"]
    assert same == corruptor._parameters("train", 0, "t", 0.302, "full")["halo_params"]
    others = {json.dumps(corruptor._parameters("train", 0, f"u{i}", 0.301, "full")["halo_params"]) for i in range(6)}
    assert len(others) > 1


def test_flared_frames_skip_the_environment_and_the_others_are_untouched(tmp_path):
    bank = _bank(_write_halo(tmp_path / "halo"))
    recipe = dict(load_config("configs/kaggle_halo.yaml")["corruption"]["image"], clean_probability=0.0)
    assert recipe["halo_clear_probability"] == 1.0 and recipe["illum_probability"] > 0
    plain, lighter = (LowLightImageCorruptor(LowLightImageCorruptionConfig(**dict(recipe, halo_clear_probability=p)),
                                             73128, halo_bank=bank) for p in (0.0, 1.0))
    image = _frame()
    camera = ("defocus", "defocus_sigma", "motion", "motion_length", "downsample", "photon_count", "read_noise_std",
              "sensor_noise", "jpeg", "halo", "halo_params")
    flared = darkened_before = 0
    for index in range(60):
        (a, pa), (b, pb) = _call(plain, image, index, trajectory=f"t{index}"), _call(lighter, image, index,
                                                                                    trajectory=f"t{index}")
        if pb["halo"]:
            flared += 1
            darkened_before += bool(pa["low_light"])
            assert pb["halo_clear"] and pb["low_light"] is False
            assert not pb.get("illumination") and pb.get("illumination_params") is None and not pb.get("light")
            assert {key: pb[key] for key in camera} == {key: pa[key] for key in camera}
            assert not brightness_stops(pb, 64, 64).any()               # nothing scaled the light
        else:
            assert pb.pop("halo_clear") is False and pb == pa and np.array_equal(a, b)
    assert flared > 15 and darkened_before > 5                        # the switch did remove darkness
    # A named scenario keeps its meaning: blur_low_light stays dark under the flare.
    for index in range(10):
        (a, pa), (b, pb) = (_call(c, image, index, mode="blur_low_light", trajectory=f"t{index}") for c in (plain, lighter))
        assert pb["halo_clear"] is False and np.array_equal(a, b)


def test_the_bank_refuses_another_build_missing_layers_and_no_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="halo_index.csv"):
        _bank(tmp_path / "nothing")
    with pytest.raises(ValueError, match="revision"):
        _bank(_write_halo(tmp_path / "old", revision="0" * 40))
    with pytest.raises(ValueError, match="Reflective"):
        _bank(_write_halo(tmp_path / "streak", effects=("Streak",)))
    root = _write_halo(tmp_path / "packed")
    next(root.glob("*/*.separate.png")).unlink()                    # e.g. Kaggle left a tar unpacked
    with pytest.raises(FileNotFoundError, match="khong co tren dia"):
        _bank(root)
    with pytest.raises(ValueError, match="halo_root"):
        LowLightImageCorruptor(LowLightImageCorruptionConfig(**dict(HALO, halo_probability=0.5)), 73128)


def test_kaggle_halo_is_p32_plus_the_halo_keys_and_spells_them_out(tmp_path):
    relight = serializable_config(load_config("configs/kaggle_relight.yaml"))
    halo = serializable_config(load_config("configs/kaggle_halo.yaml"))
    assert set(HALO) == set(HALO_KEYS) and HALO["halo_probability"] > 0
    for config in (relight, halo):
        config.pop("_config_path", None)
        config["runtime"].pop("output_dir")
    for key in HALO_KEYS:
        halo["corruption"]["image"].pop(key)
    assert halo["data"].pop("halo_root")
    assert halo == relight
    full_relight, full_halo = load_config("configs/kaggle_relight.yaml"), load_config("configs/kaggle_halo.yaml")
    validate_config(full_halo)
    moved = copy.deepcopy(full_halo)
    moved["data"]["halo_root"] = "/elsewhere/halo"
    for phase in ("phase1", "phase2"):
        assert configuration_hash(full_relight, phase) != configuration_hash(full_halo, phase)
        assert configuration_hash(moved, phase) == configuration_hash(full_halo, phase)   # a path, not the recipe
    # p32 and every config before it hash exactly as before the halo keys existed.
    assert configuration_hash(full_relight, "phase1").startswith("d2e4c470cb9c")
    assert configuration_hash(full_relight, "phase2").startswith("602bcf1f32bd")
    missing = copy.deepcopy(full_halo)
    del missing["corruption"]["image"]["halo_gain"]
    with pytest.raises(ValueError, match="halo_gain"):
        validate_config(missing)
    rootless = copy.deepcopy(full_halo)
    del rootless["data"]["halo_root"]
    with pytest.raises(ValueError, match="halo_root"):
        validate_config(rootless)
    for key, value in (("halo_probability", 1.5), ("halo_clear_probability", -0.1), ("halo_gain", [0.0, 2.0]),
                       ("halo_gain", [3.0, 1.0]),
                       ("halo_holdout_fraction", 0.0), ("halo_holdout_fraction", 0.5), ("halo_effects", ["Lens"]),
                       ("halo_revision", ""), ("halo_typo", 1.0)):
        bad = copy.deepcopy(full_halo)
        bad["corruption"]["image"][key] = value
        with pytest.raises(ValueError):
            validate_config(bad)
    # build_corruptors builds the bank from data.halo_root.
    built = copy.deepcopy(full_halo)
    built["data"]["halo_root"] = str(_write_halo(tmp_path / "halo"))
    image_corruptor, _ = build_corruptors(built)
    assert image_corruptor.halo_bank is not None and image_corruptor.metadata()["halo"]["samples"]["train"] > 0


def _train_tiny(tmp_path, name, halo_root=None):
    """Both phases through the CLI on the smoke recipe (HALO on every frame when ``halo_root``); the phase-2 .pt."""
    root, manifest, output = tmp_path / "dataset", tmp_path / "manifest", tmp_path / name
    if not root.exists():
        _write_dataset(root)
    config = _sharp_smoke()
    if halo_root is not None:
        config["corruption"]["image"].update(HALO, halo_probability=1.0)
        config["data"]["halo_root"] = str(halo_root)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    common = ["--config", str(path), "--manifest", str(manifest), "--output", str(output)]
    if not manifest.exists():
        main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest)])
    _run_until_done(["train-phase1", *common], output / "phase1/last.pt")
    last = output / "phase2/last.pt"
    _run_until_done(["train-phase2", *common, "--backbone-checkpoint", str(output / "phase1/last.pt")], last)
    return last, manifest


def test_the_recipe_trains_both_phases_through_the_cli_and_halo_probe_scores_the_flare_left(tmp_path):
    import subprocess
    import sys
    torch.set_num_threads(1)
    halo_root = _write_halo(tmp_path / "halo")
    last, manifest = _train_tiny(tmp_path, "halo", halo_root)
    assert load_checkpoint(last)["successful_updates"] == 3
    # tools/halo_probe.py: each frame with the flare and its twin without it, through the model.
    plain, _ = _train_tiny(tmp_path, "plain")                  # trained without HALO: needs --halo-root
    for checkpoint, extra, trained in ((last, [], True), (plain, ["--halo-root", str(halo_root)], False)):
        report = tmp_path / f"probe_{trained}.json"
        result = subprocess.run([sys.executable, "tools/halo_probe.py", "--checkpoint", str(checkpoint), "--manifest",
                                 str(manifest), "--samples", "4", "--batch", "2", "--device", "cpu", "--output",
                                 str(report), *extra], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr[-3000:]
        table = json.loads(report.read_text())
        assert table["trained_with_halo"] is trained and table["split"] == "valid"
        whole = table["groups"]["tat ca"]
        assert whole["count"] == 4 and whole["region_fraction"] > 0
        assert whole["flare_cost_db"]["input"] > 0                # the flare does cost the input PSNR
        assert np.isfinite(whole["flare_light_left_region"]) and np.isfinite(whole["flare_cost_kept"])
        assert "anh sang loe con lai" in result.stdout
    no_root = subprocess.run([sys.executable, "tools/halo_probe.py", "--checkpoint", str(plain), "--manifest",
                              str(manifest), "--samples", "2", "--device", "cpu"], capture_output=True, text=True)
    assert no_root.returncode != 0 and "--halo-root" in no_root.stderr + no_root.stdout
