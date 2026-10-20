"""corruption.image.lens_*: Flare7K++'s scattering flare plus rebuilt lens optics on TartanAir frames (lens_flare.py, p39).

Pins: a config without the keys renders and reports exactly what it did before, and turning the stage on leaves every
other draw where it was; with every other stage off the input is exactly srgb(linear(clean) + flare) and the TARGET the
clean frame plus the flare's light source (the lamp stays, the flare goes), clipped like a capture; an untransformed
layer is the centre of the flare image in the layer's own linear light (8-bit ^ gamma, background pulled to 0), a zoom
below 1 keeps the light, translation moves the source by translate / FRAME_FRACTION frames and flips mirror it; the
rebuilt optics put each ghost on the line through the source and the frame centre with reflectance x energy of light,
dim with the source outside the frame and add the veil, bloom and dirt; the bank pairs each flare with its light
source, splits blocks of 10 numbers whole per source, pads a non-square layer and refuses a folder without them; the
stage runs in full and blur_low_light (env_clear frames too), never on a clean frame or a FlareImage pair, on about
lens_flare_probability of frames, deterministically and JSON-serialisably, and with lens_flare_clear_probability the
flared frames skip the environment; the dataset hands the target to both phases (tone label 1, kind "flare");
configs/kaggle_lensflare is p38 plus the lens keys and data.lens_flare_root, changes both hashes, keeps the root out of
them and must spell every key out; the notebook finds the folder however Kaggle mounted it; tools/lens_flare_preview.py
draws it.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from qjepa.cli import _manifest, _train_dataset, main
from qjepa.config import LENS_KEYS, OPTIONAL_KEYS, build_corruptors, load_config, serializable_config, validate_config
from qjepa.corruptions.image import (LowLightImageCorruptionConfig, LowLightImageCorruptor, brightness_stops,
                                     frame_kind)
from qjepa.corruptions.lens_flare import (FRAME_FRACTION, LensFlareBank, find_lens_flare_root, flare7k_layers,
                                          lens_flare_layers, optics_layers)
from qjepa.corruptions.light import linear_to_srgb, srgb_to_linear
from qjepa.training.checkpoints import configuration_hash
from test_kaggle_workflow import _write_dataset
from test_sharp_cli import _sharp_smoke

LENS = {key: value for key, value in load_config("configs/kaggle_lensflare.yaml")["corruption"]["image"].items()
        if key.startswith("lens_")}


def _write_flare7k(root, synthetic=40, real=30, size=64):
    """A Flare7K++ folder as Kaggle unpacks khoidb/flare7k: each flare is a light source (a bright disk, off centre for
    Flare-R) plus a dim horizontal streak on black, its Light_Source the disk alone. Flare-R is landscape, as the
    official 1008x756 captures are."""
    for source, count, shape, centre in (("Flare7K/Scattering_Flare", synthetic, (size, size), (size // 2, size // 2)),
                                          ("Flare-R", real, (48, 64), (24, 36))):
        for part in ("Compound_Flare", "Light_Source"):
            (root / source / part).mkdir(parents=True, exist_ok=True)
        rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
        disk = (rows - centre[0]) ** 2 + (cols - centre[1]) ** 2 <= 9
        for index in range(count):
            light = np.zeros(shape + (3,), np.uint8)
            light[disk] = (255, 230, 200 - index % 50)
            flare = light.copy()
            flare[centre[0] - 1:centre[0] + 2, 4:-4] = np.maximum(flare[centre[0] - 1:centre[0] + 2, 4:-4], 90)
            name = f"{index:06d}.png"
            Image.fromarray(flare).save(root / source / "Compound_Flare" / name)
            Image.fromarray(light).save(root / source / "Light_Source" / name)
    return root


def _bank(root):
    return LensFlareBank(root, holdout_fraction=LENS["lens_flare_holdout_fraction"])


def _frame(size=64, seed=1):
    return np.random.default_rng(seed).uniform(0.05, 0.6, (size, size, 3))


def _corruptor(bank=None, **overrides):
    values = dict(LENS, clean_probability=0.0)
    values.update(overrides)
    return LowLightImageCorruptor(LowLightImageCorruptionConfig(**values), 73128,
                                  lens_bank=bank if values["lens_flare_probability"] > 0 else None)


def _call(corruptor, image, index=0, mode="full", split="train", trajectory="t", flare_pair=None):
    return corruptor(image, split=split, realization=0, trajectory=trajectory, timestamp=0.1 * index,
                     frame_index=index, mode=mode, flare_pair=flare_pair)


# Every other stage off: no optics, no low light (clear environment), no sensor (low_light_only variant).
STILL = dict(defocus_probability=0.0, motion_probability=0.0, downsample_probability=0.0,
             env_clear_probability=1.0, low_light_only_probability=1.0)


def _params(source="Flare7K", **overrides):
    """An untransformed draw (no rotation, shear, shift, flip, blur, offset; gain 1) of the first layer."""
    drawn = {"source": source, "name": "000000.png", "gamma": 2.0, "gain": 1.0, "rotation_deg": 0.0, "scale": 1.0,
             "shear_deg": 0.0, "translate": [0.0, 0.0], "flip_x": False, "flip_y": False, "blur_sigma_px": 0.0,
             "dc_offset": 0.0, "optics": None}
    drawn.update(overrides)
    return drawn


def test_old_configs_render_and_report_exactly_as_before(tmp_path):
    bank = _bank(_write_flare7k(tmp_path / "f7k"))
    assert LowLightImageCorruptionConfig().lens_flare_probability == 0.0
    clean_probability = LowLightImageCorruptionConfig().clean_probability
    old = LowLightImageCorruptor(LowLightImageCorruptionConfig(), 73128)
    zero = _corruptor(lens_flare_probability=0.0, clean_probability=clean_probability)
    on = _corruptor(bank, lens_flare_probability=1.0, lens_flare_clear_probability=0.0,
                    clean_probability=clean_probability)
    image = _frame()
    for index in range(20):
        (a, pa), (b, pb), (c, pc) = (_call(x, image, index) for x in (old, zero, on))
        assert "lens_flare" not in pa and pa == pb and np.array_equal(a, b)
        assert old.target(image, pa) is image                              # no flare: the target is the frame
        # The lens draws from its own stream: every other draw stays put.
        assert {key: pc[key] for key in pa} == pa
        if not pc["clean"]:
            assert pc["lens_flare"] and not np.array_equal(a, c)


def test_the_input_gets_flare_and_light_the_target_keeps_only_the_light(tmp_path):
    bank = _bank(_write_flare7k(tmp_path / "f7k"))
    corruptor = _corruptor(bank, lens_flare_probability=1.0, lens_optics_probability=0.0, **STILL)
    image = _frame()
    sources = set()
    for index in range(12):
        noisy, params = _call(corruptor, image, index, trajectory=f"t{index}")
        drawn = params["lens_flare_params"]
        assert params["lens_flare"] and params["low_light"] is False and params["sensor_noise"] is False
        sources.add(drawn["source"])
        flare, light = lens_flare_layers(bank, drawn, 64, 64)
        expected = np.clip(linear_to_srgb(srgb_to_linear(image) + flare), 0, 1)
        assert np.allclose(noisy, expected, atol=1e-5), index
        target = corruptor.target(image, params)
        assert target.dtype == np.float32 and 0 <= target.min() and target.max() <= 1
        assert np.allclose(target, np.clip(linear_to_srgb(srgb_to_linear(image) + light), 0, 1), atol=1e-6)
        # The light stays in the target; the streak only in the input.
        assert float(np.sum(target - image)) >= 0 and float(np.sum(noisy - target)) > 0
        fresh = _corruptor(bank, lens_flare_probability=1.0, lens_optics_probability=0.0, **STILL)
        assert np.array_equal(fresh.target(image, params), target)        # cached or recomputed: the same
    assert sources == {"Flare7K", "Flare-R"}


def test_an_untransformed_layer_is_the_centre_of_the_flare_in_its_own_linear_light(tmp_path):
    root = _write_flare7k(tmp_path / "f7k")
    bank = _bank(root)
    with Image.open(root / "Flare7K/Scattering_Flare/Compound_Flare/000000.png") as image:
        raw = np.asarray(image.convert("RGB")).astype(np.float64) / 255
    linear = raw ** 2.0
    # remove_background: min of every channel -> 0 (0 here already), max kept.
    frame = 32
    scale = FRAME_FRACTION["Flare7K"] * 64 / frame                   # zoom exactly 1: the centre 32 x 32
    flare, light, (x, y), colour = flare7k_layers(bank, _params(scale=scale), frame, frame)
    assert np.allclose(flare, linear[16:48, 16:48], atol=1e-5)
    assert np.allclose(light[..., 0], (linear[16:48, 16:48, 0] > 0.9) * 1.0, atol=1e-5)
    assert (x, y) == pytest.approx((16.5, 16.5), abs=0.6)          # the disk sits on the frame's centre
    assert colour.max() == pytest.approx(1.0) and colour[0] == pytest.approx(1.0)
    # Shifted by translate / FRAME_FRACTION frames, mirrored by the flips.
    _, light, (x, y), _ = flare7k_layers(bank, _params(scale=scale, translate=[0.05, -0.04], flip_x=True), frame, frame)
    shift = np.array([0.05, -0.04]) / FRAME_FRACTION["Flare7K"] * frame
    assert x == pytest.approx(frame - (16.5 + shift[0]), abs=0.6) and y == pytest.approx(16.5 + shift[1], abs=0.6)
    peak = np.unravel_index(np.argmax(light[..., 0]), light.shape[:2])
    assert (peak[1] + 0.5, peak[0] + 0.5) == pytest.approx((x, y), abs=2.0)
    # Zooming out keeps the light: a 16 px frame showing the whole 64 px layer (zoom 1/4) holds 1/16 of its light.
    small, _, _, _ = flare7k_layers(bank, _params(scale=0.25 * FRAME_FRACTION["Flare7K"] * 64 / 16), 16, 16)
    assert small.sum() * 16 == pytest.approx(linear.sum(), rel=0.03)
    # Gain multiplies both layers; the offset lifts the flare only, never below zero.
    lifted, lamp, _, _ = flare7k_layers(bank, _params(scale=scale, gain=2.0, dc_offset=0.01), frame, frame)
    assert np.allclose(lamp, 2.0 * light_of(bank, scale, frame), atol=1e-5)
    assert np.allclose(lifted, 2.0 * (flare + 0.01), atol=1e-5)


def light_of(bank, scale, frame):
    return flare7k_layers(bank, _params(scale=scale), frame, frame)[1]


def test_the_optics_put_ghosts_on_the_source_centre_axis_with_their_share_of_the_light():
    one = {"kind": "disk", "t": -1.0, "radius": 0.04, "soft": 0.006, "dispersion": 0.0, "reflectance": 0.01,
           "color": [1.0, 0.5, 0.25]}
    optics = {"aperture": {"blades": 6, "rotation": 0.3, "roundness": 0.1}, "ghosts": [one], "halo": None,
              "energy": 0.2, "veil": 0.0, "bloom": 0.0, "dirt": 0.0, "dirt_seed": 0}
    size = 128
    x, y = 100.0, 30.0
    layers = optics_layers(size, size, x, y, optics, np.ones(3))
    ghost = layers["ghosts"]
    weight = ghost[..., 0]
    rows, cols = np.mgrid[0:size, 0:size] + 0.5
    centroid = (float((weight * cols).sum() / weight.sum()), float((weight * rows).sum() / weight.sum()))
    assert centroid == pytest.approx((size - x, size - y), abs=0.5)      # t = -1: the mirror of the source
    energy = 0.2 * size * size
    # Normalised on an integer grid, drawn at a sub-pixel centre: within a few per cent.
    assert ghost.sum(axis=(0, 1)) == pytest.approx(0.01 * energy * np.array([1.0, 0.5, 0.25]), rel=0.05)
    assert layers["veil"].sum() == 0 and layers["bloom"].sum() == 0 and "dirt" not in layers
    # A source outside the frame still throws ghosts and glare, dimmer the further out it is.
    lit = dict(optics, veil=0.02, bloom=0.02, dirt=0.1, dirt_seed=3)
    near, far = (sum(optics_layers(size, size, sx, -10.0, lit, np.ones(3)).values()).sum() for sx in (64.0, 260.0))
    assert near > far > 0
    full = optics_layers(size, size, x, y, lit, np.ones(3))
    assert set(full) == {"veil", "bloom", "ghosts", "dirt"} and all(layer.min() >= 0 for layer in full.values())
    # Veil and bloom carry their share of the energy (all but the tail outside the frame).
    assert 0.5 * 0.02 * energy * 3 < full["bloom"].sum() <= 0.02 * energy * 3


def test_the_bank_pairs_each_flare_with_its_light_and_splits_blocks_whole(tmp_path):
    root = _write_flare7k(tmp_path / "f7k", synthetic=60, real=40)
    bank = _bank(root)
    for source in ("Flare7K", "Flare-R"):
        blocks = {split: {int(name[:6]) // 10 for name in bank.splits[split][source]} for split in bank.splits}
        assert all(blocks.values())
        assert not (blocks["train"] & blocks["valid"]) and not (blocks["train"] & blocks["test"]) \
            and not (blocks["valid"] & blocks["test"])
    assert bank.splits == _bank(root).splits
    flare, light = bank.load("Flare-R", "000003.png")
    assert flare.shape == light.shape == (64, 64, 3)                   # 48 x 64 padded to a square
    assert flare[:8].max() == 0 and light[24 + 8, 36].max() > 200      # pad rows, the disk moved down by 8
    corruptor = _corruptor(bank, lens_flare_probability=1.0)
    for split in ("train", "valid", "test"):
        for i in range(40):
            drawn = corruptor._parameters(split, 0, f"t{i}", 0.1 * i, "full")["lens_flare_params"]
            assert drawn["name"] in bank.splits[split][drawn["source"]], split
    (root / "Flare-R/Light_Source/000005.png").unlink()
    with pytest.raises(FileNotFoundError, match="nguon sang"):
        _bank(root)
    with pytest.raises(FileNotFoundError, match="lens_flare_root"):
        _bank(tmp_path / "nothing")
    with pytest.raises(ValueError, match="lens_flare_root"):
        LowLightImageCorruptor(LowLightImageCorruptionConfig(**dict(LENS, lens_flare_probability=0.5)), 73128)


def test_the_stage_runs_in_the_lens_modes_on_about_its_probability_never_on_clean_frames_or_real_pairs(tmp_path):
    bank = _bank(_write_flare7k(tmp_path / "f7k"))
    always = _corruptor(bank, lens_flare_probability=1.0)
    plain = _corruptor(lens_flare_probability=0.0)
    image = _frame()
    for mode in ("full", "blur_low_light"):
        noisy, params = _call(always, image, 3, mode=mode)
        assert params["lens_flare"] and not np.array_equal(noisy, _call(plain, image, 3, mode=mode)[0]), mode
    for mode in ("blur_only", "low_light_only", "sensor_noise_only", "clean"):
        noisy, params = _call(always, image, 3, mode=mode)
        assert params["lens_flare"] is False and params["lens_flare_params"] is None
        assert np.array_equal(noisy, _call(plain, image, 3, mode=mode)[0]), mode
    # A FlareImage pair already holds a real flare: none on top, and its target stays its clean photo.
    pair = {"name": "x.png", "source": "flare_removal", "clear": True}
    noisy, params = _call(always, image, 3, flare_pair=pair)
    assert params["lens_flare"] is False and always.target(image, params) is image
    cleared = _corruptor(bank, lens_flare_probability=1.0, env_clear_probability=1.0)
    assert all(_call(cleared, image, i)[1]["lens_flare"] for i in range(8))
    clean = _corruptor(bank, lens_flare_probability=1.0, clean_probability=1.0)
    assert not any(_call(clean, image, i)[1]["lens_flare"] for i in range(8))
    half = _corruptor(bank)
    fired = [half._parameters("train", 0, f"t{i % 7}", 0.1 * i, "full") for i in range(600)]
    assert abs(np.mean([p["lens_flare"] for p in fired]) - LENS["lens_flare_probability"]) < 0.06
    optics = [p["lens_flare_params"]["optics"] is not None for p in fired if p["lens_flare"]]
    assert abs(np.mean(optics) - LENS["lens_optics_probability"]) < 0.1
    # Deterministic, segment-stable, serialisable.
    (a, pa), (b, pb) = _call(always, image, 3), _call(always, image, 3)
    assert np.array_equal(a, b) and pa == pb
    json.dumps(pa)
    json.dumps(always.metadata())
    assert always._parameters("train", 0, "t", 0.301, "full")["lens_flare_params"] == \
        always._parameters("train", 0, "t", 0.302, "full")["lens_flare_params"]


def test_flared_frames_skip_the_environment_and_the_others_are_untouched(tmp_path):
    bank = _bank(_write_flare7k(tmp_path / "f7k"))
    recipe = dict(load_config("configs/kaggle_lensflare.yaml")["corruption"]["image"], clean_probability=0.0)
    assert recipe["lens_flare_clear_probability"] == 1.0 and recipe["illum_probability"] > 0
    plain, lighter = (LowLightImageCorruptor(LowLightImageCorruptionConfig(**dict(recipe, lens_flare_clear_probability=p)),
                                             73128, lens_bank=bank) for p in (0.0, 1.0))
    image = _frame()
    camera = ("defocus", "defocus_sigma", "motion", "downsample", "photon_count", "sensor_noise", "jpeg",
              "lens_flare", "lens_flare_params")
    flared = darkened_before = 0
    for index in range(80):
        (a, pa), (b, pb) = (_call(c, image, index, trajectory=f"t{index}") for c in (plain, lighter))
        if pb["lens_flare"]:
            flared += 1
            darkened_before += bool(pa["low_light"])
            assert pb["lens_flare_clear"] and pb["low_light"] is False and not pb.get("illumination")
            assert {key: pb[key] for key in camera} == {key: pa[key] for key in camera}
            assert not brightness_stops(pb, 64, 64).any()
            assert frame_kind(pb) == "flare"
        else:
            assert pb.pop("lens_flare_clear") is False and pa.pop("lens_flare_clear") is False
            assert pb == pa and np.array_equal(a, b)
    assert flared > 15 and darkened_before > 5


def _smoke(tmp_path):
    root, manifest_dir = tmp_path / "dataset", tmp_path / "manifest"
    _write_dataset(root)
    config = _sharp_smoke()
    config["corruption"]["image"].update(LENS, lens_flare_probability=1.0, clean_probability=0.0)
    config["data"]["lens_flare_root"] = str(_write_flare7k(tmp_path / "f7k"))
    path = tmp_path / "lens.yaml"
    path.write_text(yaml.safe_dump(config))
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest_dir)])
    return config, _manifest(config, str(manifest_dir))[0]


def test_the_dataset_hands_the_target_with_its_light_to_both_phases(tmp_path):
    config, manifest = _smoke(tmp_path)
    validate_config(config)
    for phase in ("phase1", "phase2"):
        dataset = _train_dataset(config, manifest, phase)          # what the CLI trains on
        flared = 0
        for index in range(min(6, len(dataset.samples))):
            item = dataset[index]
            params = item["corruption"]["image"]
            if not params.get("lens_flare"):
                continue
            flared += 1
            clean = dataset._clean_frame(dataset.samples[index])
            expected = dataset.image_corruptor.target(clean, params)
            got = item["image_clean"].permute(1, 2, 0).numpy()
            if not np.allclose(got, expected, atol=1e-6):                # mirrored after the corruption
                assert np.allclose(got, expected[:, ::-1], atol=1e-6)
            assert not np.allclose(expected, clean)                       # the lamp is in the target
            assert frame_kind(params) == "flare"
            if "image_tone_label" in item:
                assert float(item["image_tone_label"]) == 1.0
        assert flared >= 3, phase


def test_kaggle_lensflare_is_p38_plus_the_lens_keys_and_spells_them_out(tmp_path):
    expnorm = serializable_config(load_config("configs/kaggle_expnorm.yaml"))
    lens = serializable_config(load_config("configs/kaggle_lensflare.yaml"))
    # p45's lens_flare_wu_* keys are optional (absent: no Wu layers), so p39 spells out every other lens key.
    required = [key for key in LENS_KEYS if key not in OPTIONAL_KEYS]
    assert set(LENS) == set(required) and LENS["lens_flare_probability"] > 0
    for config in (expnorm, lens):
        config.pop("_config_path", None)
        config["runtime"].pop("output_dir")
    for key in required:
        lens["corruption"]["image"].pop(key)
    assert lens["data"].pop("lens_flare_root")
    assert lens == expnorm
    full_expnorm, full_lens = load_config("configs/kaggle_expnorm.yaml"), load_config("configs/kaggle_lensflare.yaml")
    validate_config(full_lens)
    moved = copy.deepcopy(full_lens)
    moved["data"]["lens_flare_root"] = "/elsewhere/flare7k"
    for phase in ("phase1", "phase2"):
        assert configuration_hash(full_expnorm, phase) != configuration_hash(full_lens, phase)
        assert configuration_hash(moved, phase) == configuration_hash(full_lens, phase)
    # p38 and every config before it hash exactly as before the lens keys existed.
    assert configuration_hash(full_expnorm, "phase1").startswith("20820dddfa24")
    assert configuration_hash(full_expnorm, "phase2").startswith("da2dde02ceb9")
    missing = copy.deepcopy(full_lens)
    del missing["corruption"]["image"]["lens_veil"]
    with pytest.raises(ValueError, match="lens_veil"):
        validate_config(missing)
    rootless = copy.deepcopy(full_lens)
    del rootless["data"]["lens_flare_root"]
    with pytest.raises(ValueError, match="lens_flare_root"):
        validate_config(rootless)
    for key, value in (("lens_flare_probability", 1.5), ("lens_flare_gain", [0.0, 2.0]), ("lens_flare_gain", [3.0, 1.0]),
                       ("lens_flare_holdout_fraction", 0.0), ("lens_ghost_count", [0, 20]),
                       ("lens_flare_translate", 0.9), ("lens_typo", 1.0)):
        bad = copy.deepcopy(full_lens)
        bad["corruption"]["image"][key] = value
        with pytest.raises(ValueError):
            validate_config(bad)
    built = copy.deepcopy(full_lens)
    built["data"]["lens_flare_root"] = str(_write_flare7k(tmp_path / "f7k"))
    image_corruptor, _ = build_corruptors(built)
    assert image_corruptor.lens_bank is not None
    assert image_corruptor.metadata()["lens_flare"]["samples"]["train"]["Flare-R"] > 0


def test_the_notebook_finds_flare7k_however_kaggle_mounted_it(tmp_path):
    api = _write_flare7k(tmp_path / "a/datasets/khoidb/flare7k", synthetic=10, real=10)
    assert find_lens_flare_root(tmp_path / "a") == api
    nested = _write_flare7k(tmp_path / "b/datasets/someone/flare-out/flare7kpp_640", synthetic=10, real=10)
    assert find_lens_flare_root(tmp_path / "b") == nested
    (tmp_path / "c/datasets/someone/tartanairshard0").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="tartanairshard0"):
        find_lens_flare_root(tmp_path / "c")
    _write_flare7k(tmp_path / "a/datasets/other/flare7k-copy", synthetic=10, real=10)
    with pytest.raises(ValueError, match="hon mot lan"):
        find_lens_flare_root(tmp_path / "a")


def test_the_preview_tool_draws_clean_target_flare_and_input(tmp_path):
    from tools.lens_flare_preview import main as preview
    root = tmp_path / "dataset"
    _write_dataset(root)
    report = preview(["--config", "configs/kaggle_lensflare.yaml", "--lens-flare-root",
                      str(_write_flare7k(tmp_path / "f7k")), "--images", str(root), "--count", "3",
                      "--output", str(tmp_path / "preview")])
    assert len(report["rows"]) == 3 and (tmp_path / "preview/lens_flare_preview.png").stat().st_size > 1000
    assert all(np.isfinite(row["psnr_input_db"]) and row["light_kept_mean"] >= 0 for row in report["rows"])
