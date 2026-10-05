"""corruption.image.light_*: lamps many times brighter than the rest, and the light glares.

The look the user approved in tools/light_corruption_preview.py, on every split: train so
the model learns to take the glare out and brighten the dark, valid/test so PSNR and the
best checkpoint measure exactly that.

Pins: a config without the keys (or with light_probability 0) renders and reports exactly
what it did before, and turning the light on leaves every earlier parameter draw where it
was; a small lamp becomes many times brighter while dark pixels away from it keep their
value; light above the knee glares into its surroundings and nothing below it does, while
the inside of a wide bright area (sky) gets only 1 + wide_gain; through the whole
corruptor, at a typical exposure (x0.5), a lamp stays blown out where it would not without
the stage, including on frames that go through the downsample (float round trip, not
uint8); train, valid and test all glare; the stage is deterministic, segment-stable,
"full"-only and JSON-serialisable, and fires on about light_probability of the frames;
configs/kaggle_glare is p20_gray plus the light keys, changes both hashes, and must spell
every key out; the recipe trains both phases through the CLI with the stage on every frame.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
import yaml

from qjepa.cli import main
from qjepa.config import LIGHT_KEYS, load_config, serializable_config, validate_config
from qjepa.corruptions.image import (LowLightImageCorruptionConfig, LowLightImageCorruptor,
                                     _resize_roundtrip, _resize_roundtrip_hdr)
from qjepa.corruptions.light import apply_light, draw_light_parameters, srgb_to_linear
from qjepa.training.checkpoints import configuration_hash, load_checkpoint
from test_kaggle_workflow import _write_dataset
from test_sharp_cli import _run_until_done, _sharp_smoke

GLARE = {key: value for key, value in load_config("configs/kaggle_glare.yaml")["corruption"]["image"].items()
         if key.startswith("light_")}


def _frame(size=96, lamp=(30, 30), radius=3, sky=False, specks=0):
    """A dark frame (0.05) with one small lamp, optionally a wide bright 'sky' and bright specks."""
    image = np.full((size, size, 3), 0.05)
    yy, xx = np.mgrid[:size, :size]
    if lamp is not None:
        image[(yy - lamp[0]) ** 2 + (xx - lamp[1]) ** 2 <= radius ** 2] = 0.97
    if sky:
        image[:, size // 2:] = 0.95
    if specks:
        rng = np.random.default_rng(3)
        for y, x in rng.integers(2, size - 2, size=(specks, 2)):
            image[y - 1:y + 1, x - 1:x + 1] = 0.97
    return image


def _params(**overrides):
    """Light parameters with glare switched off unless asked for: the HDR step alone."""
    params = draw_light_parameters(np.random.default_rng(0), LowLightImageCorruptionConfig(**GLARE))
    params.update(threshold=0.6, width=0.15, gain=20.0, wide_gain=0.5, shape=1.0, knee=1e9,
                  bloom_strength=0.0, star=False, ghosts=[])
    params.update(overrides)
    return params


def _luminance(image):
    return srgb_to_linear(image) @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _corruptor(**overrides):
    values = dict(GLARE, clean_probability=0.0)
    values.update(overrides)
    return LowLightImageCorruptor(LowLightImageCorruptionConfig(**values), 73128)


def _call(corruptor, image, index=0, mode="full", trajectory="t"):
    return corruptor(image, split="train", realization=0, trajectory=trajectory, timestamp=0.1 * index,
                     frame_index=index, mode=mode)


def test_old_configs_render_and_report_exactly_as_before():
    assert LowLightImageCorruptionConfig().light_probability == 0.0
    old = LowLightImageCorruptor(LowLightImageCorruptionConfig(), 73128)
    zero = _corruptor(light_probability=0.0, clean_probability=LowLightImageCorruptionConfig().clean_probability)
    on = _corruptor(clean_probability=LowLightImageCorruptionConfig().clean_probability)
    image = _frame()
    for index in range(20):
        (a, pa), (b, pb), (_, pc) = (_call(c, image, index) for c in (old, zero, on))
        assert "light" not in pa and pa == pb and np.array_equal(a, b)
        # The light draws come from their own stream: every earlier draw stays put.
        assert {key: pc[key] for key in pa} == pa


def test_a_lamp_becomes_many_times_brighter_and_the_dark_keeps_its_value():
    image = _frame()
    scene, lit = apply_light(image, _params(), stages=True)
    lamp, dark = image[..., 0] > 0.9, image[..., 0] < 0.1
    assert _luminance(scene)[lamp].mean() > 10 * _luminance(image)[lamp].mean()
    assert np.allclose(scene[dark], image[dark], atol=1e-6)
    assert np.array_equal(scene, lit)                           # glare off: the HDR step alone
    # A frame with nothing bright comes back untouched.
    flat = np.full((32, 32, 3), 0.3)
    assert np.array_equal(apply_light(flat, _params()), flat)


def test_light_above_the_knee_glares_and_the_inside_of_the_sky_gets_only_wide_gain():
    glare = dict(knee=1.0, bloom_strength=1.0, bloom_sigmas_px=[2.0, 8.0, 30.0], bloom_weights=[0.5, 0.3, 0.2])
    yy, xx = np.mgrid[:96, :96]
    ring = (np.hypot(yy - 30, xx - 30) > 5) & (np.hypot(yy - 30, xx - 30) < 9)
    lit = apply_light(_frame(), _params(**glare))
    assert _luminance(lit)[ring].mean() > 5 * _luminance(_frame())[ring].mean()      # halo round the lamp
    # Nothing above the knee: no glare at all.
    scene, lit = apply_light(_frame(), _params(**dict(glare, knee=1e9)), stages=True)
    assert np.array_equal(scene, lit)
    # Sky: its inside (far from its edge) is brighter by 1 + wide_gain (1% for the low-resolution blur).
    sky = _frame(lamp=None, sky=True)
    scene = apply_light(sky, _params(**glare), stages=True)[0]
    inside = (slice(None), slice(72, 96))
    assert _luminance(scene[inside]).max() <= 1.5 * 1.01 * _luminance(sky[inside]).max()


def test_train_valid_and_test_all_glare():
    corruptor = _corruptor(light_probability=1.0)
    image = _frame()
    for split in ("train", "valid", "test"):
        noisy, params = corruptor(image, split=split, realization=0, trajectory="t", timestamp=0.3, frame_index=3)
        plain = _corruptor(light_probability=0.0)(image, split=split, realization=0, trajectory="t", timestamp=0.3,
                                                  frame_index=3)[0]
        assert params["light"] and not np.array_equal(noisy, plain), split


def test_a_lamp_stays_blown_out_through_the_corruptor_even_after_the_downsample():
    still = dict(defocus_probability=0.0, motion_probability=0.0, downsample_probability=1.0,
                 downsample_scale=(0.9, 0.9), exposure_gain=(0.5, 0.5), white_balance_gain=(1.0, 1.0),
                 vignette_strength=(0.0, 0.0), light_gain=(30.0, 30.0), light_probability=1.0,
                 light_knee=(1e9, 1e9))
    image = _frame(size=256, lamp=(80, 80), radius=5)
    centre = (slice(78, 83), slice(78, 83))
    with_light, params = _call(_corruptor(**still), image)
    without, _ = _call(_corruptor(**dict(still, light_probability=0.0)), image)
    assert params["light"] and params["downsample"]
    assert with_light[centre].mean() > 0.95 > without[centre].mean() + 0.05
    hdr = np.full((16, 16, 3), 5.0)
    assert np.allclose(_resize_roundtrip_hdr(hdr, 0.8), 5.0, atol=1e-4)
    assert _resize_roundtrip(hdr, 0.8).max() <= 1.0              # the uint8 path would cap every lamp


def test_the_stage_is_deterministic_segment_stable_full_only_and_serialisable():
    corruptor = _corruptor(light_probability=1.0)
    image = _frame()
    a, pa = _call(corruptor, image, 3)
    b, pb = _call(corruptor, image, 3)
    assert np.array_equal(a, b) and pa == pb and pa["light"]
    json.dumps(pa)
    # Same 50 ms segment -> same camera and the same glare; another trajectory -> other glare.
    same = corruptor._parameters("train", 0, "t", 0.301, "full")["light_params"]
    assert same == corruptor._parameters("train", 0, "t", 0.302, "full")["light_params"]
    assert same != corruptor._parameters("train", 0, "u", 0.301, "full")["light_params"]
    plain = _corruptor(light_probability=0.0)
    for mode in ("blur_only", "low_light_only", "sensor_noise_only", "blur_low_light"):
        noisy, params = _call(corruptor, image, 3, mode=mode)
        assert params["light"] is False and params["light_params"] is None
        assert np.array_equal(noisy, _call(plain, image, 3, mode=mode)[0])


def test_light_fires_on_about_light_probability_of_the_frames():
    corruptor = _corruptor()
    fired = [corruptor._parameters("train", 0, f"t{i % 7}", 0.1 * i, "full")["light"] for i in range(600)]
    assert abs(np.mean(fired) - GLARE["light_probability"]) < 0.06


def test_kaggle_glare_is_p20_gray_plus_the_light_keys_and_spells_them_out():
    gray = serializable_config(load_config("configs/kaggle_gray.yaml"))
    glare = serializable_config(load_config("configs/kaggle_glare.yaml"))
    assert set(GLARE) == set(LIGHT_KEYS) and GLARE["light_probability"] > 0
    for config in (gray, glare):
        config.pop("_config_path", None)
        config["runtime"].pop("output_dir")
    for key in LIGHT_KEYS:
        glare["corruption"]["image"].pop(key)
    assert glare == gray
    full_gray, full_glare = load_config("configs/kaggle_gray.yaml"), load_config("configs/kaggle_glare.yaml")
    validate_config(full_glare)
    for phase in ("phase1", "phase2"):
        assert configuration_hash(full_gray, phase) != configuration_hash(full_glare, phase)
    missing = copy.deepcopy(full_glare)
    del missing["corruption"]["image"]["light_knee"]
    with pytest.raises(ValueError, match="light_knee"):
        validate_config(missing)
    for key, value in (("light_probability", 1.5), ("light_gain", [0.0, 5.0]), ("light_bloom_sigma_px", [[3.0, 1.0]]),
                       ("light_warmth", [-2.0, 0.5]), ("light_typo", 1.0)):
        bad = copy.deepcopy(full_glare)
        bad["corruption"]["image"][key] = value
        with pytest.raises(ValueError):
            validate_config(bad)


def test_the_recipe_trains_both_phases_through_the_cli(tmp_path):
    torch.set_num_threads(1)
    root, manifest, output = tmp_path / "dataset", tmp_path / "manifest", tmp_path / "run"
    _write_dataset(root)
    config = _sharp_smoke()
    config["corruption"]["image"].update(GLARE, light_probability=1.0)
    path = tmp_path / "glare.yaml"
    path.write_text(yaml.safe_dump(config))
    common = ["--config", str(path), "--manifest", str(manifest), "--output", str(output)]
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest)])
    _run_until_done(["train-phase1", *common], output / "phase1/last.pt")
    last = output / "phase2/last.pt"
    _run_until_done(["train-phase2", *common, "--backbone-checkpoint", str(output / "phase1/last.pt")], last)
    assert load_checkpoint(last)["successful_updates"] == 3
