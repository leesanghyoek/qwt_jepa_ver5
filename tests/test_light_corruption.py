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
JSON-serialisable and runs in the modes with the low-light stage (full, low_light_only,
blur_low_light) but not blur_only / sensor_noise_only, and fires on about light_probability of the frames;
configs/kaggle_glare is p20_gray plus the light keys, changes both hashes, and must spell
every key out; the recipe trains both phases through the CLI with the stage on every frame.
"""

from __future__ import annotations

import copy
import json
import math

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


def test_the_stage_is_deterministic_segment_stable_lighting_modes_only_and_serialisable():
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
    # Part of the lighting: every mode with the low-light stage glares; blur-only and grain-only stay isolated.
    for mode in ("low_light_only", "blur_low_light"):
        noisy, params = _call(corruptor, image, 3, mode=mode)
        assert params["light"] and not np.array_equal(noisy, _call(plain, image, 3, mode=mode)[0]), mode
    for mode in ("blur_only", "sensor_noise_only", "clean"):
        noisy, params = _call(corruptor, image, 3, mode=mode)
        assert params["light"] is False and params["light_params"] is None
        assert np.array_equal(noisy, _call(plain, image, 3, mode=mode)[0]), mode


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


# ---------------------------------------------------------------- uneven light (illum_*)
ILLUM = {key: value for key, value in load_config("configs/kaggle_illum.yaml")["corruption"]["image"].items()
         if key.startswith("illum_")}


def test_uneven_light_varies_by_region_keeps_the_mean_in_stops_and_smudges_darken_locally():
    from qjepa.corruptions.light import illumination_field
    regions = {"gradient_stops": 2.0, "gradient_angle": 0.0, "smudges": [],
               "blobs": [{"y": 0.3, "x": 0.3, "sigma": 0.15, "stops": 2.0},
                         {"y": 0.7, "x": 0.7, "sigma": 0.15, "stops": -2.0}]}
    field = illumination_field(64, 64, regions)
    assert abs(float(np.log2(field).mean())) < 1e-4                 # mean brightness in stops kept
    flat = np.full((64, 64, 3), 0.4)
    lit = _luminance(apply_light(flat, None, illumination=regions))
    assert lit.max() / lit.min() > 4                                # not one even mask: regions differ
    smudge = {"gradient_stops": 0.0, "gradient_angle": 0.0, "blobs": [],
              "smudges": [{"y": 0.5, "x": 0.5, "size": 0.08, "elongation": 3.0, "angle": 0.0, "depth": 0.8}]}
    field = illumination_field(64, 64, smudge)
    assert field[32, 32] == pytest.approx(0.2, abs=0.02) and field[2, 2] == pytest.approx(1.0, abs=1e-3)
    assert field[32, 45] < 0.6 < field[45, 32]                      # 0.2 frame away: dark along x, not along y


def test_uneven_light_is_off_without_its_keys_and_moves_no_other_draw():
    image = _frame()
    plain = _corruptor()
    zero = _corruptor(**dict(ILLUM, illum_probability=0.0))
    uneven = _corruptor(**dict(ILLUM, illum_probability=1.0))
    for index in range(12):
        (a, pa), (b, pb), (c, pc) = (_call(x, image, index) for x in (plain, zero, uneven))
        assert "illumination" not in pa and np.array_equal(a, b) and pa == pb
        assert pc["illumination"] and {key: pc[key] for key in pa} == pa   # own stream: nothing else moves
        assert not np.array_equal(a, c)
    for mode, glares in (("blur_low_light", True), ("low_light_only", True), ("blur_only", False),
                         ("sensor_noise_only", False)):
        assert _call(uneven, image, 3, mode=mode)[1]["illumination"] is glares, mode


def test_kaggle_illum_is_p22_plus_uneven_light_and_medium_glare_and_spells_its_keys_out():
    from qjepa.config import ILLUM_KEYS
    light = serializable_config(load_config("configs/kaggle_light.yaml"))
    illum = serializable_config(load_config("configs/kaggle_illum.yaml"))
    assert set(ILLUM) == set(ILLUM_KEYS) and ILLUM["illum_probability"] > 0
    for config in (light, illum):
        config.pop("_config_path", None)
        config["runtime"].pop("output_dir")
    for key in ILLUM_KEYS:
        illum["corruption"]["image"].pop(key)
    for key in ("light_knee", "light_wide_gain", "light_gain", "light_bloom_strength", "light_ghost_strength"):
        assert illum["corruption"]["image"].pop(key) != light["corruption"]["image"].pop(key), key
    assert illum == light
    full_light, full_illum = load_config("configs/kaggle_light.yaml"), load_config("configs/kaggle_illum.yaml")
    validate_config(full_illum)
    for phase in ("phase1", "phase2"):
        assert configuration_hash(full_light, phase) != configuration_hash(full_illum, phase)
    missing = copy.deepcopy(full_illum)
    del missing["corruption"]["image"]["illum_smudge_depth"]
    with pytest.raises(ValueError, match="illum_smudge_depth"):
        validate_config(missing)
    for key, value in (("illum_probability", 2.0), ("illum_smudge_depth", [0.5, 1.0]), ("illum_blob_size", [0.0, 0.2]),
                       ("illum_smudge_elongation", [0.5, 2.0]), ("illum_blobs", [2, 12])):
        bad = copy.deepcopy(full_illum)
        bad["corruption"]["image"][key] = value
        with pytest.raises(ValueError):
            validate_config(bad)


# ---------------------------------------------------------------- fog (fog_*) and clear environments
ENV = load_config("configs/kaggle_env.yaml")["corruption"]["image"]


def _fog(**overrides):
    params = {"density": 1.2, "airlight": 1.0, "coolness": 0.0, "horizon": 0.4, "tilt": 0.0, "patchiness": 0.0,
              "patch_seed": 3, "scatter_px": 0.0}
    params.update(overrides)
    return params


def test_fog_thickens_towards_the_far_line_and_is_lit_by_the_scene():
    from qjepa.corruptions.light import fog_transmission
    t = fog_transmission(64, 64, _fog())
    assert t[:20].mean() < t[40:].mean() < 1.0 and t.min() == pytest.approx(math.exp(-1.2), abs=1e-3)
    # A day frame: contrast drops most where the fog is thick (top), and the fog is bright.
    day = _frame(size=64, lamp=None)
    day[::2] = 0.8                                                   # striped bright scene
    foggy = apply_light(day, None, fog=_fog())
    contrast = lambda image, rows: float(np.ptp(_luminance(image[rows])))
    assert contrast(foggy, slice(0, 16)) < 0.5 * contrast(day, slice(0, 16))
    assert contrast(foggy, slice(0, 16)) < contrast(foggy, slice(48, 64))
    # A night frame gets a dark fog: the fog light follows the scene's own brightness.
    night = np.full((64, 64, 3), 0.03)
    assert _luminance(apply_light(night, None, fog=_fog())).max() < 0.05
    assert _luminance(apply_light(day, None, fog=_fog())).mean() > 0.3


def test_fog_and_clear_frames_are_off_without_their_keys_and_move_no_other_draw():
    image = _frame()
    base = {key: value for key, value in ENV.items() if key.startswith(("light_", "illum_"))}
    plain = _corruptor(**base)
    foggy = _corruptor(**{**base, **{k: v for k, v in ENV.items() if k.startswith("fog_")}, "fog_probability": 1.0})
    for index in range(10):
        (a, pa), (b, pb) = _call(plain, image, index), _call(foggy, image, index)
        assert "fog" not in pa and pb["fog"] and {key: pb[key] for key in pa} == pa
        assert not np.array_equal(a, b)
    assert _call(foggy, image, 3, mode="blur_only")[1]["fog"] is False
    cleared = _corruptor(**dict(base, env_clear_probability=1.0))
    for index in range(10):
        (_, pa), (c, pc) = _call(plain, image, index), _call(cleared, image, index)
        assert pc["env_clear"] and pc["light"] is False and pc["illumination"] is False and pc["low_light"] is False
        # Everything else -- the camera blur and the sensor noise -- is drawn as before.
        assert {key: pc[key] for key in ("defocus", "motion", "photon_count", "read_noise_std")} == \
            {key: pa[key] for key in ("defocus", "motion", "photon_count", "read_noise_std")}


def test_kaggle_env_is_p23_plus_fog_clear_frames_and_less_motion_blur():
    from qjepa.config import FOG_KEYS
    illum = serializable_config(load_config("configs/kaggle_illum.yaml"))
    env = serializable_config(load_config("configs/kaggle_env.yaml"))
    for config in (illum, env):
        config.pop("_config_path", None)
        config["runtime"].pop("output_dir")
    for key in (*FOG_KEYS, "env_clear_probability"):
        env["corruption"]["image"].pop(key)
    assert env["corruption"]["image"].pop("motion_probability") < illum["corruption"]["image"].pop("motion_probability")
    assert env["corruption"]["image"].pop("motion_length_px") != illum["corruption"]["image"].pop("motion_length_px")
    assert env == illum
    full = load_config("configs/kaggle_env.yaml")
    validate_config(full)
    for phase in ("phase1", "phase2"):
        assert configuration_hash(full, phase) != configuration_hash(load_config("configs/kaggle_illum.yaml"), phase)
    missing = copy.deepcopy(full)
    del missing["corruption"]["image"]["fog_horizon"]
    with pytest.raises(ValueError, match="fog_horizon"):
        validate_config(missing)
    for key, value in (("fog_probability", 1.5), ("fog_density", [0.0, 1.0]), ("fog_airlight", [0.5, 1.5]),
                       ("env_clear_probability", -0.1)):
        bad = copy.deepcopy(full)
        bad["corruption"]["image"][key] = value
        with pytest.raises(ValueError):
            validate_config(bad)
