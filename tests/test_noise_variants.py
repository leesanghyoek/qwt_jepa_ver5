"""Noise B's per-frame variety: mostly dark and grainy, sometimes only one of them.

The variant is drawn after every other parameter, so the probabilities cannot
move any earlier draw, and a config without them corrupts exactly as before.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from qjepa.config import load_config
from qjepa.corruptions.image import LowLightImageCorruptionConfig, LowLightImageCorruptor

VARIED = dict(noise_only_probability=0.1, low_light_only_probability=0.1)


def _corruptor(**changes):
    config = replace(LowLightImageCorruptionConfig(), clean_probability=0.0, **changes)
    return LowLightImageCorruptor(config, master_seed=11)


def _draws(corruptor, count=3000):
    return [corruptor._parameters("train", 0, "traj", 0.05 * segment, "full") for segment in range(count)]


def test_the_probabilities_move_no_other_draw():
    plain, varied = _draws(_corruptor(), 200), _draws(_corruptor(**VARIED), 200)
    assert all(p["low_light"] and p["sensor_noise"] for p in plain)
    flags = {"low_light", "sensor_noise"}
    for a, b in zip(plain, varied):
        assert {k: v for k, v in a.items() if k not in flags} == {k: v for k, v in b.items() if k not in flags}


def test_about_one_frame_in_ten_is_noise_only_and_one_in_ten_dark_only():
    draws = _draws(_corruptor(**VARIED))
    noise_only = np.mean([not p["low_light"] for p in draws])
    dark_only = np.mean([not p["sensor_noise"] for p in draws])
    assert 0.08 < noise_only < 0.12 and 0.08 < dark_only < 0.12
    assert not any(not p["low_light"] and not p["sensor_noise"] for p in draws)


def _frame_with(corruptor, **flags):
    for segment in range(3000):
        params = corruptor._parameters("train", 0, "traj", 0.05 * segment, "full")
        if all(params[key] == value for key, value in flags.items()):
            return 0.05 * segment
    raise AssertionError(f"no frame with {flags}")


def test_a_noise_only_frame_keeps_its_brightness_and_a_dark_only_frame_has_no_grain():
    corruptor = _corruptor(**VARIED)
    clean = np.random.default_rng(0).uniform(0.2, 0.8, (48, 48, 3)).astype(np.float32)
    call = dict(split="train", realization=0, trajectory="traj", frame_index=3)
    bright = _frame_with(corruptor, low_light=False)
    noisy, _ = corruptor(clean, timestamp=bright, **call)
    assert abs(float(noisy.mean()) - float(clean.mean())) < 0.03       # not darkened
    optics_only, _ = corruptor(clean, timestamp=bright, mode="blur_only", **call)
    assert np.abs(noisy - optics_only).mean() > 1e-3                     # but grainy
    dark = _frame_with(corruptor, sensor_noise=False)
    darkened, _ = corruptor(clean, timestamp=dark, **call)
    no_sensor, _ = corruptor(clean, timestamp=dark, mode="blur_low_light", **call)
    assert np.array_equal(darkened, no_sensor)                            # no grain at all
    assert float(darkened.mean()) < float(clean.mean()) - 0.05


def test_named_scenarios_ignore_the_variant():
    corruptor = _corruptor(**VARIED)
    clean = np.full((16, 16, 3), 0.6, dtype=np.float32)
    bright = _frame_with(corruptor, low_light=False)
    darkened, _ = corruptor(clean, split="train", realization=0, trajectory="traj", timestamp=bright,
                            frame_index=0, mode="low_light_only")
    assert float(darkened.mean()) < 0.55


def test_probabilities_that_do_not_fit_are_rejected():
    with pytest.raises(ValueError, match="noise_only_probability"):
        _corruptor(noise_only_probability=0.7, low_light_only_probability=0.5)


def test_the_recipe_trains_on_noise_b():
    image = load_config("configs/kaggle_tartanair_v2.yaml")["corruption"]["image"]
    assert image["defocus_sigma_px"] == [0.30, 0.95]
    assert image["photon_count"] == [2500.0, 15000.0]
    assert image["noise_only_probability"] == image["low_light_only_probability"] == 0.10
