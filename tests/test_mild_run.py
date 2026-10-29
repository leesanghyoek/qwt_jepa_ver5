"""p54_mild: p53_veil with p44's camera blur and darkness (configs/kaggle_mild.yaml).

Pins:
- p54 is p53 plus five corruption.image keys, each p44's value; every hash changes (phase 1 trains again);
- its camera is p44's: no gyro blur, a straight motion blur on ~15% of frames, 2-4 px, defocus sigma <= 0.95 px,
  exposure gain >= 0.5;
- on the same frames and draws, the input keeps clearly more of the clean frame's edges than under p53.
"""

from __future__ import annotations

import copy

import numpy as np
from scipy import ndimage

from qjepa.config import build_corruptors, load_config, serializable_config, validate_config
from qjepa.training.checkpoints import configuration_hash

P44 = load_config("configs/kaggle_uformer.yaml")
P53 = load_config("configs/kaggle_veil.yaml")
P54 = load_config("configs/kaggle_mild.yaml")
CAMERA = ("motion_from_imu", "motion_probability", "motion_length_px", "defocus_sigma_px", "exposure_gain")
TIMES = np.arange(0, 1.28, 0.01)


def test_p54_is_p53_with_p44s_camera_and_trains_every_phase_again():
    p53, p54 = (copy.deepcopy(serializable_config(config)) for config in (P53, P54))
    assert p54["runtime"].pop("output_dir").endswith("/p54_mild")
    p53["runtime"].pop("output_dir")
    for key in CAMERA:
        assert p54["corruption"]["image"].pop(key) == P44["corruption"]["image"][key], key
        p53["corruption"]["image"].pop(key, None)
    assert p54 == p53
    validate_config(P54)
    for phase in ("phase1", "flare", "phase2"):
        assert configuration_hash(P54, phase) != configuration_hash(P53, phase), phase


def _frames():
    """Two textured frames: smooth colour regions plus fine detail, as a photo has."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(2):
        base = ndimage.zoom(rng.random((16, 16, 3)), (16, 16, 1), order=1)
        detail = ndimage.gaussian_filter(rng.random((256, 256, 3)), (1.0, 1.0, 0))
        out.append(np.clip(0.7 * base + 0.6 * (detail - 0.5) + 0.15, 0, 1).astype(np.float32))
    return out


def _corrupt(config, frames, draws=40):
    corruptor, _ = build_corruptors(config)
    rng = np.random.default_rng(1)
    for index, clean in enumerate(frames):
        for draw in range(draws):
            gyro = np.zeros((len(TIMES), 3))
            gyro[:, 1:] = rng.uniform(0.3, 2.5, 2) * rng.choice([-1, 1], 2)
            noisy, params = corruptor(clean, split="train", realization=draw, trajectory=f"t{index}", timestamp=0.64,
                                      frame_index=draw, mode="full", gyro=gyro, imu_times=TIMES)
            yield clean, noisy, params


def test_p54s_camera_is_p44s():
    rows = [params for _, _, params in _corrupt(P54, _frames(), draws=150) if not params["clean"]]
    assert not any(params["motion_from_imu"] for params in rows)
    moved = [params for params in rows if params["motion"]]
    assert 0.08 < len(moved) / len(rows) < 0.22
    assert all(2 <= params["motion_length"] <= 4 for params in moved)
    assert all(params["defocus_sigma"] <= 0.95 for params in rows)
    assert all(params["exposure_gain"] >= 0.5 for params in rows)


def _edges(image):
    y = image @ np.array([0.299, 0.587, 0.114])
    return float(np.hypot(np.diff(y, axis=0)[:, :-1], np.diff(y, axis=1)[:-1]).mean())


def test_the_input_keeps_more_edges_than_under_p53():
    frames = _frames()
    kept = {name: float(np.median([_edges(noisy) / _edges(clean) for clean, noisy, _ in _corrupt(config, frames)]))
            for name, config in (("p53", P53), ("p54", P54))}
    assert kept["p54"] > kept["p53"] + 0.15, kept
