"""configs/kaggle_flareblur.yaml (p37_flareblur): p36_flareimage, mo hon mot chut.

Pins: p37 is p36 with only three blur keys raised (defocus 0.42 -> 0.5 of frames, sigma up to 0.85 -> 0.95 px,
motion blur 0.1 -> 0.15 of frames); both hashes change while p36's stay; on every frame the blur only grows -- a
frame blurred under p36 is blurred under p37 with a sigma no smaller and the same motion streak, every other draw
of the frame unchanged -- and blur-only frames lose PSNR against the clean frame.
"""

from __future__ import annotations

import copy

import numpy as np

from qjepa.config import build_corruptors, load_config, serializable_config, validate_config
from qjepa.training.checkpoints import configuration_hash

BLUR = {"defocus_probability": 0.5, "defocus_sigma_px": [0.3, 0.95], "motion_probability": 0.15}


def _configs():
    return load_config("configs/kaggle_flareimage.yaml"), load_config("configs/kaggle_flareblur.yaml")


def test_kaggle_flareblur_is_p36_with_only_three_blur_keys_raised_and_changes_both_hashes():
    p36, p37 = (serializable_config(copy.deepcopy(config)) for config in _configs())
    assert {key: p37["corruption"]["image"][key] for key in BLUR} == BLUR
    assert {key: p36["corruption"]["image"][key] for key in BLUR} == {
        "defocus_probability": 0.42, "defocus_sigma_px": [0.3, 0.85], "motion_probability": 0.1}
    assert p37["runtime"].pop("output_dir").endswith("outputs/p37_flareblur")
    p36["runtime"].pop("output_dir")
    for config in (p36, p37):
        for key in BLUR:
            config["corruption"]["image"].pop(key)
    assert p37 == p36
    p36, p37 = _configs()
    validate_config(p37)
    for phase in ("phase1", "phase2"):
        assert configuration_hash(p37, phase) != configuration_hash(p36, phase)
    # p36 hashes as it did when its notebook commit was pinned.
    assert configuration_hash(p36, "phase1").startswith("472a932cd3e3")
    assert configuration_hash(p36, "phase2").startswith("a07e3f510503")


def test_every_frame_is_at_least_as_blurred_under_p37_and_nothing_else_moves():
    p36, p37 = (build_corruptors(config)[0] for config in _configs())
    before_count = after_count = motion_before = motion_after = 0
    frames = ([("train", f"Env{k % 7}/Data_easy/P{k % 11:03d}", 0.05 * k) for k in range(2000)]
              + [("test", "EnvT/Data_hard/P001", 0.05 * k) for k in range(400)])
    for split, trajectory, timestamp in frames:
        a = p36._parameters(split, 0, trajectory, timestamp, "full")
        b = p37._parameters(split, 0, trajectory, timestamp, "full")
        if a["defocus"]:
            assert b["defocus"] and b["defocus_sigma"] >= a["defocus_sigma"]
        if a["motion"]:
            assert b["motion"]
        moved = {"defocus", "defocus_sigma", "motion"}
        assert {key: value for key, value in a.items() if key not in moved} == \
            {key: value for key, value in b.items() if key not in moved}
        before_count += a["defocus"]
        after_count += b["defocus"]
        motion_before += a["motion"]
        motion_after += b["motion"]
    total = len(frames)
    assert abs(before_count / total - 0.42) < 0.04 and abs(after_count / total - 0.5) < 0.04
    assert abs(motion_before / total - 0.1) < 0.03 and abs(motion_after / total - 0.15) < 0.03


def test_blur_only_frames_lose_psnr_under_p37():
    p36, p37 = (build_corruptors(config)[0] for config in _configs())
    rng = np.random.default_rng(0)
    texture = np.clip(0.5 + 0.25 * rng.standard_normal((64, 64, 3)), 0, 1).astype(np.float32)
    psnr = {"p36": [], "p37": []}
    for k in range(60):
        for name, corruptor in (("p36", p36), ("p37", p37)):
            blurred, _ = corruptor(texture, split="train", realization=0, trajectory=f"Env/Data_easy/P{k:03d}",
                                   timestamp=0.0, frame_index=0, mode="blur_only")
            mse = float(np.mean((np.clip(blurred, 0, 1) - texture) ** 2))
            psnr[name].append(-10 * np.log10(max(mse, 1e-12)))
    before, after = np.array(psnr["p36"]), np.array(psnr["p37"])
    assert np.all(after <= before + 1e-6)
    assert after.mean() < before.mean() - 0.3
