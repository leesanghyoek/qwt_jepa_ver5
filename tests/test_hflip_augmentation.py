"""(d) phase1/phase2.augment_hflip -- mirror training pairs, image and IMU together.

Phase 2 trained without any augmentation. A horizontal flip doubles the views of
every frame, but the IMU has to be mirrored with it or fusion learns a world that
does not exist. lcam_front is the IMU body frame (x forward, y right, z down); a
mirror across the forward-down plane flips the polar vector's y and every axial
component but y: accel (ax, -ay, az), gyro (-gx, gy, -gz). Pins: the signs match
the measured gyro-to-image geometry (yaw moves the image sideways, so a mirrored
yaw must move it the other way); the dataset flips clean, noisy and the noise-free
reference alike and only when asked; the draw is reproducible and about half;
bad values fail; the flag sits in each phase's own hash.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from qjepa.cli import _dataset
from qjepa.config import load_config, validate_config
from qjepa.corruptions.motion import exposure_path
from qjepa.data import PairedCameraImuDataset
from qjepa.data.dataset import IMU_MIRROR_SIGNS, mirror_draw, mirror_imu
from qjepa.data.manifest import build_manifest
from qjepa.training.checkpoints import configuration_hash
from test_kaggle_workflow import _write_dataset


def test_the_mirror_signs_follow_the_measured_camera_geometry():
    assert IMU_MIRROR_SIGNS == (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
    times = np.arange(20) * 0.01
    gyro = np.stack([np.full(20, 0.3), np.full(20, -0.2), np.full(20, 0.5)], axis=1)
    imu = np.concatenate([np.zeros((20, 3)), gyro], axis=1)
    mirrored = mirror_imu(imu)
    u, v, roll = exposure_path(gyro, times, 0.1, 0.05, focal_length_px=128.0)
    mu, mv, mroll = exposure_path(mirrored[:, 3:], times, 0.1, 0.05, focal_length_px=128.0)
    assert np.allclose(mu, -u) and np.allclose(mv, v) and mroll == pytest.approx(-roll)
    assert np.array_equal(mirror_imu(mirrored), imu)


def _config(**phase):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    for name, value in phase.items():
        config[name]["augment_hflip"] = value
    validate_config(config)
    return config


def test_the_dataset_flips_every_view_together_and_only_when_asked(tmp_path):
    root = tmp_path / "dataset"
    _write_dataset(root)
    manifest = build_manifest(root, window=32)
    config = _config()
    plain = _dataset(config, manifest, "train", fixed_realization=False, sensor_reference=True)
    flipped = _dataset(config, manifest, "train", fixed_realization=False, sensor_reference=True,
                       hflip_probability=1.0)
    a, b = plain[0], flipped[0]
    for key in ("image_clean", "image_noisy", "image_noise_free"):
        assert torch.equal(b[key], a[key].flip(-1)), key
    signs = torch.tensor(IMU_MIRROR_SIGNS)
    for key in ("imu_clean_phys", "imu_noisy_phys"):
        assert torch.equal(b[key], a[key] * signs), key
    assert torch.equal(a["image_degradation"], b["image_degradation"])
    assert "image_noise_free" not in _dataset(config, manifest, "train", fixed_realization=False)[0]


def test_the_draw_is_reproducible_and_about_half():
    draws = [mirror_draw(73133, f"sample-{index}", 0, 0.5) for index in range(400)]
    assert draws == [mirror_draw(73133, f"sample-{index}", 0, 0.5) for index in range(400)]
    assert 0.4 < sum(draws) / len(draws) < 0.6
    # A new epoch (realization) draws again; probability 0 and 1 are certain.
    assert draws != [mirror_draw(73133, f"sample-{index}", 1, 0.5) for index in range(400)]
    assert not any(mirror_draw(73133, f"sample-{index}", 0, 0.0) for index in range(50))
    assert all(mirror_draw(73133, f"sample-{index}", 0, 1.0) for index in range(50))


def test_bad_values_fail():
    with pytest.raises(ValueError, match="augment_hflip"):
        _config(phase2="yes")
    with pytest.raises(ValueError, match="hflip_probability"):
        PairedCameraImuDataset([object()], hflip_probability=1.5)


def test_each_phase_hashes_its_own_flag():
    off = _config()
    phase1, phase2 = _config(phase1=True), _config(phase2=True)
    assert configuration_hash(phase1, "phase1") != configuration_hash(off, "phase1")
    assert configuration_hash(phase1, "phase2") == configuration_hash(off, "phase2")
    assert configuration_hash(phase2, "phase2") != configuration_hash(off, "phase2")
    assert configuration_hash(phase2, "phase1") == configuration_hash(off, "phase1")
