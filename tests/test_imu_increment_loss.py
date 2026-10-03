"""phase2.imu_increment_weight -- score the IMU on what it integrates to (Brossard 2020).

Every IMU term so far scores one sample at a time, so a slow offset and white
jitter of the same RMS cost the same. Integrated over a window, the offset grows
with the window and the jitter only with its square root: an orientation or
velocity built from the restored IMU drifts with the offset, not the jitter. The
term sums the error over non-overlapping windows and divides by sqrt(window), so
white error weighs the same at every window while drift weighs sqrt(window) more.
Pins: zero when the IMU matches; an offset costs more than jitter of equal RMS;
configs without the key score as before; bad windows fail; the term is in the
phase-2 hash only.
"""

from __future__ import annotations

import copy

import pytest
import torch

from qjepa.config import load_config, validate_config
from qjepa.training.checkpoints import configuration_hash
from qjepa.training.losses import imu_increment_loss, phase2_reconstruction_loss


def test_a_matching_imu_costs_nothing_and_drift_costs_more_than_jitter():
    generator = torch.Generator().manual_seed(0)
    clean = torch.randn(4, 3, 128, generator=generator)
    assert float(imu_increment_loss(clean, clean, (8, 32))) == 0.0
    jitter = torch.randn(4, 3, 128, generator=generator) * 0.1
    offset = torch.full_like(jitter, float(jitter.square().mean().sqrt()))
    assert float(offset.square().mean()) == pytest.approx(float(jitter.square().mean()), rel=1e-5)
    drift_cost = float(imu_increment_loss(clean + offset, clean, (8, 32)))
    jitter_cost = float(imu_increment_loss(clean + jitter, clean, (8, 32)))
    assert drift_cost > 3.0 * jitter_cost
    with pytest.raises(ValueError, match="window"):
        imu_increment_loss(clean, clean, (5,))


def test_the_reconstruction_loss_adds_it_only_when_asked():
    generator = torch.Generator().manual_seed(1)
    image, imu = torch.rand(2, 3, 8, 8, generator=generator), torch.randn(2, 6, 32, generator=generator)
    noisy = imu + 0.2
    plain, parts = phase2_reconstruction_loss(image, image, noisy, imu, beta=0.05)
    assert "imu_accel_increment" not in parts
    weighted, parts = phase2_reconstruction_loss(image, image, noisy, imu, beta=0.05,
                                                 increment_weight=0.5, increment_windows=(8, 32))
    assert {"imu_accel_increment", "imu_gyro_increment"} <= set(parts)
    expected = 0.5 * (parts["imu_accel_increment"] + parts["imu_gyro_increment"])
    assert float(weighted - plain) == pytest.approx(float(expected), rel=1e-5)


def _config(**phase2):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(phase2)
    validate_config(config)
    return config


@pytest.mark.parametrize("change, message", [
    (dict(imu_increment_weight=-1.0), "imu_increment_weight"),
    (dict(imu_increment_weight=0.5, imu_increment_windows=[5]), "imu_increment_windows"),
    (dict(imu_increment_weight=0.5, imu_increment_windows=[]), "imu_increment_windows"),
])
def test_bad_settings_are_refused(change, message):
    with pytest.raises(ValueError, match=message):
        _config(**change)


def test_the_term_is_in_the_phase2_hash_only():
    on, off = _config(imu_increment_weight=0.5, imu_increment_windows=[8, 32]), _config()
    assert configuration_hash(on, "phase1") == configuration_hash(off, "phase1")
    assert configuration_hash(on, "phase2") != configuration_hash(off, "phase2")
