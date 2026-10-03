"""(b) model.encoder_norm: centre -- encoders that can tell a strong signal from a weak one.

GroupNorm divides by each group's spread, so the IMU encoder returns the same
feature for x and 3x, and the image encoder barely moves when a frame is three
times darker. A random-init probe read 0% of 4-16 px edges through GroupNorm and
25% with a norm that only subtracts the group mean. Pins: the centre norm keeps
scale and removes the mean; the encoders use it only when asked; configs without
the key build the GroupNorm encoders they were trained with; bad values are
refused; the norm changes both hashes (phase 1 must retrain).
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from qjepa.config import build_backbone, load_config, validate_config
from qjepa.models.blocks import CentreNorm
from qjepa.training.checkpoints import configuration_hash


def _config(norm=None):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    if norm is not None:
        config["model"]["encoder_norm"] = norm
    validate_config(config)
    return config


def test_the_centre_norm_removes_each_groups_mean_and_keeps_its_scale():
    norm = CentreNorm(2, 4)
    x = torch.randn(3, 4, 5, 5) + 7.0
    y = norm(x)
    for group in (slice(0, 2), slice(2, 4)):
        assert torch.allclose(y[:, group].mean(dim=(1, 2, 3)), torch.zeros(3), atol=1e-5)
    assert torch.allclose(norm(3.0 * x), 3.0 * y, atol=1e-4)          # GroupNorm would return y
    with pytest.raises(ValueError, match="divisible"):
        CentreNorm(3, 4)


def test_the_imu_encoder_tells_a_strong_window_from_a_weak_one_only_with_the_centre_norm():
    torch.manual_seed(0)
    window = torch.randn(2, 6, 32)
    for norm, scale_blind in (("group", True), ("centre", False)):
        torch.manual_seed(1)
        backbone = build_backbone(_config(norm))
        with torch.no_grad():
            weak = backbone.encode_imu_dense(window)
            strong = backbone.encode_imu_dense(3.0 * window)
        change = float((strong - weak).norm() / weak.norm())
        assert (change < 1e-3) == scale_blind, (norm, change)


def test_only_the_encoders_change_and_old_configs_build_groupnorm():
    plain = build_backbone(_config())
    centre = build_backbone(_config("centre"))
    assert any(isinstance(m, nn.GroupNorm) for m in plain.image_encoder.modules())
    for encoder in (centre.image_encoder, centre.imu_encoder):
        assert not any(isinstance(m, nn.GroupNorm) for m in encoder.modules())
        assert any(isinstance(m, CentreNorm) for m in encoder.modules())
    # Same parameter names and shapes: only the arithmetic of the norm differs.
    assert {k: v.shape for k, v in plain.state_dict().items()} == {k: v.shape for k, v in centre.state_dict().items()}


def test_a_bad_norm_is_refused_and_the_norm_is_in_both_hashes():
    with pytest.raises(ValueError, match="encoder_norm"):
        _config("batch")
    for phase in ("phase1", "phase2"):
        assert configuration_hash(_config("centre"), phase) != configuration_hash(_config(), phase)
