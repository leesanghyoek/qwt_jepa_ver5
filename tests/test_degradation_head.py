"""(c) phase1.degradation_weight and predictor_degradation_condition -- a latent that knows how bad the frame is.

JEPA maps a noisy frame and its clean twin to the same target, which teaches the
encoder to drop how blurred or noisy the frame was. A blind decoder that does not
know the blur has to average over every blur it might be, and kernel mismatch is
what over-smooths (IKC, Gu 2019). The head regresses the corruption's own
parameters from ZI, so ZI must keep them (DASR, Wang 2021); the predictor reads
the head's estimate as an action, as in IWM (Garrido 2024). Pins: the vector reads
the parameters the corruptor drew, zero for a clean frame; the smoke batch
carries it; the FiLM starts as the identity; the head trains the encoder; the
condition is detached; old configs build no head; bad settings fail.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from qjepa.cli import _synthetic_batch
from qjepa.config import build_phase1_model, load_config, seed_everything, validate_config
from qjepa.corruptions.image import DEGRADATION_FEATURES, degradation_vector
from qjepa.data import ImuNormalizer
from qjepa.execution import Phase1Forward
from qjepa.models.predictors import SpatialPredictor
from qjepa.training.checkpoints import configuration_hash
from qjepa.training.phase1 import Phase1Trainer


def _config(**phase1):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase1"].update(phase1)
    validate_config(config)
    return config


def _params(**changes):
    params = {"mode": "full", "clean": False, "low_light": True, "sensor_noise": True, "defocus": True,
              "defocus_sigma": 0.9, "motion": True, "motion_length": 5, "motion_from_imu": False,
              "downsample": True, "downsample_scale": 0.85, "exposure_gain": 0.4, "tone_gamma": 0.6,
              "photon_count": 2500.0, "read_noise_std": 2.0 / 255.0, "jpeg": True, "jpeg_quality": 60}
    params.update(changes)
    return params


def test_the_vector_reads_what_the_corruptor_drew():
    vector = dict(zip(DEGRADATION_FEATURES, degradation_vector(_params())))
    assert len(DEGRADATION_FEATURES) == 8
    assert vector["defocus_sigma"] == pytest.approx(0.9 / 1.5)
    assert vector["motion_length"] == pytest.approx(0.5)
    assert vector["darkness"] == pytest.approx(0.6) and vector["jpeg"] == pytest.approx(0.4)
    assert all(0.0 <= value <= 1.5 for value in vector.values())
    assert not degradation_vector(_params(clean=True)).any()
    # Stages that did not run leave their entries at zero.
    blur_only = dict(zip(DEGRADATION_FEATURES, degradation_vector(_params(mode="blur_only"))))
    assert blur_only["defocus_sigma"] > 0 and blur_only["darkness"] == 0 and blur_only["shot_noise"] == 0
    noise_only = dict(zip(DEGRADATION_FEATURES, degradation_vector(_params(mode="sensor_noise_only"))))
    assert noise_only["defocus_sigma"] == 0 and noise_only["shot_noise"] > 0
    imu_blur = dict(zip(DEGRADATION_FEATURES, degradation_vector(_params(motion_from_imu=True, path_span_px=3.0))))
    assert imu_blur["motion_length"] == pytest.approx(0.3)


def test_the_smoke_batch_carries_the_vector():
    batch = _synthetic_batch(_config())
    assert batch["image_degradation"].shape == (2, len(DEGRADATION_FEATURES))
    assert batch["image_degradation"].dtype == torch.float32


def test_the_conditioned_predictor_starts_as_the_unconditioned_one():
    torch.manual_seed(0)
    plain = SpatialPredictor(16, 32, spatial_dims=2)
    torch.manual_seed(0)
    conditioned = SpatialPredictor(16, 32, spatial_dims=2, condition_dim=8)
    dense = torch.randn(2, 16, 4, 4)
    assert torch.allclose(plain(dense)[0], conditioned(dense, condition=torch.rand(2, 8))[0], atol=1e-6)
    with pytest.raises(ValueError, match="condition"):
        conditioned(dense)


def test_the_head_trains_the_encoder_and_the_condition_is_detached():
    config = _config(degradation_weight=0.1, predictor_degradation_condition=True)
    seed_everything(0)
    model = build_phase1_model(config, ImuNormalizer())
    assert model.degradation_head is not None and model.image_predictor.condition is not None
    batch = _synthetic_batch(config)
    features = Phase1Forward(model)(batch["image_noisy"], batch["imu_noisy_phys"], batch["image_clean"],
                                    batch["imu_clean_phys"], batch["image_time"], batch["imu_times"])
    assert features["degradation_estimate"].shape == batch["image_degradation"].shape
    # The JEPA loss cannot reach the head through the condition.
    features["prediction_i"].sum().backward()
    assert all(p.grad is None for p in model.degradation_head.parameters())
    model.zero_grad(set_to_none=True)
    head = [p.detach().clone() for p in model.degradation_head.parameters()]
    trainer = Phase1Trainer(model, config, torch.device("cpu"))
    metrics = trainer.step(batch)
    assert metrics["skipped"] is False and math.isfinite(metrics["degradation"])
    assert any(not torch.equal(a, b) for a, b in zip(head, model.degradation_head.parameters()))


def test_old_configs_build_no_head_and_keep_their_hash():
    config = _config()
    model = build_phase1_model(config, ImuNormalizer())
    assert model.degradation_head is None and model.image_predictor.condition is None
    assert not any(key.startswith("degradation_head") for key in model.state_dict())
    metrics = Phase1Trainer(model, config, torch.device("cpu")).step(_synthetic_batch(config))
    assert "degradation" not in metrics
    with_head = _config(degradation_weight=0.1)
    assert configuration_hash(with_head, "phase1") != configuration_hash(config, "phase1")
    assert configuration_hash(with_head, "phase2") == configuration_hash(config, "phase2")


@pytest.mark.parametrize("change, message", [
    (dict(degradation_weight=-0.1), "degradation_weight"),
    (dict(predictor_degradation_condition=True), "predictor_degradation_condition"),
    (dict(degradation_weight=0.1, predictor_degradation_condition="yes"), "predictor_degradation_condition"),
    (dict(degradation_weight=0.1, predictor_degradation_condition=True, predictor_type="token",
          image_mask_ratio=0.0, imu_mask_ratio=0.0, multiscale_fine_weight=0.0, multiscale_finer_weight=0.0),
     "spatial"),
])
def test_bad_settings_are_refused(change, message):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase1"].update(change)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_the_vector_is_float32_and_finite_for_every_mode():
    for mode in ("full", "clean", "low_light_only", "blur_only", "sensor_noise_only", "blur_low_light"):
        vector = degradation_vector(_params(mode=mode))
        assert vector.dtype == np.float32 and np.isfinite(vector).all()
