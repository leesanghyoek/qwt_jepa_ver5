"""p14: the in-place small-detail metric, the VGG feature loss, the wider top level.

The metric must credit only detail at the right place: noise and invented
texture add power but no in-place detail. The feature loss must stay frozen,
out of checkpoints, and out of smoke runs (its weights are a 528 MB download);
the tests use randomly initialised VGG layers for that reason.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

import qjepa.training.perceptual as perceptual_module
from qjepa.config import build_decoders, build_phase1_model, load_config, seed_everything, validate_config
from qjepa.data import ImuNormalizer
from qjepa.evaluation.metrics import spectral_ratios
from qjepa.models import RestorationSystem
from qjepa.training.perceptual import PerceptualLoss, load_vgg16_features
from qjepa.training.phase2 import Phase2Trainer


def _textured(seed=0, size=64):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, size, size, generator=generator)


def test_in_place_detail_is_one_for_the_clean_frame_and_ignores_noise():
    clean = _textured()
    same = spectral_ratios(clean, clean)
    # Not exactly 1: the denominator carries a floor of 0.1% of the frame's power.
    assert same["image_fine_detail_in_place"] == pytest.approx(1.0, abs=0.01)
    assert same["image_edge_in_place"] == pytest.approx(1.0, abs=0.01)
    noisy = clean + 0.3 * torch.randn(clean.shape, generator=torch.Generator().manual_seed(1))
    ratios = spectral_ratios(noisy, clean)
    assert ratios["image_stripe_power"] > 1.5                      # the noise adds power ...
    assert ratios["image_fine_detail_in_place"] == pytest.approx(1.0, abs=0.1)   # ... but no detail


def test_in_place_detail_falls_for_blur_shift_and_contrast_loss():
    clean = _textured()
    blurred = F.avg_pool2d(F.pad(clean, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    assert spectral_ratios(blurred, clean)["image_fine_detail_in_place"] < 0.35
    shifted = torch.roll(clean, shifts=1, dims=-1)                 # right detail, wrong place
    assert spectral_ratios(shifted, clean)["image_fine_detail_in_place"] < 0.5
    assert spectral_ratios(0.5 * clean, clean)["image_fine_detail_in_place"] == pytest.approx(0.5, abs=0.01)


def _random_vgg(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(perceptual_module, "load_vgg16_features", lambda pretrained=True: load_vgg16_features(False))


def test_the_feature_loss_is_frozen_and_sees_smoothing(monkeypatch):
    _random_vgg(monkeypatch)
    loss = PerceptualLoss()
    loss.train(True)
    assert not loss.training and not any(p.requires_grad for p in loss.parameters())
    clean = _textured(size=32)
    assert float(loss(clean, clean)) == pytest.approx(0.0, abs=1e-6)
    smooth = F.avg_pool2d(F.pad(clean, (2, 2, 2, 2), mode="replicate"), 5, stride=1).requires_grad_(True)
    value = loss(smooth, clean)
    assert float(value.detach()) > 0.05
    value.backward()
    assert smooth.grad is not None and float(smooth.grad.abs().sum()) > 0


def test_the_recipe_widens_the_top_level_and_turns_the_feature_loss_on():
    phase2 = load_config("configs/kaggle_tartanair_v2.yaml")["phase2"]
    assert phase2["split_edge_naf_widths"][0] == 48
    assert phase2["split_edge_naf_enc_blocks"][0] == phase2["split_edge_naf_dec_blocks"][0] == 3
    assert phase2["perceptual_weight"] > 0
    assert load_config("configs/smoke.yaml")["phase2"]["perceptual_weight"] == 0.0


def test_a_negative_feature_weight_is_rejected():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"]["perceptual_weight"] = -0.1
    with pytest.raises(ValueError, match="perceptual_weight"):
        validate_config(config)


def test_phase2_trains_with_the_feature_loss_and_keeps_vgg_out_of_the_checkpoint(monkeypatch):
    _random_vgg(monkeypatch)
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(perceptual_weight=0.05, perceptual_crop=16, split_edge_naf_widths=[4, 6, 8, 12, 16],
                            split_edge_naf_enc_blocks=[1, 1, 1, 1], split_edge_naf_middle_blocks=1,
                            split_edge_naf_dec_blocks=[1, 1, 1, 1])
    validate_config(config)
    seed_everything(3)
    phase1 = build_phase1_model(config, ImuNormalizer())
    seed_everything(config["phase2"]["decoder_initialization_seed"])
    system = RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config))
    trainer = Phase2Trainer(system, config, torch.device("cpu"), "phase1.pt")
    times = torch.arange(32).float().mul(0.01).repeat(1, 1)
    clean, imu = torch.rand(1, 3, 32, 32), torch.randn(1, 6, 32)
    batch = {"image_clean": clean, "image_noisy": (clean * 0.35).clamp(0, 1), "imu_clean_phys": imu,
             "imu_noisy_phys": imu + 0.05 * torch.randn_like(imu), "image_time": times.mean(dim=1),
             "imu_times": times, "sample_id": ["s"]}
    metrics = trainer.step([batch] * config["phase2"]["gradient_accumulation"])
    assert metrics["skipped"] is False and metrics["image_perceptual"] > 0
    payload = trainer.checkpoint_payload(config)
    # Same keys as a system that never had a feature loss: VGG is not in the checkpoint.
    fresh = RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config))
    assert set(payload["system"]) == set(fresh.state_dict())
    assert not any(p is q for p in trainer.parameters for q in trainer.perceptual.parameters())
