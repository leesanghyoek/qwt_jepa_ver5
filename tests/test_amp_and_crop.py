"""p16: fp16 mixed precision for phase 2, and the VGG loss on a random crop.

AMP only runs on CUDA; on CPU the same config must train in fp32 unchanged,
which is what these tests can check here. They also pin that the forward hands
fp32 to every loss, that NAFNet's LayerNorm keeps fp32 statistics under fp16,
that the scaler state reaches the checkpoint, and that the crop really shrinks
what VGG sees.
"""

from __future__ import annotations

import copy

import pytest
import torch

import qjepa.training.perceptual as perceptual_module
from qjepa.config import build_decoders, build_phase1_model, load_config, seed_everything, validate_config
from qjepa.data import ImuNormalizer
from qjepa.execution import RestorationForward
from qjepa.models import RestorationSystem
from qjepa.models.decoders import LayerNorm2d
from qjepa.training.perceptual import PerceptualLoss, load_vgg16_features
from qjepa.training.phase2 import Phase2Trainer

SMALL_NAF = dict(split_edge_naf_widths=[4, 6, 8, 12, 16], split_edge_naf_enc_blocks=[1, 1, 1, 1],
                 split_edge_naf_middle_blocks=1, split_edge_naf_dec_blocks=[1, 1, 1, 1],
                 split_edge_refiner_width=8, split_edge_refiner_blocks=1)


def _batch():
    times = torch.arange(32).float().mul(0.01).repeat(1, 1)
    clean, imu = torch.rand(1, 3, 32, 32), torch.randn(1, 6, 32)
    return {"image_clean": clean, "image_noisy": (clean * 0.35).clamp(0, 1), "imu_clean_phys": imu,
            "imu_noisy_phys": imu + 0.05 * torch.randn_like(imu), "image_time": times.mean(dim=1),
            "imu_times": times, "sample_id": ["s"]}


def _system(config):
    seed_everything(3)
    phase1 = build_phase1_model(config, ImuNormalizer())
    seed_everything(config["phase2"]["decoder_initialization_seed"])
    return RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config))


def test_precision_accepts_amp_and_rejects_anything_else():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    for precision in ("fp32", "amp_fp16"):
        config["phase2"]["precision"] = precision
        validate_config(config)
    config["phase2"]["precision"] = "bf16"
    with pytest.raises(ValueError, match="precision"):
        validate_config(config)


def test_the_recipe_trains_phase2_in_amp_with_a_heavier_cropped_vgg_term():
    config = load_config("configs/kaggle_tartanair_v2.yaml")
    assert config["phase2"]["precision"] == "amp_fp16" and config["phase1"]["precision"] == "fp32"
    assert config["phase2"]["perceptual_weight"] == 0.5 and config["phase2"]["perceptual_crop"] == 128


@pytest.mark.parametrize("crop", [3, 64])
def test_a_crop_that_does_not_fit_is_rejected(crop):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))       # 32 x 32 frames
    config["phase2"].update(perceptual_crop=crop, perceptual_weight=0.5)
    with pytest.raises(ValueError, match="perceptual_crop"):
        validate_config(config)


def test_on_cpu_an_amp_config_trains_in_fp32_and_checkpoints_the_scaler():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(SMALL_NAF, precision="amp_fp16")
    validate_config(config)
    trainer = Phase2Trainer(_system(config), config, torch.device("cpu"), "phase1.pt")
    assert trainer.amp is False and trainer.forward_model.amp is False
    metrics = trainer.step([_batch()] * config["phase2"]["gradient_accumulation"])
    assert metrics["skipped"] is False and "amp_overflow" not in metrics
    assert "scaler" in trainer.checkpoint_payload(config)


def test_the_forward_hands_fp32_to_the_losses():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(SMALL_NAF)
    forward = RestorationForward(_system(config))
    batch = _batch()
    out = forward(batch["image_noisy"], batch["imu_noisy_phys"], batch["image_time"], batch["imu_times"])
    assert all(value.dtype == torch.float32 for value in out.values() if value.is_floating_point())


def test_layernorm_keeps_fp32_statistics_for_fp16_input():
    torch.manual_seed(0)
    norm = LayerNorm2d(64)
    x = torch.randn(2, 64, 8, 8) * 3 + 5
    reference = norm(x)
    half = norm(x.half())
    assert half.dtype == torch.float16
    assert torch.allclose(half.float(), reference, atol=1e-2)


def test_the_crop_is_what_vgg_sees(monkeypatch):
    torch.manual_seed(0)
    features = load_vgg16_features(False)
    seen = []
    features[0].register_forward_hook(lambda module, inputs, output: seen.append(tuple(inputs[0].shape[-2:])))
    loss = PerceptualLoss(features, crop=16)
    image = torch.rand(1, 3, 32, 32)
    assert float(loss(image, image)) == pytest.approx(0.0, abs=1e-6)
    assert seen == [(16, 16), (16, 16)]                      # target and prediction, same window
    whole = PerceptualLoss(features, crop=0)
    seen.clear()
    whole(image, image)
    assert seen == [(32, 32), (32, 32)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 autocast needs CUDA")
def test_amp_step_on_cuda_runs_and_reports_the_scale():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(SMALL_NAF, precision="amp_fp16")
    config["runtime"]["gpu_count"] = 1
    trainer = Phase2Trainer(_system(config), config, torch.device("cuda"), "phase1.pt")
    metrics = trainer.step([_batch()] * config["phase2"]["gradient_accumulation"])
    assert trainer.amp is True and metrics["skipped"] is False and metrics["amp_scale"] > 0
