"""Washed-out colour: the metric that shows it and the two remedies.

p8's restored frames were paler than the clean ones. Per-pixel L1 answers an
uncertain hue with the median, which leans to grey, and the colour trunk sees
too little of the frame to undo a global gamma and white balance. These pin the
vividness metric, the global tone/colour head, and the per-image colour
statistics loss -- and that configs without the keys build p8's colour branch.
"""

from __future__ import annotations

import copy

import pytest
import torch

from qjepa.config import build_decoders, build_phase1_model, load_config, seed_everything, validate_config
from qjepa.data import ImuNormalizer
from qjepa.evaluation.metrics import image_metrics
from qjepa.models import RestorationSystem
from qjepa.models.color_edge import chroma, luminance
from qjepa.models.decoders import ColorBranch, GlobalToneColor
from qjepa.training.losses import color_statistics_l1
from qjepa.training.phase2 import Phase2Trainer


def _images(seed=0, size=32):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(2, 3, size, size, generator=generator).clamp(0.05, 1.0)


def _greyer(image, amount):
    """Pull every pixel towards its own luminance: the washed-out look."""
    return (1 - amount) * image + amount * luminance(image)


def test_the_metric_reads_a_washed_out_image_as_less_saturated():
    clean = _images()
    same = image_metrics(clean, clean)
    assert same["image_saturation"] == pytest.approx(same["image_saturation_clean"])
    pale = image_metrics(_greyer(clean, 0.5), clean)
    assert pale["image_saturation"] == pytest.approx(0.5 * pale["image_saturation_clean"], rel=1e-4)
    flat = image_metrics(0.5 + 0.5 * (clean - 0.5), clean)
    assert flat["image_contrast"] == pytest.approx(0.5 * flat["image_contrast_clean"], rel=1e-4)


def test_the_global_head_starts_as_the_identity():
    torch.manual_seed(0)
    head = GlobalToneColor(16, 8)
    image = _images()
    assert torch.allclose(head(torch.randn(2, 16, 4, 4), image), image, atol=1e-6)
    branch = ColorBranch(16, 8, 2, global_tone=True)
    assert torch.allclose(branch(torch.randn(2, 16, 4, 4), image), image, atol=1e-6)


def test_the_global_head_can_express_the_inverse_of_gain_and_gamma():
    """Corruption: y = (k * x) ** gamma per channel. Inverse: M . y ** (1 / gamma), M = diag(1 / k)."""
    clean = _images()
    gain, gamma = torch.tensor([0.9, 0.5, 1.1]), 0.6
    corrupted = (clean * gain[None, :, None, None]) ** gamma
    head = GlobalToneColor(16, 8)
    bias = torch.zeros(13)
    bias[0] = torch.log(torch.tensor(1 / gamma))
    bias[1:10] = (torch.diag(1 / gain) - torch.eye(3)).flatten()
    with torch.no_grad():
        head.head[-1].bias.copy_(bias)
    assert torch.allclose(head(torch.randn(2, 16, 4, 4), corrupted), clean, atol=1e-5)


def test_the_statistics_loss_pushes_a_pale_image_back_to_its_saturation():
    clean = _images(size=16)
    light_clean = luminance(clean)
    pale = _greyer(clean, 0.6).requires_grad_(True)
    assert float(color_statistics_l1(clean, light_clean, clean, light_clean)) == pytest.approx(0.0, abs=1e-6)
    loss = color_statistics_l1(pale, luminance(pale), clean, light_clean)
    assert float(loss.detach()) > 0.01
    loss.backward()
    with torch.no_grad():
        stepped = pale - 5.0 * pale.grad
    saturation = lambda x: float(chroma(x).square().sum(1).sqrt().mean())
    assert saturation(stepped) > saturation(pale)


def test_the_statistics_loss_has_finite_gradients_on_black_and_flat_frames():
    black = torch.zeros(1, 3, 8, 8, requires_grad=True)
    target = _images(size=8)[:1]
    color_statistics_l1(black, luminance(black), target, luminance(target)).backward()
    assert torch.isfinite(black.grad).all()


def test_configs_without_the_keys_build_p8s_colour_branch():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].pop("split_color_global", None)
    config["phase2"].pop("split_color_stats_weight", None)
    validate_config(config)
    colour = build_decoders(config).image.color
    assert colour.global_tone is None
    assert not any(name.startswith("global_tone") for name in colour.state_dict())


@pytest.mark.parametrize("change, message", [
    (dict(split_color_global="yes"), "split_color_global"),
    (dict(split_color_global=True, split_branch_arch="unet", split_color_unet_widths=[4, 6, 8]),
     "split_color_global"),
    (dict(split_color_stats_weight=-1.0), "split_color_stats_weight"),
])
def test_invalid_colour_settings_are_rejected(change, message):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(change)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_the_recipe_turns_both_remedies_on():
    phase2 = load_config("configs/kaggle_tartanair_v2.yaml")["phase2"]
    assert phase2["split_color_global"] is True
    assert phase2["split_color_stats_weight"] > 0


def test_phase2_trains_the_global_head_and_logs_the_statistics_term():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(split_color_global=True, split_color_stats_weight=1.0)
    validate_config(config)
    seed_everything(3)
    phase1 = build_phase1_model(config, ImuNormalizer())
    seed_everything(config["phase2"]["decoder_initialization_seed"])
    system = RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config))
    head = system.decoders.image.color.global_tone
    before = [p.detach().clone() for p in head.parameters()]
    trainer = Phase2Trainer(system, config, torch.device("cpu"), "phase1.pt")
    times = torch.arange(32).float().mul(0.01).repeat(1, 1)
    clean, imu = torch.rand(1, 3, 32, 32), torch.randn(1, 6, 32)
    batch = {"image_clean": clean, "image_noisy": (clean * 0.35).clamp(0, 1) ** 0.6, "imu_clean_phys": imu,
             "imu_noisy_phys": imu + 0.05 * torch.randn_like(imu), "image_time": times.mean(dim=1),
             "imu_times": times, "sample_id": ["s"]}
    metrics = trainer.step([batch] * config["phase2"]["gradient_accumulation"])
    assert metrics["skipped"] is False
    assert "image_color_stats_l1" in metrics
    assert any(not torch.equal(a, b) for a, b in zip(before, head.parameters()))
