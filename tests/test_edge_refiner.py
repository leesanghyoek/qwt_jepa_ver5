"""p15: the small CNN that sharpens and smooths the edge map, and the roughness term.

Pins: the refiner starts as the identity and cannot move colour; the roughness
term is zero for the reference and for anything smoother, and grows with grain
and ringing but not with texture the reference has; configs without the keys
build p14's decoder; a phase-2 step trains the refiner and logs both new terms.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from qjepa.config import build_decoders, build_phase1_model, load_config, seed_everything, validate_config
from qjepa.data import ImuNormalizer
from qjepa.evaluation.metrics import image_metrics
from qjepa.models import RestorationSystem
from qjepa.models.color_edge import chroma, excess_roughness, luminance
from qjepa.models.decoders import EdgeRefiner, SplitColorEdgeDecoder
from qjepa.training.phase2 import Phase2Trainer

NAF = dict(widths=(4, 6, 8, 12), enc_blocks=(1, 1, 1), middle_blocks=1, dec_blocks=(1, 1, 1), aux_factors=(2,))
REFINER_KEYS = ("split_edge_refiner_blocks", "split_edge_refiner_width", "split_edge_stage1_weight",
                "split_edge_smooth_weight", "split_edge_refiner_scale")


def _textured(seed=0, size=64, channels=1):
    return torch.rand(1, channels, size, size, generator=torch.Generator().manual_seed(seed))


def test_the_refiner_starts_as_the_identity_and_cannot_touch_colour():
    torch.manual_seed(0)
    assert torch.equal(EdgeRefiner(3, 8, 2)(_textured(), _textured(1, channels=2)), _textured())
    decoder = SplitColorEdgeDecoder(32, color_width=8, color_blocks=2, branch_arch="nafnet_edge", naf=NAF,
                                    refiner={"width": 8, "blocks": 2})
    latent, image = torch.randn(1, 32, 8, 8), _textured(2, channels=3)
    before, parts = decoder(latent, image)
    assert torch.equal(parts["image_detail_stage1"], parts["image_detail"])      # identity at start
    with torch.no_grad():
        decoder.refiner.tail.weight.normal_(0.0, 0.5)
    after, _ = decoder(latent, image)
    assert not torch.allclose(luminance(after), luminance(before))
    assert torch.allclose(chroma(after), chroma(before), atol=1e-6)


def test_roughness_reads_zero_for_the_reference_and_for_anything_smoother():
    reference = F.avg_pool2d(_textured(size=66), 3, stride=1)                   # some texture
    assert float(excess_roughness(reference, reference)) == 0.0
    assert float(excess_roughness(0.5 * reference, reference)) == 0.0           # smoother: not penalised
    grainy = reference + 0.02 * torch.randn(reference.shape, generator=torch.Generator().manual_seed(3))
    assert float(excess_roughness(grainy, reference)) > 0.001


def test_roughness_penalises_grain_on_flat_areas_not_on_strong_texture():
    rows, cols = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    checker = ((rows + cols) % 2).float()[None, None]                            # gradient 1 everywhere
    flat = torch.zeros_like(checker)
    noise = 0.02 * torch.randn(flat.shape, generator=torch.Generator().manual_seed(4))
    assert float(excess_roughness(flat + noise, flat)) > 0.01
    assert float(excess_roughness(checker + noise, checker)) < 1e-4


def test_the_metric_reads_zero_for_the_clean_frame():
    clean = torch.rand(1, 3, 32, 32)
    assert image_metrics(clean, clean)["image_excess_roughness"] == 0.0
    assert image_metrics((clean + 0.05 * torch.randn_like(clean)).clamp(0, 1), clean)["image_excess_roughness"] > 0


def test_configs_without_the_keys_build_no_refiner():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    for key in REFINER_KEYS:
        config["phase2"].pop(key, None)
    validate_config(config)
    image = build_decoders(config).image
    assert image.refiner is None and not any(k.startswith("refiner.") for k in image.state_dict())


@pytest.mark.parametrize("change, message", [
    (dict(split_edge_refiner_blocks=0, split_edge_stage1_weight=0.5), "split_edge_stage1_weight"),
    (dict(split_edge_smooth_weight=-1.0), "split_edge_smooth_weight"),
    (dict(split_edge_refiner_width=0), "split_edge_refiner_width"),
])
def test_inconsistent_refiner_settings_are_rejected(change, message):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(change)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_the_recipe_turns_the_refiner_and_the_roughness_term_on():
    phase2 = load_config("configs/kaggle_tartanair_v2.yaml")["phase2"]
    assert phase2["split_edge_refiner_blocks"] == 4
    assert phase2["split_edge_stage1_weight"] == 0.5 and phase2["split_edge_smooth_weight"] == 2.0
    # p17 swaps EdgeRefiner for the overcomplete one: tests/test_overcomplete_refiner.py.
    refiner = build_decoders(load_config("configs/kaggle_tartanair_v2.yaml")).image.refiner
    assert refiner is not None and sum(p.numel() for p in refiner.parameters()) < 100_000


def test_phase2_trains_the_refiner_and_logs_both_terms():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(split_edge_naf_widths=[4, 6, 8, 12, 16], split_edge_naf_enc_blocks=[1, 1, 1, 1],
                            split_edge_naf_middle_blocks=1, split_edge_naf_dec_blocks=[1, 1, 1, 1],
                            split_edge_refiner_width=8, split_edge_refiner_blocks=2, split_edge_refiner_scale=1)
    validate_config(config)
    seed_everything(3)
    phase1 = build_phase1_model(config, ImuNormalizer())
    seed_everything(config["phase2"]["decoder_initialization_seed"])
    system = RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config))
    refiner = system.decoders.image.refiner
    before = [p.detach().clone() for p in refiner.parameters()]
    trainer = Phase2Trainer(system, config, torch.device("cpu"), "phase1.pt")
    times = torch.arange(32).float().mul(0.01).repeat(1, 1)
    clean, imu = torch.rand(1, 3, 32, 32), torch.randn(1, 6, 32)
    batch = {"image_clean": clean, "image_noisy": (clean * 0.35).clamp(0, 1), "imu_clean_phys": imu,
             "imu_noisy_phys": imu + 0.05 * torch.randn_like(imu), "image_time": times.mean(dim=1),
             "imu_times": times, "sample_id": ["s"]}
    metrics = trainer.step([batch] * config["phase2"]["gradient_accumulation"])
    assert metrics["skipped"] is False
    assert {"image_edge_stage1_l1", "image_edge_roughness"} <= set(metrics)
    assert any(not torch.equal(a, b) for a, b in zip(before, refiner.parameters()))
    seed_everything(0)
    rebuilt = RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config))
    rebuilt.load_state_dict(trainer.checkpoint_payload(config)["system"], strict=True)
