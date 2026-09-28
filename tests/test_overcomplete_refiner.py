"""p17: the edge refiner as loudspeaker, then funnel -- enlarge, sharpen, shrink back.

Pins: the refiner starts as the identity and cannot move colour; its residual
blocks work on a grid ``scale`` times the input's and the output comes back at
the input's size; configs without split_edge_refiner_scale (p15/p16) still build
EdgeRefiner with its layer names, so their checkpoints load; bad scales are
rejected; the recipe costs about the MACs of p15's refiner; a phase-2 step trains it.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from qjepa.config import build_decoders, build_phase1_model, load_config, seed_everything, validate_config
from qjepa.data import ImuNormalizer
from qjepa.models import RestorationSystem
from qjepa.models.color_edge import chroma, luminance
from qjepa.models.decoders import EdgeRefiner, OvercompleteRefiner, SplitColorEdgeDecoder
from qjepa.training.phase2 import Phase2Trainer

NAF = dict(widths=(4, 6, 8, 12), enc_blocks=(1, 1, 1), middle_blocks=1, dec_blocks=(1, 1, 1), aux_factors=(2,))


def _textured(seed=0, size=64, channels=1):
    return torch.rand(1, channels, size, size, generator=torch.Generator().manual_seed(seed))


def _conv_macs(model, detail, context):
    total = 0

    def count(module, _, out):
        nonlocal total
        total += out[0].numel() * module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]

    hooks = [m.register_forward_hook(count) for m in model.modules() if isinstance(m, nn.Conv2d)]
    model(detail, context)
    for hook in hooks:
        hook.remove()
    return total


def test_the_refiner_starts_as_the_identity_and_cannot_touch_colour():
    torch.manual_seed(0)
    assert torch.equal(OvercompleteRefiner(3, 4, 2, 2)(_textured(), _textured(1, channels=2)), _textured())
    decoder = SplitColorEdgeDecoder(32, color_width=8, color_blocks=2, branch_arch="nafnet_edge", naf=NAF,
                                    refiner={"width": 4, "blocks": 2, "scale": 2})
    assert isinstance(decoder.refiner, OvercompleteRefiner)
    latent, image = torch.randn(1, 32, 8, 8), _textured(2, channels=3)
    before, parts = decoder(latent, image)
    assert torch.equal(parts["image_detail_stage1"], parts["image_detail"])      # identity at start
    with torch.no_grad():
        decoder.refiner.tail.weight.normal_(0.0, 0.5)
    after, _ = decoder(latent, image)
    assert not torch.allclose(luminance(after), luminance(before))
    assert torch.allclose(chroma(after), chroma(before), atol=1e-6)


@pytest.mark.parametrize("scale", [2, 3])
def test_it_sharpens_on_the_enlarged_grid_and_returns_at_the_input_size(scale):
    refiner = OvercompleteRefiner(3, 4, 2, scale)
    seen = []
    refiner.trunk.register_forward_hook(lambda module, inputs, out: seen.append(tuple(out.shape)))
    out = refiner(torch.rand(2, 1, 20, 28), torch.rand(2, 2, 20, 28))
    assert seen == [(2, 4, 20 * scale, 28 * scale)]
    assert out.shape == (2, 1, 20, 28)


def test_configs_without_the_scale_key_still_build_the_p15_refiner():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].pop("split_edge_refiner_scale")
    config["phase2"]["split_edge_refiner_width"] = 32
    validate_config(config)
    refiner = build_decoders(config).image.refiner
    assert type(refiner) is EdgeRefiner
    # p15/p16 checkpoints carry these names; strict loading needs them unchanged.
    assert {"head.0.weight", "trunk.0.first.weight", "tail.weight"} <= set(refiner.state_dict())


def test_scale_one_is_the_p15_refiner():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"]["split_edge_refiner_scale"] = 1
    validate_config(config)
    assert type(build_decoders(config).image.refiner) is EdgeRefiner
    with pytest.raises(ValueError, match="scale >= 2"):
        OvercompleteRefiner(3, 4, 2, 1)


@pytest.mark.parametrize("change", [
    dict(split_edge_refiner_scale=0),
    dict(split_edge_refiner_scale=True),
    dict(split_edge_refiner_scale=2.0),
    dict(split_edge_refiner_scale=2, split_edge_refiner_blocks=0, split_edge_stage1_weight=0.0),
])
def test_bad_scales_are_rejected(change):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(change)
    with pytest.raises(ValueError, match="split_edge_refiner_scale"):
        validate_config(config)


def test_the_recipe_enlarges_twice_at_about_the_cost_of_the_p15_refiner():
    phase2 = load_config("configs/kaggle_tartanair_v2.yaml")["phase2"]
    assert phase2["split_edge_refiner_scale"] == 2
    assert phase2["split_edge_refiner_width"] == 16 and phase2["split_edge_refiner_blocks"] == 4
    refiner = build_decoders(load_config("configs/kaggle_tartanair_v2.yaml")).image.refiner
    assert isinstance(refiner, OvercompleteRefiner)
    assert sum(p.numel() for p in refiner.parameters()) < sum(p.numel() for p in EdgeRefiner(3, 32, 4).parameters())
    detail, context = torch.zeros(1, 1, 64, 64), torch.zeros(1, 2, 64, 64)
    ratio = _conv_macs(refiner, detail, context) / _conv_macs(EdgeRefiner(3, 32, 4), detail, context)
    assert 0.95 < ratio < 1.05


def test_phase2_trains_the_overcomplete_refiner():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(split_edge_naf_widths=[4, 6, 8, 12, 16], split_edge_naf_enc_blocks=[1, 1, 1, 1],
                            split_edge_naf_middle_blocks=1, split_edge_naf_dec_blocks=[1, 1, 1, 1],
                            split_edge_refiner_width=4, split_edge_refiner_blocks=2, split_edge_refiner_scale=2)
    validate_config(config)
    seed_everything(3)
    phase1 = build_phase1_model(config, ImuNormalizer())
    seed_everything(config["phase2"]["decoder_initialization_seed"])
    system = RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config))
    refiner = system.decoders.image.refiner
    assert isinstance(refiner, OvercompleteRefiner)
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
