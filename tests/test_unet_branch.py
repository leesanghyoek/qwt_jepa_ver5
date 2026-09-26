"""Funnel and loudspeaker: U-Net branches for the split colour/edge decoder.

The funnel halves the resolution down to the latent's grid, where ZI joins;
the loudspeaker doubles it back with a skip at every level. These pin what the
branches must keep from p8: an identity start, colour the edge branch cannot
touch, an edge loss that cannot train the colour branch, and p8 checkpoints
that still load (split_branch_arch defaults to resnet). ``unet_edge`` -- the
recipe since p11 -- puts the U-Net on the edge branch only and keeps p8's colour.
"""

from __future__ import annotations

import copy

import pytest
import torch

from qjepa.config import build_decoders, build_phase1_model, load_config, seed_everything, validate_config
from qjepa.data import ImuNormalizer
from qjepa.models import RestorationSystem
from qjepa.models.color_edge import chroma, color_base, luminance
from qjepa.models.decoders import ColorBranch, PixelResNetDecoder, SplitColorEdgeDecoder, UNetBranch
from qjepa.training.checkpoints import state_dict_hash
from qjepa.training.losses import color_edge_split_loss
from qjepa.training.phase2 import Phase2Trainer


def _decoder():
    torch.manual_seed(0)
    return SplitColorEdgeDecoder(32, branch_arch="unet", color_widths=(4, 6, 8),
                                 edge_widths=(4, 6, 8, 12), unet_blocks=1)


def _inputs():
    generator = torch.Generator().manual_seed(0)
    return torch.randn(2, 32, 8, 8, generator=generator), torch.rand(2, 3, 64, 64, generator=generator)


def test_funnel_reaches_the_latent_grid_and_comes_back():
    branch = UNetBranch(2, 1, 32, widths=(4, 6, 8, 12))
    latent, image = _inputs()
    out = branch(latent, image[:, :2], base=image[:, :1])
    assert out.shape == (2, 1, 64, 64)
    assert torch.equal(out, image[:, :1])          # zero-initialised tail: identity on base


def test_starts_at_the_input_luminance_and_colour():
    latent, image = _inputs()
    out, _ = _decoder()(latent, image)
    assert torch.allclose(luminance(out), luminance(image), atol=1e-6)
    assert torch.allclose(chroma(out), chroma(color_base(image, 2)), atol=1e-6)


def test_the_latent_reaches_both_branches():
    decoder = _decoder()
    with torch.no_grad():
        for tail in (decoder.color.tail, decoder.edge.tail):
            tail.weight.normal_(0.0, 0.1)
    latent, image = _inputs()
    with torch.no_grad():
        out, _ = decoder(latent, image)
        zeroed, _ = decoder(torch.zeros_like(latent), image)
    assert float((luminance(out) - luminance(zeroed)).abs().max()) > 1e-4
    assert float((chroma(out) - chroma(zeroed)).abs().max()) > 1e-4


def test_the_edge_branch_cannot_change_colour_and_its_loss_cannot_train_colour():
    decoder = _decoder()
    with torch.no_grad():
        for tail in (decoder.color.tail, decoder.edge.tail):
            tail.weight.normal_(0.0, 0.1)
    latent, image = _inputs()
    before, _ = decoder(latent, image)
    with torch.no_grad():
        decoder.edge.tail.weight.normal_(0.0, 0.5)
    after, parts = decoder(latent, image)
    assert torch.allclose(chroma(after), chroma(before), atol=1e-6)
    _, loss_parts = color_edge_split_loss(parts["image_color_base"], parts["image_illumination"],
                                          parts["image_detail"], after, torch.rand_like(image),
                                          color_scale=2, illumination_scale=8, color_weight=1.0,
                                          edge_weight=1.0, gradient_weight=0.0)
    loss_parts["image_edge_detail_l1"].backward()
    assert all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in decoder.color.parameters())


def test_p8_checkpoints_still_build_the_resnet_branches():
    """Configs without split_branch_arch are p8 runs: same classes, same names."""
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(image_decoder="split_color_edge", split_color_width=8, split_color_blocks=2,
                            split_edge_width=16, split_edge_blocks=2, split_color_scale=2,
                            split_illumination_scale=8, split_color_weight=1.0, split_edge_weight=1.0,
                            split_gradient_weight=1.0)
    config["phase2"].pop("split_branch_arch", None)
    image = build_decoders(config).image
    assert isinstance(image.color, ColorBranch) and isinstance(image.edge, PixelResNetDecoder)
    names = set(image.state_dict())
    assert {"color.head.0.weight", "color.fuse.weight", "edge.trunk.0.first.weight", "edge.tail.2.weight"} <= names


def test_invalid_unet_settings_are_rejected():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(split_branch_arch="unet", split_color_unet_widths=[4, 6, 8], split_unet_blocks=1)
    config["phase2"]["split_edge_unet_widths"] = [8]
    with pytest.raises(ValueError, match="split_edge_unet_widths"):
        validate_config(config)
    config["phase2"]["split_branch_arch"] = "transformer"
    with pytest.raises(ValueError, match="split_branch_arch"):
        validate_config(config)


def test_unet_edge_keeps_the_p8_colour_branch_and_puts_the_unet_on_the_edges():
    torch.manual_seed(0)
    decoder = SplitColorEdgeDecoder(32, color_width=8, color_blocks=2, branch_arch="unet_edge",
                                    edge_widths=(4, 6, 8, 12))
    p8 = SplitColorEdgeDecoder(32, color_width=8, color_blocks=2)
    assert isinstance(decoder.color, ColorBranch) and isinstance(decoder.edge, UNetBranch)
    assert set(decoder.color.state_dict()) == set(p8.color.state_dict())
    latent, image = _inputs()
    out, _ = decoder(latent, image)                  # identity start, as p8
    assert torch.allclose(luminance(out), luminance(image), atol=1e-6)


def test_the_edge_loss_cannot_train_the_colour_branch_with_unet_edge():
    torch.manual_seed(0)
    decoder = SplitColorEdgeDecoder(32, color_width=8, color_blocks=2, branch_arch="unet_edge",
                                    edge_widths=(4, 6, 8, 12))
    with torch.no_grad():
        for parameter in list(decoder.color.tail.parameters()) + list(decoder.edge.tail.parameters()):
            parameter.normal_(0.0, 0.1)
    latent, image = _inputs()
    out, parts = decoder(latent, image)
    _, loss_parts = color_edge_split_loss(parts["image_color_base"], parts["image_illumination"],
                                          parts["image_detail"], out, torch.rand_like(image), color_scale=2,
                                          illumination_scale=8, color_weight=1.0, edge_weight=1.0,
                                          gradient_weight=0.0)
    loss_parts["image_edge_detail_l1"].backward()
    assert all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in decoder.color.parameters())
    assert any(p.grad is not None and float(p.grad.abs().max()) > 0.0 for p in decoder.edge.parameters())


def test_the_recipe_edge_unet_bottom_sits_on_the_latent_grid():
    config = load_config("configs/kaggle_tartanair_v2.yaml")
    phase2 = config["phase2"]
    assert phase2["split_branch_arch"] == "unet_edge"
    levels = len(phase2["split_edge_unet_widths"])
    side = config["data"]["image_size"][0]
    assert side // 2 ** (levels - 1) == 16             # ZI is 128 x 16 x 16
    edge = build_decoders(config).image.edge
    assert isinstance(edge, UNetBranch)
    latent = torch.randn(1, 128, 16, 16)
    x = torch.rand(1, 2, side, side)                   # luminance + illumination
    assert edge(latent, x, base=x[:, :1]).shape == (1, 1, side, side)


def test_unet_edge_needs_only_the_edge_widths():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(split_branch_arch="unet_edge", split_edge_unet_widths=[4, 6, 8, 12],
                            split_unet_blocks=1)
    config["phase2"].pop("split_color_unet_widths", None)
    validate_config(config)
    config["phase2"].pop("split_edge_unet_widths")
    with pytest.raises(ValueError, match="split_edge_unet_widths"):
        validate_config(config)


@pytest.mark.parametrize("arch", ["unet", "unet_edge"])
def test_phase2_trains_both_unet_branches_and_leaves_the_backbone_alone(arch):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(split_branch_arch=arch, split_color_unet_widths=[4, 6, 8],
                            split_edge_unet_widths=[4, 6, 8, 12], split_unet_blocks=1,
                            split_color_global=arch != "unet")   # the global head needs the ResNet colour branch
    validate_config(config)
    seed_everything(3)
    phase1 = build_phase1_model(config, ImuNormalizer())
    seed_everything(config["phase2"]["decoder_initialization_seed"])
    system = RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config))
    assert isinstance(system.decoders.image.edge, UNetBranch)
    trainer = Phase2Trainer(system, config, torch.device("cpu"), "phase1.pt")
    backbone = state_dict_hash(system.backbone)
    before = {k: v.clone() for k, v in system.decoders.image.state_dict().items()}
    times = torch.arange(32).float().mul(0.01).repeat(1, 1)
    clean, imu = torch.rand(1, 3, 32, 32), torch.randn(1, 6, 32)
    batch = {"image_clean": clean, "image_noisy": (clean * 0.35).clamp(0, 1), "imu_clean_phys": imu,
             "imu_noisy_phys": imu + 0.05 * torch.randn_like(imu), "image_time": times.mean(dim=1),
             "imu_times": times, "sample_id": ["s"]}
    metrics = trainer.step([batch] * config["phase2"]["gradient_accumulation"])
    assert metrics["skipped"] is False
    after = system.decoders.image.state_dict()
    for branch in ("color.", "edge."):
        assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith(branch))
    assert state_dict_hash(system.backbone) == backbone
    seed_everything(0)
    rebuilt = RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config))
    rebuilt.load_state_dict(trainer.checkpoint_payload(config)["system"], strict=True)
