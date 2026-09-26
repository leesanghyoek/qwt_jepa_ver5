"""The NAFNet edge branch (p13) and its two new loss terms.

Pins: every NAF block and the whole branch start as the identity; the latent
reaches the output from the 16x16 bottom; the auxiliary heads start at the
averaged-down input; the edge loss still cannot train the colour branch; the
FFT term is phase-aware (it sees a shift that leaves every modulus unchanged);
and a phase-2 step trains the branch, logs both terms and checkpoints.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from qjepa.config import build_decoders, build_phase1_model, load_config, seed_everything, validate_config
from qjepa.data import ImuNormalizer
from qjepa.models import RestorationSystem
from qjepa.models.color_edge import chroma
from qjepa.models.decoders import ColorBranch, NAFBlock, NAFNetBranch, SplitColorEdgeDecoder
from qjepa.training.checkpoints import state_dict_hash
from qjepa.training.losses import color_edge_split_loss, fft_l1
from qjepa.training.phase2 import Phase2Trainer

NAF = dict(widths=(4, 6, 8, 12), enc_blocks=(1, 1, 2), middle_blocks=2, dec_blocks=(1, 1, 1), aux_factors=(2, 4))


def _inputs(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(2, 32, 8, 8, generator=generator), torch.rand(2, 3, 64, 64, generator=generator)


def _decoder():
    torch.manual_seed(0)
    return SplitColorEdgeDecoder(32, color_width=8, color_blocks=2, branch_arch="nafnet_edge", naf=NAF)


def test_a_fresh_naf_block_and_branch_are_the_identity():
    x = torch.randn(2, 8, 16, 16)
    assert torch.equal(NAFBlock(8)(x), x)
    latent, image = _inputs()
    branch = NAFNetBranch(2, 1, 32, **NAF)
    out, aux = branch.forward_with_aux(latent, image[:, :2], base=image[:, :1])
    assert torch.equal(out, image[:, :1])
    assert set(aux) == {2, 4}
    for factor, value in aux.items():
        assert torch.allclose(value, F.avg_pool2d(image[:, :1], factor))


def test_the_latent_reaches_the_output_through_the_bottom():
    torch.manual_seed(0)
    branch = NAFNetBranch(2, 1, 32, **NAF)
    with torch.no_grad():
        branch.ending.weight.normal_(0.0, 0.1)
        for parameter in branch.parameters():
            if parameter.dim() == 4 and parameter.shape[0] == 1 and parameter.shape[2:] == (1, 1):
                parameter.fill_(0.5)                        # open the blocks' beta/gamma gates
    latent, image = _inputs()
    a = branch(latent, image[:, :2], base=image[:, :1])
    b = branch(latent + 1.0, image[:, :2], base=image[:, :1])
    assert not torch.allclose(a, b)


def test_the_split_still_keeps_edges_out_of_the_colour_branch():
    decoder = _decoder()
    assert isinstance(decoder.color, ColorBranch) and isinstance(decoder.edge, NAFNetBranch)
    with torch.no_grad():
        decoder.edge.ending.weight.normal_(0.0, 0.5)
    latent, image = _inputs()
    out, parts = decoder(latent, image)
    assert {"image_detail_aux2", "image_detail_aux4"} <= set(parts)
    assert torch.allclose(chroma(out), chroma(parts["image_color_base"]), atol=1e-6)
    _, loss_parts = color_edge_split_loss(
        parts["image_color_base"], parts["image_illumination"], parts["image_detail"], out, torch.rand_like(image),
        color_scale=2, illumination_scale=8, color_weight=1.0, edge_weight=1.0, gradient_weight=0.0,
        fft_weight=1.0, aux_details={2: parts["image_detail_aux2"], 4: parts["image_detail_aux4"]}, aux_weight=0.5)
    (loss_parts["image_edge_detail_l1"] + loss_parts["image_edge_fft_l1"] + loss_parts["image_edge_aux_l1"]).backward()
    assert all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in decoder.color.parameters())


def test_the_fft_term_is_phase_aware():
    image = torch.rand(1, 1, 32, 32)
    shifted = torch.roll(image, shifts=3, dims=-1)          # same modulus everywhere, other phase
    same_modulus = torch.allclose(torch.fft.rfft2(shifted).abs(), torch.fft.rfft2(image).abs(), atol=1e-4)
    assert same_modulus
    assert float(fft_l1(image, image)) == 0.0
    assert float(fft_l1(shifted, image)) > 0.05


@pytest.mark.parametrize("change, message", [
    (dict(split_edge_naf_widths=[8]), "split_edge_naf_widths"),
    (dict(split_edge_naf_enc_blocks=[1, 1]), "split_edge_naf_enc_blocks"),
    (dict(split_edge_naf_middle_blocks=0), "split_edge_naf_middle_blocks"),
    (dict(split_edge_aux_factors=[3]), "split_edge_aux_factors"),
    (dict(split_edge_aux_factors=[], split_edge_aux_weight=0.5), "split_edge_aux_factors"),
    (dict(split_edge_fft_weight=-1.0), "split_edge_fft_weight"),
])
def test_inconsistent_nafnet_settings_are_rejected(change, message):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(change)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_the_recipe_is_the_nafnet_edge_branch_on_the_latent_grid():
    config = load_config("configs/kaggle_tartanair_v2.yaml")
    phase2 = config["phase2"]
    assert phase2["split_branch_arch"] == "nafnet_edge"
    assert config["data"]["image_size"][0] // 2 ** (len(phase2["split_edge_naf_widths"]) - 1) == 16
    assert phase2["split_edge_fft_weight"] > 0 and phase2["split_edge_aux_weight"] > 0
    edge = build_decoders(config).image.edge
    assert isinstance(edge, NAFNetBranch)
    assert 2.5e6 < sum(p.numel() for p in edge.parameters()) < 4e6


def test_phase2_trains_the_nafnet_branch_and_logs_both_terms():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(split_edge_naf_widths=[4, 6, 8, 12, 16], split_edge_naf_enc_blocks=[1, 1, 1, 1],
                            split_edge_naf_middle_blocks=1, split_edge_naf_dec_blocks=[1, 1, 1, 1])
    validate_config(config)
    seed_everything(3)
    phase1 = build_phase1_model(config, ImuNormalizer())
    seed_everything(config["phase2"]["decoder_initialization_seed"])
    system = RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config))
    assert isinstance(system.decoders.image.edge, NAFNetBranch)
    trainer = Phase2Trainer(system, config, torch.device("cpu"), "phase1.pt")
    backbone = state_dict_hash(system.backbone)
    before = {k: v.clone() for k, v in system.decoders.image.edge.state_dict().items()}
    times = torch.arange(32).float().mul(0.01).repeat(1, 1)
    clean, imu = torch.rand(1, 3, 32, 32), torch.randn(1, 6, 32)
    batch = {"image_clean": clean, "image_noisy": (clean * 0.35).clamp(0, 1), "imu_clean_phys": imu,
             "imu_noisy_phys": imu + 0.05 * torch.randn_like(imu), "image_time": times.mean(dim=1),
             "imu_times": times, "sample_id": ["s"]}
    metrics = trainer.step([batch] * config["phase2"]["gradient_accumulation"])
    assert metrics["skipped"] is False
    assert {"image_edge_fft_l1", "image_edge_aux_l1"} <= set(metrics)
    after = system.decoders.image.edge.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)
    assert state_dict_hash(system.backbone) == backbone
    seed_everything(0)
    rebuilt = RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config))
    rebuilt.load_state_dict(trainer.checkpoint_payload(config)["system"], strict=True)
