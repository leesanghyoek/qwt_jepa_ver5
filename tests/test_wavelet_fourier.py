"""(D) Wavelet-Fourier blocks in the NAFNet edge branch (p16_fourier_imu).

Pins: the Haar split is orthonormal and its merge exact; a block starts as the
identity and, once trained, reaches the whole frame; configs without the keys build
p16's NAFNet unchanged; a phase-2 step trains the blocks; bad settings are refused;
the run's config is p16_imu_smooth plus the two keys.
"""

from __future__ import annotations

import copy

import pytest
import torch

from qjepa.config import build_decoders, build_phase1_model, load_config, seed_everything, validate_config
from qjepa.data import ImuNormalizer
from qjepa.models import RestorationSystem
from qjepa.models.decoders import NAFNetBranch, WaveletFourierBlock, haar_merge, haar_split
from qjepa.training.checkpoints import configuration_hash
from qjepa.training.phase2 import Phase2Trainer

FOURIER = ("split_edge_naf_fourier_levels", "split_edge_naf_fourier_width")
SMALL_NAF = dict(split_edge_naf_widths=[4, 6, 8, 12, 16], split_edge_naf_enc_blocks=[1, 1, 1, 1],
                 split_edge_naf_middle_blocks=1, split_edge_naf_dec_blocks=[1, 1, 1, 1],
                 split_edge_refiner_width=8, split_edge_refiner_blocks=1)


def test_the_haar_split_is_orthonormal_and_the_merge_exact():
    x = torch.randn(2, 5, 16, 16, generator=torch.Generator().manual_seed(0))
    bands = haar_split(x)
    assert bands.shape == (2, 20, 8, 8)
    assert torch.allclose(haar_merge(bands), x, atol=1e-6)
    assert torch.allclose(bands.square().sum(), x.square().sum())
    a, b, c, d = x[..., 0::2, 0::2], x[..., 0::2, 1::2], x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    assert torch.allclose(bands[:, 0::4], (a + b + c + d) / 2, atol=1e-6)      # LL
    assert torch.allclose(bands[:, 3::4], (a - b - c + d) / 2, atol=1e-6)      # HH


def test_a_block_starts_as_the_identity_and_then_sees_the_whole_frame():
    torch.manual_seed(0)
    block = WaveletFourierBlock(8, width=4)
    x = torch.randn(1, 8, 32, 32)
    assert torch.equal(block(x), x)
    with torch.no_grad():
        block.gamma.fill_(1.0)
    poked = x.clone()
    # Not the same amount on every channel: the block's LayerNorm would remove that.
    poked[:, :, 0, 0] += 10.0 * torch.randn(8, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        change = (block(poked) - block(x) - (poked - x)).abs().amax(dim=1)[0]
    assert float(change[31, 31]) > 1e-4 and float(change[16, 5]) > 1e-4        # far corner, far middle


def _naf(fourier_levels=()):
    return NAFNetBranch(2, 1, 8, (4, 6, 8), (1, 1), 1, (1, 1), (2,), fourier_levels, 4)


def test_configs_without_the_keys_build_p16s_nafnet_and_the_blocks_start_neutral():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(SMALL_NAF)
    for key in FOURIER:
        config["phase2"].pop(key, None)
    edge = build_decoders(config).image.edge
    assert not any("fourier" in key for key in edge.state_dict())
    torch.manual_seed(1)
    plain, extended = _naf(), _naf((0, 1))
    extended.load_state_dict(plain.state_dict(), strict=False)
    latent, x = torch.randn(1, 8, 4, 4), torch.randn(1, 2, 16, 16)
    with torch.no_grad():
        for head in (plain.ending, extended.ending):                          # a non-trivial output
            head.weight.normal_(0.0, 0.1, generator=torch.Generator().manual_seed(2))
    assert torch.allclose(plain(latent, x), extended(latent, x), atol=1e-6)
    assert {"fourier_enc.0.gamma", "fourier_dec.1.gamma"} <= set(extended.state_dict())


@pytest.mark.parametrize("change, message", [
    (dict(split_edge_naf_fourier_levels=[4]), "fourier_levels"),
    (dict(split_edge_naf_fourier_levels=[0, 0]), "fourier_levels"),
    (dict(split_edge_naf_fourier_levels=[True]), "fourier_levels"),
    (dict(split_edge_naf_fourier_width=0), "fourier_width"),
])
def test_bad_settings_are_rejected(change, message):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(SMALL_NAF, split_branch_arch="nafnet_edge", image_decoder="split_color_edge", **change)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_phase2_trains_the_blocks():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(SMALL_NAF, split_edge_naf_fourier_levels=[0, 1], split_edge_naf_fourier_width=4)
    validate_config(config)
    seed_everything(3)
    phase1 = build_phase1_model(config, ImuNormalizer())
    seed_everything(config["phase2"]["decoder_initialization_seed"])
    system = RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config))
    gammas = [block.gamma for block in (*system.decoders.image.edge.fourier_enc.values(),
                                        *system.decoders.image.edge.fourier_dec.values())]
    assert len(gammas) == 4 and all(float(g.abs().max()) == 0.0 for g in gammas)
    trainer = Phase2Trainer(system, config, torch.device("cpu"), "phase1.pt")
    times = torch.arange(32).float().mul(0.01).repeat(1, 1)
    clean, imu = torch.rand(1, 3, 32, 32), torch.randn(1, 6, 32)
    batch = {"image_clean": clean, "image_noisy": (clean * 0.35).clamp(0, 1), "imu_clean_phys": imu,
             "imu_noisy_phys": imu + 0.05 * torch.randn_like(imu), "image_time": times.mean(dim=1),
             "imu_times": times, "sample_id": ["s"]}
    # NAFNet's last conv starts at zero, so nothing inside it gets a gradient on the
    # first update (its NAFBlocks neither); from the second on the blocks train.
    for _ in range(2):
        assert trainer.step([batch] * config["phase2"]["gradient_accumulation"])["skipped"] is False
    assert all(float(g.abs().max()) > 0.0 for g in gammas)


def test_the_run_is_p16_imu_smooth_plus_the_blocks():
    fourier, imu = load_config("configs/kaggle_fourier.yaml"), load_config("configs/kaggle_imu.yaml")
    assert fourier["phase2"]["split_edge_naf_fourier_levels"] == [0, 1]
    assert fourier["phase2"]["split_edge_naf_fourier_width"] == 16
    rest = {k: v for k, v in fourier["phase2"].items() if k not in FOURIER}
    assert rest == imu["phase2"]                                                 # everything else as p16_imu_smooth
    assert configuration_hash(fourier, "phase1") == configuration_hash(imu, "phase1")
    assert fourier["runtime"]["output_dir"].endswith("p16_fourier_imu")
    edge = build_decoders(fourier).image.edge
    added = sum(p.numel() for name, p in edge.named_parameters() if "fourier" in name)
    assert 100_000 < added < 200_000                                             # ~0.14 M on 3.2 M
