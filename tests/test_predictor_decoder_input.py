"""(e) phase2.decoder_predictor_input -- phase 2 reads what the JEPA predictor learned.

Phase 1 trains the predictor to turn the noisy latent ZI into the teacher's clean
latent, then threw it away: phase 2 decoded ZI. IWM (Garrido 2024) keeps the
predictor for the downstream task. Here the frozen phase-1 image predictor reads
ZI (and, with the degradation head, the head's estimate), and its prediction
joins ZI through a zero-initialised 1x1 conv. Pins: at initialisation the output
is exactly the run without it; the predictor is frozen and outside the optimizer
while the merge trains; the weights are the phase-1 ones; a phase-2 checkpoint
rebuilds the same system for evaluation; bad settings fail; the flag is in the
phase-2 hash only, so phase 1 is reused.
"""

from __future__ import annotations

import copy

import pytest
import torch

from qjepa.cli import _system_from_phase2
from qjepa.config import (build_decoders, build_phase1_model, load_config, phase2_latent_modules,
                          seed_everything, validate_config)
from qjepa.data import ImuNormalizer
from qjepa.models import RestorationSystem
from qjepa.training.checkpoints import atomic_torch_save, configuration_hash
from qjepa.training.phase2 import Phase2Trainer

SMALL_NAF = dict(split_edge_naf_widths=[4, 6, 8, 12, 16], split_edge_naf_enc_blocks=[1, 1, 1, 1],
                 split_edge_naf_middle_blocks=1, split_edge_naf_dec_blocks=[1, 1, 1, 1],
                 split_edge_refiner_width=8, split_edge_refiner_blocks=1)


def _config(phase1=None, **phase2):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase1"].update(phase1 or {})
    config["phase2"].update(SMALL_NAF, **phase2)
    validate_config(config)
    return config


def _batch():
    generator = torch.Generator().manual_seed(7)
    times = torch.arange(32).float().mul(0.01).repeat(1, 1)
    clean, imu = torch.rand(1, 3, 32, 32, generator=generator), torch.randn(1, 6, 32, generator=generator)
    return {"image_clean": clean, "image_noisy": (clean * 0.35).clamp(0, 1), "imu_clean_phys": imu,
            "imu_noisy_phys": imu + 0.05 * torch.randn(imu.shape, generator=generator),
            "image_time": times.mean(dim=1), "imu_times": times, "sample_id": ["s"]}


def _system(config):
    seed_everything(3)
    phase1 = build_phase1_model(config, ImuNormalizer())
    seed_everything(config["phase2"]["decoder_initialization_seed"])
    decoders = build_decoders(config)
    return RestorationSystem(phase1.backbone, phase1.normalizer, decoders,
                             *phase2_latent_modules(config, phase1)), phase1


def _args(batch):
    return batch["image_noisy"], batch["imu_noisy_phys"], batch["image_time"], batch["imu_times"]


def test_the_prediction_starts_neutral():
    config = _config(decoder_predictor_input=True)
    system, phase1 = _system(config)
    assert system.latent_predictor is phase1.image_predictor and system.decoders.predictor_merge is not None
    plain = copy.deepcopy(config)
    plain["phase2"]["decoder_predictor_input"] = False
    reference = RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(plain))
    reference.decoders.load_state_dict(
        {k: v for k, v in system.decoders.state_dict().items() if not k.startswith("predictor_merge")})
    with torch.no_grad():
        for model in (system, reference):
            model.decoders.image.edge.ending.weight.normal_(0.0, 0.1, generator=torch.Generator().manual_seed(4))
        assert torch.allclose(system(*_args(_batch())).image, reference(*_args(_batch())).image, atol=1e-6)


def test_the_predictor_is_frozen_and_the_merge_trains():
    config = _config(decoder_predictor_input=True)
    system, _ = _system(config)
    predictor = [p.detach().clone() for p in system.latent_predictor.parameters()]
    merge = [p.detach().clone() for p in system.decoders.predictor_merge.parameters()]
    trainer = Phase2Trainer(system, config, torch.device("cpu"), "phase1.pt")
    predictor_ids = {id(p) for p in system.latent_predictor.parameters()}
    assert not predictor_ids & {id(p) for p in trainer.parameters}
    for _ in range(3):
        assert trainer.step([_batch()] * config["phase2"]["gradient_accumulation"])["skipped"] is False
    assert all(torch.equal(a, b) for a, b in zip(predictor, system.latent_predictor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(merge, system.decoders.predictor_merge.parameters()))
    trainer.assert_backbone_frozen()


def test_the_conditioned_predictor_brings_the_degradation_head_along():
    config = _config(dict(degradation_weight=0.1, predictor_degradation_condition=True),
                     decoder_predictor_input=True)
    system, phase1 = _system(config)
    assert system.degradation_head is phase1.degradation_head
    assert not any(p.requires_grad for p in system.degradation_head.parameters())
    assert torch.isfinite(system(*_args(_batch())).image).all()


def test_a_phase2_checkpoint_rebuilds_the_same_system(tmp_path):
    config = _config(dict(degradation_weight=0.1, predictor_degradation_condition=True),
                     decoder_predictor_input=True)
    system, _ = _system(config)
    trainer = Phase2Trainer(system, config, torch.device("cpu"), "phase1.pt")
    trainer.step([_batch()] * config["phase2"]["gradient_accumulation"])
    atomic_torch_save(trainer.checkpoint_payload(config), tmp_path / "last.pt")
    rebuilt, _ = _system_from_phase2(str(tmp_path / "last.pt"), torch.device("cpu"))
    system.eval()
    with torch.no_grad():
        assert torch.allclose(rebuilt(*_args(_batch())).image, system(*_args(_batch())).image, atol=1e-6)


def test_a_checkpoint_whose_predictor_was_changed_is_refused(tmp_path):
    # strict=True only checks shapes: the hash is what ties the predictor to its phase-1 parent.
    config = _config(decoder_predictor_input=True)
    system, _ = _system(config)
    trainer = Phase2Trainer(system, config, torch.device("cpu"), "phase1.pt")
    payload = trainer.checkpoint_payload(config)
    assert payload["metadata"]["latent_predictor_hash"]
    key = next(k for k in payload["system"] if k.startswith("latent_predictor."))
    payload["system"][key] = payload["system"][key] + 1.0
    atomic_torch_save(payload, tmp_path / "last.pt")
    with pytest.raises(ValueError, match="predictor"):
        _system_from_phase2(str(tmp_path / "last.pt"), torch.device("cpu"))


def test_bad_settings_fail_and_the_flag_is_in_the_phase2_hash_only():
    with pytest.raises(ValueError, match="decoder_predictor_input"):
        _config(dict(predictor_type="token", image_mask_ratio=0.0, imu_mask_ratio=0.0,
                     multiscale_fine_weight=0.0, multiscale_finer_weight=0.0), decoder_predictor_input=True)
    with pytest.raises(ValueError, match="decoder_predictor_input"):
        _config(decoder_predictor_input="yes")
    with pytest.raises(ValueError, match="decoder_predictor_input"):
        _config(decoder_predictor_input=True, image_decoder="qwt_coefficients")
    on, off = _config(decoder_predictor_input=True), _config()
    assert configuration_hash(on, "phase1") == configuration_hash(off, "phase1")
    assert configuration_hash(on, "phase2") != configuration_hash(off, "phase2")


def test_the_phase1_anchor_never_gets_the_merge():
    config = _config(dict(decoder_enabled=True, coefficient_reconstruction_loss_weight=0.45),
                     decoder_predictor_input=True)
    model = build_phase1_model(config, ImuNormalizer())
    assert model.decoders.predictor_merge is None
