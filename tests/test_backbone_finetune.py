"""(f) phase2.backbone_finetune_after_updates -- LP-FT: decoders first, then the encoder too.

Phase 2 kept the JEPA backbone frozen for the whole run, so the latent could never
adapt to restoration. LP-FT (Kumar 2022) trains the head first and then fine-tunes
everything; here the backbone joins the optimizer after N updates at a fraction
of the learning rate. Pins: before N the backbone is untouched; from N it trains
at the scaled rate while the IMU normalizer stays frozen; the checkpoint says so
and evaluation accepts it; a resume after N restores the two-group optimizer;
under DDP the backbone gradient matches one process; bad settings fail; configs
without the key keep the frozen-backbone guarantee.
"""

from __future__ import annotations

import copy

import pytest
import torch

from qjepa.cli import _system_from_phase2
from qjepa.config import build_decoders, build_phase1_model, load_config, seed_everything, validate_config
from qjepa.data import ImuNormalizer
from qjepa.distributed import rank_and_world, spawn
from qjepa.models import RestorationSystem
from qjepa.training.checkpoints import atomic_torch_save, configuration_hash, state_dict_hash
from qjepa.training.phase2 import Phase2Trainer

SMALL_NAF = dict(split_edge_naf_widths=[4, 6, 8, 12, 16], split_edge_naf_enc_blocks=[1, 1, 1, 1],
                 split_edge_naf_middle_blocks=1, split_edge_naf_dec_blocks=[1, 1, 1, 1],
                 split_edge_refiner_width=8, split_edge_refiner_blocks=1)


def _config(**phase2):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(SMALL_NAF, max_successful_updates=4, **phase2)
    validate_config(config)
    return config


def _microbatches(size=1):
    generator = torch.Generator().manual_seed(5)
    times = torch.arange(32).float().mul(0.01).repeat(size, 1)
    clean, imu = torch.rand(size, 3, 32, 32, generator=generator), torch.randn(size, 6, 32, generator=generator)
    return [{"image_clean": clean, "image_noisy": (clean * 0.35).clamp(0, 1), "imu_clean_phys": imu,
             "imu_noisy_phys": imu + 0.05 * torch.randn(imu.shape, generator=generator),
             "image_time": times.mean(dim=1), "imu_times": times, "sample_id": ["a", "b"][:size]}]


def _trainer(config):
    seed_everything(3)
    phase1 = build_phase1_model(config, ImuNormalizer())
    seed_everything(config["phase2"]["decoder_initialization_seed"])
    system = RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config))
    return Phase2Trainer(system, config, torch.device("cpu"), "phase1.pt"), system


def test_the_backbone_waits_then_trains_at_the_scaled_rate():
    config = _config(backbone_finetune_after_updates=2, backbone_finetune_lr_scale=0.1)
    trainer, system = _trainer(config)
    frozen = state_dict_hash(system.backbone)
    normalizer = state_dict_hash(system.normalizer)
    trainer.step(_microbatches())
    assert not trainer.backbone_finetuning and len(trainer.optimizer.param_groups) == 1
    trainer.step(_microbatches())
    # Switched at the end of update 2, so the checkpoint saved there has both groups;
    # the backbone itself has not moved yet.
    assert state_dict_hash(system.backbone) == frozen and trainer.backbone_finetuning
    assert len(trainer.optimizer.param_groups) == 2
    metrics = trainer.step(_microbatches())
    lrs = [group["lr"] for group in trainer.optimizer.param_groups]
    assert lrs[1] == pytest.approx(0.1 * lrs[0]) and metrics["learning_rate"] == pytest.approx(lrs[0])
    assert state_dict_hash(system.backbone) != frozen
    assert state_dict_hash(system.normalizer) == normalizer
    trainer.assert_backbone_frozen()                 # only the normalizer is still promised frozen
    payload = trainer.checkpoint_payload(config)
    assert payload["metadata"]["backbone_finetuned"] is True
    assert payload["metadata"]["frozen_backbone_hash"] == frozen          # the phase-1 parent
    assert payload["metadata"]["backbone_current_hash"] == state_dict_hash(system.backbone)


def test_evaluation_accepts_the_finetuned_backbone_and_a_resume_restores_both_groups(tmp_path):
    config = _config(backbone_finetune_after_updates=1, backbone_finetune_lr_scale=0.1)
    trainer, system = _trainer(config)
    for _ in range(2):
        trainer.step(_microbatches())
    atomic_torch_save(trainer.checkpoint_payload(config), tmp_path / "last.pt")
    rebuilt, _ = _system_from_phase2(str(tmp_path / "last.pt"), torch.device("cpu"))
    assert state_dict_hash(rebuilt.backbone) == state_dict_hash(system.backbone)
    payload = torch.load(tmp_path / "last.pt", weights_only=False)
    resumed, resumed_system = _trainer(config)
    resumed.successful_updates = int(payload["successful_updates"])
    resumed.prepare_backbone_finetune()
    resumed_system.load_state_dict(payload["system"], strict=True)
    resumed.optimizer.load_state_dict(payload["optimizer"])
    assert len(resumed.optimizer.param_groups) == 2
    assert resumed.step(_microbatches())["skipped"] is False


def test_without_the_key_the_backbone_is_still_guarded():
    trainer, system = _trainer(_config())
    trainer.step(_microbatches())
    assert not trainer.backbone_finetuning
    with torch.no_grad():
        next(system.backbone.parameters()).add_(1.0)
    with pytest.raises(RuntimeError, match="Frozen backbone"):
        trainer.assert_backbone_frozen()


@pytest.mark.parametrize("change, message", [
    (dict(backbone_finetune_after_updates=-1), "backbone_finetune_after_updates"),
    (dict(backbone_finetune_after_updates=4), "backbone_finetune_after_updates"),
    (dict(backbone_finetune_after_updates=1, backbone_finetune_lr_scale=0.0), "backbone_finetune_lr_scale"),
    (dict(backbone_finetune_lr_scale=0.1), "backbone_finetune_lr_scale"),
])
def test_bad_settings_are_refused(change, message):
    with pytest.raises(ValueError, match=message):
        _config(**change)


def test_the_setting_is_in_the_phase2_hash_only():
    on, off = _config(backbone_finetune_after_updates=2, backbone_finetune_lr_scale=0.1), _config()
    assert configuration_hash(on, "phase1") == configuration_hash(off, "phase1")
    assert configuration_hash(on, "phase2") != configuration_hash(off, "phase2")


def _backbone_gradients(folder):
    rank, world = rank_and_world()
    config = _config(batch_size=2, backbone_finetune_after_updates=1, backbone_finetune_lr_scale=0.1)
    trainer, system = _trainer(config)
    for _ in range(2):
        shares = [{key: value[rank::world] for key, value in batch.items()} for batch in _microbatches(2)]
        trainer.step(shares)
    if rank == 0:
        torch.save({name: p.grad.clone() for name, p in system.backbone.named_parameters() if p.grad is not None},
                   folder / f"backbone_{world}.pt")


def test_under_ddp_the_unfrozen_backbone_gets_the_one_process_gradient(tmp_path):
    # DDP fixes its gradient buckets when it is built; a parameter that starts
    # frozen is left out of them, so the trainer must rebuild DDP when it unfreezes.
    torch.set_num_threads(1)
    _backbone_gradients(tmp_path)
    spawn(_backbone_gradients, tmp_path, world=2, cuda=False)
    single, ddp = torch.load(tmp_path / "backbone_1.pt"), torch.load(tmp_path / "backbone_2.pt")
    assert single and single.keys() == ddp.keys()
    for name, gradient in single.items():
        scale = float(gradient.norm()) + 1e-12
        assert float((gradient - ddp[name]).norm()) / scale < 1e-4, name
