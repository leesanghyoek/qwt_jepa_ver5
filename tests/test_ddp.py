"""Phase 2 as two processes (DDP) trains exactly what one process trains.

On Kaggle's torch 2.10 + cuDNN 9.10, DataParallel's replica threads race in fp32
("misaligned address"); runtime.parallel ddp runs one process per GPU instead.
Two gloo processes on CPU stand in for the GPUs here. The ranks split every
microbatch, gather the outputs and compute the loss of the whole batch, so the
weights must match a single process to float precision -- through gradient
accumulation, validations and restarts from checkpoints.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch
import yaml

from qjepa.cli import RESTART_EXIT_CODE, _RankShare, main
from qjepa.config import build_decoders, build_phase1_model, load_config, seed_everything, serializable_config, validate_config
from qjepa.data import ImuNormalizer
from qjepa.distributed import rank_and_world, share_rank0_rng, spawn
from qjepa.models import RestorationSystem
from qjepa.training.checkpoints import load_checkpoint
from qjepa.training.phase2 import Phase2Trainer
from test_kaggle_workflow import _write_dataset


def test_the_rank_shares_rebuild_each_batch_in_order():
    batches = [[0, 1, 2, 3], [4, 5, 6, 7]]
    shares = [list(_RankShare(batches, rank, 2)) for rank in range(2)]
    assert [first + second for first, second in zip(*shares)] == batches
    with pytest.raises(ValueError, match="split"):
        list(_RankShare([[0, 1, 2]], 0, 2))


def test_the_parallel_setting_is_validated():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["runtime"].update(parallel="ddp", gpu_count=2)
    config["phase2"]["batch_size"] = 2
    validate_config(config)
    config["phase2"]["batch_size"] = 3
    with pytest.raises(ValueError, match="even"):
        validate_config(config)
    config["runtime"]["parallel"] = "threads"
    with pytest.raises(ValueError, match="parallel"):
        validate_config(config)


def test_a_rank_sees_one_gpu_and_launches_nothing(monkeypatch):
    # Kaggle T4 x2, p18's first run: each rank is given only its own GPU, re-entered
    # the launch check and died with "DDP asked for 2 GPUs; only 1 visible". The CPU
    # runs never took that branch: it is CUDA only.
    import qjepa.cli as cli

    config = {"runtime": {"parallel": "ddp", "gpu_count": 2}}
    args = type("Args", (), {"gpus": None})()
    cuda = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(cli, "rank_and_world", lambda: (1, 2))
    assert cli._ddp_world(config, args, cuda) == 1
    # The launcher still refuses two ranks when only one GPU is there.
    monkeypatch.setattr(cli, "rank_and_world", lambda: (0, 1))
    with pytest.raises(ValueError, match="only 1 visible"):
        cli._ddp_world(config, args, cuda)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert cli._ddp_world(config, args, cuda) == 2


def _draw_after_sharing(folder):
    rank, _ = rank_and_world()
    if rank == 1:
        torch.rand(7)                                # rank 1 drifts, as when only rank 0 validates
    share_rank0_rng()
    torch.save(torch.rand(3), folder / f"draw{rank}.pt")


def test_every_rank_takes_rank0_rng(tmp_path):
    spawn(_draw_after_sharing, tmp_path, world=2, cuda=False)
    assert torch.equal(torch.load(tmp_path / "draw0.pt"), torch.load(tmp_path / "draw1.pt"))


def _exit_75(_):
    raise SystemExit(RESTART_EXIT_CODE)


def test_a_rank_exit_code_reaches_the_launcher(tmp_path):
    with pytest.raises(SystemExit) as stop:
        spawn(_exit_75, None, world=2, cuda=False)
    assert stop.value.code == RESTART_EXIT_CODE


def _logged(run, key):
    records = [json.loads(line) for line in (run / "phase2/train.jsonl").read_text().splitlines()]
    return [record[key] for record in records if key in record]


def test_two_processes_train_the_weights_of_one(tmp_path):
    torch.set_num_threads(1)
    root, manifest = tmp_path / "dataset", tmp_path / "manifest"
    _write_dataset(root)
    config = serializable_config(load_config("configs/smoke.yaml"))
    config["phase2"].update(batch_size=2, gradient_accumulation=2, max_successful_updates=3)
    single = tmp_path / "single.yaml"
    single.write_text(yaml.safe_dump(config))
    # Restart after every checkpoint: resume must also be exact under DDP.
    config["runtime"].update(parallel="ddp", gpu_count=2, restart_above_rss_gib=1e-6)
    ddp = tmp_path / "ddp.yaml"
    ddp.write_text(yaml.safe_dump(config))
    main(["build-manifest", "--config", str(single), "--data-root", str(root), "--output", str(manifest)])
    phase1 = tmp_path / "phase1_run"
    main(["train-phase1", "--config", str(single), "--manifest", str(manifest), "--output", str(phase1)])
    finals = {}
    for name, path in (("single", single), ("ddp", ddp)):
        output = tmp_path / name
        arguments = ["train-phase2", "--config", str(path), "--manifest", str(manifest), "--output", str(output),
                     "--backbone-checkpoint", str(phase1 / "phase1/last.pt")]
        restarts = 0
        while True:
            last = output / "phase2/last.pt"
            try:
                main([*arguments, *(["--resume", str(last)] if last.exists() else [])])
                break
            except SystemExit as stop:
                assert stop.code == RESTART_EXIT_CODE
                restarts += 1
        assert restarts == (2 if name == "ddp" else 0)
        payload = load_checkpoint(output / "phase2/last.pt")
        assert payload["successful_updates"] == 3
        finals[name] = payload["system"]
    assert json.loads((tmp_path / "ddp/phase2/execution.json").read_text())["backend"] == "ddp"
    # AdamW barely feels the gradient's scale, so the weights alone would not catch
    # a DDP gradient off by the world size; the logged norm does.
    for key in ("loss", "gradient_norm"):
        assert _logged(tmp_path / "ddp", key) == pytest.approx(_logged(tmp_path / "single", key), rel=1e-5), key
    assert finals["single"].keys() == finals["ddp"].keys()
    # AdamW moves a weight by about the learning rate whatever its gradient's size, so
    # float noise that flips a near-zero gradient's sign can part the runs by up to
    # two steps per update (seen: 9e-5 in the colour trunk). The gradients themselves
    # are checked to 1e-4 below, per parameter.
    drift = 2.0 * sum(_logged(tmp_path / "single", "learning_rate"))
    for key, value in finals["single"].items():
        assert torch.allclose(value, finals["ddp"][key], atol=drift), key


def _step_config():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(batch_size=2, gradient_accumulation=2, imu_refiner_blocks=1, imu_refiner_width=4,
                            imu_jitter_weight=1.0)
    return config


def _global_microbatches():
    generator = torch.Generator().manual_seed(5)
    times = torch.arange(32).float().mul(0.01).repeat(2, 1)
    batches = []
    for _ in range(2):
        clean, imu = torch.rand(2, 3, 32, 32, generator=generator), torch.randn(2, 6, 32, generator=generator)
        batches.append({"image_clean": clean, "image_noisy": (clean * 0.35).clamp(0, 1), "imu_clean_phys": imu,
                        "imu_noisy_phys": imu + 0.05 * torch.randn(imu.shape, generator=generator),
                        "image_time": times.mean(dim=1), "imu_times": times, "sample_id": ["a", "b"]})
    return batches


def _gradients_of_one_update(folder):
    rank, world = rank_and_world()
    config = _step_config()
    seed_everything(3)
    phase1 = build_phase1_model(config, ImuNormalizer())
    seed_everything(config["phase2"]["decoder_initialization_seed"])
    system = RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config))
    trainer = Phase2Trainer(system, config, torch.device("cpu"), "phase1.pt")
    shares = [{key: value[rank::world] for key, value in batch.items()} for batch in _global_microbatches()]
    trainer.step(shares)
    if rank == 0:
        torch.save({name: p.grad.clone() for name, p in system.decoders.named_parameters() if p.grad is not None},
                   folder / f"gradients_{world}.pt")


def test_every_parameter_gets_the_one_process_gradient(tmp_path):
    # The weights alone are a loose check: AdamW moves every weight by about the
    # learning rate whatever its gradient's size, so float noise that flips the
    # sign of a near-zero gradient shows up at ~lr. The gradients must match.
    torch.set_num_threads(1)
    _gradients_of_one_update(tmp_path)
    spawn(_gradients_of_one_update, tmp_path, world=2, cuda=False)
    single, ddp = torch.load(tmp_path / "gradients_1.pt"), torch.load(tmp_path / "gradients_2.pt")
    assert single.keys() == ddp.keys() and any(name.startswith("imu_refiner.") for name in single)
    for name, gradient in single.items():
        scale = float(gradient.norm()) + 1e-12
        assert float((gradient - ddp[name]).norm()) / scale < 1e-4, name
