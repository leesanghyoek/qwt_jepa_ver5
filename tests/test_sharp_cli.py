"""The sharpness plan end to end through the CLI, restarting after every checkpoint.

The unit tests pin each change; this runs them together the way the notebook does:
build-manifest, phase 1 with the centre norm, the sensor-noise Jacobian direction,
the degradation head and mirrored pairs, phase 2 with the predictor input, LP-FT,
mirrored pairs and the IMU increments -- every update followed by a restart that
resumes from the checkpoint -- then evaluate. Pins: both phases finish; the resumed
phase 2 has unfrozen the backbone and kept the phase-1 predictor; evaluation
rebuilds the system from the phase-2 checkpoint alone.
"""

from __future__ import annotations

import json

import pytest
import torch
import yaml

from qjepa.cli import RESTART_EXIT_CODE, main
from qjepa.config import load_config, serializable_config
from qjepa.training.checkpoints import load_checkpoint
from test_kaggle_workflow import _write_dataset


def _sharp_smoke():
    config = serializable_config(load_config("configs/smoke.yaml"))
    config["model"]["encoder_norm"] = "centre"
    config["encoder_sensitivity"]["noise_direction"] = "sensor_noise"
    config["phase1"].update(max_successful_updates=2, degradation_weight=0.1,
                            predictor_degradation_condition=True, augment_hflip=True)
    config["phase2"].update(max_successful_updates=3, decoder_predictor_input=True,
                            backbone_finetune_after_updates=1, backbone_finetune_lr_scale=0.1,
                            augment_hflip=True, imu_increment_weight=0.5, imu_increment_windows=[8, 32])
    config["runtime"]["restart_above_rss_gib"] = 1e-6          # restart after every checkpoint
    # Two updates cannot bring a centre-norm encoder from ~1e-4 to O(1), and under centre norm
    # the gate bounds the raw RMS itself (test_a_centre_norm_gate_bounds_the_raw_scale_itself).
    config["monitor"]["raw_scale_ratio_warning"] = [1e-6, 1e6]
    return config


def _run_until_done(arguments, last):
    restarts = 0
    while True:
        try:
            main([*arguments, *(["--resume", str(last)] if last.exists() else [])])
            return restarts
        except SystemExit as stop:
            assert stop.code == RESTART_EXIT_CODE
            restarts += 1


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_the_plan_trains_resumes_and_evaluates(tmp_path, capsys):
    torch.set_num_threads(1)
    root, manifest, output = tmp_path / "dataset", tmp_path / "manifest", tmp_path / "run"
    _write_dataset(root)
    path = tmp_path / "sharp.yaml"
    path.write_text(yaml.safe_dump(_sharp_smoke()))
    common = ["--config", str(path), "--manifest", str(manifest), "--output", str(output)]
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest)])
    assert _run_until_done(["train-phase1", *common], output / "phase1/last.pt") == 1
    phase1 = load_checkpoint(output / "phase1/last.pt")
    assert any(key.startswith("degradation_head.") for key in phase1["model"])
    assert any("degradation" in record for record in _records(output / "phase1/train.jsonl"))
    # Every update says how long it waited for data (tests/test_training_speed.py).
    assert all(record["data_wait_seconds"] >= 0 for record in _records(output / "phase1/train.jsonl")
               if "loss" in record and "successful_updates" in record)

    last = output / "phase2/last.pt"
    assert _run_until_done(["train-phase2", *common, "--backbone-checkpoint",
                            str(output / "phase1/last.pt")], last) == 2
    payload = load_checkpoint(last)
    assert payload["successful_updates"] == 3 and payload["metadata"]["backbone_finetuned"] is True
    assert len(payload["optimizer"]["param_groups"]) == 2
    predictor = {k[len("latent_predictor."):]: v for k, v in payload["system"].items()
                 if k.startswith("latent_predictor.")}
    assert predictor and all(torch.equal(v, phase1["model"][f"image_predictor.{k}"]) for k, v in predictor.items())
    assert any("imu_gyro_increment" in record for record in _records(output / "phase2/train.jsonl"))
    assert any("data_wait_seconds" in record for record in _records(output / "phase2/train.jsonl"))

    evaluation = output / "eval"
    main(["evaluate", "--checkpoint", str(last), "--manifest", str(manifest), "--device", "cpu",
          "--output", str(evaluation), "--panels", "0", "--split", "valid"])
    assert json.loads((evaluation / "metrics.json").read_text())["requested"]["image_count"] > 0

    # Its decoders were trained on the fine-tuned backbone: starting a new run from them
    # with the phase-1 backbone would pair them with the wrong encoder, so it is refused.
    with pytest.raises(SystemExit):
        main(["train-phase2", "--config", str(path), "--manifest", str(manifest), "--output",
              str(tmp_path / "again"), "--backbone-checkpoint", str(output / "phase1/last.pt"),
              "--decoder-init-checkpoint", str(last)])
    assert "fine-tuned" in capsys.readouterr().err
