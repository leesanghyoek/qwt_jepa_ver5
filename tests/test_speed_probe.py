"""The phase-2 speed probe runs end to end, and cuDNN benchmarking is a runtime switch.

p16 in fp16 ran 5x slower than p14 in fp32 on Kaggle; the probe times each
block per setting so the choice is measured, not guessed. On CPU it can only
check that every block runs and reports a time.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

from qjepa.config import load_config, validate_config
from qjepa.training.checkpoints import configuration_hash

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from decoder_speed_probe import probe  # noqa: E402


def test_the_probe_times_every_block_on_cpu():
    rows = probe(Path("configs/smoke.yaml"), torch.device("cpu"), batch=1, steps=1, warmup=0)
    assert [row["setting"] for row in rows] == ["fp32", "fp32 + benchmark"]
    for row in rows:
        assert all(value > 0 for key, value in row.items() if key not in ("setting", "edge refiner"))


def test_cudnn_benchmark_is_validated_and_outside_the_hash():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    before = [configuration_hash(config, phase) for phase in ("phase1", "phase2")]
    config["runtime"]["cudnn_benchmark"] = False
    validate_config(config)
    assert before == [configuration_hash(config, phase) for phase in ("phase1", "phase2")]
    config["runtime"]["cudnn_benchmark"] = "yes"
    with pytest.raises(ValueError, match="cudnn_benchmark"):
        validate_config(config)


def test_the_recipe_is_back_in_fp32_with_benchmarking_on():
    config = load_config("configs/kaggle_tartanair_v2.yaml")
    assert config["phase2"]["precision"] == "fp32"
    assert config["runtime"]["cudnn_benchmark"] is True
