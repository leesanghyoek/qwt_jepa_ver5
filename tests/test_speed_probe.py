"""The phase-2 speed probe runs end to end, and cuDNN benchmarking is a runtime switch.

p16 in fp16 ran 5x slower than p14 in fp32 on Kaggle; the probe times each
block per setting so the choice is measured, not guessed. On CPU it can only
check that every block runs and reports a time.
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import pytest
import torch

from qjepa.cli import _pace
from qjepa.config import load_config, validate_config
from qjepa.training.checkpoints import configuration_hash

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from decoder_speed_probe import probe  # noqa: E402


def test_the_probe_times_every_block_on_cpu():
    rows = probe(Path("configs/smoke.yaml"), torch.device("cpu"), batch=1, steps=1, warmup=0)
    assert [row["setting"] for row in rows] == ["fp32"]
    rows = probe(Path("configs/smoke.yaml"), torch.device("cpu"), batch=1, steps=1, warmup=0, benchmark=True)
    assert [row["setting"] for row in rows] == ["fp32", "fp32 + benchmark"]
    for row in rows:
        assert all(value > 0 for key, value in row.items() if key not in ("setting", "edge refiner"))


@pytest.mark.parametrize("switch, value", [("cudnn_benchmark", True), ("cudnn_enabled", False), ("gpu_count", 1)])
def test_runtime_switches_are_validated_and_outside_the_hash(switch, value):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    before = [configuration_hash(config, phase) for phase in ("phase1", "phase2")]
    config["runtime"][switch] = value
    validate_config(config)
    assert before == [configuration_hash(config, phase) for phase in ("phase1", "phase2")]
    config["runtime"][switch] = "yes"
    with pytest.raises(ValueError, match=switch):
        validate_config(config)


def test_the_recipe_trains_on_two_gpus_as_two_processes_with_cudnn():
    # torch 2.10 + cuDNN 9.10, Kaggle T4 x2: fp32 + DataParallel + cuDNN died with
    # "misaligned address" (a race between replica threads). p18 runs DDP instead, one
    # process per GPU, and fp16 -- the notebook measures it first and may fall back.
    runtime = load_config("configs/kaggle_tartanair_v2.yaml")["runtime"]
    assert load_config("configs/kaggle_tartanair_v2.yaml")["phase2"]["precision"] == "amp_fp16"
    assert runtime["gpu_count"] == 2 and runtime["parallel"] == "ddp" and runtime["cudnn_enabled"] is True
    assert runtime["cudnn_benchmark"] is False
    # The RSS limit is per process: two of them must fit in Kaggle's ~30 GB.
    assert 2 * runtime["restart_above_rss_gib"] <= 24


def test_the_probe_writes_its_rows_for_the_notebook(tmp_path, monkeypatch):
    import json

    import decoder_speed_probe

    out = tmp_path / "probe.json"
    monkeypatch.setattr(sys, "argv", ["probe", "--config", "configs/smoke.yaml", "--device", "cpu", "--batch", "1",
                                      "--steps", "1", "--warmup", "0", "--json", str(out)])
    assert decoder_speed_probe.main() == 0
    written = json.loads(out.read_text())
    assert written["batch"] == 1 and [row["setting"] for row in written["rows"]] == ["fp32"]
    assert written["rows"][0]["whole step"] > 0


def test_the_log_reports_seconds_per_update():
    started = time.perf_counter() - 3.0
    assert 1.4 < float(_pace(started, 10, 12).split()[0]) < 1.7
    assert _pace(time.perf_counter(), 5, 5).endswith(" s/update")        # no division by zero
