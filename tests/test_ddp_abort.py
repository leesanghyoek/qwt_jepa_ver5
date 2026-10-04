"""A rank that stops must not leave the others waiting for NCCL's 60-minute timeout.

Kaggle, p20 phase 1 (two T4s): the latent gate failed on rank 0 at updates 3000, 4000
and 5000. Rank 0 saved and raised; rank 1 sat in share_rank0_rng's broadcast for an
hour each time, until NCCL aborted the run. Pins: an error inside rank0_section
reaches the waiting rank through that broadcast; a failing rank leaves at once with
exit code 1 and its traceback; a failed gate under DDP ends the CLI in seconds, with
rank 1 naming the reason.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch.distributed as dist
import yaml

from qjepa.cli import main
from qjepa.config import load_config, serializable_config
from qjepa.distributed import rank0_section, rank_and_world, share_rank0_rng, spawn
from test_kaggle_workflow import _write_dataset

REPO = Path(__file__).resolve().parent.parent


def _abort_on_rank0(path: str) -> None:
    if rank_and_world()[0] == 0:
        try:
            with rank0_section():
                raise ValueError("gate says no")
        except ValueError:
            time.sleep(3)                     # let rank 1 report before spawn stops it
            raise
    try:
        share_rank0_rng()
    except RuntimeError as error:
        Path(path).write_text(str(error))
        raise


def test_an_error_in_rank0_section_reaches_the_waiting_rank(tmp_path):
    report = tmp_path / "rank1.txt"
    started = time.time()
    with pytest.raises(SystemExit) as stop:
        spawn(_abort_on_rank0, str(report), world=2, cuda=False)
    assert stop.value.code == 1 and time.time() - started < 60
    assert report.read_text() == "rank 0 stopped: ValueError: gate says no"


def _fail_on_rank1(_) -> None:
    if rank_and_world()[0] == 1:
        raise ValueError("rank 1 broke")
    dist.barrier()                            # rank 0 waits in a collective rank 1 never joins


def test_a_failing_rank_leaves_at_once_with_its_traceback(capfd):
    started = time.time()
    with pytest.raises(SystemExit) as stop:
        spawn(_fail_on_rank1, None, world=2, cuda=False)
    assert stop.value.code == 1 and time.time() - started < 60
    assert "ValueError: rank 1 broke" in capfd.readouterr().err


def test_a_failed_gate_under_ddp_stops_both_ranks_in_seconds(tmp_path):
    _write_dataset(tmp_path / "dataset")
    config = serializable_config(load_config("configs/smoke.yaml"))
    config["phase1"].update(batch_size=2, max_successful_updates=3)
    config["phase2"].update(batch_size=2, gradient_accumulation=1)
    config["runtime"].update(parallel="ddp", gpu_count=2)
    # Any diversity counts as too little: the gate fails at the first check.
    config["monitor"].update(relative_rank_std_warning=1.0e9, consecutive_warning_checks=1)
    path = tmp_path / "ddp.yaml"
    path.write_text(yaml.safe_dump(config))
    main(["build-manifest", "--config", str(path), "--data-root", str(tmp_path / "dataset"),
          "--output", str(tmp_path / "manifest")])
    started = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "qjepa", "train-phase1", "--config", str(path), "--manifest",
         str(tmp_path / "manifest"), "--output", str(tmp_path / "run")],
        cwd=REPO, capture_output=True, text=True, timeout=300)
    assert result.returncode == 1 and time.time() - started < 300
    assert "rank 0 stopped: RuntimeError: Latent diversity/scale gate failed" in result.stderr
    assert (tmp_path / "run/phase1/last.pt").is_file()          # saved before it stopped
