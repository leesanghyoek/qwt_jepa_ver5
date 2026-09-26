"""Checkpoint, exit, resume: a run that outlives a leak it cannot see.

On Kaggle the p11 phase-2 process grew ~6.7 MiB per update and was killed at
update 4097. With runtime.restart_above_rss_gib set, the CLI exits with code 75
right after a checkpoint once RSS passes the limit, and the notebook resumes.
These pin that the exit happens only after a saved checkpoint, never on the
last update, that the key stays out of the hash, and that the restarted run
ends with exactly the weights of an uninterrupted one.
"""

from __future__ import annotations

import pytest
import torch
import yaml

from qjepa.cli import RESTART_EXIT_CODE, _restart_if_memory_high, main
from qjepa.config import load_config, serializable_config, validate_config
from qjepa.training.checkpoints import configuration_hash, load_checkpoint
from test_kaggle_workflow import _write_dataset


def _runtime(limit):
    return {"runtime": {"restart_above_rss_gib": limit}}


def test_the_guard_exits_only_above_the_limit_and_never_on_the_last_update():
    memory = {"rss_mib": 20 * 1024.0}
    _restart_if_memory_high(_runtime(None), memory, 10, 100)          # not set: never
    _restart_if_memory_high(_runtime(32), memory, 10, 100)            # below the limit
    _restart_if_memory_high(_runtime(16), memory, 100, 100)           # finished: nothing to resume
    with pytest.raises(SystemExit) as exit_info:
        _restart_if_memory_high(_runtime(16), memory, 10, 100)
    assert exit_info.value.code == RESTART_EXIT_CODE


def test_the_limit_is_outside_the_hash_and_validated():
    config = load_config("configs/smoke.yaml")
    before = [configuration_hash(config, phase) for phase in ("phase1", "phase2")]
    config["runtime"]["restart_above_rss_gib"] = 16
    validate_config(config)
    assert before == [configuration_hash(config, phase) for phase in ("phase1", "phase2")]
    config["runtime"]["restart_above_rss_gib"] = 0
    with pytest.raises(ValueError, match="restart_above_rss_gib"):
        validate_config(config)


def test_a_restarted_phase2_ends_with_the_weights_of_an_uninterrupted_one(tmp_path):
    torch.set_num_threads(1)
    root, manifest = tmp_path / "dataset", tmp_path / "manifest"
    _write_dataset(root)
    config = serializable_config(load_config("configs/smoke.yaml"))
    config["phase2"]["max_successful_updates"] = 3
    plain = tmp_path / "plain.yaml"
    plain.write_text(yaml.safe_dump(config))
    config["runtime"]["restart_above_rss_gib"] = 1e-6            # always above: exit at every checkpoint
    guarded = tmp_path / "guarded.yaml"
    guarded.write_text(yaml.safe_dump(config))
    main(["build-manifest", "--config", str(plain), "--data-root", str(root), "--output", str(manifest)])
    finals = {}
    for name, path in (("plain", plain), ("guarded", guarded)):
        output = tmp_path / name
        common = ["--config", str(path), "--manifest", str(manifest), "--output", str(output)]
        main(["train-phase1", *common])
        phase2 = [*common, "--backbone-checkpoint", str(output / "phase1/last.pt")]
        restarts = 0
        while True:
            last = output / "phase2/last.pt"
            try:
                main(["train-phase2", *phase2, *(["--resume", str(last)] if last.exists() else [])])
                break
            except SystemExit as stop:
                assert stop.code == RESTART_EXIT_CODE
                restarts += 1
        assert restarts == (2 if name == "guarded" else 0)        # after updates 1 and 2, not 3
        payload = load_checkpoint(output / "phase2/last.pt")
        assert payload["successful_updates"] == 3
        finals[name] = payload["system"]
    assert {k: v.shape for k, v in finals["plain"].items()} == {k: v.shape for k, v in finals["guarded"].items()}
    for key, value in finals["plain"].items():
        torch.testing.assert_close(finals["guarded"][key], value, rtol=0, atol=0, msg=key)
