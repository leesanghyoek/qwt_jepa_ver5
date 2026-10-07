"""evaluate --scenarios, --amp and its progress lines -- a test split scored faster, on two GPUs at once.

p26's --protocol ran over half an hour on one GPU and printed nothing until it ended. Pins:
--scenarios scores a subset of the protocol, in the protocol's order, with exactly the numbers the
whole protocol gives those scenarios (so two processes, one per GPU, together give the protocol);
it needs --protocol and known names; each scenario announces itself; --amp (CUDA only) moves the
metrics by float noise and is recorded in evaluation_config.json.
"""

from __future__ import annotations

import json

import pytest
import torch
import yaml

from qjepa.cli import PROTOCOL_SCENARIOS, main
from qjepa.config import load_config, serializable_config
from test_kaggle_workflow import _write_dataset


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A smoke run through both phases: (config path, manifest, phase-2 checkpoint)."""
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("evaluate_options")
    _write_dataset(root / "dataset")
    path = root / "smoke.yaml"
    path.write_text(yaml.safe_dump(serializable_config(load_config("configs/smoke.yaml"))))
    common = ["--config", str(path), "--manifest", str(root / "manifest"), "--output", str(root / "run")]
    main(["build-manifest", "--config", str(path), "--data-root", str(root / "dataset"),
          "--output", str(root / "manifest")])
    main(["train-phase1", *common])
    main(["train-phase2", *common, "--backbone-checkpoint", str(root / "run/phase1/last.pt")])
    return root / "manifest", root / "run/phase2/best_joint_validation.pt", root


def _evaluate(trained, name, *extra, device="cpu"):
    manifest, checkpoint, root = trained
    output = root / name
    main(["evaluate", "--checkpoint", str(checkpoint), "--manifest", str(manifest), "--device", device,
          "--gpus", "1", "--split", "valid", "--panels", "0", "--output", str(output), *extra])
    return json.loads((output / "metrics.json").read_text()), json.loads((output / "evaluation_config.json").read_text())


def test_a_subset_gives_the_protocols_own_numbers_in_its_order(trained, capsys):
    whole, _ = _evaluate(trained, "whole", "--protocol")
    assert list(whole) == list(PROTOCOL_SCENARIOS)
    capsys.readouterr()
    part, config = _evaluate(trained, "part", "--protocol", "--scenarios", "blur_only, clean_clean")
    assert list(part) == ["clean_clean", "blur_only"] == list(config["scenarios"])
    for name, values in part.items():
        for key, value in values.items():
            assert value == pytest.approx(whole[name][key], rel=1e-6, abs=1e-9), (name, key)
    printed = capsys.readouterr().out
    assert "[1/2] clean_clean" in printed and "[2/2] blur_only" in printed


@pytest.mark.parametrize("extra, message", [
    (("--scenarios", "blur_only"), "add --protocol"),
    (("--protocol", "--scenarios", "blur_only,fog_only"), "fog_only"),
])
def test_bad_scenarios_are_refused(trained, capsys, extra, message):
    with pytest.raises(SystemExit):
        _evaluate(trained, "refused", *extra)
    assert message in capsys.readouterr().err


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 autocast is measured on CUDA")
def test_amp_moves_the_metrics_by_float_noise_only(trained):
    plain, plain_config = _evaluate(trained, "fp32", device="cuda")
    half, half_config = _evaluate(trained, "fp16", "--amp", device="cuda")
    assert plain_config["amp"] is False and half_config["amp"] is True
    a, b = plain["requested"], half["requested"]
    assert a["image_count"] == b["image_count"]
    assert abs(a["image_psnr_db"] - b["image_psnr_db"]) < 0.05
    assert abs(a["image_ssim"] - b["image_ssim"]) < 1e-3
    assert a["baseline_image_psnr_db"] == pytest.approx(b["baseline_image_psnr_db"])
