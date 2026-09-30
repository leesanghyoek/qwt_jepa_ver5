"""The post-sharpening probe: an unsharp mask on luminance, scored and picked on validation.

Pins: amount 0 changes nothing and the boost never moves the colour; the 2-4 px power
reads 1 for the clean frame, less when blurred, more when sharpened; the pick keeps
the PSNR budget and never goes past the clean frame's detail energy; the probe runs
end to end on a phase-2 checkpoint and saves its crops.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

from qjepa.cli import main
from qjepa.config import load_config
from qjepa.models.color_edge import chroma
from test_kaggle_workflow import _write_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from sharpen_probe import fine_power, gaussian_blur, pick, probe, report, sharpen  # noqa: E402


def _texture(seed=0):
    image = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(seed))
    return 0.25 + 0.5 * gaussian_blur(image, 0.7)            # inside (0, 1): no clamping below


def test_no_boost_changes_nothing_and_a_boost_keeps_the_colour():
    image = _texture()
    assert torch.allclose(sharpen(image, 1.2, 0.0), image)
    boosted = sharpen(image, 1.2, 0.5)
    assert not torch.allclose(boosted, image)
    assert torch.allclose(chroma(boosted), chroma(image), atol=1e-6)
    flat = torch.full((1, 3, 16, 16), 0.4)
    assert torch.allclose(gaussian_blur(flat, 2.0), flat)


def test_fine_power_reads_one_for_clean_less_when_blurred_more_when_sharpened():
    clean = _texture(1)
    assert abs(fine_power(clean, clean) - 1.0) < 1e-6
    blurred = gaussian_blur(clean, 1.0)
    assert fine_power(blurred, clean) < 0.8
    assert fine_power(sharpen(blurred, 1.0, 1.0), clean) > fine_power(blurred, clean)


def test_the_pick_keeps_the_psnr_budget_and_stops_at_clean_power():
    rows = {"input": {"image_psnr_db": 19.0, "fine_power": 0.3},
            "model": {"image_psnr_db": 25.0, "fine_power": 0.5},
            "sigma 1.2 · amount 0.5": {"image_psnr_db": 24.9, "fine_power": 0.8},
            "sigma 1.2 · amount 1.0": {"image_psnr_db": 24.7, "fine_power": 0.95},   # PSNR -0.3
            "sigma 0.8 · amount 2.0": {"image_psnr_db": 24.95, "fine_power": 1.3}}   # past clean
    assert pick(rows, 0.2) == "sigma 1.2 · amount 0.5"
    assert pick(rows, 0.5) == "sigma 1.2 · amount 1.0"
    assert pick({"input": rows["input"], "model": rows["model"]}, 0.2) == "model"


def test_the_probe_runs_on_a_phase2_checkpoint(tmp_path):
    torch.set_num_threads(1)
    root, manifest = tmp_path / "dataset", tmp_path / "manifest"
    _write_dataset(root)
    config = yaml.safe_load(yaml.safe_dump({k: v for k, v in load_config("configs/smoke.yaml").items()
                                            if k != "_config_path"}))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    common = ["--config", str(path), "--manifest", str(manifest), "--output", str(tmp_path / "run")]
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest)])
    main(["train-phase1", *common])
    main(["train-phase2", *common, "--backbone-checkpoint", str(tmp_path / "run/phase1/last.pt")])
    panel = tmp_path / "panel.png"
    results = probe(tmp_path / "run/phase2/last.pt", str(manifest), torch.device("cpu"), None, 0.2, panel)
    assert results["frames"] > 0 and panel.is_file()
    assert {"input", "model", "sigma 1.2 · amount 1.0"} <= set(results["rows"]) and results["pick"] in results["rows"]
    assert "<- chọn" in report(results)
