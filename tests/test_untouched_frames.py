"""Frames the corruption left exactly as they were are counted apart in validation and evaluation.

No stage of the corruption runs on ~2% of frames (a cleared environment, no blur, a variant without grain): the input
is the target and its PSNR is image_metrics' 120 dB floor, which no output reaches (fp16 rounding alone: ~80 dB). On
a 128-frame bank, 3 such frames raised the input mean 22.76 -> 25.04 dB and the plain frames' 33.72 -> 39.00 dB.

Pins: the overall means keep every frame (comparable with earlier runs); untouched_count and untouched_image_psnr_db
say how many there were and what the model made of them; the per-kind means and touched_* leave them out, so the
kinds and the untouched frames add up to every frame; kind_summary prints them apart.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from qjepa.cli import _dataset, _evaluate_with_overlap, _loader, _manifest, kind_summary, main
from qjepa.config import build_decoders, build_normalizer, build_phase1_model, phase2_backbone, phase2_latent_modules
from qjepa.data.dataset import PairedCameraImuDataset
from qjepa.models import RestorationSystem
from test_exposure_standardize import _config
from test_kaggle_workflow import _write_dataset


def test_untouched_frames_are_counted_apart(tmp_path, monkeypatch):
    torch.manual_seed(0)
    root, manifest_dir = tmp_path / "dataset", tmp_path / "manifest"
    _write_dataset(root)
    config = _config()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest_dir)])
    manifest, _ = _manifest(config, str(manifest_dir))
    phase1 = build_phase1_model(config, build_normalizer(manifest["meta"]))
    system = RestorationSystem(phase2_backbone(config, phase1), phase1.normalizer, build_decoders(config),
                               *phase2_latent_modules(config, phase1))
    original = PairedCameraImuDataset.__getitem__

    def every_third_untouched(self, index):
        item = original(self, index)
        if index % 3 == 0:
            item["image_noisy"] = item["image_clean"].clone()             # what the corruption sometimes does
        return item

    monkeypatch.setattr(PairedCameraImuDataset, "__getitem__", every_third_untouched)
    dataset = _dataset(config, manifest, "valid", fixed_realization=True)
    loader = _loader(config, dataset, 2, train=False)
    result = _evaluate_with_overlap(system, loader, dataset, torch.device("cpu"), maximum_batches=10_000)
    count = result["image_count"]
    untouched = len(range(0, count, 3))
    assert result["untouched_count"] == untouched
    kinds = sum(result.get(f"{kind}_count", 0) for kind in ("plain", "relit", "flare"))
    assert kinds + untouched == count                                     # every frame is somewhere, once
    # The overall means keep the untouched frames at 120 dB in; touched_* leaves them out.
    assert result["baseline_image_psnr_db"] > result["touched_baseline_image_psnr_db"] + 5
    expected = (result["baseline_image_psnr_db"] * count - 120.0 * untouched) / (count - untouched)
    assert result["touched_baseline_image_psnr_db"] == pytest.approx(expected, abs=1e-3)
    assert np.isfinite(result["untouched_image_psnr_db"]) and result["untouched_image_psnr_db"] > 60   # ~identity
    text = kind_summary(result)
    assert f"vao da sach n={untouched}: ra " in text


def test_kind_summary_without_untouched_frames_is_unchanged():
    metrics = {"v_plain_count": 4, "v_plain_baseline_image_psnr_db": 36.71, "v_plain_image_psnr_db": 27.85,
               "v_plain_worse_fraction": 1.0, "v_untouched_count": 0}
    assert kind_summary(metrics, "v_") == "tot 36.71->27.85 (n=4, 100% te hon vao)"
