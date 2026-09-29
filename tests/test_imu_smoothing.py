"""IMU smoothing: (A) the fixed post-filter probe, (B) the IMU refiner, (C) the jitter term.

p16 left IMU errors that are mostly jitter (0.70 accel / 0.78 gyro of the total,
sample to sample). These pin that the refiner starts as the identity and sees
0.65 s, that configs without the keys build and score exactly what they did, that
the jitter term charges only change beyond the clean signal's, that phase 2 trains
the refiner and logs the term, and that the probe scores fixed filters on a
phase-2 checkpoint.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
import yaml

from qjepa.cli import main
from qjepa.config import build_decoders, build_phase1_model, load_config, seed_everything, validate_config
from qjepa.data import ImuNormalizer
from qjepa.models import RestorationSystem
from qjepa.models.decoders import ImuRefiner
from qjepa.training.checkpoints import configuration_hash
from qjepa.training.losses import excess_jitter, phase2_reconstruction_loss
from qjepa.training.phase2 import Phase2Trainer
from test_kaggle_workflow import _write_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from imu_postfilter_probe import gaussian, median, probe, report  # noqa: E402

IMU_KEYS = ("imu_refiner_blocks", "imu_refiner_width", "imu_jitter_weight")
SMALL_NAF = dict(split_edge_naf_widths=[4, 6, 8, 12, 16], split_edge_naf_enc_blocks=[1, 1, 1, 1],
                 split_edge_naf_middle_blocks=1, split_edge_naf_dec_blocks=[1, 1, 1, 1],
                 split_edge_refiner_width=8, split_edge_refiner_blocks=1)


def _system(config, seed=3):
    seed_everything(seed)
    phase1 = build_phase1_model(config, ImuNormalizer())
    seed_everything(config["phase2"]["decoder_initialization_seed"])
    return RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config))


def _batch():
    times = torch.arange(32).float().mul(0.01).repeat(1, 1)
    clean, imu = torch.rand(1, 3, 32, 32), torch.randn(1, 6, 32)
    return {"image_clean": clean, "image_noisy": (clean * 0.35).clamp(0, 1), "imu_clean_phys": imu,
            "imu_noisy_phys": imu + 0.05 * torch.randn_like(imu), "image_time": times.mean(dim=1),
            "imu_times": times, "sample_id": ["s"]}


def _smoke(**phase2):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(SMALL_NAF, **phase2)
    validate_config(config)
    return config


def test_the_refiner_starts_as_the_identity_and_sees_65_samples():
    torch.manual_seed(0)
    refiner = ImuRefiner(6, width=8, blocks=4)
    restored, noisy = torch.randn(2, 6, 128), torch.randn(2, 6, 128)
    assert torch.equal(refiner(restored, noisy), restored)
    with torch.no_grad():
        refiner.tail.weight.normal_(0.0, 0.5)
    base = refiner(restored, noisy)
    poked = noisy.clone()
    poked[:, :, 64] += 5.0
    moved = (refiner(restored, poked) - base).abs().amax(dim=(0, 1)) > 1e-6
    reach = moved.nonzero().flatten()
    assert int(reach.min()) == 64 - 32 and int(reach.max()) == 64 + 32       # 65 samples, 0.65 s
    assert sum(p.numel() for p in ImuRefiner(6, 32, 4).parameters()) < 30_000


def test_configs_without_the_keys_build_and_score_what_they_did():
    config = _smoke()
    for key in IMU_KEYS:
        config["phase2"].pop(key, None)
    decoders = build_decoders(config)
    assert decoders.imu_refiner is None and not any(k.startswith("imu_refiner.") for k in decoders.state_dict())
    _, parts = phase2_reconstruction_loss(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8),
                                          torch.randn(1, 6, 16), torch.randn(1, 6, 16))
    assert not any("jitter" in key for key in parts)


def test_at_initialisation_the_refiner_changes_nothing_and_rescores_its_output():
    plain = _system(_smoke())
    refined = _system(_smoke(imu_refiner_blocks=2, imu_refiner_width=8))
    refined.load_state_dict(plain.state_dict(), strict=False)
    batch = _batch()
    with torch.no_grad():
        a = plain(batch["image_noisy"], batch["imu_noisy_phys"], batch["image_time"], batch["imu_times"])
        b = refined(batch["image_noisy"], batch["imu_noisy_phys"], batch["image_time"], batch["imu_times"])
        coefficients, _ = refined.backbone.imu_transform.analysis(b.imu_normalized)
    assert torch.allclose(a.imu_normalized, b.imu_normalized, atol=1e-6)
    assert torch.allclose(b.imu_coefficients, coefficients)             # Haar terms score the output


def test_jitter_charges_only_change_beyond_the_clean_signal():
    clean = torch.sin(torch.linspace(0, 12, 128)).repeat(2, 6, 1)
    assert float(excess_jitter(clean, clean)) == 0.0
    assert float(excess_jitter(0.5 * clean, clean)) == 0.0                     # smoother: free
    noisy = clean + 0.05 * torch.randn(clean.shape, generator=torch.Generator().manual_seed(1))
    assert float(excess_jitter(noisy, clean)) > 0.01
    assert float(excess_jitter(gaussian(noisy, 1.0), clean)) < float(excess_jitter(noisy, clean))
    with pytest.raises(ValueError):
        excess_jitter(noisy[:, :3], clean)


def test_the_loss_adds_the_jitter_term_with_its_weight():
    torch.manual_seed(0)
    image, imu, imu_clean = torch.rand(1, 3, 8, 8), torch.randn(1, 6, 16), torch.randn(1, 6, 16)
    base, _ = phase2_reconstruction_loss(image, image, imu, imu_clean)
    total, parts = phase2_reconstruction_loss(image, image, imu, imu_clean, jitter_weight=2.0)
    assert {"imu_accel_jitter", "imu_gyro_jitter"} <= set(parts)
    assert float(total) == pytest.approx(float(base) + 2.0 * float(parts["imu_accel_jitter"] + parts["imu_gyro_jitter"]))


@pytest.mark.parametrize("change, message", [
    (dict(imu_refiner_blocks=-1), "imu_refiner_blocks"),
    (dict(imu_refiner_width=0), "imu_refiner_width"),
    (dict(imu_jitter_weight=-0.5), "imu_jitter_weight"),
])
def test_bad_settings_are_rejected(change, message):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(change)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_phase2_trains_the_refiner_and_logs_the_jitter():
    config = _smoke(imu_refiner_blocks=2, imu_refiner_width=8, imu_jitter_weight=2.0)
    system = _system(config)
    before = [p.detach().clone() for p in system.decoders.imu_refiner.parameters()]
    trainer = Phase2Trainer(system, config, torch.device("cpu"), "phase1.pt")
    metrics = trainer.step([_batch()] * config["phase2"]["gradient_accumulation"])
    assert metrics["skipped"] is False and {"imu_accel_jitter", "imu_gyro_jitter"} <= set(metrics)
    assert any(not torch.equal(a, b) for a, b in zip(before, system.decoders.imu_refiner.parameters()))
    rebuilt = _system(config, seed=0)
    rebuilt.load_state_dict(trainer.checkpoint_payload(config)["system"], strict=True)


def test_the_imu_recipe_is_p16_plus_the_imu_keys():
    imu = load_config("configs/kaggle_imu.yaml")
    phase2 = imu["phase2"]
    assert phase2["split_edge_naf_widths"] == [48, 64, 96, 128, 160]
    assert phase2["split_edge_naf_enc_blocks"] == [3, 2, 4, 4] and phase2["split_edge_naf_dec_blocks"] == [3, 2, 2, 2]
    assert phase2["split_edge_refiner_scale"] == 1 and phase2["split_edge_refiner_width"] == 32
    assert phase2["precision"] == "amp_fp16" and phase2["perceptual_weight"] == 0.5    # speed only
    assert (phase2["imu_refiner_blocks"], phase2["imu_refiner_width"], phase2["imu_jitter_weight"]) == (4, 32, 2.0)
    base = load_config("configs/kaggle_tartanair_v2.yaml")
    assert configuration_hash(imu, "phase1") == configuration_hash(base, "phase1")     # same phase 1 recipe
    assert imu["runtime"]["parallel"] == "ddp" and imu["runtime"]["output_dir"].endswith("p16_imu_smooth")


def test_the_filters_keep_a_constant_and_a_median_drops_a_spike():
    flat = torch.full((1, 6, 32), 2.0)
    assert torch.allclose(gaussian(flat, 1.5), flat) and torch.equal(median(flat, 5), flat)
    spiky = flat.clone()
    spiky[:, :, 10] += 9.0
    assert torch.equal(median(spiky, 3), flat)


def test_the_probe_scores_fixed_filters_on_a_phase2_checkpoint(tmp_path):
    torch.set_num_threads(1)
    root, manifest = tmp_path / "dataset", tmp_path / "manifest"
    _write_dataset(root)
    config = yaml.safe_load(yaml.safe_dump({k: v for k, v in load_config("configs/smoke.yaml").items()
                                            if k != "_config_path"}))
    config["phase2"].update(imu_refiner_blocks=1, imu_refiner_width=4, imu_jitter_weight=1.0)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    common = ["--config", str(path), "--manifest", str(manifest), "--output", str(tmp_path / "run")]
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest)])
    main(["train-phase1", *common])
    main(["train-phase2", *common, "--backbone-checkpoint", str(tmp_path / "run/phase1/last.pt")])
    results = probe(tmp_path / "run/phase2/last.pt", str(manifest), torch.device("cpu"), None)
    assert results["windows"] > 0
    rows = results["rows"]
    assert {"input (chua xu ly)", "model", "model + Gauss 1", "model + median 3"} <= set(rows)
    for sensor in ("accel", "gyro"):
        assert rows["model + Gauss 2"][sensor]["excess_jitter"] <= rows["model"][sensor]["excess_jitter"]
        assert all(value >= 0 for value in rows["model"][sensor].values())
    text = report(results)
    assert "ACCEL" in text and "GYRO" in text and ("Tot nhat" in text or "Khong bo loc" in text)
