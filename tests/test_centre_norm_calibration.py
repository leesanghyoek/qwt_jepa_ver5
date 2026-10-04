"""model.encoder_norm_calibration: centre-norm encoders start at O(1), not at 1e-4.

A centre norm keeps its input's scale and every conv (default init) and SiLU shrinks
it: on TartanAir the centre-norm encoder starts at FI RMS 1.9e-4 (GroupNorm 0.86).
The JEPA loss compares LayerNorm'd tokens, and below LayerNorm's eps the targets come
out at RMS 0.09 instead of 1 -- p20's JEPA loss was low for nothing over its first
~800 updates, then rose; the variance term sat at its maximum and the encoder spent
thousands of updates growing x10^4. Pins: calibration sets each centre norm's output
to RMS 1 on the calibration batch, layer by layer; the encoders and the JEPA targets
start at O(1) and the teachers equal the online encoders; a fixed gain keeps the
centre norm's point -- x and 3x still differ; the key needs centre norm and changes
the phase-1 hash; the CLI calibrates a fresh run once and never a resumed one.
"""

from __future__ import annotations

import copy

import pytest
import torch
import yaml

from qjepa.cli import RESTART_EXIT_CODE, _synthetic_batch, main
from qjepa.config import build_phase1_model, load_config, validate_config
from qjepa.data import ImuNormalizer
from qjepa.models.blocks import CentreNorm, calibrate_centre_norms
from qjepa.models.predictors import image_tokens
from qjepa.training.checkpoints import configuration_hash, load_checkpoint
from qjepa.training.losses import layer_norm_no_affine
from test_kaggle_workflow import _write_dataset
from test_sharp_cli import _sharp_smoke


def _config(calibrate=None):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["model"]["encoder_norm"] = "centre"
    if calibrate is not None:
        config["model"]["encoder_norm_calibration"] = calibrate
    validate_config(config)
    return config


def _model(config):
    torch.manual_seed(0)
    return build_phase1_model(config, ImuNormalizer()).eval()


def test_each_centre_norm_outputs_rms_one_on_the_calibration_batch():
    model = _model(_config())
    batch = _synthetic_batch(_config())
    coefficients, _ = model.backbone.image_transform.analysis(batch["image_noisy"])
    calibrate_centre_norms(model.backbone.image_encoder, coefficients)
    seen = []
    hooks = [norm.register_forward_hook(lambda _m, _i, out: seen.append(float(out.square().mean().sqrt())))
             for norm in model.backbone.image_encoder.modules() if isinstance(norm, CentreNorm)]
    with torch.no_grad():
        model.backbone.image_encoder(coefficients)
    for hook in hooks:
        hook.remove()
    assert len(seen) == 12 and all(abs(rms - 1.0) < 1e-3 for rms in seen)


def test_encoders_and_jepa_targets_start_at_order_one_and_teachers_follow():
    config = _config()
    batch = _synthetic_batch(config)
    sizes = {}
    for calibrate in (False, True):
        model = _model(config)
        if calibrate:
            model.calibrate_centre_norms(batch["image_noisy"], batch["imu_noisy_phys"],
                                         batch["image_clean"], batch["imu_clean_phys"])
            for name in ("image_encoder", "imu_encoder"):
                online, teacher = getattr(model.backbone, name), getattr(model.teachers, name)
                assert all(torch.equal(a, b) for a, b in zip(online.state_dict().values(),
                                                             teacher.state_dict().values()))
        with torch.no_grad():
            TI = model.targets(batch["image_clean"], batch["imu_clean_phys"])[0]
        sizes[calibrate] = (float(TI.square().mean().sqrt()),
                            float(layer_norm_no_affine(image_tokens(TI)).square().mean().sqrt()))
    assert sizes[False][0] < 1e-2 and sizes[False][1] < 0.5          # tiny, and LayerNorm cannot fix it
    assert 0.3 < sizes[True][0] < 3.0 and sizes[True][1] > 0.99


def test_a_calibrated_encoder_still_tells_x_from_3x():
    config = _config()
    batch = _synthetic_batch(config)
    model = _model(config)
    model.calibrate_centre_norms(batch["image_noisy"], batch["imu_noisy_phys"],
                                 batch["image_clean"], batch["imu_clean_phys"])
    window = model.normalizer.normalize(batch["imu_noisy_phys"])
    with torch.no_grad():
        weak, strong = model.backbone.encode_imu_dense(window), model.backbone.encode_imu_dense(3.0 * window)
    assert float((strong - weak).norm() / weak.norm()) > 0.1


def test_the_key_needs_centre_norm_and_changes_the_phase1_hash():
    plain, calibrated = _config(), _config(True)
    assert configuration_hash(plain, "phase1") != configuration_hash(calibrated, "phase1")
    bad = copy.deepcopy(plain)
    bad["model"].update(encoder_norm="group", encoder_norm_calibration=True)
    with pytest.raises(ValueError, match="encoder_norm_calibration"):
        validate_config(bad)
    bad["model"].update(encoder_norm="centre", encoder_norm_calibration="yes")
    with pytest.raises(ValueError, match="true or false"):
        validate_config(bad)
    assert load_config("configs/kaggle_gray.yaml")["model"]["encoder_norm_calibration"] is True


def test_the_cli_calibrates_a_fresh_run_once_and_never_a_resumed_one(tmp_path, capsys):
    torch.set_num_threads(1)
    root, manifest, output = tmp_path / "dataset", tmp_path / "manifest", tmp_path / "run"
    _write_dataset(root)
    config = _sharp_smoke()
    config["model"]["encoder_norm_calibration"] = True
    path = tmp_path / "calibrated.yaml"
    path.write_text(yaml.safe_dump(config))
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest)])
    common = ["train-phase1", "--config", str(path), "--manifest", str(manifest), "--output", str(output)]
    last = output / "phase1/last.pt"
    runs = 0
    while True:                                     # restarts after every checkpoint (exit 75)
        runs += 1
        try:
            main([*common, *(["--resume", str(last)] if last.exists() else [])])
            break
        except SystemExit as stop:
            assert stop.code == RESTART_EXIT_CODE
    assert runs > 1 and capsys.readouterr().out.count("centre norm calibration") == 1
    weights = [v for k, v in load_checkpoint(last)["model"].items()
               if k.startswith("backbone.image_encoder.") and k.endswith("net.0.net.1.weight")]
    assert weights and not torch.allclose(weights[0], torch.ones_like(weights[0]))
