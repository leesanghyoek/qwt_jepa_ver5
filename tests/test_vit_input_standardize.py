"""model.vit_input_standardize: the wavelet coefficients scaled before the ViT, as I-JEPA normalizes pixels.

p28 on Kaggle: phase 1's loss fell to 0.055, climbed to 0.15, then fell again. Measured: unscaled, the patch
embeddings start at 0.5x (image) / 0.3x (IMU) the sine-cosine positions and ~80% of the teacher's LayerNorm'd
output -- the target -- is a function of the token's position; the predictor learns that pattern first and the
loss climbs once the teacher learns content. Pins: absent, the ViT has no gains and every earlier checkpoint
and hash stays as it was; "global" gives each modality one gain, so the bands keep their ratios, and
"channel" sets every channel to the RMS -- either way each patch embedding starts with I-JEPA's variance on
normalized pixels (fan-in 768), as large as the positions; on dim frames whose spectrum falls like a photo's
position then explains much less of the target (channel least); the teacher gets the same gains; a fresh
CLI run calibrates once and the checkpoint carries the gains through a resume into phase 2's frozen
backbone; two DDP ranks calibrate alike and train what one process does; bad values fail; p29 is p28 with
"global" and half the phase-1 batch (twice as fast on Kaggle).
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F
import yaml

from qjepa.cli import main
from qjepa.config import load_config, validate_config
from qjepa.models.vit import PIXEL_FAN_IN, JointCoefficientViT, sincos_embedding
from qjepa.training.checkpoints import configuration_hash, load_checkpoint
from test_ijepa import _ijepa, _ijepa_dict, _model, _phase1_logged, _run_until_done
from test_kaggle_workflow import _write_dataset


def _photo_like(batch: int, size: int = 64, seed: int = 0, mean: float = 0.4, std: float = 0.25) -> torch.Tensor:
    """Frames whose power falls with frequency, as a photo's: smooth noise around ``mean``."""
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(batch, 3, size, size, generator=generator)
    for sigma in (6, 3):
        radius = 2 * sigma
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        noise = F.conv2d(F.pad(noise, (radius, radius, 0, 0), mode="reflect"),
                         kernel.view(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3)
        noise = F.conv2d(F.pad(noise, (0, 0, radius, radius), mode="reflect"),
                         kernel.view(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3)
    return (mean + std * noise / noise.std()).clamp(0, 1)


def _imu(batch: int, length: int = 32, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 6, length, generator=generator).cumsum(-1) * 0.1


def _standardized(mode: str = "global"):
    config = _ijepa_dict()
    config["model"]["vit_input_standardize"] = mode
    validate_config(config)
    return config


def _position_share(model, images, imu) -> float:
    """Share of the LayerNorm'd teacher output explained by the token's position alone."""
    with torch.no_grad():
        teacher, _ = model.targets(images, imu)
    tokens = F.layer_norm(teacher.flatten(2).transpose(1, 2), (teacher.shape[1],))
    total = (tokens - tokens.mean((0, 1), keepdim=True)).square().mean()
    return float(1 - (tokens - tokens.mean(0, keepdim=True)).square().mean() / total)


def test_absent_the_vit_has_no_gains_and_hashes_stay():
    plain = _model(_ijepa())
    assert not plain.backbone.joint_encoder.input_standardize
    assert not any("input_scale" in key for key in plain.state_dict())
    standardized = _model(_standardized())
    extra = set(standardized.state_dict()) - set(plain.state_dict())
    assert extra == {f"{owner}.joint_encoder.{name}_input_scale"
                     for owner in ("backbone", "teachers") for name in ("image", "imu")}
    for name in ("kaggle_ijepa", "kaggle_blur", "kaggle_steady"):
        assert "vit_input_standardize" not in load_config(f"configs/{name}.yaml")["model"]


@pytest.mark.parametrize("mode", ("global", "channel"))
def test_calibration_scales_the_inputs_and_balances_content_with_position(mode):
    model = _model(_standardized(mode))
    images, imu = _photo_like(16, mean=0.25, std=0.12), _imu(16)
    measured = model.calibrate_vit_inputs(images, imu, images, imu)
    vit = model.backbone.joint_encoder
    image_coeff, _ = model.backbone.image_transform.analysis(images)
    imu_coeff, _ = model.backbone.imu_transform.analysis(model.normalizer.normalize(imu))
    assert max(measured["image"]) > 10 * min(measured["image"])     # low-pass dwarfs detail before
    for coefficients, scale, embed in ((image_coeff, vit.image_input_scale, vit.image_embed),
                                       (imu_coeff, vit.imu_input_scale, vit.imu_embed)):
        dims = (0, *range(2, coefficients.ndim))
        scaled = coefficients * scale.view(1, -1, *[1] * (coefficients.ndim - 2))
        target = math.sqrt(PIXEL_FAN_IN / (embed.in_channels * math.prod(embed.kernel_size)))
        if mode == "channel":
            assert torch.allclose(scaled.square().mean(dims).sqrt(), torch.full_like(scale, target), rtol=1e-4)
        else:                                                       # one gain: the bands keep their ratios
            assert torch.allclose(scale, scale[0].expand_as(scale))
            assert float(scaled.square().mean().sqrt()) == pytest.approx(target, rel=1e-4)
    with torch.no_grad():
        for embed, coefficients, scale, grid in (
                (vit.image_embed, image_coeff, vit.image_input_scale.view(1, -1, 1, 1), (4, 4)),
                (vit.imu_embed, imu_coeff, vit.imu_input_scale.view(1, -1, 1), (2,))):
            patches = embed(coefficients * scale).flatten(2).transpose(1, 2)
            positions = sincos_embedding(patches.shape[-1], grid, patches)
            assert 0.7 < float(patches.std() / positions.std()) < 1.5
    # The teacher reads with the same gains.
    for name in ("image_input_scale", "imu_input_scale"):
        assert torch.equal(getattr(model.teachers.joint_encoder, name), getattr(vit, name))


def test_position_explains_much_less_of_the_target():
    images, imu = _photo_like(32, seed=3, mean=0.25, std=0.12), _imu(32, seed=4)   # a dim low-light frame
    shares = {"none": _position_share(_model(_ijepa()), images, imu)}
    for mode in ("global", "channel"):
        model = _model(_standardized(mode))
        model.calibrate_vit_inputs(images, imu, images, imu)
        shares[mode] = _position_share(model, images, imu)
    # Measured 0.79 / 0.55 / 0.42.
    assert shares["none"] > 0.7 and shares["global"] < shares["none"] - 0.15 and shares["channel"] < shares["global"]


def test_a_fresh_run_calibrates_once_and_phase2_freezes_those_gains(tmp_path):
    torch.set_num_threads(1)
    root, manifest = tmp_path / "dataset", tmp_path / "manifest"
    _write_dataset(root)
    config = _ijepa_dict(max_successful_updates=4, batch_size=4)
    config["model"]["vit_input_standardize"] = "global"
    config["phase2"]["max_successful_updates"] = 2
    config["runtime"].update(restart_above_rss_gib=1e-6, checkpoint_every_updates=2)   # resume mid-run
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest)])
    run = tmp_path / "run"
    restarts = _run_until_done(["train-phase1", "--config", str(path), "--manifest", str(manifest),
                                "--output", str(run)], run / "phase1/last.pt")
    assert restarts >= 1
    phase1 = load_checkpoint(run / "phase1/last.pt")["model"]
    gains = {key: value for key, value in phase1.items() if key.endswith("_input_scale")}
    assert len(gains) == 4 and not any(torch.equal(value, torch.ones_like(value)) for value in gains.values())
    assert torch.equal(gains["backbone.joint_encoder.image_input_scale"],
                       gains["teachers.joint_encoder.image_input_scale"])
    _run_until_done(["train-phase2", "--config", str(path), "--manifest", str(manifest), "--output", str(run),
                     "--backbone-checkpoint", str(run / "phase1/last.pt")], run / "phase2/last.pt")
    system = load_checkpoint(run / "phase2/last.pt")["system"]
    for name in ("image_input_scale", "imu_input_scale"):
        assert torch.equal(system[f"backbone.joint_encoder.{name}"], gains[f"backbone.joint_encoder.{name}"])


def test_two_processes_calibrate_alike_and_train_what_one_does(tmp_path):
    torch.set_num_threads(1)
    root, manifest = tmp_path / "dataset", tmp_path / "manifest"
    _write_dataset(root)
    config = _ijepa_dict(max_successful_updates=3, batch_size=2)
    config["model"]["vit_input_standardize"] = "global"
    config["phase2"]["batch_size"] = 2                          # ddp splits every batch over two ranks
    single = tmp_path / "single.yaml"
    single.write_text(yaml.safe_dump(config))
    config["runtime"].update(parallel="ddp", gpu_count=2)
    ddp = tmp_path / "ddp.yaml"
    ddp.write_text(yaml.safe_dump(config))
    main(["build-manifest", "--config", str(single), "--data-root", str(root), "--output", str(manifest)])
    gains = {}
    for name, path in (("single", single), ("ddp", ddp)):
        main(["train-phase1", "--config", str(path), "--manifest", str(manifest), "--output", str(tmp_path / name)])
        model = load_checkpoint(tmp_path / name / "phase1/last.pt")["model"]
        gains[name] = {key: value for key, value in model.items() if key.endswith("_input_scale")}
    assert gains["single"] and all(torch.equal(value, gains["ddp"][key]) for key, value in gains["single"].items())
    # Each rank calibrates on the whole bank: a rank with other gains would move the gathered loss at once.
    first = {name: _phase1_logged(tmp_path / name, "loss")[0] for name in ("single", "ddp")}
    assert first["ddp"] == pytest.approx(first["single"], rel=1e-5)


@pytest.mark.parametrize("value", ["yes", 1, True])
def test_bad_values_fail(value):
    config = _ijepa_dict()
    config["model"]["vit_input_standardize"] = value
    with pytest.raises(ValueError, match="vit_input_standardize"):
        validate_config(config)
    cnn = copy.deepcopy(load_config("configs/smoke.yaml"))
    cnn["model"]["vit_input_standardize"] = "global"
    with pytest.raises(ValueError, match="vit_input_standardize"):
        validate_config(cnn)


def test_p29_is_p28_with_the_key_and_a_halved_phase1_batch():
    p28, p29 = load_config("configs/kaggle_steady.yaml"), load_config("configs/kaggle_inputnorm.yaml")
    validate_config(p29)
    differ = {(section, key) for section in ("data", "model", "corruption", "phase1", "phase2", "monitor", "runtime")
              for key in {*p28.get(section, {}), *p29.get(section, {})}
              if p28.get(section, {}).get(key) != p29.get(section, {}).get(key)}
    assert differ == {("model", "vit_input_standardize"), ("phase1", "batch_size"), ("runtime", "output_dir")}
    assert p29["model"]["vit_input_standardize"] == "global"
    # Twice as fast on Kaggle's four CPUs: half the frames to corrupt per update, still a whole batch per GPU.
    assert p29["phase1"]["batch_size"] * 2 == p28["phase1"]["batch_size"] and p29["phase1"]["batch_size"] % 2 == 0
    for phase in ("phase1", "phase2"):
        assert configuration_hash(p28, phase) != configuration_hash(p29, phase)
