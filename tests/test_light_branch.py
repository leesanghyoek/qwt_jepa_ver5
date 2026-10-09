"""phase2.split_light_*: LightBranch takes the glare off and lifts the dark before colour and edges.

Pins: the branch starts as the identity, so the decoder with it starts as the decoder without
it (the residual-at-init invariant); its output is J = max(I_lin - V, 0) * exp(g) in linear light
for the maps it predicts; every output pixel sees the whole frame; trained alone on frames
darkened and veiled in linear light it more than halves the low-frequency error; configs without
the keys build no light layers; configs/kaggle_light is p21_glare plus the lighter glare and the
branch keys, changes both hashes and must spell the keys out; the recipe trains both phases
through the CLI and logs image_light_l1.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from qjepa.cli import _synthetic_batch, main
from qjepa.config import (SPLIT_LIGHT_KEYS, build_decoders, build_phase1_model, load_config, serializable_config,
                          validate_config)
from qjepa.data import ImuNormalizer
from qjepa.models import RestorationSystem
from qjepa.models.color_edge import downsample
from qjepa.models.decoders import LightBranch, linear_to_srgb, srgb_to_linear
from qjepa.training.checkpoints import configuration_hash, load_checkpoint
from test_kaggle_workflow import _write_dataset
from test_sharp_cli import _records, _run_until_done, _sharp_smoke

LIGHT = {"split_light_branch": True, "split_light_width": 8, "split_light_scale": 4, "split_light_levels": 3,
         "split_light_weight": 1.0, "split_light_loss_scale": 8}


def _config(light: bool):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    if light:
        config["phase2"].update(LIGHT)
    validate_config(config)
    return config


def _veiled(n: int = 8, size: int = 64, seed: int = 0):
    """Smooth clean frames, darkened (x0.15-0.65) and veiled by a wide blob, both in linear light."""
    generator = torch.Generator().manual_seed(seed)
    clean = F.interpolate(torch.rand(n, 3, 8, 8, generator=generator), size=(size, size), mode="bilinear",
                          align_corners=False)
    clean = (clean + 0.15 * torch.rand(n, 3, size, size, generator=generator)).clamp(0, 1)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    cy, cx = torch.randint(0, size, (2, n, 1, 1), generator=generator)
    blob = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (size / 4) ** 2)).unsqueeze(1)
    veil = 0.6 * torch.rand(n, 1, 1, 1, generator=generator) * blob
    exposure = 0.15 + 0.5 * torch.rand(n, 1, 1, 1, generator=generator)
    return linear_to_srgb((srgb_to_linear(clean) * exposure + veil).clamp(0, 1)), clean


def test_the_branch_starts_as_the_identity_and_the_decoder_as_without_it():
    plain_config, light_config = _config(False), _config(True)
    torch.manual_seed(0)
    model = build_phase1_model(plain_config, ImuNormalizer()).eval()
    plain = build_decoders(plain_config)
    light = build_decoders(light_config)
    missing, unexpected = light.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and missing and all(key.startswith("image.light.") for key in missing)
    batch = _synthetic_batch(plain_config)
    inputs = (batch["image_noisy"], batch["imu_noisy_phys"], batch["image_time"], batch["imu_times"])
    with torch.no_grad():
        without = RestorationSystem(model.backbone, model.normalizer, plain).eval()(*inputs)
        with_light = RestorationSystem(model.backbone, model.normalizer, light).eval()(*inputs)
    assert torch.allclose(with_light.image, without.image, atol=1e-5)
    assert torch.allclose(with_light.image_parts["image_light"], batch["image_noisy"], atol=1e-5)
    assert "image_light" not in without.image_parts


def test_the_output_is_the_veil_taken_off_and_the_gain_applied_in_linear_light():
    branch = LightBranch(latent_channels=4, width=8, scale=4, levels=2)
    with torch.no_grad():
        branch.tail.bias[:3] = math.atanh(0.1)                   # V = 0.1
        branch.tail.bias[3] = 3 * math.atanh(math.log(2) / 3)    # g = log 2: gain x2
    image = torch.rand(2, 3, 32, 32)
    expected = linear_to_srgb((srgb_to_linear(image) - 0.1).clamp_min(0) * 2).clamp(0, 1)
    assert torch.allclose(branch(torch.zeros(2, 4, 2, 2), image), expected, atol=1e-5)


def test_every_output_pixel_sees_the_whole_frame():
    torch.manual_seed(1)
    branch = LightBranch(latent_channels=4, width=8, scale=4, levels=3)
    torch.nn.init.normal_(branch.tail.weight, std=0.1)
    image = (0.2 + 0.6 * torch.rand(1, 3, 64, 64)).requires_grad_()
    branch(torch.zeros(1, 4, 4, 4), image)[..., -8:, -8:].sum().backward()
    # A local network has exactly zero gradient from the far corner; this one does not.
    assert image.grad[..., :4, :4].abs().sum() > 0


def test_trained_alone_it_takes_off_a_veil_and_lifts_the_exposure():
    torch.manual_seed(0)
    branch = LightBranch(latent_channels=4, width=24, scale=4, levels=3)
    optimizer = torch.optim.Adam(branch.parameters(), 2e-3)
    held_noisy, held_clean = _veiled(32, seed=99)
    low = lambda image: float(F.l1_loss(downsample(image, 8), downsample(held_clean, 8)))
    before = low(held_noisy)
    for step in range(200):
        noisy, clean = _veiled(seed=step)
        loss = F.l1_loss(downsample(branch(torch.zeros(8, 4, 4, 4), noisy), 8), downsample(clean, 8))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        after = low(branch(torch.zeros(32, 4, 4, 4), held_noisy))
    assert after < 0.5 * before, (before, after)


def test_old_configs_build_no_light_layers_and_kaggle_light_spells_its_keys_out():
    for path in Path("configs").glob("*.yaml"):
        if path.name not in ("kaggle_light.yaml", "kaggle_illum.yaml", "kaggle_env.yaml",
                             "kaggle_local.yaml", "kaggle_ijepa.yaml",
                             "kaggle_ijepa_target.yaml", "kaggle_blur.yaml",
                             "kaggle_steady.yaml", "kaggle_inputnorm.yaml",
                             "kaggle_exposure.yaml", "kaggle_bilateral.yaml",
                             "kaggle_relight.yaml", "kaggle_halo.yaml"):  # p23-p33 extend p22
            assert "split_light_branch" not in load_config(path)["phase2"], path
    assert build_decoders(_config(False)).image.light is None
    glare = serializable_config(load_config("configs/kaggle_glare.yaml"))
    light = serializable_config(load_config("configs/kaggle_light.yaml"))
    assert light["phase2"]["split_light_branch"] is True and set(SPLIT_LIGHT_KEYS) <= set(light["phase2"])
    for config in (glare, light):
        config.pop("_config_path", None)
        config["runtime"].pop("output_dir")
    for key in ("split_light_branch", *SPLIT_LIGHT_KEYS):
        light["phase2"].pop(key)
    lighter = {"light_knee", "light_wide_gain", "light_gain", "light_bloom_strength", "light_ghost_strength"}
    for key in lighter:
        assert light["corruption"]["image"].pop(key) != glare["corruption"]["image"].pop(key), key
    assert light == glare
    full_glare, full_light = load_config("configs/kaggle_glare.yaml"), load_config("configs/kaggle_light.yaml")
    for phase in ("phase1", "phase2"):
        assert configuration_hash(full_glare, phase) != configuration_hash(full_light, phase)
    for key, value in (("split_light_width", None), ("split_light_scale", 3), ("split_light_levels", 0),
                       ("split_light_weight", -1.0), ("split_light_branch", "yes")):
        bad = copy.deepcopy(full_light)
        if value is None:
            del bad["phase2"][key]
        else:
            bad["phase2"][key] = value
        with pytest.raises(ValueError):
            validate_config(bad)
    bad = copy.deepcopy(_config(True))
    bad["phase2"]["image_decoder"] = "resnet_pixel"
    with pytest.raises(ValueError, match="split_color_edge"):
        validate_config(bad)


def test_the_recipe_trains_both_phases_through_the_cli(tmp_path):
    torch.set_num_threads(1)
    root, manifest, output = tmp_path / "dataset", tmp_path / "manifest", tmp_path / "run"
    _write_dataset(root)
    config = _sharp_smoke()
    config["phase2"].update(LIGHT)
    path = tmp_path / "light.yaml"
    path.write_text(yaml.safe_dump(config))
    common = ["--config", str(path), "--manifest", str(manifest), "--output", str(output)]
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest)])
    _run_until_done(["train-phase1", *common], output / "phase1/last.pt")
    last = output / "phase2/last.pt"
    _run_until_done(["train-phase2", *common, "--backbone-checkpoint", str(output / "phase1/last.pt")], last)
    checkpoint = load_checkpoint(last)
    assert checkpoint["successful_updates"] == 3
    assert any(key.startswith("decoders.image.light.") for key in checkpoint["system"])
    records = [record for record in _records(output / "phase2/train.jsonl") if "image_light_l1" in record]
    assert records and all(math.isfinite(record["image_light_l1"]) for record in records)
