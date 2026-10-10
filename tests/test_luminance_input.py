"""model.image_input: luminance -- JEPA learns edges on the grey frame, colour goes round it.

The QWT collapses RGB to luminance Y before analysing it, so the encoder, the EMA
teacher and the phase-1 anchor only ever see edges; phase 2's split decoder takes the
colour from the RGB frame the backbone kept and puts it back. Pins: the luminance QWT
is the same filter bank applied to Y (linear in the RGB coefficients) and inverts to
Y; the latent and the teacher's target are blind to a change of colour at fixed Y,
while the RGB backbone is not; phase 2 starts at the input's luminance and colour, so
the colour does reach the decoder; the phase-1 anchor decodes 16 Y coefficients;
configs without the key build the 48-channel RGB backbone; the key changes both
hashes; bad values and a coefficient phase-2 decoder are refused; configs/kaggle_gray
differs from p19_sharp by this key alone; the recipe trains, resumes and evaluates
through the CLI.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from qjepa.cli import _synthetic_batch, main
from qjepa.config import (build_backbone, build_decoders, build_phase1_model, load_config,
                          serializable_config, validate_config)
from qjepa.data import ImuNormalizer
from qjepa.models import LatentDecoders, RestorationSystem
from qjepa.models.color_edge import chroma, color_base, luminance
from qjepa.training.checkpoints import configuration_hash, load_checkpoint
from qjepa.transforms import QuaternionWaveletTransform2D
from qjepa.transforms.qwt import LUMINANCE_WEIGHTS
from test_kaggle_workflow import _write_dataset
from test_sharp_cli import _run_until_done, _sharp_smoke

FIRST_CONV = "stages.0.net.0.net.0.weight"   # the image encoder reads the QWT coefficients here


def _config(image_input=None, anchor=False):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    if image_input is not None:
        config["model"]["image_input"] = image_input
    if anchor:
        config["phase1"].update(decoder_enabled=True, coefficient_reconstruction_loss_weight=0.45)
    validate_config(config)
    return config


def _recolour(image):
    """The same frame in other colours: (0.587, -0.299, 0) has zero luminance."""
    direction = torch.tensor([0.587, -0.299, 0.0]).view(1, 3, 1, 1)
    pattern = torch.rand(image.shape[0], 1, *image.shape[-2:], generator=torch.Generator().manual_seed(7))
    recoloured = image + 0.3 * pattern * direction
    assert torch.allclose(luminance(recoloured), luminance(image), atol=1e-6)
    return recoloured


def _inputs(config):
    batch = _synthetic_batch(config)
    return batch["image_noisy"], batch["imu_noisy_phys"], batch["image_time"], batch["imu_times"]


def test_the_luminance_qwt_is_the_rgb_filter_bank_on_y_and_inverts_to_y():
    rgb_transform = QuaternionWaveletTransform2D()
    y_transform = QuaternionWaveletTransform2D(image_input="luminance")
    image = torch.rand(2, 3, 32, 32, dtype=torch.float64)
    rgb, _ = rgb_transform.analysis(image)
    y, layout = y_transform.analysis(image)
    assert rgb.shape[1] == rgb_transform.coeff_channels == 48
    assert y.shape[1] == y_transform.coeff_channels == 16
    weights = torch.tensor(LUMINANCE_WEIGHTS, dtype=torch.float64).view(1, 3, 1, 1, 1)
    assert torch.allclose(y, (rgb.reshape(2, 3, 16, 16, 16) * weights).sum(1), atol=1e-12)
    # Same weights as the split decoder's luminance, and Y comes back exactly.
    assert torch.allclose(y_transform.synthesis(y, layout), luminance(image), atol=1e-10)
    assert torch.allclose(y_transform.analysis(luminance(image))[0], y, atol=1e-12)
    with pytest.raises(ValueError, match="image_input"):
        QuaternionWaveletTransform2D(image_input="grey")


def test_the_latent_and_the_teacher_target_are_blind_to_colour_only_in_luminance_mode():
    for image_input, blind in (("rgb", False), ("luminance", True)):
        config = _config(image_input)
        torch.manual_seed(0)
        model = build_phase1_model(config, ImuNormalizer()).eval()
        image, imu, image_time, imu_times = _inputs(config)
        with torch.no_grad():
            latents = [model.backbone.encode_online(x, model.normalizer.normalize(imu), image_time, imu_times)
                       for x in (image, _recolour(image))]
            targets = [model.targets(x, imu)[0] for x in (image, _recolour(image))]
        for a, b in ((latents[0].ZI, latents[1].ZI), (latents[0].FI, latents[1].FI), tuple(targets)):
            assert torch.allclose(a, b, atol=1e-5) == blind, image_input
        assert (latents[0].image_rgb is not None) == blind
        assert latents[0].image_coefficients.shape[1] == (16 if blind else 48)


def test_phase2_starts_at_the_input_luminance_and_colour():
    config = _config("luminance")
    torch.manual_seed(0)
    model = build_phase1_model(config, ImuNormalizer())
    system = RestorationSystem(model.backbone, model.normalizer, build_decoders(config)).eval()
    image, imu, image_time, imu_times = _inputs(config)
    with torch.no_grad():
        restored = system(image, imu, image_time, imu_times)
    scale = int(config["phase2"]["split_color_scale"])
    assert restored.image.shape == image.shape
    assert torch.allclose(luminance(restored.image), luminance(image), atol=1e-5)
    # The colour came from the RGB frame the backbone kept: the QWT never saw it.
    assert torch.allclose(chroma(restored.image), chroma(color_base(image, scale)), atol=1e-5)
    assert chroma(restored.image).abs().mean() > 1e-2
    assert restored.image_coefficients.shape[1] == 16


def test_the_phase1_anchor_decodes_luminance_coefficients():
    config = _config("luminance", anchor=True)
    model = build_phase1_model(config, ImuNormalizer())
    image, imu, image_time, imu_times = _inputs(config)
    latent = model.backbone.encode_online(image, model.normalizer.normalize(imu), image_time, imu_times)
    coefficients, _ = model.reconstruct(latent)
    target, _ = model.backbone.image_transform.analysis(image)
    assert coefficients.shape == target.shape and coefficients.shape[1] == 16


def test_old_configs_build_rgb_and_the_key_changes_both_hashes():
    for path in Path("configs").glob("*.yaml"):
        if path.name not in ("kaggle_gray.yaml", "kaggle_glare.yaml", "kaggle_light.yaml",
                             "kaggle_illum.yaml", "kaggle_env.yaml",
                             "kaggle_local.yaml", "kaggle_ijepa.yaml",
                             "kaggle_ijepa_target.yaml", "kaggle_blur.yaml",
                             "kaggle_steady.yaml", "kaggle_inputnorm.yaml",
                             "kaggle_exposure.yaml", "kaggle_bilateral.yaml",
                             "kaggle_relight.yaml", "kaggle_halo.yaml", "kaggle_chroma.yaml"):  # p21-p34 extend p20
            assert "image_input" not in load_config(path)["model"], path
    plain, grey = _config(), _config("luminance")
    assert build_backbone(plain).image_transform.coeff_channels == 48
    assert build_backbone(plain).image_encoder.state_dict()[FIRST_CONV].shape[1] == 48
    assert build_backbone(grey).image_encoder.state_dict()[FIRST_CONV].shape[1] == 16
    for phase in ("phase1", "phase2"):
        assert configuration_hash(plain, phase) != configuration_hash(grey, phase)


def test_bad_values_and_a_coefficient_image_decoder_are_refused():
    config = _config()
    config["model"]["image_input"] = "grey"
    with pytest.raises(ValueError, match="image_input"):
        validate_config(config)
    config = _config()
    config["model"]["image_input"] = "luminance"
    config["phase2"]["image_decoder"] = "qwt_coefficients"
    with pytest.raises(ValueError, match="pixel image decoder|resnet_pixel"):
        validate_config(config)
    backbone = build_backbone(_config("luminance"))
    with pytest.raises(ValueError, match="pixel image decoder"):
        RestorationSystem(backbone, ImuNormalizer(), LatentDecoders(image_coefficient_channels=16))


def test_kaggle_gray_is_p19_sharp_plus_the_key():
    sharp = serializable_config(load_config("configs/kaggle_sharp.yaml"))
    grey = serializable_config(load_config("configs/kaggle_gray.yaml"))
    assert grey["model"].pop("image_input") == "luminance"
    for config in (sharp, grey):
        config.pop("_config_path", None)
        config["runtime"].pop("output_dir")
    assert grey == sharp


def test_the_recipe_trains_resumes_and_evaluates(tmp_path):
    torch.set_num_threads(1)
    root, manifest, output = tmp_path / "dataset", tmp_path / "manifest", tmp_path / "run"
    _write_dataset(root)
    config = _sharp_smoke()
    config["model"]["image_input"] = "luminance"
    config["phase1"].update(decoder_enabled=True, coefficient_reconstruction_loss_weight=0.45)
    path = tmp_path / "gray.yaml"
    path.write_text(yaml.safe_dump(config))
    common = ["--config", str(path), "--manifest", str(manifest), "--output", str(output)]
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest)])
    _run_until_done(["train-phase1", *common], output / "phase1/last.pt")
    phase1 = load_checkpoint(output / "phase1/last.pt")
    assert phase1["model"][f"backbone.image_encoder.{FIRST_CONV}"].shape[1] == 16
    last = output / "phase2/last.pt"
    _run_until_done(["train-phase2", *common, "--backbone-checkpoint", str(output / "phase1/last.pt")], last)
    assert load_checkpoint(last)["successful_updates"] == 3
    evaluation = output / "eval"
    main(["evaluate", "--checkpoint", str(last), "--manifest", str(manifest), "--device", "cpu",
          "--output", str(evaluation), "--panels", "0", "--split", "valid"])
    assert json.loads((evaluation / "metrics.json").read_text())["requested"]["image_count"] > 0
