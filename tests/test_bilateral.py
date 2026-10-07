"""p31: phase2.split_tone_grid -- brightness and colour as a bilateral grid of affine colour transforms.

On p28's noise ~95% of the input's low-frequency error is a per-region a * x + b of the input (74% one per
frame, 21% more per 4 x 4 region), so the stage predicts transforms instead of painting brightness. Pins:
slicing is trilinear in position and brightness -- a constant grid gives the same coefficients everywhere,
a grid that grows across the frame or up the brightness levels gives coefficients that grow with them;
the affine is applied to the pixel itself (zero = identity, a gain scales, an offset shifts); the stage
starts as the identity with the luminance as its guide, and reads the latent and the exposure statistics;
it can bend brightness per region; in the decoder it replaces the light branch and hands its output on as
image_light; phase 2 trains with it and logs the tone stage's L1; bad settings fail; p31 is p30 with the
light branch swapped for the grid, keeping p28's phase-1 hash.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch
import yaml

from qjepa.cli import main
from qjepa.config import build_decoders, load_config, validate_config
from qjepa.models.color_edge import luminance
from qjepa.models.decoders import EXPOSURE_FEATURES, BilateralGridTone, apply_affine, slice_grid
from qjepa.training.checkpoints import configuration_hash, load_checkpoint
from test_ijepa import _ijepa_dict, _run_until_done
from test_kaggle_workflow import _write_dataset

GRID = {"split_light_branch": False, "split_tone_grid": True, "split_tone_grid_size": 4, "split_tone_grid_bins": 4,
        "split_tone_grid_width": 8, "split_tone_grid_weight": 1.0}


def test_slicing_is_trilinear_in_position_and_brightness():
    guide = torch.rand(1, 1, 32, 32)
    constant = torch.full((1, 2, 4, 4, 4), 0.3)
    assert torch.allclose(slice_grid(constant, guide), torch.full((1, 2, 32, 32), 0.3), atol=1e-6)
    across = torch.arange(4.0).view(1, 1, 1, 1, 4).expand(1, 1, 4, 4, 4).contiguous()      # grows with x
    sliced = slice_grid(across, guide)[0, 0]
    assert torch.all(sliced[:, 1:] >= sliced[:, :-1] - 1e-6) and sliced[:, -1].mean() > sliced[:, 0].mean() + 2.5
    upward = torch.arange(4.0).view(1, 1, 4, 1, 1).expand(1, 1, 4, 4, 4).contiguous()      # grows with brightness
    ramp = torch.linspace(0, 1, 32).view(1, 1, 1, 32).expand(1, 1, 32, 32)
    levels = slice_grid(upward, ramp)[0, 0, 0]
    assert torch.all(levels[1:] >= levels[:-1] - 1e-6) and levels[-1] > levels[0] + 2.5


def test_the_affine_acts_on_the_pixel_itself():
    image = torch.rand(2, 3, 8, 8)
    zero = torch.zeros(2, 12, 8, 8)
    assert torch.equal(apply_affine(zero, image), image)
    gain = zero.clone().view(2, 3, 4, 8, 8)
    for c in range(3):
        gain[:, c, c] = -0.5                                     # A = 0.5 I
    assert torch.allclose(apply_affine(gain.view(2, 12, 8, 8), image), 0.5 * image)
    offset = zero.clone().view(2, 3, 4, 8, 8)
    offset[:, :, 3] = 0.1
    assert torch.allclose(apply_affine(offset.view(2, 12, 8, 8), image), image + 0.1)


def test_the_stage_starts_as_the_identity_and_reads_the_latent_and_the_statistics():
    tone = BilateralGridTone(16, size=4, bins=4, width=8, exposure_stats=True)
    image, latent = torch.rand(2, 3, 32, 32) * 0.8, torch.randn(2, 16, 2, 2)
    assert torch.allclose(tone(latent, image), image) and torch.allclose(tone.guide_map(image), luminance(image))
    assert tone.global_head[0].in_features == 8 + EXPOSURE_FEATURES
    torch.manual_seed(0)
    torch.nn.init.normal_(tone.out.weight, std=0.05)
    out = tone(latent, image)
    assert not torch.allclose(out, image)
    assert not torch.allclose(out, tone(latent + 1.0, image))                 # the latent matters
    with torch.no_grad():
        tone.global_head[0].weight[:, -EXPOSURE_FEATURES:] = 0.0
    assert not torch.allclose(out, tone(latent, image))                       # so do the statistics


def test_it_can_brighten_one_region_and_leave_another():
    tone = BilateralGridTone(16, size=4, bins=2, width=8)
    image = torch.full((1, 3, 32, 32), 0.3)
    grid = torch.zeros(1, 12, 2, 4, 4)
    grid.view(1, 3, 4, 2, 4, 4)[:, [0, 1, 2], [0, 1, 2], :, :, :2] = 1.0       # left half: gain 2
    tone.coefficients = lambda latent, image: grid
    with torch.no_grad():
        out = tone(torch.zeros(1, 16, 2, 2), image)[0, 0]
    assert float(out[:, :4].mean()) == pytest.approx(0.6, abs=1e-5)                  # brightened twice
    assert float(out[:, -4:].mean()) == pytest.approx(0.3, abs=1e-5)                 # left as it was


def test_in_the_decoder_it_replaces_the_light_branch():
    config = _ijepa_dict()
    config["phase2"].update(GRID)
    validate_config(config)
    decoder = build_decoders(config).image
    assert decoder.light is None and isinstance(decoder.tone_grid, BilateralGridTone)
    plain = _ijepa_dict()
    assert build_decoders(plain).image.tone_grid is None
    image = torch.rand(2, 3, 64, 64)
    latent = torch.randn(2, config["model"]["embedding_dim"], 4, 4)
    _, parts = decoder(latent, image)
    assert torch.allclose(parts["image_light"], image)                        # identity at initialisation


def test_phase2_trains_with_the_grid_and_logs_the_tone_l1(tmp_path):
    torch.set_num_threads(1)
    root, manifest = tmp_path / "dataset", tmp_path / "manifest"
    _write_dataset(root)
    config = _ijepa_dict(max_successful_updates=2, batch_size=2)
    config["phase2"].update(GRID, max_successful_updates=2, batch_size=2, lowfreq_mse_weight=10.0)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest)])
    run = tmp_path / "run"
    common = ["--config", str(path), "--manifest", str(manifest), "--output", str(run)]
    _run_until_done(["train-phase1", *common], run / "phase1/last.pt")
    _run_until_done(["train-phase2", *common, "--backbone-checkpoint", str(run / "phase1/last.pt")],
                    run / "phase2/last.pt")
    rows = [json.loads(line) for line in (run / "phase2/train.jsonl").read_text().splitlines()]
    assert len([row for row in rows if "image_light_l1" in row and "image_lowfreq_mse" in row]) == 2
    system = load_checkpoint(run / "phase2/last.pt")["system"]
    assert any(key.startswith("decoders.image.tone_grid.") for key in system)
    assert not any(key.startswith("decoders.image.light.") for key in system)


@pytest.mark.parametrize("change, message", [
    ({"split_tone_grid": "yes"}, "split_tone_grid must be"),
    ({"split_light_branch": True}, "replaces the light branch"),
    ({"split_tone_grid_bins": 1}, "split_tone_grid_bins"),
    ({"split_tone_grid_weight": -1.0}, "split_tone_grid_weight"),
])
def test_bad_settings_fail(change, message):
    config = copy.deepcopy(load_config("configs/kaggle_bilateral.yaml"))
    config["phase2"].update(change)
    with pytest.raises(ValueError, match=message):
        validate_config(config)
    missing = copy.deepcopy(load_config("configs/kaggle_bilateral.yaml"))
    del missing["phase2"]["split_tone_grid_size"]
    with pytest.raises(ValueError, match="split_tone_grid_size"):
        validate_config(missing)


def test_p31_is_p30_with_the_light_branch_swapped_for_the_grid():
    p28, p30, p31 = (load_config(f"configs/{name}.yaml") for name in ("kaggle_steady", "kaggle_exposure", "kaggle_bilateral"))
    validate_config(p31)
    differ = {(section, key) for section in ("data", "model", "corruption", "phase1", "phase2", "monitor", "runtime")
              for key in {*p30.get(section, {}), *p31.get(section, {})}
              if p30.get(section, {}).get(key) != p31.get(section, {}).get(key)}
    assert differ == {("phase2", key) for key in ("split_light_branch", "split_tone_grid", "split_tone_grid_size",
                                                   "split_tone_grid_bins", "split_tone_grid_width",
                                                   "split_tone_grid_weight")} | {("runtime", "output_dir")}
    assert configuration_hash(p28, "phase1") == configuration_hash(p31, "phase1")        # reuses p28's phase 1
