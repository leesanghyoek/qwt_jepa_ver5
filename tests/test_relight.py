"""p32: phase2.split_relight -- the stop map the corruption applied, predicted and divided out before the grid.

On p31 the bilateral grid left dark regions dark (relit frames 8% too dark on average, 0.099 low-frequency RMSE
against 0.036 for a 4 x 4 regional oracle) and changed 98% of the frames nobody had darkened. Dividing a noisy
frame by the TRUE stop map and fitting one affine per channel leaves 0.031 (README p32_relight). Pins: the stop
map is the uneven light's field in stops, scaled by the low-light stage (vignette and exposure gain at the sRGB
exponent, the whole map by tone_gamma), and zero on a clean frame or one without either stage; dividing a frame
lit by a known field by that field gives the frame back; the dataset carries the map only when asked, mirrored
with the frame; the stage starts as the identity (stops 0), reads the latent, the coordinates and the exposure
statistics, and brightens exactly by the stops it predicts; only the stop L1 trains the map (the image losses
reach the frame, not the map); in the decoder it sits before the grid and needs it;
absent, the decoder keeps p31's layers; phase 2 trains with it and logs the stop L1, and
brightness_probe scores its map; bad settings fail; p32 is
p31 plus the relight keys, keeping p28's phase-1 hash.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
import yaml

from qjepa.cli import _dataset, main
from qjepa.config import build_decoders, load_config, validate_config
from qjepa.corruptions.image import SRGB_EXPONENT, _vignette, brightness_stops
from qjepa.corruptions.light import illumination_field, linear_to_srgb, srgb_to_linear
from qjepa.models.decoders import EXPOSURE_FEATURES, RELIGHT_STOP_RANGE, RelightStops
from qjepa.training.checkpoints import configuration_hash, load_checkpoint
from test_bilateral import GRID
from test_ijepa import _ijepa_dict, _run_until_done
from test_kaggle_workflow import _write_dataset

RELIGHT = {**GRID, "split_relight": True, "split_relight_size": 16, "split_relight_widths": [8, 8, 8],
           "split_relight_weight": 0.5}
ILLUMINATION = {"gradient_stops": 2.0, "gradient_angle": 0.3,
                "blobs": [{"y": 0.3, "x": 0.6, "sigma": 0.2, "stops": -2.5}],
                "smudges": [{"y": 0.7, "x": 0.3, "size": 0.1, "elongation": 2.0, "angle": 1.0, "depth": 0.8}],
                "max_brighten_stops": 0.75}


def _params(**change):
    params = {"mode": "full", "clean": False, "low_light": False, "sensor_noise": True, "illumination": False,
              "vignette_strength": 0.3, "exposure_gain": 0.6, "tone_gamma": 0.8}
    params.update(change)
    return params


def test_the_stop_map_is_the_field_scaled_by_the_low_light_stage():
    h, w = 32, 48
    assert not brightness_stops(_params(), h, w).any()                             # grain only
    assert not brightness_stops(_params(clean=True, illumination=True, illumination_params=ILLUMINATION), h, w).any()
    field = np.log2(illumination_field(h, w, ILLUMINATION))
    uneven = brightness_stops(_params(illumination=True, illumination_params=ILLUMINATION), h, w)
    assert uneven.shape == (h, w) and uneven.dtype == np.float32
    assert np.allclose(uneven, field, atol=1e-5) and uneven.min() < -2.0          # the smudge and the dark blob
    dark = brightness_stops(_params(low_light=True, illumination=True, illumination_params=ILLUMINATION), h, w)
    sensor = SRGB_EXPONENT * np.log2(_vignette(h, w, 0.3)[..., 0] * 0.6)
    assert np.allclose(dark, 0.8 * (field + sensor), atol=1e-5)
    assert dark[0, 0] < dark[h // 2, w // 2 - 1] - 0.5 * (field[h // 2, w // 2 - 1] - field[0, 0]) - 0.1   # vignette


def test_dividing_by_the_map_gives_the_frame_back():
    rng = np.random.default_rng(0)
    clean = rng.uniform(0.05, 0.6, (32, 32, 3)).astype(np.float32)
    stops = brightness_stops(_params(illumination=True, illumination_params=ILLUMINATION), 32, 32)
    lit = linear_to_srgb(srgb_to_linear(clean) * np.exp2(stops)[..., None])
    image = torch.from_numpy(lit.transpose(2, 0, 1))[None]
    relight = RelightStops(4, size=32, widths=(4, 4))
    relight.stops = lambda latent, image: torch.from_numpy(stops)[None, None]
    relit, predicted = relight(torch.zeros(1, 4, 2, 2), image)
    assert torch.allclose(relit[0], torch.from_numpy(clean.transpose(2, 0, 1)), atol=1e-4)
    assert float((image - relit).abs().max()) > 0.1                                # it did move the light


def test_the_dataset_carries_the_map_only_when_asked_and_mirrors_it(tmp_path):
    root, manifest = tmp_path / "dataset", tmp_path / "manifest"
    _write_dataset(root)
    config = _ijepa_dict()
    illum = {k: v for k, v in load_config("configs/kaggle_relight.yaml")["corruption"]["image"].items()
             if k.startswith("illum_")}
    config["corruption"]["image"].update(illum, illum_probability=1.0, env_clear_probability=0.0)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest)])
    from qjepa.cli import _manifest
    loaded, _ = _manifest(config, str(manifest))
    assert "image_stops" not in _dataset(config, loaded, "train", fixed_realization=True)[0]
    plain = _dataset(config, loaded, "train", fixed_realization=True, brightness_target=True)
    mirrored = _dataset(config, loaded, "train", fixed_realization=True, brightness_target=True, hflip_probability=1.0)
    item, flipped = plain[0], mirrored[0]
    params = item["corruption"]["image"]
    expected = brightness_stops(params, *item["image_noisy"].shape[-2:])
    assert item["image_stops"].shape == (1, *item["image_noisy"].shape[-2:])
    assert torch.allclose(item["image_stops"][0], torch.from_numpy(expected))
    assert torch.allclose(flipped["image_stops"], item["image_stops"].flip(-1))
    assert torch.allclose(flipped["image_noisy"], item["image_noisy"].flip(-1))


def test_the_stage_starts_as_the_identity_and_reads_its_inputs():
    relight = RelightStops(16, size=16, widths=(8, 8, 8), exposure_stats=True)
    image, latent = torch.rand(2, 3, 32, 32) * 0.8, torch.randn(2, 16, 4, 4)
    relit, stops = relight(latent, image)
    assert stops.shape == (2, 1, 16, 16) and not stops.any()
    assert torch.allclose(relit, image, atol=1e-5)
    assert relight.global_head[0].in_features == 8 + EXPOSURE_FEATURES
    torch.manual_seed(0)
    torch.nn.init.normal_(relight.out.weight, std=0.1)
    stops = relight.stops(latent, image)
    assert not torch.allclose(stops, relight.stops(latent + 1.0, image))         # the latent matters
    with torch.no_grad():
        relight.global_head[0].weight[:, -EXPOSURE_FEATURES:] = 0.0
    assert not torch.allclose(stops, relight.stops(latent, image))               # so do the statistics
    # Coordinates: a uniform frame still gets a map that varies across it (vignette, gradients).
    flat = relight.stops(latent[:1], torch.full((1, 3, 32, 32), 0.4))
    assert float(flat.detach().std()) > 0


def test_it_brightens_by_exactly_the_stops_and_is_bounded():
    relight = RelightStops(4, size=8, widths=(4, 4))
    image = torch.full((1, 3, 16, 16), 0.2)
    for stops, gain in ((-2.0, 4.0), (1.0, 0.5), (-20.0, 2 ** -RELIGHT_STOP_RANGE[0])):
        relight.stops = lambda latent, image, s=stops: torch.full((1, 1, 8, 8), s)
        with torch.no_grad():
            relit, _ = relight(torch.zeros(1, 4, 2, 2), image)
        expected = linear_to_srgb(np.minimum(srgb_to_linear(np.array([0.2], np.float32)) * gain, 1.0))
        assert float(relit.mean()) == pytest.approx(float(np.clip(expected[0], 0, 1)), abs=1e-5)


def test_in_the_decoder_it_sits_before_the_grid_and_needs_it():
    config = _ijepa_dict()
    config["phase2"].update(RELIGHT)
    validate_config(config)
    decoder = build_decoders(config).image
    assert isinstance(decoder.relight, RelightStops) and decoder.tone_grid is not None
    image = torch.rand(2, 3, 64, 64)
    latent = torch.randn(2, config["model"]["embedding_dim"], 4, 4)
    _, parts = decoder(latent, image)
    assert parts["image_stops"].shape == (2, 1, 16, 16)
    assert torch.allclose(parts["image_light"], image, atol=1e-5)                 # identity at initialisation
    # Absent, the decoder keeps p31's layers: its checkpoints load.
    p31 = _ijepa_dict()
    p31["phase2"].update(GRID)
    old = build_decoders(p31).image
    assert old.relight is None and set(old.state_dict()) == {key for key in decoder.state_dict()
                                                            if not key.startswith("relight.")}


def test_phase2_trains_with_it_and_logs_the_stop_l1(tmp_path):
    torch.set_num_threads(1)
    root, manifest = tmp_path / "dataset", tmp_path / "manifest"
    _write_dataset(root)
    config = _ijepa_dict(max_successful_updates=2, batch_size=2)
    config["phase2"].update(RELIGHT, max_successful_updates=2, batch_size=2)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest)])
    run = tmp_path / "run"
    common = ["--config", str(path), "--manifest", str(manifest), "--output", str(run)]
    _run_until_done(["train-phase1", *common], run / "phase1/last.pt")
    _run_until_done(["train-phase2", *common, "--backbone-checkpoint", str(run / "phase1/last.pt")],
                    run / "phase2/last.pt")
    rows = [json.loads(line) for line in (run / "phase2/train.jsonl").read_text().splitlines()]
    logged = [row["image_stops_l1"] for row in rows if "image_stops_l1" in row]
    assert len(logged) == 2 and all(value >= 0 for value in logged)
    system = load_checkpoint(run / "phase2/last.pt")["system"]
    assert any(key.startswith("decoders.image.relight.") for key in system)
    # tools/brightness_probe.py scores the predicted map against the corruption's.
    import subprocess
    import sys
    output = tmp_path / "brightness.json"
    result = subprocess.run([sys.executable, "tools/brightness_probe.py", "--checkpoint", str(run / "phase2/last.pt"),
                             "--manifest", str(manifest), "--samples", "4", "--batch", "2", "--device", "cpu",
                             "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    row = json.loads(output.read_text())["tat ca"]
    assert row["stop doan"] >= 0 and row["|stop that|"] >= 0 and "stop doan" in result.stdout


@pytest.mark.parametrize("change, message", [
    ({"split_relight": "yes"}, "split_relight must be"),
    ({"split_tone_grid": False}, "split_tone_grid: true"),
    ({"split_relight_size": 4}, "split_relight_size must be"),
    ({"split_relight_size": 20}, "halve cleanly"),
    ({"split_relight_widths": [32]}, "split_relight_widths"),
    ({"split_relight_weight": -1.0}, "split_relight_weight"),
])
def test_bad_settings_fail(change, message):
    config = copy.deepcopy(load_config("configs/kaggle_relight.yaml"))
    config["phase2"].update(change)
    with pytest.raises(ValueError, match=message):
        validate_config(config)
    missing = copy.deepcopy(load_config("configs/kaggle_relight.yaml"))
    del missing["phase2"]["split_relight_widths"]
    with pytest.raises(ValueError, match="split_relight_widths"):
        validate_config(missing)


def test_p32_is_p31_plus_the_relight_keys_and_keeps_p28s_phase1():
    p28, p31, p32 = (load_config(f"configs/{name}.yaml") for name in ("kaggle_steady", "kaggle_bilateral", "kaggle_relight"))
    validate_config(p32)
    differ = {(section, key) for section in ("data", "model", "corruption", "phase1", "phase2", "monitor", "runtime")
              for key in {*p31.get(section, {}), *p32.get(section, {})}
              if p31.get(section, {}).get(key) != p32.get(section, {}).get(key)}
    assert differ == {("phase2", key) for key in ("split_relight", "split_relight_size", "split_relight_widths",
                                                   "split_relight_weight")} | {("runtime", "output_dir")}
    assert configuration_hash(p28, "phase1") == configuration_hash(p32, "phase1")        # reuses p28's phase 1
    assert configuration_hash(p31, "phase2") != configuration_hash(p32, "phase2")


def test_only_its_own_loss_trains_the_map():
    relight = RelightStops(4, size=8, widths=(4, 4))
    torch.nn.init.normal_(relight.out.weight, std=0.1)
    image = (torch.rand(1, 3, 16, 16) * 0.5).requires_grad_(True)
    relit, stops = relight(torch.randn(1, 4, 2, 2), image)
    relit.sum().backward()                                                     # an image loss
    assert image.grad.abs().sum() > 0                                          # reaches the frame
    assert all(p.grad is None or not p.grad.any() for p in relight.parameters())   # not the map
    stops.abs().mean().backward()                                              # the stop L1
    assert relight.out.weight.grad.abs().sum() > 0
