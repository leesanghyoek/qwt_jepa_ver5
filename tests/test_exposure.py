"""p30: phase2.split_exposure_stats and phase2.lowfreq_mse_weight -- phase 2 aimed at brightness, not edges.

On p28 ~95% of the restored frames' squared error sat in the QWT low-pass band, and on p26 (same decoder)
frames exposed right lost 9 dB to a brightness the decoder should have left alone. Pins: the exposure
statistics are the input's luminance quantiles (ordered), a histogram summing to 1 and each channel's
maximum, scale with the frame's brightness, and carry no gradient; absent, the tone head and the light
branch keep their old shapes, so p28's checkpoints load; on, both widen by those 27 numbers and use them,
and both still start as the identity; the low-frequency MSE is the MSE of 8x8 block means -- blind to
detail inside a block, the square of a uniform brightness error; phase 2 trains with both on and logs the
term; bad settings fail; p30 is p28 with these two keys, so it keeps p28's phase-1 hash.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch
import torch.nn.functional as F
import yaml

from qjepa.cli import main
from qjepa.config import build_decoders, load_config, validate_config
from qjepa.models.decoders import EXPOSURE_FEATURES, EXPOSURE_QUANTILES, GlobalToneColor, LightBranch, exposure_statistics
from qjepa.training.checkpoints import configuration_hash, load_checkpoint
from qjepa.training.losses import lowfreq_mse
from test_ijepa import _ijepa_dict, _run_until_done
from test_kaggle_workflow import _write_dataset


def _frames(batch=4, size=64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(batch, 3, size, size, generator=generator) * 0.8


def test_the_statistics_are_quantiles_a_histogram_and_maxima_without_gradient():
    image = _frames().requires_grad_(True)
    stats = exposure_statistics(image)
    assert stats.shape == (4, EXPOSURE_FEATURES) and not stats.requires_grad
    quantiles = stats[:, :len(EXPOSURE_QUANTILES)]
    histogram = stats[:, len(EXPOSURE_QUANTILES):len(EXPOSURE_QUANTILES) + 16]
    assert torch.all(quantiles[:, 1:] >= quantiles[:, :-1])
    assert torch.allclose(histogram.sum(1), torch.ones(4))
    assert torch.allclose(stats[:, -3:], image.detach().flatten(2).amax(-1))
    darker = exposure_statistics(image.detach() * 0.5)
    # Half the exposure: quantiles and maxima halve, the histogram moves to the lower half.
    assert torch.allclose(darker[:, :len(EXPOSURE_QUANTILES)], quantiles * 0.5, atol=1e-5)
    assert torch.allclose(darker[:, -3:], stats[:, -3:] * 0.5, atol=1e-6)
    assert float(darker[:, len(EXPOSURE_QUANTILES) + 8:len(EXPOSURE_QUANTILES) + 16].sum()) == 0.0


def test_absent_the_decoder_keeps_its_shapes_on_the_head_and_context_widen():
    p28, p30 = load_config("configs/kaggle_steady.yaml"), load_config("configs/kaggle_exposure.yaml")
    old, new = build_decoders(p28).image, build_decoders(p30).image
    width = p28["phase2"]["split_color_width"]
    assert old.color.global_tone.head[0].in_features == 2 * width + 6
    assert old.light.context.in_features == p28["phase2"]["split_light_width"]
    assert new.color.global_tone.head[0].in_features == 2 * width + 6 + EXPOSURE_FEATURES
    assert new.light.context.in_features == p30["phase2"]["split_light_width"] + EXPOSURE_FEATURES
    changed = {key for key, value in old.state_dict().items() if new.state_dict()[key].shape != value.shape}
    assert changed == {"color.global_tone.head.0.weight", "light.context.weight"}


def test_both_start_as_the_identity_and_use_the_statistics():
    image, latent = _frames(), torch.randn(4, 16, 4, 4)
    tone = GlobalToneColor(16, 8, exposure_stats=True)
    exponent, matrix, bias = tone.parameters_for(latent, image)
    assert torch.allclose(exponent, torch.ones(4)) and torch.allclose(matrix, torch.eye(3).expand(4, 3, 3))
    assert torch.allclose(bias, torch.zeros(4, 3))
    light = LightBranch(16, 8, scale=4, levels=2, exposure_stats=True)
    assert torch.allclose(light(latent, image), image.clamp(0, 1), atol=1e-5)
    # Wake the heads up: the statistics' columns change the answer.
    torch.manual_seed(0)
    for module in (tone.head[-1], light.tail):
        torch.nn.init.normal_(module.weight, std=0.1)
    raw = tone.head(torch.cat((tone.features(image).mean(dim=(-2, -1)), F.relu(tone.latent(latent)).mean(dim=(-2, -1)),
                               image.mean(dim=(-2, -1)), image.flatten(2).std(dim=-1), exposure_statistics(image)), 1))
    before, after = tone.parameters_for(latent, image)[0], None
    with torch.no_grad():
        tone.head[0].weight[:, -EXPOSURE_FEATURES:] = 0.0
    after = tone.parameters_for(latent, image)[0]
    assert raw.shape == (4, 13) and not torch.allclose(before, after)
    out = light(latent, image)
    with torch.no_grad():
        light.context.weight[:, -EXPOSURE_FEATURES:] = 0.0
    assert not torch.allclose(out, light(latent, image))


def test_lowfreq_mse_sees_block_means_only():
    clean = _frames(size=32)
    # A pattern that averages to zero over every 8x8 block: detail, invisible to the term.
    checker = (torch.arange(8)[:, None] + torch.arange(8)[None, :]) % 2 * 2.0 - 1.0
    detail = 0.05 * checker.repeat(4, 4)
    assert float(lowfreq_mse(clean + detail, clean)) == pytest.approx(0.0, abs=1e-10)
    assert float(lowfreq_mse(clean + 0.1, clean)) == pytest.approx(0.01, rel=1e-5)
    assert torch.allclose(lowfreq_mse(clean * 0.7, clean), F.mse_loss(F.avg_pool2d(clean * 0.7, 8), F.avg_pool2d(clean, 8)))


def test_phase2_trains_with_both_and_logs_the_term(tmp_path):
    torch.set_num_threads(1)
    root, manifest = tmp_path / "dataset", tmp_path / "manifest"
    _write_dataset(root)
    config = _ijepa_dict(max_successful_updates=2, batch_size=2)
    config["phase2"].update(max_successful_updates=2, batch_size=2, split_exposure_stats=True, lowfreq_mse_weight=10.0)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest)])
    run = tmp_path / "run"
    common = ["--config", str(path), "--manifest", str(manifest), "--output", str(run)]
    _run_until_done(["train-phase1", *common], run / "phase1/last.pt")
    _run_until_done(["train-phase2", *common, "--backbone-checkpoint", str(run / "phase1/last.pt")],
                    run / "phase2/last.pt")
    rows = [json.loads(line) for line in (run / "phase2/train.jsonl").read_text().splitlines()]
    logged = [row["image_lowfreq_mse"] for row in rows if "image_lowfreq_mse" in row]
    assert len(logged) == 2 and all(value >= 0 for value in logged)
    system = load_checkpoint(run / "phase2/last.pt")["system"]
    head = system["decoders.image.color.global_tone.head.0.weight"]
    assert head.shape[1] == 2 * config["phase2"].get("split_color_width", 32) + 6 + EXPOSURE_FEATURES


@pytest.mark.parametrize("change, message", [
    ({"split_exposure_stats": "yes"}, "split_exposure_stats"),
    ({"split_exposure_stats": True, "split_color_global": False, "split_light_branch": False}, "turn one of them on"),
    ({"lowfreq_mse_weight": -1.0}, "lowfreq_mse_weight"),
    ({"lowfreq_mse_weight": True}, "lowfreq_mse_weight"),
])
def test_bad_settings_fail(change, message):
    config = copy.deepcopy(load_config("configs/kaggle_steady.yaml"))
    config["phase2"].update(change)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_p30_is_p28_with_the_two_keys_and_keeps_its_phase1():
    p28, p30 = load_config("configs/kaggle_steady.yaml"), load_config("configs/kaggle_exposure.yaml")
    validate_config(p30)
    differ = {(section, key) for section in ("data", "model", "corruption", "phase1", "phase2", "monitor", "runtime")
              for key in {*p28.get(section, {}), *p30.get(section, {})}
              if p28.get(section, {}).get(key) != p30.get(section, {}).get(key)}
    assert differ == {("phase2", "split_exposure_stats"), ("phase2", "lowfreq_mse_weight"), ("runtime", "output_dir")}
    assert configuration_hash(p28, "phase1") == configuration_hash(p30, "phase1")       # reuses p28's phase 1
    assert configuration_hash(p28, "phase2") != configuration_hash(p30, "phase2")
