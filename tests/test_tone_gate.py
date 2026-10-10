"""p35: phase2.split_tone_gate -- how much of the tone stage each frame gets, learnt from the corruption.

On p33's flare-free valid frames (mild blur and grain, light untouched) the input stood at 36.71 dB and the tone
stage's output at 30.20 dB: it relit frames that needed none. Pins: corruptions.image.frame_kind says "plain" for a
frame whose light was untouched (clean, blur-only, grain-only, a cleared environment) and "relit" / "flare" for the
rest; the training dataset carries that as image_tone_label only when phase 2 asks; the gate sends a plain frame
through the tone stage untouched when it says so and the whole tone stage when it says so, the decoder starts as the
identity, and only the gate's own cross-entropy trains it (the image losses reach the tone stage, not the gate);
absent, the decoder keeps p34's layers; bad settings fail; phase 2 trains with it and logs the gate's cross-entropy
and accuracy; validation reports PSNR in and out by kind of frame; p35 is p34 plus the gate keys, with p33's
phase-1 hash.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
import yaml

from qjepa.cli import _dataset, kind_summary, main
from qjepa.config import TONE_GATE_KEYS, build_decoders, load_config, serializable_config, validate_config
from qjepa.corruptions.image import FRAME_KINDS, frame_kind
from qjepa.training.checkpoints import configuration_hash, load_checkpoint
from test_bilateral import GRID
from test_ijepa import _ijepa_dict, _run_until_done
from test_kaggle_workflow import _write_dataset

GATE = {"split_tone_gate": True, "split_tone_gate_width": 8, "split_tone_gate_weight": 0.5}


def _params(**change):
    params = {"mode": "full", "clean": False, "low_light": False, "sensor_noise": True}
    params.update(change)
    return params


def test_frame_kind_says_which_frames_had_their_light_changed():
    assert frame_kind(_params()) == "plain"                                   # blur and grain only
    assert frame_kind(_params(clean=True, low_light=True)) == "plain"
    assert frame_kind(_params(mode="blur_only")) == "plain" and frame_kind(_params(mode="sensor_noise_only")) == "plain"
    assert frame_kind(_params(low_light=True)) == "relit"
    assert frame_kind(_params(mode="blur_low_light")) == "relit"               # the named scenario keeps its stage
    for key in ("illumination", "light", "fog"):
        assert frame_kind(_params(**{key: True})) == "relit", key
    assert frame_kind(_params(halo=True)) == "flare" and frame_kind(_params(halo=True, low_light=True)) == "flare"


def _config(**phase2):
    config = _ijepa_dict(max_successful_updates=2, batch_size=2)
    config["phase2"].update(GRID, **phase2)
    validate_config(config)
    return config


def test_the_dataset_carries_the_label_only_when_asked(tmp_path):
    root, manifest = tmp_path / "dataset", tmp_path / "manifest"
    _write_dataset(root)
    config = _config(**GATE)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest)])
    from qjepa.data import read_manifest
    loaded = read_manifest(manifest)
    plain = _dataset(config, loaded, "train", fixed_realization=False)
    labelled = _dataset(config, loaded, "train", fixed_realization=False, tone_label=True)
    assert "image_tone_label" not in plain[0]
    for index in range(len(labelled)):
        item = labelled[index]
        assert item["image_tone_label"].item() == float(frame_kind(item["corruption"]["image"]) != "plain")


def test_the_gate_lets_a_frame_through_untouched_or_gives_it_the_whole_tone_stage():
    torch.manual_seed(0)
    config = _config(**GATE)
    decoder = build_decoders(config).image.eval()
    for module in (decoder.tone_grid,):                      # give the grid something to do
        for parameter in module.parameters():
            torch.nn.init.normal_(parameter, std=0.2)
    image, latent = torch.rand(2, 3, 64, 64), torch.randn(2, config["model"]["embedding_dim"], 4, 4)
    outputs = {}
    for name, bias in (("shut", -50.0), ("open", 50.0)):
        torch.nn.init.zeros_(decoder.tone_gate.head[-1].weight)
        torch.nn.init.constant_(decoder.tone_gate.head[-1].bias, bias)
        with torch.no_grad():
            outputs[name] = decoder(latent, image)[1]["image_light"]
    with torch.no_grad():
        graded = decoder.tone_grid(latent, image)
    assert torch.allclose(outputs["shut"], image, atol=1e-6)                 # the tone stage cannot touch it
    assert torch.allclose(outputs["open"], graded, atol=1e-6)                # the whole tone stage
    assert not torch.allclose(graded, image, atol=1e-3)


def test_the_decoder_starts_as_the_identity_and_only_the_cross_entropy_trains_the_gate():
    torch.manual_seed(0)
    config = _config(**GATE)
    decoder = build_decoders(config).image
    image, latent = torch.rand(2, 3, 64, 64), torch.randn(2, config["model"]["embedding_dim"], 4, 4)
    restored, parts = decoder(latent, image)
    assert torch.allclose(parts["image_light"], image, atol=1e-5)
    (restored - torch.rand_like(restored)).abs().mean().backward()            # an image loss alone
    gate = [p.grad for p in decoder.tone_gate.parameters()]
    assert all(g is None or g.abs().sum() == 0 for g in gate)                # never reaches the gate
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.tone_grid.parameters())
    decoder.zero_grad()
    restored, parts = decoder(latent, image)
    torch.nn.functional.binary_cross_entropy_with_logits(parts["image_tone_logit"], torch.ones(2, 1)).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.tone_gate.parameters())
    # Absent, the decoder keeps p34's layers: its checkpoints load.
    old = build_decoders(_config()).image
    assert old.tone_gate is None and set(old.state_dict()) == {key for key in decoder.state_dict()
                                                               if not key.startswith("tone_gate.")}


@pytest.mark.parametrize("change, message", [
    ({"split_tone_gate": "yes"}, "split_tone_gate must be"),
    ({"split_tone_gate_width": 2}, "split_tone_gate_width"),
    ({"split_tone_gate_weight": -1.0}, "split_tone_gate_weight"),
])
def test_bad_settings_fail(change, message):
    config = _ijepa_dict()
    config["phase2"].update(GRID, **GATE)
    config["phase2"].update(change)
    with pytest.raises(ValueError, match=message):
        validate_config(config)
    missing = _ijepa_dict()
    missing["phase2"].update(GRID, split_tone_gate=True, split_tone_gate_width=8)
    with pytest.raises(ValueError, match="split_tone_gate_weight"):
        validate_config(missing)
    gridless = _ijepa_dict()
    gridless["phase2"].update(GATE)
    with pytest.raises(ValueError, match="split_tone_grid"):
        validate_config(gridless)


def test_phase2_trains_with_it_and_validation_reports_psnr_by_kind_of_frame(tmp_path, capsys):
    torch.set_num_threads(1)
    root, manifest, run = tmp_path / "dataset", tmp_path / "manifest", tmp_path / "run"
    _write_dataset(root)
    config = _config(**GATE)
    config["phase2"].update(max_successful_updates=2, batch_size=2)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest)])
    common = ["--config", str(path), "--manifest", str(manifest), "--output", str(run)]
    _run_until_done(["train-phase1", *common], run / "phase1/last.pt")
    _run_until_done(["train-phase2", *common, "--backbone-checkpoint", str(run / "phase1/last.pt")],
                    run / "phase2/last.pt")
    rows = [json.loads(line) for line in (run / "phase2/train.jsonl").read_text().splitlines()]
    updates = [row for row in rows if "image_tone_bce" in row]
    assert len(updates) == 2 and all(row["image_tone_bce"] > 0 and 0 <= row["image_tone_accuracy"] <= 1
                                     for row in updates)
    validations = [row for row in rows if "validation_image_psnr_db" in row]
    kinds = [f"validation_{kind}_count" for kind in FRAME_KINDS if f"validation_{kind}_count" in validations[-1]]
    assert kinds and sum(validations[-1][key] for key in kinds) == validations[-1]["validation_image_count"]
    assert "theo loai anh:" in capsys.readouterr().out
    system = load_checkpoint(run / "phase2/last.pt")["system"]
    assert any(key.startswith("decoders.image.tone_gate.") for key in system)


def test_kind_summary_reads_the_evaluation_keys():
    metrics = {"v_plain_count": 4, "v_plain_baseline_image_psnr_db": 36.71, "v_plain_image_psnr_db": 27.85,
               "v_plain_worse_fraction": 1.0}
    assert kind_summary(metrics, "v_") == "tot 36.71->27.85 (n=4, 100% te hon vao)"
    assert kind_summary({}, "v_") == ""


def test_kaggle_tonegate_is_p34_plus_the_gate_keys_with_p33s_phase1_hash():
    p34 = serializable_config(load_config("configs/kaggle_chroma.yaml"))
    p35 = serializable_config(load_config("configs/kaggle_tonegate.yaml"))
    for config in (p34, p35):
        config.pop("_config_path", None)
        config["runtime"].pop("output_dir")
    assert p35["phase2"].pop("split_tone_gate") is True
    for key in TONE_GATE_KEYS:
        p35["phase2"].pop(key)
    assert p35 == p34
    full_halo, full_p35 = load_config("configs/kaggle_halo.yaml"), load_config("configs/kaggle_tonegate.yaml")
    validate_config(full_p35)
    assert configuration_hash(full_p35, "phase1") == configuration_hash(full_halo, "phase1")    # reuse p33's phase 1
    assert configuration_hash(full_p35, "phase2") != configuration_hash(load_config("configs/kaggle_chroma.yaml"),
                                                                        "phase2")
