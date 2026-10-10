"""p34: phase2.split_chroma_detail -- the chroma finer than the colour branch's grid, at full resolution.

``compose`` took the frame's chroma from the colour branch alone, at 1/color_scale: a perfect model was capped
at 33.0 dB on clean TartanAir frames (color_scale 2) and the decoder did not start as the identity. On p33's
valid frames left with mild blur and grain, the input stood at 36.7 dB and the restored frame at 27.9 dB.

Pins: any RGB offset is exactly its luminance on three channels plus chroma_to_rgb of its chroma, and
chroma_to_rgb carries no luminance; on a frame with sharp colour edges the colour grid alone caps the frame and
adding the frame's own chroma detail gives it back exactly; with the key the decoder starts as the identity,
chroma included, without it the colour starts blurred; the branch's correction is zero at initialisation and
adds no luminance; absent, the decoder keeps p33's layers (its checkpoints load); bad settings fail; phase 2
trains with it and the correction moves; p34 is p33 plus the chroma keys, with p33's phase-1 hash.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch
import yaml

from qjepa.cli import main
from qjepa.config import CHROMA_KEYS, build_decoders, load_config, serializable_config, validate_config
from qjepa.models.color_edge import (chroma, chroma_to_rgb, color_base, compose, luminance, split_targets)
from qjepa.training.checkpoints import configuration_hash, load_checkpoint
from test_ijepa import _run_until_done
from test_kaggle_workflow import _write_dataset

CHROMA = {"split_chroma_detail": True, "split_chroma_width": 8, "split_chroma_blocks": 2}


def _edges(size=32, seed=0):
    """Blocks of saturated colour 3 px wide: colour edges finer than a /2 grid can hold."""
    generator = torch.Generator().manual_seed(seed)
    blocks = torch.rand(2, 3, size // 3 + 1, size // 3 + 1, generator=generator)
    return blocks.repeat_interleave(3, -1).repeat_interleave(3, -2)[..., :size, :size].contiguous()


def _psnr(a, b):
    return float(10 * torch.log10(1 / (a - b).square().mean().clamp_min(1e-12)))


def test_an_rgb_offset_is_its_luminance_plus_its_chroma():
    offset = torch.randn(4, 3, 8, 8)
    assert torch.allclose(luminance(offset).expand(-1, 3, -1, -1) + chroma_to_rgb(chroma(offset)), offset, atol=1e-6)
    cbcr = torch.randn(4, 2, 8, 8)
    assert torch.allclose(chroma(chroma_to_rgb(cbcr)), cbcr, atol=1e-6)
    assert luminance(chroma_to_rgb(cbcr)).abs().max() < 1e-6                 # chroma only: no light added


def test_the_colour_grid_caps_a_perfect_model_and_the_frames_chroma_detail_lifts_the_cap():
    frame = _edges()
    base, light, detail = split_targets(frame, 2, 8)
    capped = compose(base, light, detail)
    assert torch.allclose(luminance(capped), luminance(frame), atol=1e-6)    # the luminance was always exact
    assert _psnr(capped, frame) < 30                                          # the colour was not
    coarse = color_base(frame, 2)
    lifted = capped + chroma_to_rgb(chroma(frame) - chroma(coarse))
    assert torch.allclose(lifted, frame, atol=1e-5)


def _smoke(**phase2):
    config = serializable_config(load_config("configs/smoke.yaml"))
    config["phase2"].update(phase2)
    validate_config(config)
    return config


def test_the_decoder_starts_as_the_identity_only_with_the_chroma_detail():
    torch.manual_seed(0)
    frame = _edges()
    old, new = build_decoders(_smoke()).image.eval(), build_decoders(_smoke(**CHROMA)).image.eval()
    latent = torch.randn(2, _smoke()["model"]["embedding_dim"], 2, 2)
    with torch.no_grad():
        before, _ = old(latent, frame)
        after, parts = new(latent, frame)
    assert (before - frame).abs().max() > 0.1                                # colour blurred from update 0
    assert torch.allclose(after, frame, atol=1e-5)                           # the input, chroma included
    assert torch.allclose(parts["image_chroma_detail"], chroma(frame) - chroma(color_base(frame, 2)), atol=1e-6)
    assert new.chroma.tail.weight.abs().sum() == 0 and new.chroma.tail.bias.abs().sum() == 0
    # Absent, the decoder keeps p33's layers: its checkpoints load.
    assert old.chroma is None and set(old.state_dict()) == {key for key in new.state_dict()
                                                            if not key.startswith("chroma.")}


@pytest.mark.parametrize("change, message", [
    ({"split_chroma_detail": "yes"}, "split_chroma_detail must be"),
    ({"split_chroma_width": 2}, "split_chroma_width"),
    ({"split_chroma_blocks": 0}, "split_chroma_blocks"),
    ({"split_chroma_blocks": 1.5}, "split_chroma_blocks"),
])
def test_bad_settings_fail(change, message):
    config = serializable_config(load_config("configs/smoke.yaml"))
    config["phase2"].update(CHROMA, **change)
    with pytest.raises(ValueError, match=message):
        validate_config(config)
    missing = serializable_config(load_config("configs/smoke.yaml"))
    missing["phase2"].update(split_chroma_detail=True, split_chroma_width=8)
    with pytest.raises(ValueError, match="split_chroma_blocks"):
        validate_config(missing)
    other = serializable_config(load_config("configs/smoke.yaml"))
    other["phase2"].update(CHROMA, image_decoder="resnet_pixel")
    with pytest.raises(ValueError, match="split_color_edge"):
        validate_config(other)


def test_kaggle_chroma_is_p33_plus_the_chroma_keys_with_p33s_phase1_hash():
    halo = serializable_config(load_config("configs/kaggle_halo.yaml"))
    p34 = serializable_config(load_config("configs/kaggle_chroma.yaml"))
    for config in (halo, p34):
        config.pop("_config_path", None)
        config["runtime"].pop("output_dir")
    assert p34["phase2"].pop("split_chroma_detail") is True
    for key in CHROMA_KEYS:
        p34["phase2"].pop(key)
    assert p34 == halo
    full_halo, full_p34 = load_config("configs/kaggle_halo.yaml"), load_config("configs/kaggle_chroma.yaml")
    validate_config(full_p34)
    assert configuration_hash(full_p34, "phase1") == configuration_hash(full_halo, "phase1")   # reuse p33's phase 1
    assert configuration_hash(full_p34, "phase2") != configuration_hash(full_halo, "phase2")


def test_phase2_trains_with_it_and_the_correction_moves(tmp_path):
    torch.set_num_threads(1)
    root, manifest, run = tmp_path / "dataset", tmp_path / "manifest", tmp_path / "run"
    _write_dataset(root)
    config = serializable_config(load_config("configs/smoke.yaml"))
    config["phase2"].update(CHROMA, max_successful_updates=2)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest)])
    common = ["--config", str(path), "--manifest", str(manifest), "--output", str(run)]
    _run_until_done(["train-phase1", *common], run / "phase1/last.pt")
    _run_until_done(["train-phase2", *common, "--backbone-checkpoint", str(run / "phase1/last.pt")],
                    run / "phase2/last.pt")
    system = load_checkpoint(run / "phase2/last.pt")["system"]
    tail = [value for key, value in system.items() if key.endswith("chroma.tail.weight")]
    assert len(tail) == 1 and tail[0].abs().sum() > 0                         # the image losses reached it
    rows = [json.loads(line) for line in (run / "phase2/train.jsonl").read_text().splitlines()]
    assert sum("image_l1" in row for row in rows) == 2
    # tools/halo_probe.py: with the chroma detail, what the decoder returns before learning is the input itself.
    import subprocess
    import sys
    from test_halo_flare import _write_halo
    report = tmp_path / "probe.json"
    result = subprocess.run([sys.executable, "tools/halo_probe.py", "--checkpoint", str(run / "phase2/last.pt"),
                             "--manifest", str(manifest), "--samples", "4", "--batch", "2", "--device", "cpu",
                             "--halo-root", str(_write_halo(tmp_path / "halo")), "--output", str(report)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    for version in ("with", "without"):
        stage = json.loads(report.read_text())["stages"][version]
        assert stage["identity"] == pytest.approx(stage["input"])
