"""qjepa/models/uformer_flare.py: Flare7K++'s Uformer, ported for its released checkpoint, and the probe that uses it.

Pins: the port has the original's 308 parameter/buffer names and shapes, so Flare7K++'s ``net_g_last.pth`` loads
strictly (checked once against github.com/ykdai/Flare7K basicsr/archs/uformer_arch.py: same names, same shapes and the
same output to the last bit on a 512x512 input with shared random weights; the tiling matched test_large.py within its
own uint8 rounding, at most 3.6/255); the loader takes {'params_ema'}, {'params'} or a bare state dict; the tiling of
remove_flare covers every pixel once -- a model that hands its input back returns the frame unchanged, wide or tall;
tools/flare_pretrained_probe.py scores the input, the current flare stage and the Uformer on held-out FlareImage pairs.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from qjepa.config import load_config, serializable_config
from qjepa.models.uformer_flare import TILE, Uformer, load_flare7kpp, remove_flare
from qjepa.training.flare_stage import build_flare_layer
from test_flare_pairs import SHORT_SIDE, _write_flareimage


def test_the_port_has_the_released_checkpoints_names_and_shapes():
    state = Uformer().state_dict()
    assert len(state) == 308
    assert sum(v.numel() for k, v in state.items() if not k.endswith("relative_position_index")) == 20_473_888
    expected = {
        "input_proj.proj.0.weight": (32, 3, 3, 3),
        "encoderlayer_0.blocks.0.attn.relative_position_bias_table": (225, 1),
        "encoderlayer_0.blocks.1.attn.relative_position_index": (64, 64),
        "dowsample_3.conv.0.weight": (512, 256, 4, 4),
        "conv.blocks.1.attn.qkv.to_kv.weight": (1024, 512),
        "upsample_0.deconv.0.weight": (512, 256, 2, 2),
        "decoderlayer_3.blocks.1.mlp.fc2.weight": (64, 256),
        "output_proj.proj.0.weight": (6, 64, 3, 3),
    }
    for name, shape in expected.items():
        assert tuple(state[name].shape) == shape, name


@pytest.mark.parametrize("wrap", ["params_ema", "params", None])
def test_the_loader_takes_every_basicsr_layout_strictly_in_eval_mode(tmp_path, wrap):
    torch.manual_seed(0)
    source = Uformer()
    state = source.state_dict()
    path = tmp_path / "net_g_last.pth"
    torch.save({wrap: state} if wrap else state, path)
    model = load_flare7kpp(path)
    assert not model.training
    assert all(torch.equal(model.state_dict()[k], v) for k, v in state.items())
    broken = {k: v for k, v in state.items() if not k.startswith("output_proj")}
    torch.save({"params": broken}, path)
    with pytest.raises(RuntimeError):
        load_flare7kpp(path)


class _HandBack(nn.Module):
    """Stands in for the Uformer: flare-free = its input, flare = 0; counts its 512x512 tiles."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.tiles = 0

    def forward(self, x):
        assert x.shape[-2:] == (TILE, TILE)
        self.tiles += 1
        return torch.cat([x, torch.zeros_like(x)], dim=1)


@pytest.mark.parametrize("shape,tiles", [((40, 90), 4), ((90, 40), 4), ((64, 64), 1), ((48, 50), 2)])
def test_remove_flare_covers_every_pixel_once_wide_or_tall(shape, tiles):
    image = np.random.default_rng(1).random((*shape, 3)).astype(np.float32)
    model = _HandBack()
    out = remove_flare(model, image)
    assert out.shape == image.shape and model.tiles == tiles
    assert np.abs(out - image).max() < 1e-4


def test_the_probe_scores_input_flare_stage_and_uformer_on_held_out_pairs(tmp_path):
    from tools.flare_pretrained_probe import main
    torch.manual_seed(0)
    root = _write_flareimage(tmp_path / "flare_dataset")
    config = serializable_config(load_config("configs/kaggle_flarestage.yaml"))
    config["data"].update(flare_pairs_root=str(root), flare_pairs_short_side=SHORT_SIDE)
    stage = build_flare_layer(config["flare"])
    torch.save({"metadata": {"phase": "flare_stage"}, "config": config, "model": stage.state_dict()},
               tmp_path / "flare.pt")
    torch.save({"params": Uformer().state_dict()}, tmp_path / "uformer.pth")
    report = main(["--uformer", str(tmp_path / "uformer.pth"), "--flare-stage", str(tmp_path / "flare.pt"),
                   "--output", str(tmp_path / "probe"), "--max-pairs", "3", "--show", "1", "--device", "cpu"])
    assert len(report["rows"]) == 3 and report["split"] == "test"
    for row in report["rows"]:
        assert 0 < row["flare_fraction"] < 1
        for method in ("vao", "nhanh", "uformer"):
            assert np.isfinite(row[method]["ca"]) and np.isfinite(row[method]["loe"])
    assert "uformer_hon_nhanh_loe" in report["summary"]["tat ca"]
    assert (tmp_path / "probe/pair_00.png").is_file() and not (tmp_path / "probe/pair_01.png").exists()
    assert json.loads((tmp_path / "probe/report.json").read_text())["summary"]["tat ca"]["count"] == 3
