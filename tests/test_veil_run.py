"""p53_veil: a veil grid after the flare stage's Uformer (UformerFlareLayer's docstring, configs/kaggle_veil.yaml).

Pins:
- p53 is p52 plus flare.veil_grid / veil_only and a new flare-stage schedule; same phase-1 hash as p49, a new flare and
  phase-2 hash; phase 2 builds the layer with the grid, p52 without; bad keys are refused;
- the grid starts as the identity: the layer gives exactly the Uformer's answer;
- the grid learns to take off a broad glow wherever it lies in the frame (new frames, after a short training);
- train-flare with veil_only starts the grid from a stage that has none, trains the grid only (the Uformer's weights
  end bit for bit as they started), and phase 2 loads that layer, grid included, and never moves it.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from qjepa.cli import main
from qjepa.config import _restore_settings, load_config, serializable_config, validate_config
from qjepa.models.decoders import BilateralGridTone
from qjepa.models.uformer_flare import UformerFlareLayer
from qjepa.training.checkpoints import configuration_hash, load_checkpoint
from test_flare_stage import _train_flare
from test_ijepa import _run_until_done
from test_uformer_stage import _HandBack, _uformer_stage

P49 = load_config("configs/kaggle_glint.yaml")
P52 = load_config("configs/kaggle_detail.yaml")
P53 = load_config("configs/kaggle_veil.yaml")
SCHEDULE = {"batch_size": 8, "max_successful_updates": 2000, "learning_rate": 5e-4, "minimum_lr": 1e-5,
            "warmup_updates": 100, "kinds": {"composite": 0.3, "pair": 0.3, "plain": 0.15, "scene": 0.25}}


def test_p53_is_p52_with_the_veil_grid_and_reuses_p49_phase1():
    p52, p53 = (copy.deepcopy(serializable_config(config)) for config in (P52, P53))
    assert p53["runtime"].pop("output_dir").endswith("/p53_veil")
    p52["runtime"].pop("output_dir")
    assert (p53["flare"].pop("veil_grid"), p53["flare"].pop("veil_only")) == (16, True)
    assert {key: p53["flare"].pop(key) for key in SCHEDULE} == SCHEDULE
    for key in SCHEDULE:
        p52["flare"].pop(key)
    assert p53 == p52
    validate_config(P53)
    assert configuration_hash(P53, "phase1") == configuration_hash(P49, "phase1")
    assert configuration_hash(P53, "flare") != configuration_hash(P52, "flare")
    assert configuration_hash(P53, "phase2") != configuration_hash(P52, "phase2")
    assert _restore_settings(P53)["flare"]["veil_grid"] == 16
    assert "veil_grid" not in _restore_settings(P52)["flare"]


@pytest.mark.parametrize("change, message", [
    ({"veil_grid": 1}, "veil_grid must be an integer >= 2"),
    ({"veil_grid": True}, "veil_grid must be an integer >= 2"),
    ({"veil_only": "yes"}, "veil_only must be true or false"),
    ({"veil_grid": None}, "veil_only trains only the veil grid"),
    ({"uformer_init": "release"}, "veil_only keeps the Uformer"),
])
def test_bad_veil_keys_are_refused(change, message):
    config = copy.deepcopy(P53)
    config["flare"].update(change)
    if config["flare"]["veil_grid"] is None:
        del config["flare"]["veil_grid"]
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_the_veil_grid_needs_the_uformer():
    config = load_config("configs/kaggle_flarestage.yaml")
    config["flare"]["veil_grid"] = 8
    with pytest.raises(ValueError, match="veil_grid follows the Uformer"):
        validate_config(config)


class _TakesSome(_HandBack):
    """Stands in for the Uformer: takes 20% off the frame, as if that were the flare."""

    def forward(self, x):
        return super().forward(0.8 * x)


def test_the_grid_starts_as_the_identity():
    torch.manual_seed(0)
    image = torch.rand(2, 3, 32, 48)
    plain, veiled = UformerFlareLayer(4.0), UformerFlareLayer(4.0, veil_grid=4)
    plain.net = veiled.net = _TakesSome()
    with torch.no_grad():
        (a, flare_a), (b, flare_b) = plain(None, image), veiled(None, image)
    assert torch.equal(a, b) and torch.equal(flare_a, flare_b)
    assert flare_a.abs().mean() > 0.05


def _glow_frames(count, generator):
    """A smooth scene, and the same scene under a broad warm glow centred anywhere in the frame (added light)."""
    scene = F.interpolate(torch.rand(count, 3, 8, 8, generator=generator), size=64, mode="bilinear",
                          align_corners=False) * 0.6 + 0.1 * torch.rand(count, 3, 64, 64, generator=generator)
    centre = torch.rand(count, 2, generator=generator) * 64
    rows = torch.arange(64.0)
    distance = ((rows[None, :, None] - centre[:, 0, None, None]) ** 2
                + (rows[None, None, :] - centre[:, 1, None, None]) ** 2)
    glow = torch.exp(-distance / (2 * 16.0 ** 2))[:, None] * torch.tensor([0.5, 0.4, 0.2])[None, :, None, None]
    return (scene + glow).clamp(0, 1), scene.clamp(0, 1)


def test_the_grid_learns_to_take_off_a_glow_wherever_it_lies():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    grid = BilateralGridTone(0, size=8)
    optimizer = torch.optim.Adam(grid.parameters(), lr=2e-3)
    stream = torch.Generator().manual_seed(0)
    for _ in range(150):
        veiled, scene = _glow_frames(8, stream)
        loss = F.l1_loss(grid(None, veiled), scene)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    veiled, scene = _glow_frames(16, torch.Generator().manual_seed(1))         # frames it never saw
    with torch.no_grad():
        left = (grid(None, veiled) - scene).abs().mean() / (veiled - scene).abs().mean()
    assert left < 0.4


def test_train_flare_trains_only_the_grid_and_phase2_loads_it_frozen(tmp_path):
    torch.set_num_threads(2)
    config, manifest_dir, _ = _uformer_stage(tmp_path)
    path, previous = _train_flare(config, tmp_path, "previous")              # a stage without the grid (p51's kind)
    main(["train-flare", "--config", str(path), "--manifest", str(manifest_dir), "--output", str(previous)])
    start = load_checkpoint(previous / "flare/best.pt")
    assert not any(key.startswith("veil.") for key in start["model"])

    config["flare"].update(veil_grid=4, veil_only=True, uformer_init="stage", learning_rate=1e-2)
    config["data"]["uformer_checkpoint"] = str(previous / "flare/best.pt")
    config["phase2"].update(split_flare_layer=True, split_flare_widths=list(config["flare"]["widths"]),
                            split_flare_blocks=config["flare"]["blocks"], split_flare_weight=0.0,
                            split_flare_latent=False, split_flare_pretrained=True, split_flare_freeze=True,
                            split_flare_model="uformer_flare7kpp", split_flare_scale=4.0,
                            max_successful_updates=2, backbone_finetune_after_updates=1)
    validate_config(config)
    path, run = _train_flare(config, tmp_path, "run")
    common = ["--config", str(path), "--manifest", str(manifest_dir), "--output", str(run)]
    main(["train-flare", *common])
    last = load_checkpoint(run / "flare/last.pt")
    for key, value in start["model"].items():
        assert torch.equal(last["model"][key], value), key                     # the Uformer did not move
    assert last["model"]["veil.out.weight"].abs().sum() > 0                    # the grid did
    assert last["optimizer"]["state"] and len(last["optimizer"]["param_groups"][0]["params"]) == \
        sum(1 for key in last["model"] if key.startswith("veil."))

    _run_until_done(["train-phase1", *common], run / "phase1/last.pt")
    _run_until_done(["train-phase2", *common, "--backbone-checkpoint", str(run / "phase1/last.pt")],
                    run / "phase2/last.pt")
    best, final = load_checkpoint(run / "flare/best.pt"), load_checkpoint(run / "phase2/last.pt")
    assert any(key.startswith("veil.") for key in best["model"])
    for key, value in best["model"].items():
        assert torch.equal(final["system"][f"decoders.image.flare.{key}"], value), key
    assert final["metadata"]["frozen_flare_hash"] == best["metadata"]["layer_hash"]
