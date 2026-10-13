"""p41: phase2.split_flare_layer, a full-resolution flare layer the split decoder subtracts before anything else.

On p40's checkpoint (tools/flare_image_test.py, no training) the decoder took ~1/4 of the flare off held-out FlareImage
pairs and ~1/3 off the very pairs it trained on, and outside the flare PSNR fell on 15 of 16 training pairs (median
-4.8 dB): it changed the whole frame whenever it saw a flare. The flare layer has its own target -- the pair's flared
photo minus its clean photo, zero on every TartanAir frame -- and a pair's tone label no longer says "flare".

Pins: absent, the decoder has no new layers and every dataset item is as before; on, the decoder starts as p40's
(layer 0, output unchanged) and every stage reads the frame minus the layer; the flare loss reaches the layer; a pair
item's target is its flared crop minus its clean crop, turned and mirrored with them, a TartanAir item's is zero and
its every other key unchanged; a "clear" pair's tone label is 0 with the layer and 1 without; phase 2 trains through
the CLI on a phase-1 checkpoint trained without the key and logs the flare loss; p41 is p40 plus the four phase-2 keys,
with p40's phase-1 hash; bad settings fail.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from qjepa.cli import _train_dataset
from qjepa.config import build_decoders, load_config, serializable_config, validate_config
from qjepa.data import collate_paired
from qjepa.training.checkpoints import configuration_hash, load_checkpoint
from test_bilateral import GRID
from test_flare_pairs import _smoke
from test_ijepa import _run_until_done
from test_tone_gate import GATE

LAYER = {"split_flare_layer": True, "split_flare_widths": [8, 8, 8], "split_flare_blocks": 1,
         "split_flare_weight": 3.0}


def _config(tmp_path, **phase2):
    config, manifest, _, manifest_dir = _smoke(tmp_path, phase1_batch=4, phase2_batch=4)
    config["phase2"].update(GRID, **GATE, **phase2)
    validate_config(config)
    return config, manifest, manifest_dir


def test_absent_no_layers_and_on_the_decoder_starts_as_p40s_and_the_loss_reaches_the_layer(tmp_path):
    config, _, _ = _config(tmp_path)
    torch.manual_seed(0)
    old = build_decoders(config).image
    assert old.flare is None and not any(key.startswith("flare.") for key in old.state_dict())
    layered = copy.deepcopy(config)
    layered["phase2"].update(LAYER)
    validate_config(layered)
    new = build_decoders(layered).image
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected and missing and all(key.startswith("flare.") for key in missing)
    generator = torch.Generator().manual_seed(1)
    image = torch.rand(2, 3, 32, 32, generator=generator)
    latent = torch.randn(2, config["model"]["embedding_dim"], 8, 8, generator=generator)
    with torch.no_grad():
        before, _ = old(latent, image)
        after, parts = new(latent, image)
    assert torch.equal(parts["image_flare"], torch.zeros_like(image))       # zero-initialised: no flare taken off
    assert torch.allclose(after, before, atol=1e-6)                           # p40's decoder, as it was
    # Every stage reads the frame without the layer: a layer of 0.3 on a patch is gone from what they read.
    with torch.no_grad():
        new.flare.net.tail.bias.fill_(0.1)
        _, parts = new(latent, image)
    assert torch.allclose(parts["image_deflared"], image - parts["image_flare"])
    target = torch.zeros_like(image)
    loss = F.l1_loss(new(latent, image)[1]["image_flare"], target)
    loss.backward()
    assert new.flare.net.tail.weight.grad.abs().sum() > 0


def test_items_carry_the_pairs_flare_and_tartanair_stays_as_it_was(tmp_path):
    config, manifest, _ = _config(tmp_path)
    plain = _train_dataset(config, manifest, "phase2")
    layered_config = copy.deepcopy(config)
    layered_config["phase2"].update(LAYER)
    validate_config(layered_config)
    layered = _train_dataset(layered_config, manifest, "phase2")
    assert plain.flare_layer is False and layered.flare_layer is True
    first = len(plain.samples)
    for index in range(4):                                                    # TartanAir: zero flare, all else equal
        a, b = plain[index], layered[index]
        assert "image_flare_target" not in a and torch.equal(b.pop("image_flare_target"), torch.zeros(3, 32, 32))
        assert torch.equal(a["image_noisy"], b["image_noisy"]) and torch.equal(a["image_clean"], b["image_clean"])
        assert torch.equal(a["image_tone_label"], b["image_tone_label"])
    seen = 0
    for draw in range(24):
        a, b = plain[first + draw], layered[first + draw]
        assert torch.equal(a["image_noisy"], b["image_noisy"]) and torch.equal(a["image_clean"], b["image_clean"])
        flare = b["image_flare_target"]
        pair = b["corruption"]["image"]["flare_pair"]
        gt, flared = layered.flare_bank.load(next(p for p in layered.flare_bank.pairs("train") if p.name == pair["name"]))
        # The test pairs: flared = gt + a bright square. The target is that square, wherever the crop, turn and
        # mirror put it: never negative, and the clean crop plus it is a crop of the flared photo.
        assert float(flare.min()) >= -1e-6
        assert float(flare.max()) <= float((flared.astype(np.float32) - gt).max()) / 255 + 1e-6
        seen += int(float(flare.max()) > 0.1)
        assert pair["clear"] is True
        assert float(a["image_tone_label"]) == 1.0 and float(b["image_tone_label"]) == 0.0   # the flare is the layer's
    assert seen > 0
    batch = collate_paired([layered[0], layered[first]])
    assert batch["image_flare_target"].shape == (2, 3, 32, 32)
    # Phase 1 never reads it.
    assert _train_dataset(layered_config, manifest, "phase1").flare_layer is False


def test_the_target_turns_and_mirrors_with_the_pair(tmp_path):
    config, manifest, _ = _config(tmp_path, **LAYER)
    config["data"]["flare_pairs_augment"] = "dihedral"
    validate_config(config)
    dataset = _train_dataset(config, manifest, "phase2")
    first = len(dataset.samples)
    for draw in range(16):
        item = dataset[first + draw]
        pair = item["corruption"]["image"]["flare_pair"]
        gt, flared = dataset.flare_bank.load(next(p for p in dataset.flare_bank.pairs("train") if p.name == pair["name"]))
        difference = flared.astype(np.float32) / 255 - gt.astype(np.float32) / 255
        clean = item["image_clean"].permute(1, 2, 0).numpy()
        target = item["image_flare_target"].permute(1, 2, 0).numpy()
        # Find the crop of gt the item shows (any of the 8 views and the phase mirror), then check the same view of
        # the difference is the target.
        found = False
        for turns in range(4):
            for flip in (False, True):
                for mirror in (False, True):
                    view = lambda x: (lambda y: y[:, ::-1] if mirror else y)(
                        (lambda y: y[:, ::-1] if flip else y)(np.rot90(x, turns)))
                    g, d = view(gt.astype(np.float32) / 255), view(difference)
                    for top in range(g.shape[0] - 31):
                        for left in range(g.shape[1] - 31):
                            if np.array_equal(g[top:top + 32, left:left + 32], clean):
                                found = found or np.allclose(d[top:top + 32, left:left + 32], target, atol=1e-6)
        assert found, draw


@pytest.mark.parametrize("change, message", [
    ({"split_flare_layer": "yes"}, "true or false"),
    ({"split_flare_widths": [8]}, "at least two"),
    ({"split_flare_blocks": 0}, "split_flare_blocks"),
    ({"split_flare_weight": -1}, "split_flare_weight"),
])
def test_bad_settings_fail(tmp_path, change, message):
    config, _, _ = _config(tmp_path)
    config["phase2"].update(LAYER, **change)
    with pytest.raises(ValueError, match=message):
        validate_config(config)
    missing = copy.deepcopy(config)
    missing["phase2"].update(LAYER)
    del missing["phase2"]["split_flare_weight"]
    with pytest.raises(ValueError, match="missing: split_flare_weight"):
        validate_config(missing)
    without_pairs = copy.deepcopy(config)
    without_pairs["phase2"].update(LAYER)
    without_pairs["data"]["flare_pairs_fraction"] = 0.0
    with pytest.raises(ValueError, match="flare_pairs_fraction"):
        validate_config(without_pairs)


def test_phase2_trains_the_layer_on_a_phase1_trained_without_the_key(tmp_path):
    torch.set_num_threads(1)
    config, _, manifest_dir = _config(tmp_path)
    run = tmp_path / "run"
    before, after = tmp_path / "p40.yaml", tmp_path / "p41.yaml"
    before.write_text(yaml.safe_dump(config))
    _run_until_done(["train-phase1", "--config", str(before), "--manifest", str(manifest_dir), "--output", str(run)],
                    run / "phase1/last.pt")
    config["phase2"].update(LAYER)
    after.write_text(yaml.safe_dump(config))
    _run_until_done(["train-phase2", "--config", str(after), "--manifest", str(manifest_dir), "--output", str(run),
                     "--backbone-checkpoint", str(run / "phase1/last.pt")], run / "phase2/last.pt")
    final = load_checkpoint(run / "phase2/last.pt")
    assert final["successful_updates"] == 3
    assert any(key.startswith("decoders.image.flare.") for key in final["system"])
    rows = [json.loads(line) for line in (run / "phase2/train.jsonl").read_text().splitlines()]
    flare = [row["image_flare_l1"] for row in rows if "image_flare_l1" in row]
    assert len(flare) == 3 and all(np.isfinite(flare)) and flare[0] > 0


def test_kaggle_flarelayer_is_p40_plus_the_four_keys_with_p40s_phase1_hash():
    p40, p41 = load_config("configs/kaggle_flarefocus.yaml"), load_config("configs/kaggle_flarelayer.yaml")
    validate_config(p41)
    assert configuration_hash(p41, "phase1") == configuration_hash(p40, "phase1")
    assert configuration_hash(p41, "phase2") != configuration_hash(p40, "phase2")
    # p40 hashes as it did when it was trained (notebook f86e386 on cd903d5).
    assert configuration_hash(p40, "phase1").startswith("d4d03229be4b")
    assert configuration_hash(p40, "phase2").startswith("55cab3f2f5ce")
    a, b = (serializable_config(copy.deepcopy(config)) for config in (p40, p41))
    assert b["runtime"].pop("output_dir").endswith("outputs/p41_flarelayer")
    a["runtime"].pop("output_dir")
    assert {key: b["phase2"].pop(key) for key in ("split_flare_layer", "split_flare_widths", "split_flare_blocks",
                                                   "split_flare_weight")} == {
        "split_flare_layer": True, "split_flare_widths": [16, 32, 48, 64, 96], "split_flare_blocks": 1,
        "split_flare_weight": 3.0}
    assert a == b
