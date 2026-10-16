"""p44_uformer: Flare7K++'s Uformer as the flare layer -- fine-tuned by the flare stage from its released weights, the
best checkpoint on held-out pairs kept, frozen in phase 2 -- and phase 2's image loss balanced across kinds of frame.

Pins: UformerFlareLayer has p42's layer contract (frame without the flare, the flare), takes any frame size (upsampled,
padded to multiples of 128, square tiles) and hands a frame back unchanged when its Uformer does; checkpointing changes
neither outputs nor gradients; kind_weights gives every kind of a microbatch the same total weight with mean 1, and the
hooks scale exactly the image gradients of each frame; train-flare starts from the checkpoint in data.uformer_checkpoint
(its sha256 recorded), validates at update 0 and keeps flare/best.pt, the best of every validation, update 0 included;
phase 2 builds the Uformer layer, loads best.pt and never moves it; p44 is p43 plus its keys, with p43's phase-1 hash,
p43's hashes unchanged and the checkpoint path out of every hash; bad settings fail.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch
from torch import nn

from qjepa.cli import main
from qjepa.config import load_config, serializable_config, validate_config
from qjepa.models.uformer_flare import Uformer, UformerFlareLayer, file_sha256
from qjepa.training.checkpoints import configuration_hash, load_checkpoint
from qjepa.training.phase2 import _weight_frames, kind_weights
from test_flare_stage import _stage, _train_flare
from test_ijepa import _run_until_done

P43 = load_config("configs/kaggle_brightgt.yaml")
P44 = load_config("configs/kaggle_uformer.yaml")


class _HandBack(nn.Module):
    """Stands in for the Uformer: flare-free = its input. Square inputs only, as the real one."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, x):
        assert x.shape[-1] == x.shape[-2] and x.shape[-1] % 128 == 0, x.shape
        return torch.cat([x * self.weight, torch.zeros_like(x)], dim=1)


@pytest.mark.parametrize("size,scale", [((32, 32), 4.0), ((32, 48), 4.0), ((40, 24), 4.0), ((256, 256), 2.0)])
def test_the_layer_hands_any_frame_back_when_its_uformer_does(size, scale):
    layer = UformerFlareLayer(scale)
    layer.net = _HandBack()
    image = torch.rand(2, 3, *size)
    clean, flare = layer(None, image)
    assert clean.shape == image.shape and flare.shape == image.shape
    assert flare.abs().max() < 1e-5 and (clean - image).abs().max() < 1e-5


def test_the_layer_runs_in_fp32_under_autocast_and_passes_gradients():
    torch.manual_seed(0)
    layer = UformerFlareLayer(4.0)
    image = torch.rand(1, 3, 32, 32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _, flare = layer(None, image)
    assert flare.dtype == torch.float32
    flare.abs().mean().backward()
    assert all(p.grad is not None for p in layer.net.parameters() if p.requires_grad)


def test_checkpointing_changes_neither_outputs_nor_gradients():
    torch.manual_seed(0)
    plain, saved = Uformer(), Uformer()
    saved.load_state_dict(plain.state_dict())
    saved.set_checkpointing(True)
    x = torch.rand(1, 3, 128, 128)
    outputs = []
    for model in (plain, saved):
        out = model(x)
        out.square().mean().backward()
        outputs.append(out.detach())
    assert torch.allclose(outputs[0], outputs[1], atol=1e-6)
    for (name, a), b in zip(plain.named_parameters(), saved.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-6, rtol=1e-4), name


def test_kind_weights_give_every_kind_the_same_share_and_the_hooks_scale_image_gradients():
    codes = torch.tensor([4, 4, 0, 2])
    weights = kind_weights(codes, 5)
    assert weights.sum().item() == pytest.approx(4.0)
    assert weights.tolist() == pytest.approx([2 / 3, 2 / 3, 4 / 3, 4 / 3])
    assert kind_weights(torch.tensor([1, 1, 1]), 5).tolist() == pytest.approx([1.0, 1.0, 1.0])
    source = torch.ones(4, 3, requires_grad=True)
    imu_source = torch.ones(4, 2, requires_grad=True)
    restored = {"image": source * 2.0, "imu_normalized": imu_source * 2.0}
    _weight_frames(restored, weights)
    (restored["image"].sum() + restored["imu_normalized"].sum()).backward()
    assert source.grad[:, 0].tolist() == pytest.approx((2.0 * weights).tolist())
    assert imu_source.grad.unique().tolist() == [2.0]                     # IMU untouched


def _uformer_stage(tmp_path):
    config, manifest, manifest_dir = _stage(tmp_path, model="uformer_flare7kpp", uformer_scale=4.0, select_best=True,
                                            batch_size=2, max_successful_updates=4, validation_every_updates=2,
                                            validation_composites=2, learning_rate=1e-4)
    torch.manual_seed(0)
    weights = tmp_path / "net_g_last.pth"
    torch.save({"params": Uformer().state_dict()}, weights)
    config["data"]["uformer_checkpoint"] = str(weights)
    validate_config(config)
    return config, manifest_dir, weights


def test_train_flare_fine_tunes_from_the_checkpoint_and_keeps_the_best_validation(tmp_path):
    torch.set_num_threads(2)
    config, manifest_dir, weights = _uformer_stage(tmp_path)
    path, run = _train_flare(config, tmp_path, "run")
    main(["train-flare", "--config", str(path), "--manifest", str(manifest_dir), "--output", str(run)])
    last, best = load_checkpoint(run / "flare/last.pt"), load_checkpoint(run / "flare/best.pt")
    assert last["successful_updates"] == 4 and last["metadata"]["pretrained_sha256"] == file_sha256(weights)
    rows = [json.loads(line) for line in (run / "flare/train.jsonl").read_text().splitlines()]
    checks = {row["successful_updates"]: row["validation"] for row in rows if "validation" in row}
    assert sorted(checks) == [0, 2, 4]
    from qjepa.training.flare_stage import selection_score
    scores = {update: selection_score(report) for update, report in checks.items()}
    chosen = max(scores, key=lambda update: (scores[update], -update))           # the first of equal bests
    assert best["metadata"]["selected_update"] == chosen
    assert best["metadata"]["selection_score"] == pytest.approx(scores[chosen])
    assert best["metadata"]["configuration_hash"] == configuration_hash(config, "flare")
    if chosen == 0:                                                               # the released weights themselves
        released = torch.load(weights, weights_only=True)["params"]
        assert all(torch.equal(best["model"][f"net.{k}"], v) for k, v in released.items())
    missing = copy.deepcopy(config)
    missing["data"]["uformer_checkpoint"] = str(tmp_path / "absent.pth")
    path, other = _train_flare(missing, tmp_path, "other")
    with pytest.raises(SystemExit):
        main(["train-flare", "--config", str(path), "--manifest", str(manifest_dir), "--output", str(other)])


def test_phase2_builds_the_uformer_loads_best_pt_and_never_moves_it(tmp_path):
    torch.set_num_threads(2)
    config, manifest_dir, _ = _uformer_stage(tmp_path)
    config["phase2"].update(split_flare_layer=True, split_flare_widths=list(config["flare"]["widths"]),
                            split_flare_blocks=config["flare"]["blocks"], split_flare_weight=0.0,
                            split_flare_latent=False, split_flare_pretrained=True, split_flare_freeze=True,
                            split_flare_model="uformer_flare7kpp", split_flare_scale=4.0, kind_balance=True,
                            max_successful_updates=2, backbone_finetune_after_updates=1)
    validate_config(config)
    path, run = _train_flare(config, tmp_path, "run")
    common = ["--config", str(path), "--manifest", str(manifest_dir), "--output", str(run)]
    _run_until_done(["train-phase1", *common], run / "phase1/last.pt")
    main(["train-flare", *common])
    best = load_checkpoint(run / "flare/best.pt")
    _run_until_done(["train-phase2", *common, "--backbone-checkpoint", str(run / "phase1/last.pt")],
                    run / "phase2/last.pt")
    final = load_checkpoint(run / "phase2/last.pt")
    assert final["successful_updates"] == 2
    for key, value in best["model"].items():
        assert torch.equal(final["system"][f"decoders.image.flare.{key}"], value), key
    assert final["metadata"]["frozen_flare_hash"] == best["metadata"]["layer_hash"]


def test_kaggle_uformer_is_p43_plus_its_keys_with_p43s_phase1_hash():
    validate_config(P44)
    assert configuration_hash(P44, "phase1") == configuration_hash(P43, "phase1")
    assert configuration_hash(P43, "phase2").startswith("9194075590e1")        # p43 as trained
    assert configuration_hash(P43, "flare").startswith("116e1b937920")
    a, b = (serializable_config(copy.deepcopy(config)) for config in (P43, P44))
    assert b["runtime"].pop("output_dir").endswith("outputs/p44_uformer")
    a["runtime"].pop("output_dir")
    for key, value in {"split_flare_model": "uformer_flare7kpp", "split_flare_scale": 2.0, "kind_balance": True}.items():
        assert b["phase2"].pop(key) == value
    for key, value in {"model": "uformer_flare7kpp", "uformer_scale": 2.0, "select_best": True}.items():
        assert b["flare"].pop(key) == value
    for key in ("batch_size", "max_successful_updates", "learning_rate", "minimum_lr", "warmup_updates",
                "validation_every_updates", "checkpoint_every_updates"):
        a["flare"].pop(key), b["flare"].pop(key)
    assert a == b
    moved = copy.deepcopy(P44)
    moved["data"]["uformer_checkpoint"] = "/elsewhere/net_g_last.pth"
    for phase in ("phase1", "phase2", "flare"):
        assert configuration_hash(moved, phase) == configuration_hash(P44, phase)


@pytest.mark.parametrize("change, message", [
    (lambda c: c["flare"].update(model="uformer"), "flare.model"),
    (lambda c: c["flare"].update(uformer_scale=8), "uformer_scale"),
    (lambda c: (c["flare"].pop("model"), c["phase2"].pop("split_flare_model")), "upsampling: set"),
    (lambda c: c["phase2"].update(split_flare_model="unet"), "must equal flare.model"),
    (lambda c: c["phase2"].update(split_flare_scale=1.0), "must equal flare.model"),
    (lambda c: c["flare"].update(select_best="yes"), "select_best"),
    (lambda c: c["phase2"].update(kind_balance=1), "kind_balance"),
])
def test_bad_settings_fail(change, message):
    config = copy.deepcopy(P44)
    change(config)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_the_checkpoint_is_found_in_input_or_its_zips_and_downloaded_otherwise(tmp_path, monkeypatch):
    import sys
    import types
    import zipfile
    from qjepa.models.uformer_flare import download_flare7kpp, find_flare7kpp
    root, cache = tmp_path / "input", tmp_path / "cache"
    shard = root / "datasets/me/tartanairshard0"
    (shard / "Env/flare7kpp").mkdir(parents=True)
    (shard / "Env/flare7kpp/net_g_last.pth").write_bytes(b"shard")             # inside a shard: never walked
    assert find_flare7kpp(root, cache, skip=[shard]) is None
    archive = root / "datasets/me/weights"
    archive.mkdir(parents=True)
    with zipfile.ZipFile(archive / "ckpt.zip", "w") as bundle:
        bundle.writestr("flare7k/net_g_last.pth", b"old")
        bundle.writestr("flare7kpp/net_g_last.pth", b"new")
    found = find_flare7kpp(root, cache, skip=[shard])
    assert found == cache / "net_g_last.pth" and found.read_bytes() == b"new"       # the Flare7K++ one, extracted
    assert find_flare7kpp(root, cache, skip=[shard]) == found                       # cached
    (root / "datasets/me/plain/flare7kpp").mkdir(parents=True)
    (root / "datasets/me/plain/flare7kpp/net_g_last.pth").write_bytes(b"file")
    found.unlink()
    assert find_flare7kpp(root, cache, skip=[shard]).read_bytes() == b"file"

    calls = []

    def fake(url, output, quiet=False, refuse=False):
        calls.append(url)
        if "QUOTA" in url:
            raise RuntimeError("Too many users have viewed or downloaded this file recently.")
        with zipfile.ZipFile(output, "w") as bundle:
            bundle.writestr("flare7kpp/net_g_last.pth", b"x" * 2_000_000)

    monkeypatch.setitem(sys.modules, "gdown", types.SimpleNamespace(download=fake))
    assert download_flare7kpp(tmp_path / "quota", "QUOTA") is None
    got = download_flare7kpp(tmp_path / "copy", "MYCOPY")
    assert got.stat().st_size == 2_000_000 and "id=MYCOPY" in calls[-1]
    assert sorted(p.name for p in got.parent.iterdir()) == ["net_g_last.pth"]
    download_flare7kpp(tmp_path / "official")
    assert calls[-1].endswith("id=17AX9BJ-GS0in9Ey7vw3BVPISm67Rpzho")
