"""p51_restormer: phase 2 restores with the released Restormer (qjepa/models/restormer.py), fine-tuned, not a NAFNet
trained from zero.

Pins:
- the port has the release's 494 tensor names and 26 126 644 parameters (checked once against github.com/swz30/
  Restormer basicsr/models/archs/restormer_arch.py: same output to 1e-5 on shared random weights, 1e-6 with
  motion_deblurring.pth; that file on a VisDrone frame with 5-15 px motion blur: +4 to +5 dB, a sharp frame 49 dB);
- the repo's two layers start at zero, so at init the output does not depend on the extra channels or on ZI, and
  it is the released model's answer plus ``base``; both layers get a gradient; any frame size comes back whole;
- load_restormer_into takes only the release's sha256 and a state that fills every released tensor;
- p51 is p50 plus the restore_net keys with the NAFNet's shape keys null; same phase-1 hash as p49, same flare
  hash as p50, its own phase-2 hash; earlier configs validate and hash as before; bad keys are refused;
- RestoreRGBDecoder built from p51's settings restores with RestormerNet, reading the frame without its flare and the
  gyro path through extra_embed.
"""

from __future__ import annotations

import copy

import pytest
import torch

from qjepa.config import _restore_settings, load_config, serializable_config, validate_config
from qjepa.models import restormer as restormer_module
from qjepa.models.decoders import RestoreRGBDecoder
from qjepa.models.restormer import EXTRA_LAYERS, RestormerNet, file_sha256, load_restormer_into
from qjepa.training.checkpoints import configuration_hash

P49 = load_config("configs/kaggle_glint.yaml")
P50 = load_config("configs/kaggle_noghost.yaml")
P51 = load_config("configs/kaggle_restormer.yaml")


def test_the_port_has_the_released_names_and_parameter_count():
    state = {key: value for key, value in RestormerNet(3, 128).state_dict().items() if not key.startswith(EXTRA_LAYERS)}
    assert len(state) == 494
    assert sum(value.numel() for value in state.values()) == 26_126_644
    expected = {
        "patch_embed.proj.weight": (48, 3, 3, 3),
        "encoder_level1.0.norm1.body.weight": (48,),
        "encoder_level1.0.attn.temperature": (1, 1, 1),
        "encoder_level1.0.ffn.project_in.weight": (254, 48, 1, 1),
        "latent.7.attn.qkv_dwconv.weight": (1152, 1, 3, 3),
        "down3_4.body.0.weight": (96, 192, 3, 3),
        "up2_1.body.0.weight": (192, 96, 3, 3),
        "refinement.3.ffn.project_out.weight": (96, 255, 1, 1),
        "output.weight": (3, 96, 3, 3),
    }
    for name, shape in expected.items():
        assert tuple(state[name].shape) == shape, name
    assert len(RestormerNet(3, 128).pretrained_parameters()) == 494


def test_at_init_the_repo_layers_add_nothing_and_both_learn():
    torch.manual_seed(0)
    net = RestormerNet(7, 16).eval()
    frame, extra = torch.rand(2, 3, 24, 40), torch.rand(2, 4, 24, 40)
    latent = torch.randn(2, 16, 2, 3)
    with torch.no_grad():
        plain = net(None, torch.cat((frame, torch.zeros_like(extra)), 1))
        full = net(latent, torch.cat((frame, extra), 1))
        based = net(latent, torch.cat((frame, extra), 1), base=torch.zeros_like(frame))
    assert torch.equal(plain, full)
    assert torch.allclose(full - based, frame, atol=1e-6)
    odd = torch.rand(1, 7, 21, 30)                     # not a multiple of 8: padded inside, cropped back
    with torch.no_grad():
        assert net(None, odd).shape == (1, 3, 21, 30)
    net.train()
    net(latent, torch.cat((frame, extra), 1)).square().mean().backward()
    for layer in (net.extra_embed, net.latent_in):
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_the_loader_takes_only_the_release(tmp_path, monkeypatch):
    source = RestormerNet(3, 8)
    released = {key: value for key, value in source.state_dict().items() if not key.startswith(EXTRA_LAYERS)}
    path = tmp_path / "motion_deblurring.pth"
    torch.save({"params": released}, path)
    monkeypatch.setitem(restormer_module.RESTORMER_WEIGHTS, "motion_deblurring", ("unused", file_sha256(path)))
    target = RestormerNet(6, 8)
    assert load_restormer_into(target, path, "motion_deblurring") == file_sha256(path)
    assert torch.equal(target.output.weight, source.output.weight)
    assert target.extra_embed.weight.abs().sum() == 0
    del released["output.weight"]
    torch.save({"params": released}, path)
    with pytest.raises(ValueError, match="sha256"):
        load_restormer_into(target, path, "motion_deblurring")
    monkeypatch.setitem(restormer_module.RESTORMER_WEIGHTS, "motion_deblurring", ("unused", file_sha256(path)))
    with pytest.raises(ValueError, match="not this Restormer"):
        load_restormer_into(target, path, "motion_deblurring")


def test_p51_is_p50_with_restormer_and_reuses_p49_phase1_and_p50_flare_stage():
    p50, p51 = copy.deepcopy(serializable_config(P50)), copy.deepcopy(serializable_config(P51))
    assert p51["runtime"]["output_dir"].endswith("/p51_restormer")
    assert p51["phase2"]["restore_net"] == "restormer"
    assert p51["phase2"]["restore_restormer_weights"] == "motion_deblurring"
    assert p51["phase2"]["restore_pretrained_lr_scale"] == 0.25
    for key in ("restore_net", "restore_restormer_weights", "restore_pretrained_lr_scale"):
        del p51["phase2"][key]
    for key in ("restore_widths", "restore_enc_blocks", "restore_middle_blocks", "restore_dec_blocks",
                "restore_fourier_levels", "restore_fourier_width"):
        assert p51["phase2"][key] is None
        p51["phase2"][key] = p50["phase2"][key]
    p51["runtime"]["output_dir"] = p50["runtime"]["output_dir"]
    assert p51 == p50
    validate_config(P51)
    assert configuration_hash(P51, "phase1") == configuration_hash(P49, "phase1")
    assert configuration_hash(P51, "flare") == configuration_hash(P50, "flare")
    assert configuration_hash(P51, "phase2") != configuration_hash(P50, "phase2")
    # The weights' path is out of the hash, as the Uformer's.
    moved = copy.deepcopy(P51)
    moved["data"]["restormer_checkpoint"] = "/kaggle/working/restormer/motion_deblurring.pth"
    assert configuration_hash(moved, "phase2") == configuration_hash(P51, "phase2")


@pytest.mark.parametrize("change, message", [
    ({"restore_net": "swinir"}, "restore_net must be one of"),
    ({"restore_widths": [48, 64]}, "set restore_widths to null"),
    ({"restore_restormer_weights": "deraining"}, "restore_restormer_weights must be one of"),
    ({"restore_pretrained_lr_scale": 0.0}, r"restore_pretrained_lr_scale must be in \(0, 1\]"),
])
def test_bad_restormer_keys_are_refused(change, message):
    config = copy.deepcopy(P51)
    config["phase2"].update(change)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_restormer_keys_need_restore_net_restormer():
    config = copy.deepcopy(P50)
    config["phase2"]["restore_pretrained_lr_scale"] = 0.25
    with pytest.raises(ValueError, match="belong to phase2.restore_net: restormer"):
        validate_config(config)


def test_the_decoder_restores_with_restormer_reading_the_deflared_frame_and_the_gyro_path():
    settings = _restore_settings(P51)
    assert settings["naf"] is None and settings["restormer"] == {"weights": "motion_deblurring"}
    decoder = RestoreRGBDecoder(128, **dict(settings, flare=None))
    assert isinstance(decoder.net, RestormerNet)
    assert decoder.net.extra_embed.in_channels == 3 + decoder.motion.channels
    nafnet = RestoreRGBDecoder(128, **dict(_restore_settings(P50), flare=None))
    assert not isinstance(nafnet.net, RestormerNet)


def test_phase2_starts_from_the_released_weights_and_fine_tunes_them_at_their_own_rate(tmp_path, monkeypatch, capsys):
    from qjepa.cli import main
    from qjepa.config import RESTORE_NAF_KEYS
    from qjepa.training.checkpoints import load_checkpoint
    from test_restore import _restore_smoke
    from test_scene_pairs import _train_tiny
    torch.set_num_threads(2)
    config, _, manifest_dir = _restore_smoke(tmp_path, lowlight=False)
    config["phase2"].update({key: None for key in RESTORE_NAF_KEYS}, restore_net="restormer",
                            restore_restormer_weights="motion_deblurring", restore_pretrained_lr_scale=0.25)
    torch.manual_seed(3)
    released = {key: value for key, value in RestormerNet(3, 8).state_dict().items() if not key.startswith(EXTRA_LAYERS)}
    weights = tmp_path / "motion_deblurring.pth"
    torch.save({"params": released}, weights)
    monkeypatch.setitem(restormer_module.RESTORMER_WEIGHTS, "motion_deblurring", ("unused", file_sha256(weights)))
    config["data"]["restormer_checkpoint"] = str(weights)
    validate_config(config)
    output = _train_tiny(tmp_path, config, manifest_dir, "run")
    assert f"Restormer motion_deblurring | sha256 {file_sha256(weights)[:12]}" in capsys.readouterr().out
    # Without the weights' path a fresh phase 2 stops before training.
    import yaml
    del config["data"]["restormer_checkpoint"]
    path = tmp_path / "missing.yaml"
    path.write_text(yaml.safe_dump(serializable_config(config)))
    with pytest.raises(SystemExit):
        main(["train-phase2", "--config", str(path), "--manifest", str(manifest_dir), "--output",
              str(tmp_path / "missing"), "--backbone-checkpoint", str(output / "phase1/last.pt")])
    assert "set data.restormer_checkpoint" in capsys.readouterr().err
    payload = load_checkpoint(output / "phase2/last.pt")
    assert payload["successful_updates"] == 3
    groups = payload["optimizer"]["param_groups"]
    # Decoders, the released Restormer at 0.25 x, then (smoke config) LP-FT's backbone group.
    assert [group.get("lr_scale", 1.0) for group in groups[:2]] == [1.0, 0.25]
    assert len(groups[1]["params"]) == 494
    state = payload["system"]
    shift = max((state[f"decoders.image.net.{key}"].cpu() - value).abs().max().item() for key, value in released.items())
    assert 0 < shift < 1e-2                                # fine-tuned from the release, not replaced
    assert state["decoders.image.net.latent_in.weight"].abs().sum() > 0


def test_the_weights_are_found_in_input_by_name_and_sha256_and_skipped_folders_are_not_walked(tmp_path, monkeypatch):
    from qjepa.models.restormer import find_restormer
    good = tmp_path / "input/datasets/me/restormer/motion_deblurring.pth"
    good.parent.mkdir(parents=True)
    good.write_bytes(b"released")
    monkeypatch.setitem(restormer_module.RESTORMER_WEIGHTS, "motion_deblurring", ("unused", file_sha256(good)))
    other = tmp_path / "input/datasets/me/shard/motion_deblurring.pth"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"released")
    assert find_restormer(tmp_path / "input", "motion_deblurring", skip=[good.parent]) == other
    other.write_bytes(b"something else")
    assert find_restormer(tmp_path / "input", "motion_deblurring", skip=[good.parent]) is None
    assert find_restormer(tmp_path / "input", "motion_deblurring") == good
