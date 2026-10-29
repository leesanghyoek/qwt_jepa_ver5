"""p52_detail: the Restormer adds detail only, the light is the tone steps' (curve + bilateral grid), and the Restormer
trains on one 128 x 128 window per frame (RestoreRGBDecoder's docstring, configs/kaggle_detail.yaml).

Pins:
- p52 is p51 plus the four restore_* detail keys and motion_field null; same phase-1 hash as p49, same flare hash as
  p51, its own phase-2 hash; bad keys are refused, and the keys need the Restormer;
- crop_frames cuts ZI's 1/8 map where the frame's window lies (corners on the 8-px grid);
- with restore_highpass_sigma the network cannot move brightness: a constant lift of its answer does not reach the
  output, a 2-px pattern does; without the key it does (p51);
- the light grid starts as the identity and learns; in training one window per frame comes out with its corner, in
  evaluation the whole frame; checkpointing is off when asked;
- phase 2 trains on the windows and validates whole frames, end to end.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from qjepa.config import _restore_settings, load_config, serializable_config, validate_config
from qjepa.models import restormer as restormer_module
from qjepa.models.decoders import RestoreRGBDecoder, crop_frames
from qjepa.models.restormer import EXTRA_LAYERS, RestormerNet, file_sha256
from qjepa.training.checkpoints import configuration_hash

P49 = load_config("configs/kaggle_glint.yaml")
P50 = load_config("configs/kaggle_noghost.yaml")
P51 = load_config("configs/kaggle_restormer.yaml")
P52 = load_config("configs/kaggle_detail.yaml")
DETAIL = {"restore_train_crop": 128, "restore_highpass_sigma": 4.0, "restore_light_grid": 16,
          "restore_checkpointing": False}


def test_p52_is_p51_with_the_detail_keys_and_reuses_p49_phase1_and_the_flare_stage():
    p51, p52 = copy.deepcopy(serializable_config(P51)), copy.deepcopy(serializable_config(P52))
    assert p52["runtime"]["output_dir"].endswith("/p52_detail")
    assert {key: p52["phase2"][key] for key in DETAIL} == DETAIL
    assert p52["phase2"]["motion_field"] is None and p51["phase2"]["motion_field"] is not None
    for key in DETAIL:
        del p52["phase2"][key]
    p52["phase2"]["motion_field"] = p51["phase2"]["motion_field"]
    p52["runtime"]["output_dir"] = p51["runtime"]["output_dir"]
    assert p52 == p51
    validate_config(P52)
    assert configuration_hash(P52, "phase1") == configuration_hash(P49, "phase1")
    assert configuration_hash(P52, "flare") == configuration_hash(P51, "flare")
    assert configuration_hash(P52, "phase2") != configuration_hash(P51, "phase2")
    settings = _restore_settings(P52)
    assert settings["restormer"] == {"weights": "motion_deblurring", "checkpointing": False}
    assert (settings["train_crop"], settings["highpass_sigma"], settings["light_grid"], settings["motion"]) == \
        (128, 4.0, 16, None)
    # p51 builds as before.
    old = _restore_settings(P51)
    assert old["restormer"] == {"weights": "motion_deblurring"}
    assert (old["train_crop"], old["highpass_sigma"], old["light_grid"]) == (None, None, None)


@pytest.mark.parametrize("change, message", [
    ({"restore_train_crop": 100}, "restore_train_crop must be a multiple of 8"),
    ({"restore_train_crop": 256}, "smaller than the frame"),
    ({"restore_highpass_sigma": 0}, "restore_highpass_sigma must be positive"),
    ({"restore_highpass_sigma": 50.0}, "inside the restored window"),
    ({"restore_light_grid": 1}, "restore_light_grid must be an integer >= 2"),
    ({"restore_checkpointing": "no"}, "restore_checkpointing must be true or false"),
])
def test_bad_detail_keys_are_refused(change, message):
    config = copy.deepcopy(P52)
    config["phase2"].update(change)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_detail_keys_need_the_restormer():
    config = copy.deepcopy(P50)
    config["phase2"]["restore_train_crop"] = 128
    with pytest.raises(ValueError, match="belong to phase2.restore_net: restormer"):
        validate_config(config)


def test_the_window_of_zi_map_is_where_the_frames_window_is():
    torch.manual_seed(0)
    frame = torch.rand(3, 2, 64, 64)
    offsets = [[0, 8], [16, 40], [48, 48]]
    whole_map = F.avg_pool2d(frame, 8)
    assert torch.allclose(crop_frames(whole_map, offsets, 16, scale=8),
                          F.avg_pool2d(crop_frames(frame, offsets, 16), 8))
    assert torch.equal(crop_frames(frame, torch.tensor(offsets), 16)[1], frame[1, :, 16:32, 40:56])


class _Lift(nn.Module):
    """Stands in for the Restormer: answers base + a constant lift + a 2-px checkerboard."""

    def __init__(self, pattern: torch.Tensor) -> None:
        super().__init__()
        self.pattern = pattern

    def forward(self, latent, x, base=None):
        return base + 0.3 + self.pattern


def _decoder(**detail) -> RestoreRGBDecoder:
    torch.manual_seed(0)
    return RestoreRGBDecoder(8, restormer={"weights": "motion_deblurring", "checkpointing": False}, tone_width=8,
                             **detail)


def test_with_the_highpass_the_network_adds_detail_and_no_brightness():
    rows = torch.arange(32)
    pattern = 0.05 * (-1.0) ** (rows[:, None] + rows[None, :]).float()
    image, latent = torch.rand(2, 3, 32, 32), torch.randn(2, 8, 2, 2)
    for sigma, lift in ((2.0, 0.0), (None, 0.3)):
        decoder = _decoder(highpass_sigma=sigma).eval()
        decoder.net = _Lift(pattern)
        with torch.no_grad():
            out, parts = decoder(latent, image)
        added = out - parts["image_light"]
        assert torch.allclose(added, lift + pattern.expand_as(added), atol=1e-4)


def test_the_light_grid_starts_as_the_identity_and_learns():
    decoder = _decoder(light_grid=4)
    image, latent = torch.rand(2, 3, 32, 32), torch.randn(2, 8, 4, 4)
    with torch.no_grad():
        _, parts = decoder.eval()(latent, image)
        toned = decoder.tone(latent, image, decoder.exposure_norm)
    assert torch.allclose(parts["image_light"], toned, atol=1e-6)
    decoder.train()
    out, parts = decoder(latent, image)
    (out.square().mean() + parts["image_light"].square().mean()).backward()
    assert decoder.light_grid.out.weight.grad.abs().sum() > 0


def test_training_restores_one_window_per_frame_and_evaluation_the_whole_frame():
    decoder = _decoder(train_crop=16, highpass_sigma=2.0, light_grid=4)
    assert decoder.net.checkpointing is False
    image, latent = torch.rand(5, 3, 32, 32), torch.randn(5, 8, 4, 4)
    out, parts = decoder.train()(latent, image)
    assert out.shape == (5, 3, 16, 16) and parts["image_light"].shape == (5, 3, 32, 32)
    corners = parts["image_crop"]
    assert corners.shape == (5, 2) and bool(((corners % 8 == 0) & (corners >= 0) & (corners <= 16)).all())
    with torch.no_grad():
        out, parts = decoder(latent, image)                  # no gradient: whole frames, as in validation
    assert out.shape == (5, 3, 32, 32) and "image_crop" not in parts
    out, parts = decoder.eval()(latent, image)
    assert out.shape == (5, 3, 32, 32) and "image_crop" not in parts


def test_phase2_trains_on_windows_and_validates_whole_frames(tmp_path, monkeypatch):
    from qjepa.config import RESTORE_NAF_KEYS
    from qjepa.training.checkpoints import load_checkpoint
    from test_restore import _restore_smoke
    from test_scene_pairs import _train_tiny
    torch.set_num_threads(2)
    config, _, manifest_dir = _restore_smoke(tmp_path, lowlight=False)
    config["phase2"].update({key: None for key in RESTORE_NAF_KEYS}, restore_net="restormer",
                            restore_restormer_weights="motion_deblurring", restore_pretrained_lr_scale=0.25,
                            restore_train_crop=16, restore_highpass_sigma=2.0, restore_light_grid=4,
                            restore_checkpointing=False, motion_field=None)
    torch.manual_seed(3)
    released = {key: value for key, value in RestormerNet(3, 8).state_dict().items() if not key.startswith(EXTRA_LAYERS)}
    weights = tmp_path / "motion_deblurring.pth"
    torch.save({"params": released}, weights)
    monkeypatch.setitem(restormer_module.RESTORMER_WEIGHTS, "motion_deblurring", ("unused", file_sha256(weights)))
    config["data"]["restormer_checkpoint"] = str(weights)
    validate_config(config)
    output = _train_tiny(tmp_path, config, manifest_dir, "run")
    payload = load_checkpoint(output / "phase2/last.pt")
    assert payload["successful_updates"] == 3
    assert payload["system"]["decoders.image.light_grid.out.weight"].abs().sum() > 0
    rows = [json.loads(line) for line in (output / "phase2/train.jsonl").read_text().splitlines()]
    trained = [row for row in rows if "loss" in row]
    assert trained and all(torch.isfinite(torch.tensor(row["loss"])) for row in trained)
    assert any(row.get("kind_nhe_count", 0) + row.get("kind_mo_count", 0) + row.get("kind_toi_count", 0)
               + row.get("kind_toi_mo_count", 0) + row.get("kind_loe_count", 0) > 0 for row in trained)
    checks = [row for row in rows if "validation_image_psnr_db" in row]
    assert checks and all(torch.isfinite(torch.tensor(row["validation_image_psnr_db"])) for row in checks)
