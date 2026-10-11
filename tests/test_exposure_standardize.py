"""p38: phase2.split_exposure_standardize and phase2.split_tone_gate_lr_scale.

The global heads of the split decoder (GlobalToneColor, the bilateral grid, the stop map, the tone gate) read the
exposure statistics on scales two orders apart, and the tone gate its luminance layout and the latent's mean as
they come. Measured on p37 (README p38): an MLP like the gate still said "relit" for every frame after 250 steps on
the raw features, and the decoder meanwhile put one average correction on every frame -- near-clean frames lost
12-17 dB. Standardised, the gate learnt in tens of steps and the frames kept to 2-5 dB.

Pins: absent, the decoder has no new buffers and every head reads the raw statistics, so old checkpoints load and
old configs hash as before; on, the decoder still starts as the identity, the heads read (stats - mean) / std with
constants measured on calibration frames (the measured rows come out mean 0, std 1; a constant feature keeps the
floor), and the constants are in the checkpoint; a gate on standardised features learns plain-vs-relit frames in
far fewer steps than on the raw ones; the gate gets its own learning-rate group only with the key; the logged loss
splits exactly into image, IMU and auxiliary heads; train-phase2 calibrates a fresh run once, and a resumed run
keeps the constants; bad settings fail; p38 is p37 plus the two keys, with p37's phase-1 hash.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch
import torch.nn.functional as F
import yaml

from qjepa.cli import main
from qjepa.config import build_decoders, load_config, serializable_config, validate_config
from qjepa.models.decoders import EXPOSURE_FEATURES, FeatureStandardizer, exposure_statistics
from qjepa.training.checkpoints import configuration_hash, load_checkpoint
from test_bilateral import GRID
from test_ijepa import _ijepa_dict, _run_until_done
from test_kaggle_workflow import _write_dataset
from test_tone_gate import GATE

STANDARDIZE = {"split_exposure_standardize": True, "split_tone_gate_lr_scale": 10}
RELIGHT = {"split_relight": True, "split_relight_size": 16, "split_relight_widths": [8, 8, 8],
           "split_relight_weight": 0.5}
CHROMA = {"split_chroma_detail": True, "split_chroma_width": 8, "split_chroma_blocks": 1}   # p34: starts as identity


def _config(**phase2):
    config = _ijepa_dict(max_successful_updates=2, batch_size=2)
    config["phase2"].update(GRID, **GATE, **RELIGHT, **CHROMA)
    config["phase2"].update(split_exposure_stats=True, split_color_global=True, **phase2)
    validate_config(config)
    return config


def _frames(count=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    texture = torch.rand(count, 3, 64, 64, generator=generator)
    gain = torch.linspace(0.2, 1.0, count).view(-1, 1, 1, 1)             # dark to bright frames
    return (texture * gain).clamp(0, 1)


def test_absent_nothing_changes_and_on_the_decoder_still_starts_as_the_identity():
    torch.manual_seed(0)
    old = build_decoders(_config()).image
    new = build_decoders(_config(**STANDARDIZE)).image
    assert old.exposure_norm is None and old.tone_gate.layout_norm is None
    added = set(new.state_dict()) - set(old.state_dict())
    assert added == {"exposure_norm.mean", "exposure_norm.std", "tone_gate.layout_norm.mean",
                     "tone_gate.layout_norm.std"}
    channels = _config()["model"]["embedding_dim"]
    image, latent = _frames(2), torch.randn(2, channels, 4, 4)
    new.calibrate_exposure(new.exposure_rows(torch.randn(16, channels, 4, 4), _frames(16, seed=1)))
    with torch.no_grad():
        restored, parts = new(latent, image)
    assert torch.allclose(restored, image, atol=1e-5)                     # zero-initialised heads: still the identity
    assert torch.allclose(parts["image_light"], image, atol=1e-5)


def test_calibration_standardises_the_rows_it_measured_and_the_heads_read_them():
    torch.manual_seed(0)
    decoder = build_decoders(_config(**STANDARDIZE)).image
    frames, latent = _frames(32, seed=2), torch.randn(32, _config()["model"]["embedding_dim"], 4, 4)
    rows = decoder.exposure_rows(latent, frames)
    measured = decoder.calibrate_exposure(rows)
    assert len(measured["exposure_std"]) == EXPOSURE_FEATURES
    standardised = decoder.exposure_norm(rows["exposure"])
    varying = rows["exposure"].std(0) > 1e-2
    assert torch.allclose(standardised.mean(0), torch.zeros(EXPOSURE_FEATURES), atol=1e-4)
    assert torch.allclose(standardised.std(0)[varying], torch.ones(int(varying.sum())), atol=1e-4)
    assert torch.all(decoder.exposure_norm.std >= 1e-2)                   # a constant feature keeps the floor
    gate_rows = decoder.tone_gate.layout_norm(rows["gate_layout"])
    assert torch.allclose(gate_rows.mean(0), torch.zeros(gate_rows.shape[1]), atol=1e-4)
    # The grid's global head reads the standardised statistics: a frame's grid moves with the constants.
    for parameter in decoder.tone_grid.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    with torch.no_grad():
        before = decoder.tone_grid.coefficients(latent[:2], frames[:2], decoder.exposure_norm)
        decoder.exposure_norm.mean += 1.0
        after = decoder.tone_grid.coefficients(latent[:2], frames[:2], decoder.exposure_norm)
    assert not torch.allclose(before, after)


def _gate_steps(standardize: bool, seed: int = 0) -> int:
    """Steps until a gate-sized MLP tells dark (relit) frames from bright (plain) ones on 90% of a held-out set."""
    torch.manual_seed(seed)

    def features(count, generator_seed):
        generator = torch.Generator().manual_seed(generator_seed)
        relit = torch.rand(count, generator=generator) < 0.6
        scene = torch.rand(count, 1, 1, 1, generator=generator) * 0.5 + 0.5
        texture = torch.rand(count, 3, 32, 32, generator=generator)
        exposure = torch.where(relit.view(-1, 1, 1, 1), 0.35 + 0.3 * torch.rand(count, 1, 1, 1, generator=generator),
                               torch.ones(count, 1, 1, 1))
        return exposure_statistics((texture * scene * exposure).clamp(0, 1)), relit.float()

    train_x, train_y = features(512, 1)
    test_x, test_y = features(256, 2)
    norm = FeatureStandardizer(EXPOSURE_FEATURES)
    if standardize:
        norm.calibrate(train_x[:128])
    head = torch.nn.Sequential(torch.nn.Linear(EXPOSURE_FEATURES, 32), torch.nn.ReLU(), torch.nn.Linear(32, 32),
                               torch.nn.ReLU(), torch.nn.Linear(32, 1))
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    for step in range(1, 2001):
        index = torch.randint(0, len(train_x), (8,))
        loss = F.binary_cross_entropy_with_logits(head(norm(train_x[index])).squeeze(1), train_y[index])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            with torch.no_grad():
                accuracy = ((head(norm(test_x)).squeeze(1) > 0).float() == test_y).float().mean()
            if accuracy >= 0.9:
                return step
    return 10_000


def test_a_gate_on_standardised_statistics_learns_in_far_fewer_steps():
    # Measured: raw 80-430 steps (1890 over 8 seeds), standardised 30-120 (560). On p37's real frames the raw gate
    # was still at chance after 250 steps (README p38).
    raw = [_gate_steps(False, seed) for seed in range(8)]
    standardised = [_gate_steps(True, seed) for seed in range(8)]
    assert max(standardised) <= 150, standardised
    assert all(fast < slow for fast, slow in zip(standardised, raw)), (raw, standardised)
    assert sum(raw) >= 2.5 * sum(standardised), (raw, standardised)


def test_the_gate_gets_its_own_learning_rate_only_with_the_key(tmp_path):
    from qjepa.config import build_normalizer, build_phase1_model, phase2_backbone, phase2_latent_modules
    from qjepa.models import RestorationSystem
    from qjepa.training.phase2 import Phase2Trainer

    def trainer(config):
        torch.manual_seed(0)
        phase1 = build_phase1_model(config, build_normalizer({"normalization": {
            "mean": [0.0] * 6, "std": [1.0] * 6}}))
        system = RestorationSystem(phase2_backbone(config, phase1), phase1.normalizer, build_decoders(config),
                                   *phase2_latent_modules(config, phase1))
        return Phase2Trainer(system, config, torch.device("cpu"), "parent")

    plain = trainer(_config())
    assert len(plain.optimizer.param_groups) == 1
    scaled = trainer(_config(**STANDARDIZE))
    groups = scaled.optimizer.param_groups
    gate = {id(p) for p in scaled.system.decoders.image.tone_gate.parameters()}
    assert len(groups) == 2 and {id(p) for p in groups[1]["params"]} == gate and groups[1]["lr_scale"] == 10
    assert not gate & {id(p) for p in groups[0]["params"]}
    scaled.successful_updates = 1
    lr = scaled._set_lr()
    assert groups[0]["lr"] == pytest.approx(lr) and groups[1]["lr"] == pytest.approx(10 * lr)


@pytest.mark.parametrize("change, message", [
    ({"split_exposure_standardize": "yes"}, "split_exposure_standardize must be"),
    ({"split_tone_gate_lr_scale": 0}, "split_tone_gate_lr_scale"),
    ({"split_tone_gate_lr_scale": True}, "split_tone_gate_lr_scale"),
])
def test_bad_settings_fail(change, message):
    config = _config()
    config["phase2"].update(change)
    with pytest.raises(ValueError, match=message):
        validate_config(config)
    gateless = _ijepa_dict()
    gateless["phase2"].update(GRID, split_tone_gate_lr_scale=10)
    with pytest.raises(ValueError, match="split_tone_gate: true"):
        validate_config(gateless)
    statless = _ijepa_dict()
    statless["phase2"].update(GRID, split_exposure_standardize=True)
    with pytest.raises(ValueError, match="split_exposure_stats or split_tone_gate"):
        validate_config(statless)


def test_phase2_calibrates_a_fresh_run_once_logs_the_loss_split_and_a_resume_keeps_the_constants(tmp_path, capsys):
    torch.set_num_threads(1)
    root, manifest, run = tmp_path / "dataset", tmp_path / "manifest", tmp_path / "run"
    _write_dataset(root)
    config = _config(**STANDARDIZE)
    config["phase2"].update(max_successful_updates=2, batch_size=2)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest)])
    common = ["--config", str(path), "--manifest", str(manifest), "--output", str(run)]
    _run_until_done(["train-phase1", *common], run / "phase1/last.pt")
    # Phase 2 saves after every update and restarts itself (RSS over the tiny limit): update 2 runs from a resume.
    config["runtime"].update(checkpoint_every_updates=1, restart_above_rss_gib=1e-6)
    path.write_text(yaml.safe_dump(config))
    capsys.readouterr()
    restarts = _run_until_done(["train-phase2", *common, "--backbone-checkpoint", str(run / "phase1/last.pt")],
                               run / "phase2/last.pt")
    out = capsys.readouterr().out
    assert restarts == 1 and out.count("exposure standardization |") == 1     # the fresh run only
    final = load_checkpoint(run / "phase2/last.pt")
    assert final["successful_updates"] == 2
    state = final["system"]
    mean, std = state["decoders.image.exposure_norm.mean"], state["decoders.image.exposure_norm.std"]
    assert mean.abs().sum() > 0 and not torch.allclose(std, torch.ones_like(std))
    assert "decoders.image.tone_gate.layout_norm.std" in state
    # The constants measured at the start are those after the resume: what the frozen frames gave, not a re-measure.
    from qjepa.cli import _calibrate_decoder_exposure, _manifest, _system_from_phase2
    system, saved = _system_from_phase2(str(run / "phase2/last.pt"), torch.device("cpu"))
    loaded, _ = _manifest(saved, str(manifest))
    remeasured = copy.deepcopy(system)
    _calibrate_decoder_exposure(saved, loaded, remeasured, torch.device("cpu"))
    assert torch.allclose(remeasured.decoders.image.exposure_norm.mean, mean)
    rows = [json.loads(line) for line in (run / "phase2/train.jsonl").read_text().splitlines()]
    updates = [row for row in rows if "loss_image" in row]
    assert len(updates) == 2
    for update in updates:
        assert update["loss"] == pytest.approx(update["loss_image"] + update["loss_imu"] + update["loss_heads"],
                                               rel=1e-5, abs=1e-6)
        assert update["loss_heads"] > 0 and update["loss_imu"] > 0


def test_kaggle_expnorm_is_p37_plus_the_two_keys_with_p37s_phase1_hash():
    p37 = serializable_config(load_config("configs/kaggle_flareblur.yaml"))
    p38 = serializable_config(load_config("configs/kaggle_expnorm.yaml"))
    for config in (p37, p38):
        config.pop("_config_path", None)
        config["runtime"].pop("output_dir")
    assert {key: p38["phase2"].pop(key) for key in STANDARDIZE} == STANDARDIZE
    assert p38 == p37
    full37, full38 = load_config("configs/kaggle_flareblur.yaml"), load_config("configs/kaggle_expnorm.yaml")
    validate_config(full38)
    assert configuration_hash(full38, "phase1") == configuration_hash(full37, "phase1")      # reuse p37's phase 1
    assert configuration_hash(full38, "phase2") != configuration_hash(full37, "phase2")
    # p37 and p36 hash as they did before the keys existed.
    assert configuration_hash(full37, "phase1").startswith("20820dddfa24")
    assert configuration_hash(full37, "phase2").startswith("a31ff9dea72d")
    assert configuration_hash(load_config("configs/kaggle_flareimage.yaml"), "phase2").startswith("a07e3f510503")
