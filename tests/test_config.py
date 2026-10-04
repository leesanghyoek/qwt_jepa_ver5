import copy

import pytest

from qjepa.cli import _latent_gate
from qjepa.config import build_decoders, load_config, validate_config


def test_main_config_enforces_real_batch_eight():
    config = load_config("configs/pipeline_v3.yaml")
    bad = copy.deepcopy(config)
    bad["phase1"]["batch_size"] = 4
    with pytest.raises(ValueError, match="physical batch"):
        validate_config(bad)


def test_phase2_rejects_negative_variation_weight_and_non_positive_beta():
    config = load_config("configs/pipeline_v3.yaml")
    for key, value in (
        ("imu_variation_weight", -0.1),
        ("smooth_l1_beta", 0.0),
        ("smooth_l1_beta", -1.0),
    ):
        bad = copy.deepcopy(config)
        bad["phase2"][key] = value
        with pytest.raises(ValueError, match=key):
            validate_config(bad)


def test_phase_contracts_reject_unimplemented_paths_and_lying_config():
    config = load_config("configs/smoke.yaml")
    for section, key, value in (
        ("phase1", "decoder_enabled", True),          # bat decoder ma trong so 0
        ("phase1", "reconstruction_loss_weight", 1.0),  # khong co duong pixel-space
        ("phase2", "encoder_skips", False),           # lech voi decoder_input
        ("phase2", "reconstruction_detail_weight", -1.0),
    ):
        bad = copy.deepcopy(config)
        bad[section][key] = value
        with pytest.raises(ValueError):
            validate_config(bad)


def test_residual_mode_must_match_its_declared_output():
    config = load_config("configs/smoke.yaml")
    for residual, declared in ((True, "absolute_prediction"), (False, "input_residual")):
        bad = copy.deepcopy(config)
        bad["phase2"]["input_coefficient_residual"] = residual
        bad["phase2"]["output_coefficients"] = declared
        with pytest.raises(ValueError, match="output_coefficients"):
            validate_config(bad)


def test_latent_gate_detects_collapsed_diversity():
    reference = {
        "validation_clean_ZI_same_position_std": 1.0,
        "validation_clean_ZI_pooled_effective_rank": 4.0,
        "validation_clean_ZI_raw_rms": 1.0,
    }
    monitor = load_config("configs/smoke.yaml")["monitor"]
    passed, reasons = _latent_gate(reference, dict(reference), monitor)
    assert passed and not reasons
    collapsed = dict(reference)
    collapsed["validation_clean_ZI_same_position_std"] = 0.01
    passed, reasons = _latent_gate(reference, collapsed, monitor)
    assert not passed and reasons


def test_a_centre_norm_gate_bounds_the_raw_scale_itself():
    """p20 on Kaggle: centre-norm features start ~1e-4 and train to ~1-3, so the ratio to
    init read x18000 and failed a healthy run. Under centre norm the band bounds the raw
    RMS; GroupNorm keeps the ratio to init."""
    monitor = load_config("configs/smoke.yaml")["monitor"]
    reference = {"validation_noisy_FI_raw_rms": 1.87e-4, "validation_noisy_FI_same_position_std": 1e-4}
    trained = {"validation_noisy_FI_raw_rms": 3.4, "validation_noisy_FI_same_position_std": 1.0}
    assert not _latent_gate(reference, trained, monitor)[0]                       # GroupNorm rule
    assert _latent_gate(reference, trained, monitor, absolute_scale=True) == (True, [])
    for runaway in (1e-3, 50.0):                                                   # collapse, explosion
        passed, reasons = _latent_gate(reference, dict(trained, validation_noisy_FI_raw_rms=runaway),
                                       monitor, absolute_scale=True)
        assert not passed and "outside" in reasons[0]


def test_encoder_skips_requires_an_explicit_merge_kind():
    """Config cu khong co skip_gating phai bi chan o day, khong phai o load_state_dict.

    Neu de mac dinh, mot config cu hash GIONG HET checkpoint cu vi khoa khong ton
    tai o ca hai ben — roi build_decoders lang le dung kien truc moi, va loi chi
    lo ra khi nap state_dict, sau khi da dung sai model.
    """
    config = load_config("configs/pipeline_v3.yaml")
    assert config["phase2"]["encoder_skips"] is True
    bad = copy.deepcopy(config)
    del bad["phase2"]["skip_gating"]
    with pytest.raises(ValueError, match="skip_gating"):
        validate_config(bad)

    # Tat skip thi khoa do khong con y nghia, khong duoc doi hoi.
    without = copy.deepcopy(config)
    without["phase2"]["encoder_skips"] = False
    without["phase2"]["decoder_input"] = "fused_dense_latent_only"
    del without["phase2"]["skip_gating"]
    validate_config(without)


def test_old_checkpoint_config_without_the_new_keys_rebuilds_the_old_architecture():
    """build_decoders cung doc config nam TRONG checkpoint.

    Checkpoint train truoc khi khoa ra doi khong co no, va y nghia dung cua
    'khong co' la kien truc truoc do. Neu tra khoa bang [...] thi moi checkpoint
    cu deu nap that bai bang KeyError thay vi mot thong bao ro rang.
    """
    config = copy.deepcopy(load_config("configs/pipeline_v3.yaml"))
    for key in ("residual_sees_input", "skip_gating", "encoder_skips",
                "image_decoder", "image_resnet_width", "image_resnet_blocks"):
        config["phase2"].pop(key, None)
    config["phase2"]["decoder_input"] = "fused_dense_latent_only"
    decoders = build_decoders(config)
    assert decoders.image.sees_input is False
    assert decoders.uses_skips is False
    # Va head phai co dung so kenh cua kien truc cu, neu khong state_dict lech.
    assert decoders.image.head.in_channels == config["model"]["encoder_channels"][0]
