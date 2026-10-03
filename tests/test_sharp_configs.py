"""The two runs of the sharpness plan, as configs.

kaggle_sharp_p2 changes phase 2 only (longer, mirrored pairs, the JEPA predictor
into the decoder, LP-FT, IMU increments), so its phase-1 hash must equal
p16_infomax's and the notebook can reuse that phase 1. kaggle_sharp adds the
phase-1 fixes (sensor-noise Jacobian direction, centre norm, degradation head and
condition, mirrored pairs), so its phase 1 must retrain. Pins: each key the plan
names is set; the reuse and retrain promises hold in the hashes.
"""

from __future__ import annotations

from qjepa.config import load_config
from qjepa.training.checkpoints import configuration_hash


def test_the_phase2_only_run_reuses_the_infomax_phase1():
    infomax, sharp_p2 = load_config("configs/kaggle_infomax.yaml"), load_config("configs/kaggle_sharp_p2.yaml")
    phase2 = sharp_p2["phase2"]
    assert phase2["max_successful_updates"] >= 20000
    assert phase2["augment_hflip"] is True and phase2["decoder_predictor_input"] is True
    assert 0 < phase2["backbone_finetune_after_updates"] < phase2["max_successful_updates"]
    assert 0 < phase2["backbone_finetune_lr_scale"] <= 0.1
    assert phase2["imu_increment_weight"] > 0
    assert configuration_hash(sharp_p2, "phase1") == configuration_hash(infomax, "phase1")
    assert configuration_hash(sharp_p2, "phase2") != configuration_hash(infomax, "phase2")
    assert sharp_p2["runtime"]["output_dir"].endswith("p19_sharp_p2")


def test_the_full_run_retrains_phase1_with_every_fix():
    sharp_p2, sharp = load_config("configs/kaggle_sharp_p2.yaml"), load_config("configs/kaggle_sharp.yaml")
    assert sharp["model"]["encoder_norm"] == "centre"
    assert sharp["encoder_sensitivity"]["noise_direction"] == "sensor_noise"
    assert sharp["phase1"]["degradation_weight"] > 0 and sharp["phase1"]["predictor_degradation_condition"] is True
    assert sharp["phase1"]["augment_hflip"] is True
    assert configuration_hash(sharp, "phase1") != configuration_hash(sharp_p2, "phase1")
    changed = {"augment_hflip", "degradation_weight", "predictor_degradation_condition"}
    assert ({k: v for k, v in sharp["phase1"].items() if k not in changed}
            == {k: v for k, v in sharp_p2["phase1"].items() if k not in changed})
    assert sharp["phase2"] == sharp_p2["phase2"]
    assert sharp["runtime"]["output_dir"].endswith("p19_sharp")
