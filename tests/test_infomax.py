"""p16_infomax: four terms that make the phase-1 latent hold more, and phase 2 using it.

(1) coding rate (log-det), (2) dense InfoNCE, (3) a floor on the latent's gain for
2-4 px detail, (4) a JEPA target at 64 x 64, plus the JEPA encoder's finer stages
fed into the NAFNet edge branch. Pins: each term reads the way its maths says;
configs without the keys train and score exactly as before; a phase-1 step with
all four stays finite, trains the new head and logs every term; the phase-2 inputs
start neutral and then train; bad settings are refused; the run's config.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch

from qjepa.cli import _synthetic_batch
from qjepa.config import build_decoders, build_phase1_model, load_config, seed_everything, validate_config
from qjepa.data import ImuNormalizer
from qjepa.execution import Phase1Forward
from qjepa.models import RestorationSystem
from qjepa.models.predictors import SpatialPredictor
from qjepa.training.checkpoints import configuration_hash
from qjepa.training.losses import coding_rate_loss, dense_infonce_loss
from qjepa.training.phase1 import Phase1Trainer
from qjepa.training.phase2 import Phase2Trainer

PHASE1 = dict(predictor_type="spatial", multiscale_fine_weight=0.5, multiscale_finer_weight=0.25,
              coding_rate_weight=0.2, infonce_weight=0.05)
FLOORS = {"image": 6.9, "imu": 5.8}
NEW_PHASE1 = ("multiscale_finer_weight", "coding_rate_weight", "coding_rate_eps_squared", "infonce_weight",
              "infonce_temperature")
SMALL_NAF = dict(split_edge_naf_widths=[4, 6, 8, 12, 16], split_edge_naf_enc_blocks=[1, 1, 1, 1],
                 split_edge_naf_middle_blocks=1, split_edge_naf_dec_blocks=[1, 1, 1, 1],
                 split_edge_refiner_width=8, split_edge_refiner_blocks=1)


def _config(floors=True, **phase1):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase1"].update(phase1)
    if floors:
        config["encoder_sensitivity"]["signal_floor_log_gain"] = dict(FLOORS)
    validate_config(config)
    return config


def test_the_coding_rate_pays_for_spread_and_ignores_scale():
    generator = torch.Generator().manual_seed(0)
    spread = torch.randn(4, 32, 16, generator=generator)
    collapsed = spread[..., :1].repeat(1, 1, 16) + 0.01 * torch.randn(4, 32, 16, generator=generator)
    assert float(coding_rate_loss(spread)) < float(coding_rate_loss(collapsed)) < 0
    assert float(coding_rate_loss(3.0 * spread)) == pytest.approx(float(coding_rate_loss(spread)), rel=1e-5)


def test_infonce_rewards_telling_each_token_from_its_neighbours():
    generator = torch.Generator().manual_seed(1)
    target = torch.randn(2, 16, 32, generator=generator)
    assert float(dense_infonce_loss(target, target)) < 0.05
    smooth = target.mean(dim=1, keepdim=True).repeat(1, 16, 1)             # neighbours alike
    assert float(dense_infonce_loss(smooth, target)) > math.log(16) * 0.5
    predicted = target.clone().requires_grad_(True)
    target = target.clone().requires_grad_(True)
    dense_infonce_loss(predicted + 0.3 * torch.randn(predicted.shape, generator=generator), target).backward()
    assert predicted.grad is not None and target.grad is None                 # the target is detached


def test_the_finer_head_grows_from_the_fine_one():
    predictor = SpatialPredictor(16, 32, spatial_dims=2, fine_channels=6, finer_channels=4)
    _, fine = predictor(torch.randn(2, 16, 4, 4))
    assert fine.shape == (2, 6, 8, 8) and predictor.finer(fine).shape == (2, 4, 16, 16)
    with pytest.raises(ValueError, match="fine"):
        SpatialPredictor(16, 32, spatial_dims=2, finer_channels=4)


def test_phase1_returns_the_64_target_at_the_encoders_stage():
    config = _config(**PHASE1)
    seed_everything(0)
    model = build_phase1_model(config, ImuNormalizer())
    batch = _synthetic_batch(config)
    features = Phase1Forward(model)(batch["image_noisy"], batch["imu_noisy_phys"], batch["image_clean"],
                                    batch["imu_clean_phys"], batch["image_time"], batch["imu_times"])
    assert features["prediction_i_finer"].shape == features["target_i_finer"].shape
    assert features["target_i_finer"].shape[-1] == 2 * features["target_i_fine"].shape[-1]


def test_configs_without_the_keys_train_as_before():
    config = _config(floors=False, predictor_type="spatial", multiscale_fine_weight=0.5)
    for key in NEW_PHASE1:
        config["phase1"].pop(key, None)
    model = build_phase1_model(config, ImuNormalizer())
    assert model.image_predictor.finer is None and not model.finer_scale
    metrics = Phase1Trainer(model, config, torch.device("cpu")).step(_synthetic_batch(config))
    assert not {"coding_rate", "infonce", "jepa_image_finer", "sensitivity_keep"} & set(metrics)
    assert configuration_hash(config, "phase2") == configuration_hash(_config(**PHASE1), "phase2")


def test_a_phase1_step_with_the_four_terms_trains_and_logs_them():
    config = _config(**PHASE1)
    seed_everything(2)
    model = build_phase1_model(config, ImuNormalizer())
    finer = [p.detach().clone() for p in model.image_predictor.finer.parameters()]
    trainer = Phase1Trainer(model, config, torch.device("cpu"))
    batch = _synthetic_batch(config)
    metrics = [trainer.step(batch) for _ in range(2)][-1]
    assert metrics["skipped"] is False and math.isfinite(metrics["loss"])
    for key in ("coding_rate", "infonce", "infonce_image", "infonce_imu", "jepa_image_finer"):
        assert key in metrics and math.isfinite(metrics[key]), key
    assert metrics["coding_rate"] < 0
    assert any(not torch.equal(a, b) for a, b in zip(finer, model.image_predictor.finer.parameters()))
    keeps = [m.get("sensitivity_keep") for m in (trainer.step(batch), trainer.step(batch))]
    assert any(value is not None and value >= 0 for value in keeps)        # image or IMU probe, alternating


@pytest.mark.parametrize("change, message", [
    (dict(multiscale_finer_weight=0.25, multiscale_fine_weight=0.0), "multiscale_finer_weight"),
    (dict(coding_rate_weight=-1.0), "coding_rate_weight"),
    (dict(infonce_temperature=0.0), "infonce_temperature"),
])
def test_bad_phase1_settings_are_refused(change, message):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase1"].update(predictor_type="spatial", **change)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_a_bad_floor_is_refused():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["encoder_sensitivity"]["signal_floor_log_gain"] = {"camera": 3.0}
    with pytest.raises(ValueError, match="signal_floor_log_gain"):
        validate_config(config)


def _phase2_config(**phase2):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(SMALL_NAF, encoder_skips=True, **phase2)
    validate_config(config)
    return config


def _batch():
    times = torch.arange(32).float().mul(0.01).repeat(1, 1)
    clean, imu = torch.rand(1, 3, 32, 32), torch.randn(1, 6, 32)
    return {"image_clean": clean, "image_noisy": (clean * 0.35).clamp(0, 1), "imu_clean_phys": imu,
            "imu_noisy_phys": imu + 0.05 * torch.randn_like(imu), "image_time": times.mean(dim=1),
            "imu_times": times, "sample_id": ["s"]}


def test_the_encoder_stages_reach_nafnet_start_neutral_and_train():
    config = _phase2_config(split_edge_naf_stage_levels=[1, 2, 3])
    seed_everything(3)
    phase1 = build_phase1_model(config, ImuNormalizer())
    seed_everything(config["phase2"]["decoder_initialization_seed"])
    system = RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config))
    stage_in = system.decoders.image.edge.stage_in
    assert set(stage_in) == {"1", "2", "3"} and system.decoders.image.uses_stages
    plain = copy.deepcopy(config)
    plain["phase2"].pop("split_edge_naf_stage_levels")
    reference = RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(plain))
    reference.load_state_dict(system.state_dict(), strict=False)
    batch = _batch()
    with torch.no_grad():
        for model in (system, reference):
            model.decoders.image.edge.ending.weight.normal_(0.0, 0.1, generator=torch.Generator().manual_seed(4))
        args = (batch["image_noisy"], batch["imu_noisy_phys"], batch["image_time"], batch["imu_times"])
        assert torch.allclose(system(*args).image, reference(*args).image, atol=1e-6)
    before = [p.detach().clone() for p in stage_in.parameters()]
    trainer = Phase2Trainer(system, config, torch.device("cpu"), "phase1.pt")
    for _ in range(2):
        assert trainer.step([batch] * config["phase2"]["gradient_accumulation"])["skipped"] is False
    assert any(not torch.equal(a, b) for a, b in zip(before, stage_in.parameters()))


@pytest.mark.parametrize("change, message", [
    (dict(split_edge_naf_stage_levels=[4]), "stage_levels"),
    (dict(split_edge_naf_stage_levels=[1], encoder_skips=False), "encoder_skips"),
])
def test_bad_stage_settings_are_refused(change, message):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config["phase2"].update(SMALL_NAF, encoder_skips=True)
    config["phase2"].update(change)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_the_run_is_p16_fourier_imu_plus_the_information_terms():
    infomax, fourier = load_config("configs/kaggle_infomax.yaml"), load_config("configs/kaggle_fourier.yaml")
    phase1 = infomax["phase1"]
    assert (phase1["coding_rate_weight"], phase1["infonce_weight"], phase1["multiscale_finer_weight"]) == (0.2, 0.05, 0.25)
    assert infomax["encoder_sensitivity"]["signal_floor_log_gain"] == FLOORS
    assert infomax["phase2"]["split_edge_naf_stage_levels"] == [1, 2, 3]
    assert infomax["phase2"]["max_successful_updates"] == 3000
    changed = ("split_edge_naf_stage_levels", "max_successful_updates")
    assert ({k: v for k, v in infomax["phase2"].items() if k not in changed}
            == {k: v for k, v in fourier["phase2"].items() if k not in changed})
    assert configuration_hash(infomax, "phase1") != configuration_hash(fourier, "phase1")   # phase 1 must retrain
    assert infomax["runtime"]["checkpoint_every_updates"] == 500
