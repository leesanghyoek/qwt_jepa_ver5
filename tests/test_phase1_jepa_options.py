"""Phase-1 JEPA options: spatial predictor, token masking, multi-scale targets,
and the normalized JEPA that makes runs comparable.

Each option is off unless its key is set, so these tests also pin that configs
written before the keys existed build and train exactly the old model.
"""

from __future__ import annotations

import copy

import pytest
import torch

from qjepa.cli import _synthetic_batch, _validate_latent
from qjepa.config import build_phase1_model, load_config, seed_everything, validate_config
from qjepa.data import ImuNormalizer
from qjepa.execution import Phase1Forward
from qjepa.models.masking import block_mask, sample_seeds, span_mask
from qjepa.models.predictors import LatentPredictor, SpatialPredictor
from qjepa.training.checkpoints import configuration_hash
from qjepa.training.losses import jepa_coarse_loss, jepa_diagnostics
from qjepa.training.phase1 import Phase1Trainer, phase1_masking

ALL_OPTIONS = dict(predictor_type="spatial", predictor_kernel=3, predictor_mixing_layers=2,
                   image_mask_ratio=0.3, image_mask_block=[2, 4], imu_mask_ratio=0.25,
                   imu_mask_span=[1, 2], multiscale_fine_weight=0.5, multiscale_coarse_weight=0.25,
                   multiscale_coarse_pool=2)


NEW_KEYS = ("predictor_type", "predictor_kernel", "predictor_mixing_layers", "image_mask_ratio",
            "image_mask_block", "imu_mask_ratio", "imu_mask_span", "multiscale_fine_weight",
            "multiscale_coarse_weight", "multiscale_coarse_pool")


def _config(**phase1):
    """Smoke config as written before these keys existed, plus ``phase1``."""
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    for key in NEW_KEYS:
        config["phase1"].pop(key, None)
    config["phase1"].update(phase1)
    validate_config(config)
    return config


def test_the_recipe_turns_every_option_on():
    phase1 = load_config("configs/kaggle_tartanair_v2.yaml")["phase1"]
    assert {key: phase1[key] for key in ALL_OPTIONS} == ALL_OPTIONS
    assert phase1["teacher_momentum_end"] == 0.9995


def test_configs_without_the_keys_build_the_old_token_predictor():
    model = build_phase1_model(_config(), ImuNormalizer())
    assert isinstance(model.image_predictor, LatentPredictor)
    assert isinstance(model.imu_predictor, LatentPredictor)
    assert not model.fine_scale
    assert {name for name in model.state_dict() if "predictor" in name} == {
        f"{branch}_predictor.net.{layer}.{kind}" for branch in ("image", "imu")
        for layer in (0, 1, 3) for kind in ("weight", "bias")}
    assert phase1_masking(_config()["phase1"]) is None


def test_the_spatial_predictor_costs_about_what_the_token_one_does():
    spatial = sum(p.numel() for p in SpatialPredictor(128, 256, spatial_dims=2).parameters())
    token = sum(p.numel() for p in LatentPredictor(128, 256).parameters())
    assert token < spatial < 1.1 * token             # 71,424 vs 66,176


def test_the_spatial_predictor_reads_a_five_token_neighbourhood():
    torch.manual_seed(0)
    predictor = SpatialPredictor(8, 16, spatial_dims=2).eval()
    grid = torch.randn(1, 8, 9, 9)
    kick = 5.0 * torch.randn(8)
    base, _ = predictor(grid)
    centre = 4 * 9 + 4
    for distance, reaches in ((2, True), (3, False)):
        moved = grid.clone()
        moved[0, :, 4, 4 + distance] += kick         # not a constant: LayerNorm would erase it
        out, _ = predictor(moved)
        assert (not torch.allclose(out[0, centre], base[0, centre])) is reaches


def test_a_masked_token_is_predicted_from_its_neighbours_only():
    torch.manual_seed(0)
    predictor = SpatialPredictor(8, 16, spatial_dims=2).eval()
    grid = torch.randn(1, 8, 6, 6)
    mask = torch.zeros(1, 36, dtype=torch.bool)
    mask[0, 14] = True                               # row 2, column 2
    base, _ = predictor(grid, mask)
    hidden = grid.clone()
    hidden[0, :, 2, 2] += 5.0 * torch.randn(8)       # the masked token's own value
    same, _ = predictor(hidden, mask)
    assert torch.allclose(same, base)
    neighbour = grid.clone()
    neighbour[0, :, 2, 3] += 5.0 * torch.randn(8)
    moved, _ = predictor(neighbour, mask)
    assert not torch.allclose(moved[0, 14], base[0, 14])


def test_masks_reach_their_ratio_never_hide_everything_and_repeat_per_seed():
    for seed in range(20):
        image = block_mask(16, 16, 0.3, (2, 4), seed)
        assert 0.3 * 256 <= int(image.sum()) < 256
        assert torch.equal(image, block_mask(16, 16, 0.3, (2, 4), seed))
        imu = span_mask(8, 0.25, (1, 2), seed)
        assert 2 <= int(imu.sum()) < 8
    assert not torch.equal(block_mask(16, 16, 0.3, (2, 4), 0), block_mask(16, 16, 0.3, (2, 4), 1))
    tiny = block_mask(2, 2, 0.75, (2, 4), 0)         # blocks never span a whole axis
    assert 1 <= int(tiny.sum()) < 4


def test_a_sample_gets_the_same_mask_whether_its_batch_is_split_or_not():
    config = _config(**ALL_OPTIONS)
    seed_everything(0)
    model = build_phase1_model(config, ImuNormalizer())
    runner = Phase1Forward(model, phase1_masking(config["phase1"]))
    batch = _synthetic_batch(config)
    args = [batch[k] for k in ("image_noisy", "imu_noisy_phys", "image_clean", "imu_clean_phys",
                               "image_time", "imu_times")]
    seeds = sample_seeds(1, 7, 2)
    whole = runner(*args, mask_seeds=seeds)
    halves = [runner(*[a[i:i + 1] for a in args], mask_seeds=seeds[i:i + 1]) for i in range(2)]
    for key in ("image_mask", "imu_mask"):
        assert torch.equal(whole[key], torch.cat([h[key] for h in halves]))


def test_normalized_jepa_reads_zero_when_perfect_and_one_at_collapse():
    torch.manual_seed(0)
    target = torch.randn(4, 16, 6, 6)
    tokens = target.flatten(2).transpose(1, 2)
    perfect = jepa_diagnostics(tokens.clone(), target)
    assert float(perfect["normalized"]) == pytest.approx(0.0, abs=1e-6)
    assert float(perfect["cosine"]) == pytest.approx(1.0, abs=1e-5)
    constant = tokens.mean(dim=(0, 1), keepdim=True).expand_as(tokens)
    collapsed = jepa_diagnostics(constant, target)
    assert float(collapsed["normalized"]) == pytest.approx(1.0, abs=0.05)
    mask = torch.rand(4, 36) < 0.3
    split = jepa_diagnostics(tokens + 0.3 * torch.randn_like(tokens), target, mask)
    assert 0 < float(split["visible"]) and 0 < float(split["masked"])


def test_the_coarse_term_ignores_errors_that_cancel_within_a_block():
    torch.manual_seed(0)
    target = torch.randn(2, 16, 8, 8)
    sign = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]).repeat(4, 4)
    cancelling = target + 0.5 * sign * torch.randn(2, 16, 1, 1)
    shared = target + 0.5 * torch.randn(2, 16, 1, 1).expand(2, 16, 8, 8).contiguous()
    tokens = lambda dense: dense.flatten(2).transpose(1, 2)
    assert float(jepa_coarse_loss(tokens(cancelling), target, 2)) == pytest.approx(0.0, abs=1e-6)
    assert float(jepa_coarse_loss(tokens(shared), target, 2)) > 0.01


def test_the_fine_head_matches_the_teachers_previous_stage():
    config = _config(**ALL_OPTIONS)
    model = build_phase1_model(config, ImuNormalizer())
    batch = _synthetic_batch(config)
    out = Phase1Forward(model)(batch["image_noisy"], batch["imu_noisy_phys"], batch["image_clean"],
                               batch["imu_clean_phys"], batch["image_time"], batch["imu_times"])
    assert out["prediction_i_fine"].shape == out["target_i_fine"].shape
    assert out["target_i_fine"].shape[-1] == 2 * out["target_i"].shape[-1]
    assert not out["target_i_fine"].requires_grad
    assert "image_mask" not in out                   # no seeds, no mask: validation's view


@pytest.mark.parametrize("change, message", [
    (dict(image_mask_ratio=0.3, image_mask_block=[2, 4]), "predictor_type: spatial"),
    (dict(multiscale_fine_weight=0.5), "predictor_type: spatial"),
    (dict(predictor_type="spatial", image_mask_ratio=0.9, image_mask_block=[2, 4]), "image_mask_ratio"),
    (dict(predictor_type="spatial", imu_mask_ratio=0.2, imu_mask_span=[3, 1]), "imu_mask_span"),
    (dict(multiscale_coarse_weight=0.2), "multiscale_coarse_pool"),
    (dict(predictor_type="spatial", predictor_kernel=4), "predictor_kernel"),
    (dict(predictor_type="conv"), "predictor_type"),
])
def test_inconsistent_options_are_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        _config(**change)


def test_the_options_move_the_phase1_contract_but_not_phase2():
    old, new = _config(), _config(**ALL_OPTIONS)
    assert configuration_hash(old, "phase1") != configuration_hash(new, "phase1")
    assert configuration_hash(old, "phase2") == configuration_hash(new, "phase2")


def test_a_phase1_step_with_every_option_trains_the_new_parts_and_logs_them():
    config = _config(**ALL_OPTIONS)
    seed_everything(config["phase1"]["initialization_seed"])
    model = build_phase1_model(config, ImuNormalizer())
    trainer = Phase1Trainer(model, config, torch.device("cpu"), "manifest")
    teacher = [p.detach().clone() for p in model.teachers.parameters()]
    fine_head = [p.detach().clone() for p in model.image_predictor.fine.parameters()]
    metrics = trainer.step(_synthetic_batch(config))
    assert metrics["skipped"] is False
    for key in ("jepa_image_normalized", "jepa_imu_cosine", "jepa_image_visible", "jepa_image_masked",
                "jepa_imu_masked", "jepa_image_fine", "jepa_image_fine_normalized", "jepa_image_coarse"):
        assert key in metrics and metrics[key] == metrics[key], key
    assert model.image_predictor.mask_token.abs().sum() > 0      # learned from zero
    assert any(not torch.equal(a, b) for a, b in zip(fine_head, model.image_predictor.fine.parameters()))
    # The fine target comes from the teacher: no gradient reaches it, it moves by EMA only.
    momentum = metrics["teacher_momentum"]
    online = [model.backbone.image_encoder, model.backbone.imu_encoder]
    online_parameters = [p for encoder in online for p in encoder.parameters()]
    for before, after, source in zip(teacher, model.teachers.parameters(), online_parameters):
        assert after.grad is None
        assert torch.allclose(after, momentum * before + (1 - momentum) * source.detach(), atol=1e-6)
    rebuilt = build_phase1_model(config, ImuNormalizer())
    rebuilt.load_state_dict(model.state_dict(), strict=True)


def test_the_old_recipe_logs_the_normalized_jepa_without_masked_terms():
    config = _config()
    seed_everything(config["phase1"]["initialization_seed"])
    trainer = Phase1Trainer(build_phase1_model(config, ImuNormalizer()), config, torch.device("cpu"), "m")
    metrics = trainer.step(_synthetic_batch(config))
    assert "jepa_image_normalized" in metrics and "jepa_image_cosine" in metrics
    assert not any(key.endswith(("_masked", "_visible", "_fine", "_coarse")) for key in metrics)


def test_validation_reports_normalized_jepa_without_masking():
    config = _config(**ALL_OPTIONS)
    model = build_phase1_model(config, ImuNormalizer())
    batch = _synthetic_batch(config)
    result = _validate_latent(model, [batch], torch.device("cpu"), 1)
    for key in ("validation_jepa_image_normalized", "validation_jepa_imu_cosine", "validation_jepa_image_fine"):
        assert key in result
    assert not any(key.endswith("_masked") for key in result)


class _TwoChunks(torch.nn.Module):
    """DataParallel's scatter/gather on CPU, as in test_multi_gpu."""

    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, *args, **kwargs):
        size = args[0].shape[0]
        outputs = []
        for start, end in ((0, size // 2), (size // 2, size)):
            options = {key: value[start:end] if isinstance(value, torch.Tensor) else value
                       for key, value in kwargs.items()}
            outputs.append(self.wrapped(*[value[start:end] for value in args], **options))
        return {key: torch.cat([output[key] for output in outputs], dim=0) for key in outputs[0]}


def test_two_gpu_split_gives_the_same_masked_multiscale_step():
    torch.set_num_threads(1)
    config = _config(**ALL_OPTIONS, batch_size=8, minimum_statistics_batch=8)
    seed_everything(19)
    batch = _synthetic_batch(config)
    model = build_phase1_model(config, ImuNormalizer())
    first = Phase1Trainer(model, config, torch.device("cpu"))
    second = Phase1Trainer(copy.deepcopy(model), config, torch.device("cpu"))
    second.forward_model = _TwoChunks(second.forward_model)
    a, b = first.step(batch), second.step(batch)
    for key in ("loss", "jepa", "jepa_image_masked", "jepa_image_fine", "jepa_image_coarse",
                "jepa_image_normalized"):
        assert a[key] == pytest.approx(b[key], rel=3e-4, abs=1e-5), key
    for pa, pb in zip(first.parameters, second.parameters):
        torch.testing.assert_close(pa.grad, pb.grad, rtol=2e-3, atol=2e-5)
