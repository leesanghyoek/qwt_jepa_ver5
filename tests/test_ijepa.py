"""phase1.objective: ijepa -- phase 1 as I-JEPA (Assran 2023) and nothing else.

The user asked for I-JEPA alone, with a ViT over the QWT coefficients, the context
encoder reading the corrupted pair (the EMA target encoder reads the clean one), and
image and IMU in one I-JEPA: one ViT over both kinds of token, one predictor. Pins:
the multi-block masks follow I-JEPA's collator (targets of the drawn scale, a context
with the targets cut out, one length per batch, reproducible); the context encoder
computes the context tokens from those tokens alone, and an IMU context token moves
the image tokens (the attention is the fusion); the dense latent ZI hears the IMU and
ZU the frame; each target block is its own predictor pass, and the IMU context moves
the image prediction; the targets are the EMA teacher's
LayerNorm'd tokens of the clean pair and carry no gradient; one update logs
I-JEPA's loss and nothing else and moves the teacher by EMA; the weight-decay and
momentum schedules and parameter groups are I-JEPA's; the config refuses every
other phase-1 term; old configs keep their hash and their CNN; p26_ijepa is
p25_local with only the I-JEPA keys changed; the CLI trains, resumes and
evaluates it; two DDP processes train what one does.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch
import torch.nn.functional as F
import yaml

from qjepa.cli import RESTART_EXIT_CODE, _synthetic_batch, main
from qjepa.config import (IJEPA_ABSENT_TERMS, _validate_ijepa, build_backbone, build_corruptors, build_phase1_model,
                          load_config, seed_everything, serializable_config, validate_config)
from qjepa.data import ImuNormalizer
from qjepa.distributed import spawn
from qjepa.execution import IJEPAForward
from qjepa.models.encoders import DenseCoefficientEncoder
from qjepa.models.ijepa import IJEPAPredictor, multiblock_masks
from qjepa.models.vit import PATCH, JointCoefficientViT
from qjepa.training.checkpoints import configuration_hash, load_checkpoint
from qjepa.training.ijepa import IJEPATrainer, ijepa_parameter_groups
from qjepa.training.schedules import cosine_weight_decay, linear_momentum
from test_kaggle_workflow import _write_dataset

IJEPA_MASKS = {"targets": 4, "target_scale": (0.15, 0.2), "target_aspect": (0.75, 1.5),
               "context_scale": (0.85, 1.0)}


def _ijepa_dict(**phase1):
    """smoke.yaml as I-JEPA: 64x64 frames (a 4x4 token grid), a 32-sample IMU window (2 tokens)."""
    config = serializable_config(load_config("configs/smoke.yaml"))
    config["data"]["image_size"] = [64, 64]
    config["model"].update(encoder_type="vit", vit_depth=2, vit_heads=4)
    config["encoder_sensitivity"]["enabled"] = False
    config["phase2"].update(encoder_skips=False, decoder_input="fused_dense_latent_only")
    for key in IJEPA_ABSENT_TERMS:
        config["phase1"][key] = 0.0
    config["phase1"].update(
        objective="ijepa", decoder_enabled=False, ijepa_context_input="noisy", ijepa_targets=4,
        ijepa_target_scale=[0.15, 0.2], ijepa_target_aspect=[0.75, 1.5], ijepa_context_scale=[0.85, 1.0],
        ijepa_image_min_keep=1, ijepa_imu_min_keep=1, ijepa_predictor_dim=16, ijepa_predictor_depth=2,
        ijepa_predictor_heads=4, weight_decay=0.04, weight_decay_end=0.4, teacher_momentum_start=0.996,
        teacher_momentum_end=1.0, gradient_clip_norm=None, batch_size=4)
    config["phase1"].update(phase1)
    return config


def _ijepa(**phase1):
    config = _ijepa_dict(**phase1)
    validate_config(config)
    return config


def _model(config):
    seed_everything(0)
    return build_phase1_model(config, ImuNormalizer())


def test_masks_follow_ijepa_multiblock():
    context, targets = multiblock_masks((16, 16), 8, 7, min_keep=10, **IJEPA_MASKS)
    assert context.shape[0] == targets.shape[0] == 8 and targets.shape[1] == 4
    # One target size per batch, cut to the shortest: 15-20% of the 256 tokens, give or take rounding.
    assert 0.1 * 256 <= targets.shape[2] <= 0.25 * 256
    assert context.shape[1] >= 10
    for sample in range(8):
        hidden = set(targets[sample].flatten().tolist())
        assert not hidden & set(context[sample].tolist()), "the context saw a target token"
        assert torch.equal(context[sample], context[sample].sort().values)
    assert int(context.max()) < 256 and int(targets.max()) < 256
    # Reproducible from the seed, as resume and the DDP ranks need.
    again = multiblock_masks((16, 16), 8, 7, min_keep=10, **IJEPA_MASKS)
    assert torch.equal(again[0], context) and torch.equal(again[1], targets)
    other = multiblock_masks((16, 16), 8, 8, min_keep=10, **IJEPA_MASKS)
    assert not torch.equal(other[1], targets)


def test_imu_masks_are_spans_on_eight_tokens():
    context, targets = multiblock_masks((8,), 8, 3, min_keep=1, **IJEPA_MASKS)
    assert targets.shape[1:] == (4, 1) and context.shape[1] >= 1
    for sample in range(8):
        assert not set(targets[sample].flatten().tolist()) & set(context[sample].tolist())
        # A span: consecutive indices.
        assert torch.equal(context[sample], torch.arange(int(context[sample, 0]),
                                                         int(context[sample, 0]) + context.shape[1])) or \
            context.shape[1] < 6
    with pytest.raises(ValueError, match="cannot keep"):
        multiblock_masks((8,), 2, 3, min_keep=8, **IJEPA_MASKS)


def test_the_context_encoder_sees_only_the_context():
    torch.manual_seed(0)
    encoder = JointCoefficientViT(16, 12, 32, depth=2, heads=4)
    image, imu = torch.randn(2, 16, 4 * PATCH, 4 * PATCH), torch.randn(2, 12, 4 * PATCH)    # 4x4 + 4 tokens
    keep_image = torch.tensor([[0, 1, 4, 5, 10], [2, 3, 7, 11, 15]])
    keep_imu = torch.tensor([[0, 2], [1, 3]])
    context_image, context_imu = encoder(image, imu, keep_image=keep_image, keep_imu=keep_imu)
    assert context_image.shape == (2, 5, 32) and context_imu.shape == (2, 2, 32)
    image_changed, imu_changed = image.clone(), imu.clone()
    for sample in range(2):
        for token in set(range(16)) - set(keep_image[sample].tolist()):
            row, column = divmod(token, 4)
            image_changed[sample, :, row * PATCH:(row + 1) * PATCH, column * PATCH:(column + 1) * PATCH] += 5.0
        for token in set(range(4)) - set(keep_imu[sample].tolist()):
            imu_changed[sample, :, token * PATCH:(token + 1) * PATCH] += 5.0
    again = encoder(image_changed, imu_changed, keep_image=keep_image, keep_imu=keep_imu)
    assert torch.equal(again[0], context_image) and torch.equal(again[1], context_imu)
    # A kept IMU token moves the image tokens of its sample, and only of its sample: the fusion.
    imu_changed[0, :, :PATCH] += 5.0                                   # IMU token 0 is in sample 0's context
    heard = encoder(image, imu_changed, keep_image=keep_image, keep_imu=keep_imu)[0]
    assert not torch.allclose(heard[0], context_image[0]) and torch.equal(heard[1], context_image[1])
    # Dense: the CNNs' shapes, FI [B, D, H, W] and FU [B, D, L].
    dense_image, dense_imu = encoder(image, imu)
    assert dense_image.shape == (2, 32, 4, 4) and dense_imu.shape == (2, 32, 4)
    with pytest.raises(ValueError, match="one resolution"):
        encoder(image, imu, return_stages=True)
    with pytest.raises(ValueError, match="both modalities"):
        encoder(image, imu, keep_image=keep_image)


def test_the_latent_of_each_modality_hears_the_other():
    config = _ijepa()
    batch = _synthetic_batch(config)
    model = _model(config).eval()
    arguments = (batch["image_noisy"], batch["imu_noisy_phys"], batch["image_time"], batch["imu_times"])
    with torch.no_grad():
        base = model.encode_online(*arguments)
        imu_moved = model.encode_online(arguments[0], arguments[1] * 3.0, *arguments[2:])
        image_moved = model.encode_online(arguments[0] * 0.5, *arguments[1:])
    assert not torch.allclose(imu_moved.ZI, base.ZI) and not torch.allclose(image_moved.ZU, base.ZU)
    # No fusion module after the encoder: the joint ViT's output is the latent.
    assert torch.equal(base.ZI, base.FI) and torch.equal(base.ZU, base.FU) and model.backbone.fusion is None


def test_each_target_block_is_its_own_predictor_pass():
    torch.manual_seed(0)
    predictor = IJEPAPredictor(32, 16, depth=2, heads=4)
    context = {"image": torch.randn(2, 5, 32), "imu": torch.randn(2, 2, 32)}
    masks = {"image_context": torch.tensor([[0, 1, 4, 5, 10], [2, 3, 7, 11, 15]]),
             "imu_context": torch.tensor([[0, 2], [1, 3]]),
             "image_targets": torch.tensor([[[2, 3], [8, 9], [12, 13]], [[0, 1], [4, 5], [8, 12]]]),
             "imu_targets": torch.tensor([[[1], [3]], [[0], [2]]])}
    grids = {"image": (4, 4), "imu": (4,)}
    image, imu = predictor(context, masks, grids)
    assert image.shape == (2, 3, 2, 32) and imu.shape == (2, 2, 1, 32)
    alone = predictor(context, dict(masks, image_targets=masks["image_targets"][:, 1:2]), grids)
    assert torch.allclose(alone[0], image[:, 1:2], atol=1e-6) and torch.allclose(alone[1], imu, atol=1e-6)
    # Where the target sits is what the mask token is told: other positions, other prediction.
    moved = predictor(context, dict(masks, image_targets=masks["image_targets"].flip(-1).roll(1, dims=1)), grids)
    assert not torch.allclose(moved[0], image)
    # The IMU context informs the image prediction, and the image context the IMU one.
    assert not torch.allclose(predictor(dict(context, imu=context["imu"] + 1.0), masks, grids)[0], image)
    assert not torch.allclose(predictor(dict(context, image=context["image"] + 1.0), masks, grids)[1], imu)


def _masks(model, batch, seed=0):
    image_grid, imu_grid = model.token_grids(tuple(batch["image_noisy"].shape), int(batch["imu_noisy_phys"].shape[-1]))
    return model.sample_masks(image_grid, imu_grid, int(batch["image_noisy"].shape[0]), seed)


def test_context_reads_the_noisy_pair_and_the_teacher_the_clean_one():
    config = _ijepa()
    batch = _synthetic_batch(config)
    model = _model(config)
    forward = IJEPAForward(model)
    masks = _masks(model, batch)
    arguments = [batch[key] for key in ("image_noisy", "imu_noisy_phys", "image_clean", "imu_clean_phys",
                                        "image_time", "imu_times")]
    base = forward(*arguments, **masks)
    noisier = list(arguments)
    noisier[0] = (arguments[0] + 0.2).clamp(0, 1)
    shifted = forward(*noisier, **masks)
    assert not torch.allclose(shifted["prediction_i"], base["prediction_i"])
    assert torch.equal(shifted["target_i"], base["target_i"])
    cleaner = list(arguments)
    cleaner[2] = (arguments[2] * 0.5)
    moved = forward(*cleaner, **masks)
    assert torch.equal(moved["prediction_i"], base["prediction_i"])
    assert not torch.allclose(moved["target_i"], base["target_i"])
    # The paper's arm: both sides clean, so the noisy pair changes nothing.
    model.context_input = "clean"
    assert torch.equal(forward(*noisier, **masks)["prediction_i"], forward(*arguments, **masks)["prediction_i"])


def test_targets_are_the_ema_teacher_layer_normed_and_frozen():
    config = _ijepa()
    batch = _synthetic_batch(config)
    model = _model(config)
    masks = _masks(model, batch)
    outputs = IJEPAForward(model)(batch["image_noisy"], batch["imu_noisy_phys"], batch["image_clean"],
                                  batch["imu_clean_phys"], batch["image_time"], batch["imu_times"], **masks)
    dense, _ = model.teachers.joint_encoder(
        model.backbone.image_transform.analysis(batch["image_clean"])[0],
        model.backbone.imu_transform.analysis(model.normalizer.normalize(batch["imu_clean_phys"]))[0])
    tokens = F.layer_norm(dense.flatten(2).transpose(1, 2), (dense.shape[1],))
    expected = torch.stack([tokens[sample, masks["image_targets"][sample]] for sample in range(tokens.shape[0])])
    assert torch.allclose(outputs["target_i"], expected, atol=1e-6)
    assert outputs["target_i"].mean(-1).abs().max() < 1e-4
    assert outputs["target_i"].grad_fn is None and outputs["prediction_i"].grad_fn is not None
    assert not any(parameter.requires_grad for parameter in model.teachers.parameters())
    assert outputs["prediction_i"].shape == outputs["target_i"].shape
    assert outputs["prediction_u"].shape == outputs["target_u"].shape


def test_one_update_trains_ijepa_alone():
    config = _ijepa()
    batch = _synthetic_batch(config)
    model = _model(config)
    trainer = IJEPATrainer(model, config, torch.device("cpu"), "synthetic")
    online = {name: value.detach().clone() for name, value in model.backbone.joint_encoder.named_parameters()}
    teacher = {name: value.detach().clone() for name, value in model.teachers.joint_encoder.named_parameters()}
    metrics = trainer.step(batch)
    assert metrics["skipped"] is False and metrics["successful_updates"] == 1
    allowed = {"skipped", "loss", "jepa", "jepa_image", "jepa_imu", "gradient_norm", "teacher_momentum",
               "learning_rate", "weight_decay", "successful_updates", "image_context_tokens",
               "image_targets_tokens", "imu_context_tokens", "imu_targets_tokens"}
    assert set(metrics) == allowed
    assert metrics["loss"] == pytest.approx(0.5 * (metrics["jepa_image"] + metrics["jepa_imu"]))
    assert metrics["teacher_momentum"] == pytest.approx(0.996)
    for name, before in online.items():
        after = dict(model.backbone.joint_encoder.named_parameters())[name].detach()
        expected = 0.996 * teacher[name] + 0.004 * after
        assert torch.allclose(dict(model.teachers.joint_encoder.named_parameters())[name], expected, atol=1e-7), name
    assert any(not torch.equal(before, dict(model.backbone.joint_encoder.named_parameters())[name])
               for name, before in online.items())
    # No anchor decoder, no fusion, no degradation head anywhere in what is saved.
    payload = trainer.checkpoint_payload(config)
    assert payload["metadata"]["phase1_objective"] == "ijepa"
    assert payload["metadata"]["trained_with_reconstruction"] is False
    assert not any(key.startswith(("decoders.", "backbone.fusion.", "degradation_head.")) for key in payload["model"])


def test_schedules_and_weight_decay_groups_are_ijepas():
    assert linear_momentum(0, 100, 0.996, 1.0) == pytest.approx(0.996)
    assert linear_momentum(50, 100, 0.996, 1.0) == pytest.approx(0.998)
    assert linear_momentum(99, 100, 0.996, 1.0) < 1.0                    # the EMA step needs m < 1
    assert cosine_weight_decay(0, 100, 0.04, 0.4) == pytest.approx(0.04)
    assert cosine_weight_decay(50, 100, 0.04, 0.4) == pytest.approx(0.22)
    assert cosine_weight_decay(100, 100, 0.04, 0.4) == pytest.approx(0.4)
    model = _model(_ijepa())
    decayed, plain = ijepa_parameter_groups(model)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    assert all("bias" in names[id(p)] or p.ndim == 1 for p in plain["params"])
    assert all("bias" not in names[id(p)] and p.ndim > 1 for p in decayed["params"])
    assert plain["weight_decay"] == 0.0
    decayed_names = {names[id(p)] for p in decayed["params"]}
    assert {"predictor.mask_token", "backbone.joint_encoder.image_embed.weight",
            "backbone.joint_encoder.imu_embed.weight"} <= decayed_names
    assert "backbone.joint_encoder.image_type" in {names[id(p)] for p in plain["params"]}
    grouped = {id(p) for group in (decayed, plain) for p in group["params"]}
    assert grouped == {id(p) for p in model.online_parameters()}
    assert not grouped & {id(p) for p in model.teachers.parameters()}


@pytest.mark.parametrize("change, message", [
    *[((("phase1", key), 0.1), "alone") for key in IJEPA_ABSENT_TERMS],
    ((("phase1", "decoder_enabled"), True, ("phase1", "coefficient_reconstruction_loss_weight"), 0.45), "alone"),
    ((("phase1", "predictor_degradation_condition"), True), "alone"),
    ((("encoder_sensitivity", "enabled"), True), "alone"),
    ((("phase1", "objective"), "jepa"), "come together"),
    ((("model", "encoder_type"), "cnn"), "come together"),
    ((("phase2", "encoder_skips"), True), "one resolution"),
    ((("phase2", "decoder_predictor_input"), True), "one resolution"),
    ((("model", "encoder_norm"), "centre"), "CNN"),
    ((("model", "vit_heads"), 5), "divide"),
    ((("data", "image_size"), [72, 72]), "divisible"),
    ((("phase1", "ijepa_targets"), None), "ijepa_targets"),
    ((("phase1", "ijepa_context_input"), "both"), "ijepa_context_input"),
    ((("phase1", "ijepa_target_scale"), [0.3, 0.2]), "ijepa_target_scale"),
    ((("phase1", "teacher_momentum_end"), 1.01), "momentum"),
    ((("phase1", "gradient_clip_norm"), 0.0), "gradient_clip_norm"),
    ((("phase1", "weight_decay_end"), None), "weight_decay_end"),
])
def test_the_config_refuses_anything_but_ijepa(change, message):
    config = _ijepa_dict()
    for (section, key), value in zip(change[::2], change[1::2]):
        config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)                 # refused, whichever check sees it first
    with pytest.raises(ValueError, match=message):
        _validate_ijepa(config)


def test_old_configs_keep_their_hash_and_their_cnn():
    config = load_config("configs/kaggle_local.yaml")
    # Measured before the I-JEPA keys existed: every p25 checkpoint stays loadable.
    assert configuration_hash(config, "phase1") == "e18e7e02f5e394351b93a39a616117bfbe846c014f44bc2d7ecc10d237dc2e4f"
    assert configuration_hash(config, "phase2") == "7a81db41a77e840d08b9fb8e2ed5bb0b33999f127618ff6922c2af40c7465c32"
    backbone = build_backbone(config)
    assert backbone.encoder_type == "cnn" and backbone.fusion is not None
    assert isinstance(backbone.image_encoder, DenseCoefficientEncoder)


def test_p26_is_p25_with_only_the_ijepa_keys_changed():
    p25, p26 = load_config("configs/kaggle_local.yaml"), load_config("configs/kaggle_ijepa.yaml")
    assert p26["data"] == p25["data"] and p26["corruption"] == p25["corruption"]
    differ = lambda section: {key for key in {*p25[section], *p26[section]}
                              if p25[section].get(key) != p26[section].get(key)}
    assert differ("phase2") == {"encoder_skips", "decoder_input", "split_edge_naf_stage_levels",
                                "decoder_predictor_input"}
    assert differ("model") == {"encoder_type", "vit_depth", "vit_heads", "encoder_norm", "encoder_norm_calibration"}
    assert p26["phase1"]["objective"] == "ijepa" and p26["phase1"]["ijepa_context_input"] == "noisy"
    model = build_phase1_model(p26, ImuNormalizer())
    image = torch.rand(1, 3, 256, 256)
    latent = model.encode_online(image, torch.randn(1, 6, 128), torch.tensor([0.635]),
                                 torch.arange(128).float()[None] * 0.01)
    # The grids every phase-2 decoder reads, as the CNN gave them.
    assert latent.ZI.shape == (1, 128, 16, 16) and latent.ZU.shape == (1, 128, 8)
    assert torch.equal(latent.ZI, latent.FI) and torch.equal(latent.ZU, latent.FU)


def _draws(config_name, count=600):
    """The image corruption parameters drawn for ``count`` frames of separate trajectories."""
    image_corruptor, _ = build_corruptors(load_config(f"configs/{config_name}.yaml"))
    return [image_corruptor._parameters("train", 0, f"trajectory/{index}", 0.0, "full") for index in range(count)]


def test_p27_blurs_a_little_more_and_darkens_fewer():
    p26, p27 = load_config("configs/kaggle_ijepa.yaml"), load_config("configs/kaggle_blur.yaml")
    changed = {key for key in {*p26["corruption"]["image"], *p27["corruption"]["image"]}
               if p26["corruption"]["image"].get(key) != p27["corruption"]["image"].get(key)}
    # Downsample and motion blur stay p26's: only the defocus grows.
    assert changed == {"defocus_probability", "defocus_sigma_px", "exposure_gain", "env_clear_probability",
                       "illum_max_brighten_stops"}
    for section in ("data", "model", "phase1", "phase2"):
        assert p26[section] == p27[section], section
    # Corruption is in both hashes: p27 retrains both phases.
    for phase in ("phase1", "phase2"):
        assert configuration_hash(p26, phase) != configuration_hash(p27, phase)
    stats = {}
    for name in ("kaggle_ijepa", "kaggle_blur"):
        draws = [draw for draw in _draws(name) if not draw["clean"]]
        defocus = [draw["defocus_sigma"] for draw in draws if draw["defocus"]]
        stats[name] = {"rate": len(defocus) / len(draws), "sigma": sum(defocus) / len(defocus),
                       "clear": sum(draw.get("env_clear", False) for draw in draws) / len(draws),
                       "gain": max(draw["exposure_gain"] for draw in draws)}
    old, new = stats["kaggle_ijepa"], stats["kaggle_blur"]
    # The user: less blur than p27's first two tries, still more than p26's (measured 0.31 -> 0.42, 0.51 -> 0.58 px).
    assert old["rate"] + 0.04 < new["rate"] < old["rate"] + 0.15
    assert old["sigma"] + 0.02 < new["sigma"] < old["sigma"] + 0.1
    assert new["clear"] > old["clear"] + 0.08 and old["gain"] <= 0.9 < 0.95 < new["gain"] <= 1.0


def _run_until_done(arguments, last):
    restarts = 0
    while True:
        try:
            main([*arguments, *(["--resume", str(last)] if last.exists() else [])])
            return restarts
        except SystemExit as stop:
            assert stop.code == RESTART_EXIT_CODE
            restarts += 1


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_ijepa_trains_resumes_and_evaluates_through_the_cli(tmp_path):
    torch.set_num_threads(1)
    root, manifest, output = tmp_path / "dataset", tmp_path / "manifest", tmp_path / "run"
    _write_dataset(root)
    config = _ijepa_dict(max_successful_updates=2, batch_size=2)
    config["phase2"]["max_successful_updates"] = 2
    config["runtime"]["restart_above_rss_gib"] = 1e-6          # restart after every checkpoint
    path = tmp_path / "ijepa.yaml"
    path.write_text(yaml.safe_dump(config))
    common = ["--config", str(path), "--manifest", str(manifest), "--output", str(output)]
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest)])
    assert _run_until_done(["train-phase1", *common], output / "phase1/last.pt") == 1
    phase1 = load_checkpoint(output / "phase1/last.pt")
    assert phase1["successful_updates"] == 2 and phase1["metadata"]["latent_gate_status"] == "PASS"
    assert phase1["metadata"]["phase1_objective"] == "ijepa"
    updates = [record for record in _records(output / "phase1/train.jsonl") if "loss" in record]
    assert len(updates) == 2 and all("variance" not in record for record in updates)
    gates = [record for record in _records(output / "phase1/train.jsonl") if "latent_gate_status" in record]
    assert gates and all("validation_jepa_image" in record for record in gates)

    last = output / "phase2/last.pt"
    _run_until_done(["train-phase2", *common, "--backbone-checkpoint", str(output / "phase1/last.pt")], last)
    payload = load_checkpoint(last)
    assert payload["successful_updates"] == 2
    vit = {key: value for key, value in payload["system"].items() if key.startswith("backbone.joint_encoder.")}
    assert vit and all(torch.equal(value, phase1["model"][key]) for key, value in vit.items())
    evaluation = output / "eval"
    main(["evaluate", "--checkpoint", str(last), "--manifest", str(manifest), "--device", "cpu",
          "--output", str(evaluation), "--panels", "0", "--split", "valid"])
    assert json.loads((evaluation / "metrics.json").read_text())["requested"]["image_count"] > 0


def _phase1_logged(run, key):
    return [record[key] for record in _records(run / "phase1/train.jsonl")
            if key in record and "successful_updates" in record and "loss" in record]


def test_ijepa_on_two_processes_trains_what_one_does(tmp_path):
    torch.set_num_threads(1)
    root, manifest = tmp_path / "dataset", tmp_path / "manifest"
    _write_dataset(root)
    config = _ijepa_dict(max_successful_updates=3, batch_size=2)
    config["phase2"]["batch_size"] = 2
    single = tmp_path / "single.yaml"
    single.write_text(yaml.safe_dump(config))
    config["runtime"].update(parallel="ddp", gpu_count=2)
    ddp = tmp_path / "ddp.yaml"
    ddp.write_text(yaml.safe_dump(config))
    main(["build-manifest", "--config", str(single), "--data-root", str(root), "--output", str(manifest)])
    finals = {}
    for name, path in (("single", single), ("ddp", ddp)):
        output = tmp_path / name
        main(["train-phase1", "--config", str(path), "--manifest", str(manifest), "--output", str(output)])
        payload = load_checkpoint(output / "phase1/last.pt")
        assert payload["successful_updates"] == 3 and payload["metadata"]["latent_gate_status"] == "PASS"
        finals[name] = payload["model"]
    assert json.loads((tmp_path / "ddp/phase1/execution.json").read_text())["backend"] == "ddp"
    for key in ("loss", "jepa_image", "jepa_imu", "gradient_norm"):
        ddp_values, single_values = _phase1_logged(tmp_path / "ddp", key), _phase1_logged(tmp_path / "single", key)
        assert len(ddp_values) == len(single_values) == 3, key
        assert ddp_values[0] == pytest.approx(single_values[0], rel=1e-5), key
        assert ddp_values == pytest.approx(single_values, rel=2e-3), key
    drift = 2.0 * sum(_phase1_logged(tmp_path / "single", "learning_rate"))
    for key, value in finals["single"].items():
        if value.is_floating_point():
            assert torch.allclose(value, finals["ddp"][key], atol=max(drift, 1e-6)), key
