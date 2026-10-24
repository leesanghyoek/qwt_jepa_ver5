"""p48 (phase1.ijepa_reconstruction_*): I-JEPA plus a decoder that draws the CLEAN frame from the noisy frame's latent.

The user chose (24/10/2026) to keep I-JEPA and add a reconstruction constraint, so that the latent keeps what it takes
to redraw the clean frame instead of only what the predictor needs. Pins: without the key there is no decoder, no
state-dict key and every old hash stays (p47's phase-1 hash is pinned); the decoder reads the dense ZI of the whole
context-input frame and nothing else, and its target is the clean frame as the encoder's transform reads it (Y in
luminance); the error is L1 plus the detail weight times L1 of the differences; one update adds weight x that error
to I-JEPA's loss, sends its gradient into the encoder and never into the EMA teacher, and the checkpoint says it
trained with reconstruction; the config refuses half-spelled or misplaced keys; p48 is p47 with only the phase-1 keys
and the output directory changed, so the flare stage and phase 2 hash as p47's; the CLI trains, restarts, resumes
(the decoder count carries on) and phase 2 loads the result; two DDP processes train what one does.
"""

from __future__ import annotations

import pytest
import torch
import yaml

from qjepa.cli import _synthetic_batch, main
from qjepa.config import IJEPA_RECONSTRUCTION_KEYS, load_config, serializable_config, validate_config
from qjepa.execution import IJEPAForward
from qjepa.models.ijepa import ReconstructionHead, reconstruction_error
from qjepa.training.checkpoints import configuration_hash, load_checkpoint, require_phase1_checkpoint
from qjepa.training.ijepa import IJEPATrainer
from test_ijepa import _ijepa, _ijepa_dict, _model, _phase1_logged, _records, _run_until_done
from test_kaggle_workflow import _write_dataset

RECONSTRUCTION = {"ijepa_reconstruction_weight": 1.0, "ijepa_reconstruction_width": 8,
                  "ijepa_reconstruction_detail_weight": 0.5}
# configuration_hash(kaggle_restore.yaml, "phase1") before p48 existed.
P47_PHASE1_HASH = "e87e8314cd5a213c9a3a52863a5d3209bf1770caf8615e66677a52632365a16b"


def _masks(model, batch):
    image_grid, imu_grid = model.token_grids(tuple(batch["image_noisy"].shape), int(batch["imu_noisy_phys"].shape[-1]))
    return model.sample_masks(image_grid, imu_grid, int(batch["image_noisy"].shape[0]), "test", 0)


def _forward(model, batch):
    return IJEPAForward(model)(batch["image_noisy"], batch["imu_noisy_phys"], batch["image_clean"],
                               batch["imu_clean_phys"], batch["image_time"], batch["imu_times"],
                               **_masks(model, batch))


def test_without_the_key_there_is_no_decoder_and_p47_keeps_its_hash():
    config = _ijepa()
    model = _model(config)
    assert model.reconstruction is None and not model.reconstructs
    assert not any(key.startswith("reconstruction.") for key in model.state_dict())
    assert "reconstruction_error" not in _forward(model, _synthetic_batch(config))
    assert configuration_hash(load_config("configs/kaggle_restore.yaml"), "phase1") == P47_PHASE1_HASH


@pytest.mark.parametrize("image_input, channels", [("rgb", 3), ("luminance", 1)])
def test_the_decoder_draws_the_clean_frame_from_the_noisy_latent_alone(image_input, channels):
    config = _ijepa_dict(**RECONSTRUCTION)
    config["model"]["image_input"] = image_input
    validate_config(config)
    model = _model(config).eval()
    batch = _synthetic_batch(config)
    batch["image_clean"] = batch["image_clean"].flip(-1)          # a clean frame the noisy one is not
    with torch.no_grad():
        latent = model.encode_online(batch["image_noisy"], batch["imu_noisy_phys"], batch["image_time"],
                                     batch["imu_times"])
        drawn = model.reconstruction(latent.ZI)
        target = model.reconstruction_target(batch["image_clean"])
        error = _forward(model, batch)["reconstruction_error"]
    assert drawn.shape == target.shape == (batch["image_clean"].shape[0], channels, *batch["image_clean"].shape[-2:])
    if image_input == "luminance":
        weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        assert torch.allclose(target, (batch["image_clean"] * weights).sum(1, keepdim=True), atol=1e-6)
    assert torch.allclose(error, reconstruction_error(drawn, target, 0.5), atol=1e-6)
    assert error.shape == (batch["image_clean"].shape[0],)
    # The head is a function of ZI alone: no skip, no input image on the side.
    head_inputs = []
    handle = model.reconstruction.register_forward_pre_hook(lambda module, args: head_inputs.append(args))
    with torch.no_grad():
        _forward(model, batch)
    handle.remove()
    assert len(head_inputs) == 1 and len(head_inputs[0]) == 1 and head_inputs[0][0].shape == latent.ZI.shape


def test_the_error_is_l1_plus_the_detail_weight_times_l1_of_the_differences():
    drawn = torch.zeros(2, 1, 4, 4)
    clean = torch.zeros(2, 1, 4, 4)
    clean[1, 0, :, 2:] = 1.0                       # a vertical edge in the second sample
    plain = reconstruction_error(drawn, clean, 0.0)
    assert torch.allclose(plain, torch.tensor([0.0, 0.5]))
    # the edge: one jump of 1 in each of 4 rows of 3 horizontal differences, none vertically
    assert torch.allclose(reconstruction_error(drawn, clean, 2.0), torch.tensor([0.0, 0.5 + 2.0 * 4 / 12]))
    head = ReconstructionHead(16, 8, 3)
    assert head(torch.zeros(1, 16, 2, 3)).shape == (1, 3, 32, 48)
    with pytest.raises(ValueError, match="power-of-two"):
        ReconstructionHead(16, 8, 3, stride=12)


def test_one_update_adds_the_term_trains_the_encoder_through_it_and_never_the_teacher():
    config = _ijepa(**RECONSTRUCTION)
    model = _model(config)
    batch = _synthetic_batch(config)
    # The term alone moves the online encoder, not the teacher.
    error = _forward(model, batch)["reconstruction_error"].mean()
    error.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.joint_encoder.parameters())
    assert all(p.grad is None for p in model.teachers.parameters())
    model.zero_grad(set_to_none=True)

    trainer = IJEPATrainer(model, config, torch.device("cpu"), "synthetic")
    head = {name: value.detach().clone() for name, value in model.reconstruction.named_parameters()}
    metrics = trainer.step(batch)
    assert metrics["skipped"] is False
    assert metrics["loss"] == pytest.approx(metrics["jepa"] + 1.0 * metrics["reconstruction"], rel=1e-6)
    assert metrics["jepa"] == pytest.approx(0.5 * (metrics["jepa_image"] + metrics["jepa_imu"]))
    assert any(not torch.equal(before, dict(model.reconstruction.named_parameters())[name])
               for name, before in head.items())
    payload = trainer.checkpoint_payload(config)
    assert payload["metadata"]["trained_with_reconstruction"] is True
    assert payload["metadata"]["phase1_decoder_forward_calls"] == 1
    require_phase1_checkpoint(payload)
    assert any(key.startswith("reconstruction.") for key in payload["model"])


@pytest.mark.parametrize("phase1, message", [
    ({"ijepa_reconstruction_weight": -1.0}, "nonnegative"),
    ({"ijepa_reconstruction_weight": 1.0, "ijepa_reconstruction_detail_weight": 0.5}, "width"),
    ({"ijepa_reconstruction_weight": 1.0, "ijepa_reconstruction_width": 8}, "detail_weight"),
    ({"ijepa_reconstruction_width": 8}, "positive phase1.ijepa_reconstruction_weight"),
    ({"ijepa_reconstruction_weight": 0.0, "ijepa_reconstruction_detail_weight": 0.5}, "positive"),
])
def test_the_config_refuses_half_spelled_keys(phase1, message):
    with pytest.raises(ValueError, match=message):
        _ijepa(**phase1)


def test_the_keys_belong_to_ijepa():
    config = serializable_config(load_config("configs/smoke.yaml"))
    config["phase1"]["ijepa_reconstruction_weight"] = 1.0
    with pytest.raises(ValueError, match="belongs to phase1.objective: ijepa"):
        validate_config(config)


def test_p48_is_p47_with_only_the_phase1_keys_changed():
    p47 = serializable_config(load_config("configs/kaggle_restore.yaml"))
    p48 = serializable_config(load_config("configs/kaggle_recon.yaml"))
    assert p48["runtime"]["output_dir"].endswith("/p48_recon")
    assert {key: p48["phase1"][key] for key in IJEPA_RECONSTRUCTION_KEYS} == {
        "ijepa_reconstruction_weight": 1.0, "ijepa_reconstruction_width": 64,
        "ijepa_reconstruction_detail_weight": 1.0}
    for key in IJEPA_RECONSTRUCTION_KEYS:
        del p48["phase1"][key]
    p48["runtime"]["output_dir"] = p47["runtime"]["output_dir"]
    assert p48 == p47
    old, new = load_config("configs/kaggle_restore.yaml"), load_config("configs/kaggle_recon.yaml")
    assert configuration_hash(old, "phase1") != configuration_hash(new, "phase1")
    for phase in ("phase2", "flare"):
        assert configuration_hash(old, phase) == configuration_hash(new, phase), phase


def test_the_cli_trains_restarts_resumes_and_phase2_loads_it(tmp_path):
    torch.set_num_threads(1)
    root, manifest, output = tmp_path / "dataset", tmp_path / "manifest", tmp_path / "run"
    _write_dataset(root)
    config = _ijepa_dict(max_successful_updates=2, batch_size=2, **RECONSTRUCTION)
    config["phase2"]["max_successful_updates"] = 1
    config["runtime"]["restart_above_rss_gib"] = 1e-6          # restart after every checkpoint
    path = tmp_path / "recon.yaml"
    path.write_text(yaml.safe_dump(config))
    common = ["--config", str(path), "--manifest", str(manifest), "--output", str(output)]
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest)])
    assert _run_until_done(["train-phase1", *common], output / "phase1/last.pt") >= 1
    phase1 = load_checkpoint(output / "phase1/last.pt")
    assert phase1["successful_updates"] == 2 and phase1["metadata"]["latent_gate_status"] == "PASS"
    # Both updates ran the decoder, across the restart.
    assert phase1["metadata"]["phase1_decoder_forward_calls"] == 2
    assert phase1["metadata"]["trained_with_reconstruction"] is True
    updates = [record for record in _records(output / "phase1/train.jsonl") if "loss" in record]
    assert len(updates) == 2 and all(record["reconstruction"] > 0 for record in updates)
    gates = [record for record in _records(output / "phase1/train.jsonl") if "latent_gate_status" in record]
    assert gates and all("validation_reconstruction" in record for record in gates)
    last = output / "phase2/last.pt"
    _run_until_done(["train-phase2", *common, "--backbone-checkpoint", str(output / "phase1/last.pt")], last)
    payload = load_checkpoint(last)
    assert payload["successful_updates"] == 1
    vit = {key: value for key, value in payload["system"].items() if key.startswith("backbone.joint_encoder.")}
    assert vit and all(torch.equal(value, phase1["model"][key]) for key, value in vit.items())
    assert not any("reconstruction" in key for key in payload["system"])   # phase 1's decoder stays behind


def test_two_processes_train_what_one_does(tmp_path):
    torch.set_num_threads(1)
    root, manifest = tmp_path / "dataset", tmp_path / "manifest"
    _write_dataset(root)
    config = _ijepa_dict(max_successful_updates=3, batch_size=2, **RECONSTRUCTION)
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
        finals[name] = load_checkpoint(output / "phase1/last.pt")["model"]
    for key in ("loss", "jepa", "reconstruction", "gradient_norm"):
        ddp_values, single_values = _phase1_logged(tmp_path / "ddp", key), _phase1_logged(tmp_path / "single", key)
        assert len(ddp_values) == len(single_values) == 3, key
        # The decoder's conv weight gradients sum 64 x 64 pixels in another order per rank: 1.8e-5 measured on the
        # first norm. A wrong scale of the term across ranks would be tens of percent.
        assert ddp_values[0] == pytest.approx(single_values[0], rel=1e-4 if key == "gradient_norm" else 1e-5), key
        assert ddp_values == pytest.approx(single_values, rel=2e-3), key
    drift = 2.0 * sum(_phase1_logged(tmp_path / "single", "learning_rate"))
    for key, value in finals["single"].items():
        if value.is_floating_point():
            assert torch.allclose(value, finals["ddp"][key], atol=max(drift, 1e-6)), key
