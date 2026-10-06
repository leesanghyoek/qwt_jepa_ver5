"""phase2.backbone_weights -- which phase-1 encoder phase 2 freezes: context (online) or target (EMA teacher).

I-JEPA evaluates with its target encoder; here phase 2 reads corrupted inputs, which only the context
encoder trained on, so both are on offer and the choice is measured on Kaggle. Pins: the key changes the
phase-2 hash and not the phase-1 one, so one phase 1 serves both arms; absent, phase 2 freezes the online
encoder as before; target puts the teacher's weights in -- the joint ViT, or a CNN backbone's two encoders
with its fusion left online; the phase-1 checkpoint names both hashes and the phase-2 checkpoint the one it
froze, through a resume; p26_ijepa_target is p26_ijepa with this key alone; bad values are refused.
"""

from __future__ import annotations

import copy

import pytest
import torch
import yaml

from qjepa.cli import main
from qjepa.config import build_phase1_model, load_config, phase2_backbone, seed_everything, validate_config
from qjepa.data import ImuNormalizer
from qjepa.training.checkpoints import configuration_hash, load_checkpoint, state_dict_hash, target_backbone_hash
from test_ijepa import _ijepa, _ijepa_dict, _run_until_done
from test_kaggle_workflow import _write_dataset


def _model(config):
    seed_everything(0)
    model = build_phase1_model(config, ImuNormalizer())
    with torch.no_grad():                        # a teacher that has drifted from the online encoder
        for parameter in model.teachers.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    return model


def _states(module):
    return {key: value.clone() for key, value in module.state_dict().items()}


def test_target_puts_the_teachers_joint_vit_in_the_backbone():
    config = _ijepa()
    model = _model(config)
    online, teacher = _states(model.backbone.joint_encoder), _states(model.teachers.joint_encoder)
    assert any(not torch.equal(online[key], teacher[key]) for key in online)
    # Absent: the online encoder, as every run so far.
    assert phase2_backbone(config, model) is model.backbone
    assert all(torch.equal(value, online[key]) for key, value in model.backbone.joint_encoder.state_dict().items())
    expected = target_backbone_hash(model)
    target = copy.deepcopy(config)
    target["phase2"]["backbone_weights"] = "target"
    backbone = phase2_backbone(target, model)
    assert all(torch.equal(value, teacher[key]) for key, value in backbone.joint_encoder.state_dict().items())
    assert state_dict_hash(backbone) == expected


def test_target_on_a_cnn_backbone_takes_both_encoders_and_keeps_the_fusion():
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    model = _model(config)
    fusion = _states(model.backbone.fusion)
    teachers = {name: _states(getattr(model.teachers, name)) for name in ("image_encoder", "imu_encoder")}
    config["phase2"]["backbone_weights"] = "target"
    validate_config(config)
    backbone = phase2_backbone(config, model)
    for name, state in teachers.items():
        assert all(torch.equal(value, state[key]) for key, value in getattr(backbone, name).state_dict().items())
    assert all(torch.equal(value, fusion[key]) for key, value in backbone.fusion.state_dict().items())


def test_the_key_moves_the_phase2_hash_only_and_bad_values_fail():
    context, target = load_config("configs/kaggle_ijepa.yaml"), load_config("configs/kaggle_ijepa_target.yaml")
    assert configuration_hash(context, "phase1") == configuration_hash(target, "phase1")
    assert configuration_hash(context, "phase2") != configuration_hash(target, "phase2")
    differ = {(section, key) for section in ("data", "model", "corruption", "phase1", "phase2", "runtime")
              for key in {*context[section], *target[section]}
              if context[section].get(key) != target[section].get(key)}
    assert differ == {("phase2", "backbone_weights"), ("runtime", "output_dir")}
    bad = _ijepa_dict()
    bad["phase2"]["backbone_weights"] = "ema"
    with pytest.raises(ValueError, match="backbone_weights"):
        validate_config(bad)


def test_one_phase1_serves_both_arms_through_the_cli(tmp_path):
    torch.set_num_threads(1)
    root, manifest = tmp_path / "dataset", tmp_path / "manifest"
    _write_dataset(root)
    config = _ijepa_dict(max_successful_updates=2, batch_size=2)
    config["phase2"]["max_successful_updates"] = 2
    config["runtime"]["restart_above_rss_gib"] = 1e-6          # restart after every checkpoint
    paths = {}
    for arm in ("context", "target"):
        arm_config = copy.deepcopy(config)
        if arm == "target":
            arm_config["phase2"]["backbone_weights"] = "target"
        paths[arm] = tmp_path / f"{arm}.yaml"
        paths[arm].write_text(yaml.safe_dump(arm_config))
    main(["build-manifest", "--config", str(paths["context"]), "--data-root", str(root), "--output", str(manifest)])
    phase1_run = tmp_path / "phase1_run"
    _run_until_done(["train-phase1", "--config", str(paths["context"]), "--manifest", str(manifest),
                     "--output", str(phase1_run)], phase1_run / "phase1/last.pt")
    phase1 = load_checkpoint(phase1_run / "phase1/last.pt")
    assert phase1["metadata"]["target_backbone_hash"] != phase1["metadata"]["backbone_hash"]
    for arm, hash_key, weights in (("context", "backbone_hash", "backbone.joint_encoder."),
                                   ("target", "target_backbone_hash", "teachers.joint_encoder.")):
        output = tmp_path / arm
        last = output / "phase2/last.pt"
        # The target arm runs on the context arm's phase 1: its config differs in phase 2 only.
        _run_until_done(["train-phase2", "--config", str(paths[arm]), "--manifest", str(manifest),
                         "--output", str(output), "--backbone-checkpoint", str(phase1_run / "phase1/last.pt")], last)
        payload = load_checkpoint(last)
        assert payload["successful_updates"] == 2 and payload["metadata"]["backbone_weights"] == arm
        assert payload["metadata"]["frozen_backbone_hash"] == phase1["metadata"][hash_key]
        frozen = {key[len("backbone.joint_encoder."):]: value for key, value in payload["system"].items()
                  if key.startswith("backbone.joint_encoder.")}
        assert frozen and all(torch.equal(value, phase1["model"][weights + key]) for key, value in frozen.items())
        main(["evaluate", "--checkpoint", str(last), "--manifest", str(manifest), "--device", "cpu",
              "--output", str(output / "eval"), "--panels", "0", "--split", "valid", "--max-batches", "1"])
