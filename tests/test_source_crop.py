"""data.source_size: train on crops of the frame at its own resolution, run on whole frames.

Pins: with the key, a sample is a crop of image_size taken from the frame read at source_size --
its own pixels, not a downscale -- at a place drawn per (sample, realization), so epochs see other
crops and the fixed validation realization always the same one; full_frame returns the whole
source frame; without the key the dataset reads at image_size exactly as before; bad sizes are
refused; configs/kaggle_local is p24_env plus the source size and the local-light corruption;
the recipe trains both phases through the CLI on crops and evaluates and previews whole frames.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
import yaml

from qjepa.cli import _dataset, main
from qjepa.config import load_config, serializable_config, validate_config
from qjepa.data.dataset import load_rgb
from qjepa.data.manifest import read_manifest
from qjepa.training.checkpoints import configuration_hash
from test_kaggle_workflow import _write_dataset
from test_sharp_cli import _run_until_done, _sharp_smoke


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("crop")
    _write_dataset(root / "dataset")
    config = _sharp_smoke()
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    main(["build-manifest", "--config", str(path), "--data-root", str(root / "dataset"), "--output", str(root / "m")])
    return read_manifest(root / "m"), config


def _with_source(config, size=(64, 64)):
    config = copy.deepcopy(config)
    config["data"]["source_size"] = list(size)
    return config


def test_a_sample_is_a_crop_of_the_frame_at_its_own_resolution_moving_with_the_realization(manifest):
    samples, config = manifest
    config = _with_source(config)
    dataset = _dataset(config, samples, "train", fixed_realization=False)
    sample = dataset.samples[0]
    frame = load_rgb(sample.image_path, (64, 64))
    origins = []
    for realization in range(6):
        dataset.set_realization(realization)
        clean = dataset[0]["image_clean"].numpy().transpose(1, 2, 0)
        top, left = dataset.crop_origin(sample)
        assert clean.shape == (32, 32, 3)
        assert np.array_equal(clean, frame[top:top + 32, left:left + 32])
        origins.append((top, left))
    assert len(set(origins)) > 1                                    # another crop in another epoch
    validation = _dataset(config, samples, "valid", fixed_realization=True)
    assert validation.crop_origin(validation.samples[0]) == validation.crop_origin(validation.samples[0])
    whole = _dataset(config, samples, "valid", fixed_realization=True, full_frame=True)
    assert tuple(whole[0]["image_clean"].shape) == (3, 64, 64) and tuple(whole[0]["image_noisy"].shape) == (3, 64, 64)


def test_without_the_key_the_dataset_reads_at_image_size_as_before(manifest):
    samples, config = manifest
    plain = _dataset(config, samples, "valid", fixed_realization=True)
    assert plain.source_size is None and plain.full_frame is False
    item = plain[0]
    assert np.array_equal(item["image_clean"].numpy().transpose(1, 2, 0),
                          load_rgb(plain.samples[0].image_path, tuple(config["data"]["image_size"])))
    # full_frame means nothing without a source size: still the image_size frame.
    assert tuple(_dataset(config, samples, "valid", fixed_realization=True, full_frame=True)[0]["image_clean"].shape) \
        == tuple(item["image_clean"].shape)


def test_bad_source_sizes_are_refused_and_kaggle_local_is_p24_plus_source_size_and_local_light():
    config = load_config("configs/kaggle_local.yaml")
    validate_config(config)
    assert config["data"]["source_size"] == [640, 640] and config["data"]["image_size"] == [256, 256]
    for value in ([200, 200], [650, 640], [640], "640", [640.0, 640]):
        bad = copy.deepcopy(config)
        bad["data"]["source_size"] = value
        with pytest.raises(ValueError, match="source_size"):
            validate_config(bad)
    env = serializable_config(load_config("configs/kaggle_env.yaml"))
    local = serializable_config(load_config("configs/kaggle_local.yaml"))
    for item in (env, local):
        item.pop("_config_path", None)
        item["runtime"].pop("output_dir")
    assert local["data"].pop("source_size") == [640, 640]
    changed = {"exposure_gain", "tone_gamma", "illum_strength", "illum_blobs", "illum_blob_size", "illum_gradient",
               "defocus_probability", "defocus_sigma_px", "downsample_probability", "motion_probability",
               "motion_length_px", "light_bloom_sigma_px"}
    for key in changed:
        assert local["corruption"]["image"].pop(key) != env["corruption"]["image"].pop(key), key
    assert local == env
    for phase in ("phase1", "phase2"):
        assert configuration_hash(load_config("configs/kaggle_env.yaml"), phase) != configuration_hash(config, phase)


def test_the_recipe_trains_on_crops_and_evaluates_and_previews_whole_frames(tmp_path):
    from tools.random_pair_preview import preview
    torch.set_num_threads(1)
    root, manifest_dir, output = tmp_path / "dataset", tmp_path / "manifest", tmp_path / "run"
    _write_dataset(root)
    config = _with_source(_sharp_smoke())
    path = tmp_path / "crop.yaml"
    path.write_text(yaml.safe_dump(config))
    common = ["--config", str(path), "--manifest", str(manifest_dir), "--output", str(output)]
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest_dir)])
    _run_until_done(["train-phase1", *common], output / "phase1/last.pt")
    last = output / "phase2/last.pt"
    _run_until_done(["train-phase2", *common, "--backbone-checkpoint", str(output / "phase1/last.pt")], last)
    evaluation = output / "eval_full"
    main(["evaluate", "--checkpoint", str(last), "--manifest", str(manifest_dir), "--device", "cpu", "--split", "valid",
          "--output", str(evaluation), "--panels", "0", "--full-frame"])
    assert json.loads((evaluation / "metrics.json").read_text())["requested"]["image_count"] > 0
    report = preview(last, manifest_dir, output=tmp_path / "panels", state=tmp_path / "state.json", count=1,
                     image_mode="full", device="cpu", seed=1)
    assert report["items"] and (tmp_path / "panels" / report["items"][0]["panel"]).is_file()
