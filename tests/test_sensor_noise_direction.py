"""(a) encoder_sensitivity.noise_direction: sensor_noise -- stop teaching the encoder to ignore blur.

The Jacobian term penalises the encoder's gain along the corruption direction.
With noise_direction "corruption" that direction is (noisy - clean), and on a
blurred frame it is the frame's own detail with the sign flipped (cos -0.97 with
the signal direction): the term asked the encoder to be deaf to edges. With
"sensor_noise" it is (noisy - the same frame rendered without sensor grain), the
grain alone. Pins: the reference render is the noisy one minus the grain, with
every other stage identical; the noisy output is bit-identical to __call__; on a
blurred frame the old direction points against the detail and the new one does
not; the trainer reads the reference only when asked; bad values are refused.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from qjepa.cli import _synthetic_batch
from qjepa.config import build_phase1_model, load_config, seed_everything, validate_config
from qjepa.corruptions import LowLightImageCorruptor
from qjepa.corruptions.image import LowLightImageCorruptionConfig
from qjepa.data import ImuNormalizer
from qjepa.training.checkpoints import configuration_hash
from qjepa.training.phase1 import Phase1Trainer, image_noise_reference
from qjepa.training.sensitivity import corruption_direction, detail_direction

KWARGS = dict(split="train", realization=0, trajectory="env/P000", timestamp=12.3, frame_index=4)


def _texture(size=48, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.random((size // 4, size // 4, 3))
    image = np.kron(base, np.ones((4, 4, 1)))
    return (0.2 + 0.6 * image).astype(np.float32)


def _config(direction=None):
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    if direction is not None:
        config["encoder_sensitivity"]["noise_direction"] = direction
    validate_config(config)
    return config


@pytest.mark.parametrize("mode", ["full", "sensor_noise_only", "blur_only", "low_light_only", "clean"])
def test_the_noisy_render_is_unchanged_and_the_reference_lacks_only_the_grain(mode):
    corruptor = LowLightImageCorruptor(replace(LowLightImageCorruptionConfig(), clean_probability=0.0))
    image = _texture()
    expected, expected_params = corruptor(image, mode=mode, **KWARGS)
    noisy, reference, params = corruptor.render_with_sensor_reference(image, mode=mode, **KWARGS)
    assert np.array_equal(noisy, expected) and params == expected_params
    grain = mode == "sensor_noise_only" or (mode == "full" and params["sensor_noise"])
    assert (not np.array_equal(noisy, reference)) == grain
    if mode == "sensor_noise_only":
        # Only quantization (and JPEG) of the clean frame: no blur, no darkening.
        assert float(np.abs(reference - image).mean()) < 0.03


def test_on_a_blurred_frame_only_the_sensor_direction_stops_pointing_against_the_detail():
    config = replace(LowLightImageCorruptionConfig(), clean_probability=0.0, defocus_probability=1.0,
                     defocus_sigma_px=(1.2, 1.2), motion_probability=0.0, downsample_probability=0.0,
                     jpeg_probability=0.0, noise_only_probability=1.0, photon_count=(20000.0, 20000.0),
                     quantization_bits=(8, 8))
    corruptor = LowLightImageCorruptor(config)
    clean, noisy, reference = [], [], []
    for index in range(4):
        image = _texture(seed=index)
        corrupted, without_grain, _ = corruptor.render_with_sensor_reference(
            image, **{**KWARGS, "frame_index": index})
        for store, value in ((clean, image), (noisy, corrupted), (reference, without_grain)):
            store.append(torch.from_numpy(value.transpose(2, 0, 1)))
    clean, noisy, reference = (torch.stack(v) for v in (clean, noisy, reference))
    detail, _ = detail_direction(clean)
    old, _ = corruption_direction(clean, noisy)
    new, valid = corruption_direction(reference, noisy)
    cosine = lambda a, b: torch.nn.functional.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)
    assert bool(valid.all())
    assert float(cosine(old, detail).mean()) < -0.5
    assert abs(float(cosine(new, detail).mean())) < 0.1


def test_the_trainer_reads_the_reference_only_when_asked():
    batch = _synthetic_batch(_config())
    assert image_noise_reference(batch, _config()["encoder_sensitivity"]) is batch["image_clean"]
    sensor = _config("sensor_noise")
    assert image_noise_reference(batch, sensor["encoder_sensitivity"]) is batch["image_noise_free"]
    seed_everything(0)
    model = build_phase1_model(sensor, ImuNormalizer())
    metrics = Phase1Trainer(model, sensor, torch.device("cpu")).step(batch)      # update 0: the image probe
    assert metrics["skipped"] is False and metrics["encoder_source"] == "image"
    missing = {key: value for key, value in batch.items() if key != "image_noise_free"}
    with pytest.raises(KeyError, match="image_noise_free"):
        image_noise_reference(missing, sensor["encoder_sensitivity"])


def test_a_bad_direction_is_refused_and_the_direction_is_in_the_phase1_hash_only():
    with pytest.raises(ValueError, match="noise_direction"):
        _config("random")
    assert configuration_hash(_config("sensor_noise"), "phase1") != configuration_hash(_config(), "phase1")
    assert configuration_hash(_config("sensor_noise"), "phase2") == configuration_hash(_config(), "phase2")
