"""p47_restore and p46_nolatent: the gyro blurs the frame and the decoder reads the gyro's path, harder and more
real data, the restore_rgb decoder, and the control run without the JEPA latent.

Pins:
- the corruption: a wide blur kernel goes through the FFT and gives ndimage's values; an item whose IMU is borrowed is
  blurred by that window's own motion around that window's frame time (motion_time), at its pixel scale
  (focal_scale); a real low-light pair keeps its own grain (no sensor stage) and counts as a dark frame; with
  exposure_follows_light a frame the low-light stage leaves alone gets the shortest exposure; a gyro path under a
  pixel leaves a frame sharp in the kind label;
- the data: data.source_scale_range reads each frame at a drawn scale before the crop, the same for a sample in a
  realization, and focal_scale follows; real low-light pairs (data.lowlight_pairs_*) come in their slots of every
  training batch, the dark crop in and the bright crop out, and without the keys the batches are p46's;
- the model: GyroBlurField integrates the noisy gyro into the same path motion.exposure_path gives the blur, marks a
  window without IMU, and follows a mirrored IMU; RestoreRGBDecoder starts as the frame without its flare and its
  gyro channels reach the output; phase2.image_latent none hands the decoder zeros;
- the configs: kaggle_restore.yaml is one flat file, refuses the split decoder's stages and a gyro path without the
  gyro blur; kaggle_nolatent.yaml keeps p46's phase-1 and flare-stage hashes; no older config's hash moved (pinned
  for p46); both phases train through the CLI with restore_rgb, the gyro path, scaled crops and low-light pairs and
  print the "LOL valid" line, and phase 2 trains with image_latent none.
"""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest
import torch
from PIL import Image
from scipy import ndimage

from qjepa.cli import _dataset, _train_dataset, _training_batch_stream, lowlight_pair_bank, main
from qjepa.config import build_decoders, load_config, validate_config
from qjepa.corruptions.image import (FFT_KERNEL_SIDE, LowLightImageCorruptionConfig, LowLightImageCorruptor,
                                     convolve_reflect, frame_kind, log_kind)
from qjepa.corruptions.motion import exposure_path, imu_blur_kernel
from qjepa.data.dataset import LOWLIGHT_DRAW_BASE, SCENE_DRAW_BASE, mirror_imu
from qjepa.data.scene_pairs import find_scene_pairs_root, is_scene_pairs_root
from qjepa.models.decoders import RestoreRGBDecoder
from qjepa.models.motion_field import GyroBlurField
from qjepa.training.checkpoints import configuration_hash, load_checkpoint
from test_scene_pairs import _scene_smoke, _train_tiny, _write_scene_pairs

LOW_SHORT_SIDE = 40
LOWLIGHT = {"lowlight_pairs_fraction": 0.25, "lowlight_pairs_short_side": LOW_SHORT_SIDE}


def _write_lowlight_pairs(root, train=12, valid=4, size=(44, 60)):
    """A lowlight-pairs folder: index.csv with kind = lowlight, gt/ (bright) and input/ (the same scene, dark)."""
    (root / "gt").mkdir(parents=True)
    (root / "input").mkdir()
    height, width = size
    rows = []
    for index in range(train + valid):
        name = f"lol_{index:04d}.png"
        bright = np.zeros((height, width, 3), dtype=np.uint8)
        bright[..., 0] = np.linspace(60, 230, width, dtype=np.uint8)[None, :]
        bright[..., 1] = np.linspace(40, 200, height, dtype=np.uint8)[:, None]
        bright[..., 2] = 90 + index
        Image.fromarray(bright).save(root / "gt" / name)
        Image.fromarray((bright // 6).astype(np.uint8)).save(root / "input" / name)
        split = "train" if index < train else "val"
        rows.append({"file": name, "split": split, "scene": f"{index}.png", "scene_split": split, "flare": "lol_v1",
                     "kind": "lowlight"})
    with (root / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def _gyro_window(length=128, rate=100.0, seed=0):
    rng = np.random.default_rng(seed)
    times = 1_750_000_000.0 + np.arange(length) / rate
    gyro = np.cumsum(rng.normal(0.0, 0.2, (length, 3)), axis=0) * 0.2 + np.array([0.1, 0.6, -0.9])
    return gyro, times


# --- corruption ------------------------------------------------------------------------------------------------------

def test_wide_kernels_go_through_the_fft_and_give_ndimage_s_values():
    rng = np.random.default_rng(1)
    image = rng.random((40, 52, 3))
    for side in (FFT_KERNEL_SIDE + 2, 21, 33):
        kernel = rng.random((side, side))
        kernel /= kernel.sum()
        expected = np.stack([ndimage.convolve(image[..., c], kernel, mode="reflect") for c in range(3)], axis=-1)
        assert np.abs(convolve_reflect(image, kernel) - expected).max() < 1e-10
    small = rng.random((5, 5))
    assert np.array_equal(convolve_reflect(image, small),
                          np.stack([ndimage.convolve(image[..., c], small, mode="reflect") for c in range(3)], -1))


def _gyro_corruptor(**change):
    values = dict(motion_from_imu=True, clean_probability=0.0, defocus_probability=0.0, downsample_probability=0.0,
                  noise_only_probability=0.0, low_light_only_probability=0.0, focal_length_px=320.0,
                  exposure_seconds=(0.03, 0.06), motion_max_radius_px=40.0)
    values.update(change)
    return LowLightImageCorruptor(LowLightImageCorruptionConfig(**values), master_seed=5)


def test_a_borrowed_imu_blurs_with_its_own_motion_at_the_items_pixel_scale():
    gyro, times = _gyro_window()
    centre = float(times[64])
    corruptor = _gyro_corruptor()
    clean = np.full((48, 48, 3), 0.5, dtype=np.float32)
    shared = dict(split="train", realization=0, trajectory="pair", timestamp=0.0, frame_index=0, gyro=gyro,
                  imu_times=times)
    _, borrowed = corruptor(clean, **shared, motion_time=centre)
    kernel, report = imu_blur_kernel(gyro, times, centre, borrowed["exposure_seconds"], focal_length_px=320.0,
                                     max_radius_px=40.0)
    assert borrowed["path_span_px"] == pytest.approx(report["path_span_px"])
    assert borrowed["path_span_px"] > 2.0
    # Without motion_time the window is read at timestamp 0, long before it: np.interp holds the first sample.
    _, stale = corruptor(clean, **shared)
    held = np.repeat(gyro[:1], len(gyro), axis=0)
    _, report = imu_blur_kernel(held, times, 0.0, stale["exposure_seconds"], focal_length_px=320.0,
                                max_radius_px=40.0)
    assert stale["path_span_px"] == pytest.approx(report["path_span_px"])
    # A frame read at 0.4 of the size the focal length is given for: 0.4 of the pixels.
    _, scaled = corruptor(clean, **shared, motion_time=centre, focal_scale=0.4)
    assert scaled["path_span_px"] == pytest.approx(0.4 * borrowed["path_span_px"], rel=1e-6)


def test_a_real_low_light_photo_gets_the_camera_blur_but_no_grain():
    gyro, times = _gyro_window()
    corruptor = _gyro_corruptor(jpeg_probability=1.0)
    rng = np.random.default_rng(3)
    dark = (rng.random((48, 48, 3)) * 0.2).astype(np.float32)
    info = {"name": "x", "source": "lowlight/lol_v1", "clear": True, "grain": False, "lowlight": True}
    out, params = corruptor(dark, split="train", realization=0, trajectory="lol", timestamp=0.0, frame_index=0,
                            gyro=gyro, imu_times=times, flare_pair=info, motion_time=float(times[64]))
    assert params["sensor_noise"] is False and params["low_light"] is False and params["motion"]
    kernel, _ = imu_blur_kernel(gyro, times, float(times[64]), params["exposure_seconds"], focal_length_px=320.0,
                                max_radius_px=40.0)
    assert np.abs(out - np.clip(convolve_reflect(dark.astype(np.float64), kernel), 0, 1)).max() < 1e-6
    assert frame_kind(params) == "relit" and log_kind(params) == "toi_mo"
    # A FlareImage pair (no "grain" key) keeps its grain, and stays a flared frame.
    _, flared = corruptor(dark, split="train", realization=0, trajectory="lol", timestamp=0.0, frame_index=0,
                          gyro=gyro, imu_times=times, flare_pair={"name": "x", "source": "halo", "clear": True})
    assert flared["sensor_noise"] is True and frame_kind(flared) == "flare"


def test_a_frame_in_enough_light_gets_the_shortest_exposure_and_a_sub_pixel_path_is_sharp():
    gyro, times = _gyro_window()
    clean = np.full((48, 48, 3), 0.5, dtype=np.float32)
    shared = dict(split="train", realization=0, timestamp=float(times[64]), frame_index=0, gyro=gyro, imu_times=times)
    for follows in (False, True):
        corruptor = _gyro_corruptor(exposure_follows_light=follows, noise_only_probability=0.5,
                                    exposure_seconds=(0.004, 0.066))
        exposures = {}
        for trajectory in range(40):
            _, params = corruptor(clean, trajectory=f"t{trajectory}", **shared)
            exposures.setdefault(params["low_light"], []).append(params["exposure_seconds"])
        assert exposures[True] and exposures[False]
        if follows:
            assert set(exposures[False]) == {0.004} and min(exposures[True]) > 0.004
        else:
            assert max(exposures[False]) > 0.004                       # as before: drawn with the darkness
    # The label: a gyro path under a pixel leaves the frame 'nhe'; the drawn blur of earlier runs keeps its label.
    base = {"mode": "full", "clean": False, "low_light": False, "sensor_noise": True, "defocus": False}
    assert log_kind({**base, "motion": True, "motion_from_imu": True, "path_span_px": 0.6}) == "nhe"
    assert log_kind({**base, "motion": True, "motion_from_imu": True, "path_span_px": 1.5}) == "mo"
    assert log_kind({**base, "motion": True, "motion_from_imu": False, "motion_length": 2}) == "mo"


# --- data ------------------------------------------------------------------------------------------------------------

def _restore_smoke(tmp_path, *, lowlight=True, **data):
    config, manifest, manifest_dir = _scene_smoke(tmp_path, phase1_batch=8, phase2_batch=4)
    if lowlight:
        root = tmp_path / "lowlight"
        if not root.exists():
            _write_lowlight_pairs(root)
        config["data"].update(LOWLIGHT, lowlight_pairs_root=str(root))
    config["data"].update(data)
    image = config["corruption"]["image"]
    image.update(motion_from_imu=True, focal_length_px=32.0, exposure_seconds=[0.01, 0.066], motion_max_radius_px=8.0)
    phase2 = config["phase2"]
    for key in [key for key in phase2 if key.startswith(("split_", "image_resnet"))]:
        del phase2[key]
    phase2.update(image_decoder="restore_rgb", image_latent="zi", encoder_skips=False,
                  decoder_input="fused_dense_latent_only", decoder_predictor_input=False,
                  restore_widths=[8, 12, 16], restore_enc_blocks=[1, 1], restore_middle_blocks=1,
                  restore_dec_blocks=[1, 1], restore_fourier_levels=[0], restore_fourier_width=4,
                  restore_tone_width=8, restore_tone_weight=1.0, restore_fft_weight=1.0,
                  motion_field={"seconds": 0.066, "samples": 5, "focal_px": 24.0, "scale_px": 2.0, "max_px": 8.0})
    validate_config(config)
    return config, manifest, manifest_dir


def test_frames_come_at_a_drawn_scale_and_the_blur_follows_it(tmp_path):
    config, manifest, _ = _restore_smoke(tmp_path, lowlight=False, source_size=[64, 64],
                                         source_scale_range=[0.5, 1.0])
    dataset = _train_dataset(config, manifest, "phase2")
    sample = dataset.samples[0]
    scale, (top, left) = dataset.source_geometry(sample)
    assert 0.5 <= scale <= 1.0 and dataset.source_geometry(sample) == (scale, (top, left))
    from qjepa.data.dataset import load_rgb
    side = max(32, round(64 * scale))
    whole = load_rgb(sample.image_path, (side, side))
    assert np.array_equal(dataset._clean_frame(sample), whole[top:top + 32, left:left + 32])
    assert dataset._focal_scale(sample) == pytest.approx(scale)
    scales = set()
    for realization in range(6):
        dataset.set_realization(realization)
        scales.add(round(dataset.source_geometry(sample)[0], 6))
    assert len(scales) > 3
    # A pair photo read at short side S counts as a frame of S across its short side.
    assert dataset._focal_scale(bank=dataset.scene_bank) == pytest.approx(dataset.scene_bank.short_side / 64)
    for bad in ([0.2, 1.0], [0.0, 0.5], [0.9, 0.5], [0.5, 1.2]):
        broken = json.loads(json.dumps(config))
        broken["data"]["source_scale_range"] = bad
        with pytest.raises(ValueError, match="source_scale_range"):
            validate_config(broken)


def test_low_light_pairs_fill_their_slots_dark_crop_in_bright_crop_out(tmp_path):
    config, manifest, _ = _restore_smoke(tmp_path)
    bank = lowlight_pair_bank(config)
    assert len(bank.pairs("train")) == 12 and len(bank.pairs("valid")) == 4
    assert is_scene_pairs_root(tmp_path / "lowlight", "lowlight") and not is_scene_pairs_root(tmp_path / "lowlight")
    assert find_scene_pairs_root(tmp_path, kind="lowlight") == tmp_path / "lowlight"
    assert find_scene_pairs_root(tmp_path) == tmp_path / "visdrone"        # VisDrone's finder no longer sees two
    dataset = _train_dataset(config, manifest, "phase2")
    item = dataset[len(dataset.samples) + LOWLIGHT_DRAW_BASE + 3]
    assert item["sample_id"].startswith("lowlight/") and item["corruption"]["image"]["sensor_noise"] is False
    assert float(item["image_clean"].mean()) > 3 * float(item["image_noisy"].mean())
    stream = _training_batch_stream(config, dataset, 8, start_microbatch=0, namespace="t")
    for _ in range(3):
        batch = next(stream)
        ids = batch["sample_id"]
        assert sum(i.startswith("lowlight/") for i in ids) == 2
        assert sum(i.startswith("scenepair/") for i in ids) == 2
        assert sum(i.startswith("flareimage/") for i in ids) == 2
    # Without the low-light keys: exactly p46's batches.
    plain_config, plain_manifest, _ = _restore_smoke(tmp_path, lowlight=False)
    with_low = _training_batch_stream(config, _train_dataset(config, manifest, "phase2"), 8, start_microbatch=0,
                                      namespace="t")
    without = _training_batch_stream(plain_config, _train_dataset(plain_config, plain_manifest, "phase2"), 8,
                                     start_microbatch=0, namespace="t")
    first = next(without)["sample_id"]
    assert not any(i.startswith("lowlight/") for i in first) and len(first) == 8
    assert next(with_low)["sample_id"] != first


# --- model -----------------------------------------------------------------------------------------------------------

def test_the_gyro_path_is_the_one_the_corruption_blurs_with():
    gyro, times = _gyro_window()
    centre = float(times[60])
    field = GyroBlurField(seconds=0.066, samples=9, focal_px=200.0, scale_px=8.0, max_px=64.0, reference_width=256)
    imu = np.concatenate((np.tile([0.0, 0.0, 9.81], (len(gyro), 1)), gyro), axis=1)
    phys = torch.tensor(imu.T[None], dtype=torch.float32)
    stamps = torch.tensor(times[None], dtype=torch.float64)
    u, v, present = field.path(phys, stamps, torch.tensor([centre], dtype=torch.float64), 256)
    u_ref, v_ref, _ = exposure_path(gyro, times, centre, 0.066, focal_length_px=200.0, samples=9)
    assert present.item() == 1.0
    assert np.abs(u[0].numpy() - u_ref).max() < 1e-3 and np.abs(v[0].numpy() - v_ref).max() < 1e-3
    assert np.abs(u_ref).max() > 1.0                               # a real path, not zeros compared with zeros
    # Twice as wide a frame: twice the pixels.
    u2, _, _ = field.path(phys, stamps, torch.tensor([centre], dtype=torch.float64), 512)
    assert torch.allclose(u2, 2 * u, atol=1e-4)
    # A mirrored IMU (dataset.mirror_imu) mirrors the path: u flips, v stays.
    mirrored = torch.tensor(mirror_imu(imu).T[None], dtype=torch.float32)
    um, vm, _ = field.path(mirrored, stamps, torch.tensor([centre], dtype=torch.float64), 256)
    assert torch.allclose(um, -u, atol=1e-4) and torch.allclose(vm, v, atol=1e-4)
    # No IMU (the all-zero window): zeros and the flag 0.
    values = field(torch.zeros_like(phys), stamps, torch.tensor([centre], dtype=torch.float64), 256)
    assert values.shape == (1, field.channels) and float(values.abs().sum()) == 0.0


def test_restore_rgb_starts_as_its_input_and_its_gyro_channels_reach_the_output():
    torch.manual_seed(0)
    decoder = RestoreRGBDecoder(16, naf={"widths": (8, 12, 16), "enc_blocks": (1, 1), "middle_blocks": 1,
                                         "dec_blocks": (1, 1), "fourier_levels": (0,), "fourier_width": 4},
                                tone_width=8,
                                motion={"seconds": 0.066, "samples": 5, "focal_px": 24.0, "scale_px": 2.0,
                                        "max_px": 8.0, "reference_width": 32})
    image = torch.rand(2, 3, 32, 32)
    latent = torch.randn(2, 16, 8, 8)
    motion = torch.randn(2, decoder.motion.channels)
    out, parts = decoder(latent, image, motion=motion)
    # GlobalToneColor floors x at 1e-4 before its curve; nothing else moves at update 0.
    assert torch.allclose(out, image, atol=1e-4) and torch.allclose(parts["image_light"], image, atol=1e-4)
    # Once the net's last layer moved, a different gyro path gives a different frame.
    torch.nn.init.normal_(decoder.net.ending.weight, std=0.1)
    for block in decoder.net.modules():
        if hasattr(block, "beta"):
            torch.nn.init.normal_(block.beta, std=0.1)
    first, _ = decoder(latent, image, motion=motion)
    second, _ = decoder(latent, image, motion=motion + 1.0)
    assert not torch.allclose(first, second)
    first.mean().backward()
    assert decoder.net.intro.weight.grad[:, 6:].abs().sum() > 0      # the gyro channels get a gradient
    assert decoder.tone.head[-1].weight.grad.abs().sum() > 0


def test_image_latent_none_hands_the_decoder_zeros_and_the_control_keeps_p46_s_hashes():
    p46, control = load_config("configs/kaggle_visdrone.yaml"), load_config("configs/kaggle_nolatent.yaml")
    assert configuration_hash(control, "phase1") == configuration_hash(p46, "phase1")
    assert configuration_hash(control, "flare") == configuration_hash(p46, "flare")
    assert configuration_hash(control, "phase2") != configuration_hash(p46, "phase2")
    assert build_decoders(control).image_latent == "none" and build_decoders(p46).image_latent == "zi"
    from qjepa.models.pipeline import RestorationSystem
    from types import SimpleNamespace
    system = RestorationSystem.__new__(RestorationSystem)
    torch.nn.Module.__init__(system)
    system.decoders, system.latent_predictor = SimpleNamespace(image_latent="none"), None
    latent = SimpleNamespace(ZI=torch.randn(2, 4, 3, 3))
    assert torch.equal(system.image_latent(latent), torch.zeros(2, 4, 3, 3))
    system.decoders.image_latent = "zi"
    assert torch.equal(system.image_latent(latent), latent.ZI)


# --- configs ---------------------------------------------------------------------------------------------------------

# Computed with the code before p47 (commit 555652f): the keys of p47 left every older config's hashes as they were.
P46_HASHES = {"phase1": "bdf0c6608611", "phase2": "ebfcf2bacf30", "flare": "004fda07d1c6"}


def test_kaggle_restore_is_one_flat_file_and_refuses_what_restore_rgb_does_not_have():
    p46 = load_config("configs/kaggle_visdrone.yaml")
    assert {phase: configuration_hash(p46, phase)[:12] for phase in P46_HASHES} == P46_HASHES
    text = open("configs/kaggle_restore.yaml", encoding="utf-8").read()
    assert "\nextends:" not in text and not text.startswith("extends")
    config = load_config("configs/kaggle_restore.yaml")
    phase2, image, data = config["phase2"], config["corruption"]["image"], config["data"]
    assert phase2["image_decoder"] == "restore_rgb" and image["motion_from_imu"] is True
    assert not any(key.startswith(("split_color", "split_edge", "split_tone", "split_relight")) for key in phase2)
    assert image["exposure_gain"] == [0.08, 1.0] and data["source_scale_range"] == [0.4, 1.0]
    assert configuration_hash(config, "phase1") != P46_HASHES["phase1"]
    for change, match in (({"split_tone_grid": True}, "split_color_edge decoder|none of the split"),
                          ({"split_tone_gate_lr_scale": 10}, "split_tone_gate: true|none of the split"),
                          ({"split_edge_refiner_blocks": 2}, "none of the split"),
                          ({"restore_fft_weight": -1}, "restore_fft_weight"),
                          ({"motion_field": {"seconds": 0.066}}, "motion_field")):
        broken = json.loads(json.dumps(config))
        broken["phase2"].update(change)
        with pytest.raises(ValueError, match=match):
            validate_config(broken)
    broken = json.loads(json.dumps(config))
    broken["corruption"]["image"]["motion_from_imu"] = False
    with pytest.raises(ValueError, match="motion_from_imu"):
        validate_config(broken)
    broken = json.loads(json.dumps(config))
    del broken["data"]["lowlight_pairs_short_side"]
    with pytest.raises(ValueError, match="lowlight_pairs"):
        validate_config(broken)
    # The low-light folder is a path: out of every hash, like the other datasets' roots.
    moved = json.loads(json.dumps(config))
    moved["data"]["lowlight_pairs_root"] = "/elsewhere"
    assert all(configuration_hash(moved, phase) == configuration_hash(config, phase)
               for phase in ("phase1", "phase2", "flare"))


# --- CLI -------------------------------------------------------------------------------------------------------------

def test_both_phases_train_with_restore_rgb_the_gyro_path_scaled_crops_and_low_light_pairs(tmp_path, capsys):
    torch.set_num_threads(1)
    config, _, manifest_dir = _restore_smoke(tmp_path, source_size=[48, 48], source_scale_range=[0.7, 1.0],
                                             imu_absent_probability=0.25)
    output = _train_tiny(tmp_path, config, manifest_dir, "run")
    assert load_checkpoint(output / "phase2/last.pt")["successful_updates"] == 3
    rows = [json.loads(line) for line in (output / "phase2/train.jsonl").read_text().splitlines()]
    losses = [row for row in rows if "loss" in row]
    assert losses and all(math.isfinite(row["loss"]) for row in losses)
    assert all(math.isfinite(row["image_fft_l1"]) and math.isfinite(row["image_light_l1"]) for row in losses)
    checks = [row for row in rows if "validation_image_psnr_db" in row]
    assert checks and all(row["lowlight_pairs_validation_count"] == 4 for row in checks)
    printed = capsys.readouterr().out
    assert "LOL valid n=4" in printed and "VisDrone valid" in printed
    import subprocess
    import sys
    report = subprocess.run([sys.executable, "tools/training_report.py", "--run", str(output)], capture_output=True,
                            text=True)
    assert report.returncode == 0, report.stderr[-2000:]
    assert "LOL valid" in report.stdout and "VisDrone valid" in report.stdout
    image = tmp_path / "frame.png"
    Image.fromarray(np.full((32, 32, 3), 60, dtype=np.uint8)).save(image)
    main(["infer", "--checkpoint", str(output / "phase2/last.pt"), "--image", str(image), "--output",
          str(tmp_path / "alone"), "--device", "cpu"])
    assert (tmp_path / "alone/image_restored.png").is_file()


def test_phase2_trains_without_the_latent_on_the_split_decoder(tmp_path):
    torch.set_num_threads(1)
    config, _, manifest_dir = _scene_smoke(tmp_path)
    config["phase2"]["image_latent"] = "none"
    validate_config(config)
    output = _train_tiny(tmp_path, config, manifest_dir, "nolatent")
    payload = load_checkpoint(output / "phase2/last.pt")
    assert payload["successful_updates"] == 3 and payload["config"]["phase2"]["image_latent"] == "none"
