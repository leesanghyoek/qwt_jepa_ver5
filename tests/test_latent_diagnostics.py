"""D1 (latent oracle) and D2 (edge probe): helpers, then both tools end to end on a smoke run."""

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from qjepa.cli import main
from qjepa.config import load_config, serializable_config
from tools.edge_probe import cells, detail_targets, edge_probe
from tools.latent_oracle import finish_gap, latent_gap, oracle


def test_cells_follow_the_latent_grid_in_row_major_order():
    grid = torch.arange(6, dtype=torch.float32).view(1, 1, 2, 3)
    image = grid.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)   # 8 x 12, tile 4
    patches = cells(image, (2, 3))
    assert patches.shape == (1, 6, 16)
    for index in range(6):
        assert torch.all(patches[0, index] == index)
    # The latent itself is one value per cell, in the same order as its own flatten.
    assert torch.equal(cells(grid, (2, 3))[0, :, 0], grid.flatten())
    widened = cells(image, (2, 3), margin=2)
    assert widened.shape == (1, 6, 64)
    # Centre of the widened cell is still that cell.
    assert torch.all(widened[0, 4].view(8, 8)[2:6, 2:6] == 4)


def test_detail_bands_are_zero_on_a_flat_image_and_split_scales():
    flat = torch.full((1, 1, 32, 32), 0.4)
    for band in detail_targets(flat).values():
        assert band.abs().max() < 1e-6
    checker = torch.zeros(1, 1, 32, 32)
    checker[..., ::2, ::2] = checker[..., 1::2, 1::2] = 1.0           # 2 px period
    bands = detail_targets(checker)
    assert bands["fine"].abs().mean() > 5 * bands["edges"].abs().mean()


def test_latent_gap_is_zero_for_identical_latents_and_measures_other_frames():
    clean = torch.randn(3, 8, 2, 2)
    same = finish_gap(latent_gap(clean, clean))
    assert same["relative_distance"] == 0
    assert abs(same["cosine"] - 1) < 1e-6
    assert same["relative_distance_to_other_frame"] > 0.5
    assert "relative_distance_to_other_frame" not in finish_gap(latent_gap(clean[:1], clean[:1]))


def _write_dataset(root: Path) -> None:
    for split_index, split in enumerate(("train", "valid", "test")):
        path = root / split / f"env_{split}" / "Data_easy" / "P000"
        (path / "imu").mkdir(parents=True)
        (path / "image_lcam_front").mkdir()
        relative = np.arange(96) * 0.01
        times = 1_750_000_000 + relative
        imu = np.stack([np.sin(relative * (axis + 1) * 3 + split_index) for axis in range(6)], axis=-1)
        np.save(path / "imu/imu_time.npy", times)
        np.save(path / "imu/acc.npy", imu[:, :3])
        np.save(path / "imu/gyro.npy", imu[:, 3:])
        np.save(path / "imu/cam_time.npy", [(times[i] + times[i + 31]) / 2 for i in (0, 16, 32, 48)])
        rng = np.random.default_rng(split_index)
        for index in range(4):
            pixels = rng.integers(20, 240, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path / f"image_lcam_front/{index:06d}_lcam_front.png")


def test_both_tools_run_on_phase_checkpoints(tmp_path):
    torch.set_num_threads(1)
    root, manifest, output = tmp_path / "dataset", tmp_path / "manifest", tmp_path / "run"
    _write_dataset(root)
    config = serializable_config(load_config("configs/smoke.yaml"))
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config))
    common = ["--config", str(config_path), "--manifest", str(manifest), "--output", str(output)]
    main(["build-manifest", "--config", str(config_path), "--data-root", str(root), "--output", str(manifest)])
    main(["train-phase1", *common])
    main(["train-phase2", *common, "--backbone-checkpoint", str(output / "phase1/last.pt")])
    phase2 = output / "phase2/last.pt"

    report = oracle(phase2, manifest, output=tmp_path / "d1.json", samples=4, device="cpu")
    for scenario in ("blur_only", "full_full"):
        arms = report[scenario]["arms"]
        assert {"normal", "clean", "zero"} <= set(arms)
        assert all(np.isfinite(arm["image_psnr_restored_db"]) for arm in arms.values())
        # Same frames in every arm: the input never changes.
        assert len({arm["image_psnr_input_db"] for arm in arms.values()}) == 1
    assert json.loads((tmp_path / "d1.json").read_text())["spread_samples"] == 4

    for checkpoint in (output / "phase1/last.pt", phase2):
        result = edge_probe(checkpoint, manifest, output=tmp_path / "d2.json", samples=4,
                            cells_per_image=4, scenarios=("blur_only",), device="cpu")
        rows = result["blur_only"]
        assert {"input", "zi", "input+zi", "input+zi_clean"} <= set(rows)
        assert all(np.isfinite(rows[key][target]["recovered_pct"])
                   for key in rows for target in ("fine", "edges"))
