"""Random previews rotate whole paired samples across consecutive runs; --glare-config tests any
checkpoint, one trained before the glare existed too, on frames with lamps and glare."""

import numpy as np
import pytest
import torch
import yaml

from qjepa.cli import main
from qjepa.config import LIGHT_KEYS, load_config, serializable_config
from test_kaggle_workflow import _write_dataset
from test_sharp_cli import _run_until_done, _sharp_smoke
from tools.glare_metrics import format_report, measure, region_metrics
from tools.random_pair_preview import choose_indices, preview, with_glare


def test_next_preview_avoids_previous_pair_ids_when_pool_allows():
    ids = [f"frame-{index}" for index in range(12)]
    eligible = list(range(12))
    rng = np.random.default_rng(7)
    first = choose_indices(ids, eligible, 4, set(), rng)
    second = choose_indices(ids, eligible, 4, {ids[index] for index in first}, rng)

    assert len(first) == len(second) == 4
    assert len(set(first)) == len(set(second)) == 4
    assert set(first).isdisjoint(second)


def test_preview_seed_replays_same_selection():
    ids = [f"frame-{index}" for index in range(10)]
    draw = lambda: choose_indices(ids, list(range(10)), 4, set(), np.random.default_rng(91))
    assert draw() == draw()


def test_with_glare_adds_the_light_keys_to_a_copy_and_forces_the_probability():
    config = serializable_config(load_config("configs/kaggle_gray.yaml"))
    glared = with_glare(config, "configs/kaggle_glare.yaml", 1.0)
    assert "light_probability" not in config["corruption"]["image"]          # the checkpoint's recipe untouched
    assert glared["corruption"]["image"]["light_probability"] == 1.0
    assert set(LIGHT_KEYS) <= set(glared["corruption"]["image"])
    with pytest.raises(ValueError, match="light_"):
        with_glare(config, "configs/kaggle_gray.yaml")


def test_the_glare_preview_tests_a_checkpoint_trained_without_glare(tmp_path):
    torch.set_num_threads(1)
    root, manifest, output = tmp_path / "dataset", tmp_path / "manifest", tmp_path / "run"
    _write_dataset(root)
    config = _sharp_smoke()
    assert not any(key.startswith("light_") for key in config["corruption"]["image"])
    path = tmp_path / "plain.yaml"
    path.write_text(yaml.safe_dump(config))
    common = ["--config", str(path), "--manifest", str(manifest), "--output", str(output)]
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest)])
    _run_until_done(["train-phase1", *common], output / "phase1/last.pt")
    last = output / "phase2/last.pt"
    _run_until_done(["train-phase2", *common, "--backbone-checkpoint", str(output / "phase1/last.pt")], last)
    before = last.read_bytes()
    report = preview(last, manifest, output=tmp_path / "glare", state=tmp_path / "state.json", count=2,
                     image_mode="full", device="cpu", seed=3, glare_config="configs/kaggle_glare.yaml")
    assert report["items"] and all(item["image_corruption"]["light"] and item["image_corruption"]["light_params"]
                                   for item in report["items"])
    assert (tmp_path / "glare" / report["items"][0]["panel"]).is_file()
    plain = preview(last, manifest, output=tmp_path / "plain", state=tmp_path / "state2.json", count=2,
                    image_mode="full", device="cpu", seed=3)
    assert all("light" not in item["image_corruption"] for item in plain["items"])
    # Cell 14e: the same checkpoint measured with and without glare on the same frames.
    report = measure(last, manifest, glare_config="configs/kaggle_glare.yaml", count=2, frames="random",
                     image_mode="full", device="cpu")
    assert len(report["rows"]) == 2 and report["trained_light_probability"] == 0.0
    for row in report["rows"]:
        assert row["light"]
        for column in ("glare_input", "glare_output", "plain_input", "plain_output"):
            assert np.isfinite(row[column]["image_psnr_db"]) and 0 <= row[column]["dark_fraction"] <= 1
    text = format_report(report)
    assert "KHÔNG" in text and text.rstrip().endswith("=== hết ===")
    assert last.read_bytes() == before


def test_region_metrics_measure_the_dark_the_bright_the_halo_and_blown_pixels():
    clean = torch.zeros(1, 3, 8, 8)
    clean[..., :4] = 0.05                                   # dark left half
    clean[..., 4:] = 0.8                                    # bright right half
    halo = torch.zeros(8, 8, dtype=torch.bool)
    halo[:, 2:6] = True
    perfect = region_metrics(clean, clean, halo)
    assert perfect["dark_mae255"] == perfect["bright_mae255"] == perfect["halo_mae255"] == 0
    assert perfect["dark_brightness"] == pytest.approx(1) and perfect["bright_brightness"] == pytest.approx(1)
    assert perfect["dark_fraction"] == perfect["bright_fraction"] == perfect["halo_fraction"] == 0.5
    glared = clean.clone()
    glared[..., 2:6] += 0.3                                 # light spilled over the halo columns
    glared[..., 7] = 1.0                                    # one blown-out column
    measured = region_metrics(glared, clean, halo)
    # Halo: +0.3 on the two dark columns, clipped to 1 (+0.2) on the two bright ones -> mean 0.25.
    assert measured["dark_brightness"] > 1 and measured["halo_mae255"] == pytest.approx(0.25 * 255, rel=1e-3)
    # Blown out: column 7, and the two bright halo columns clipped to 1.
    assert measured["white_fraction"] == pytest.approx(3 / 8) and measured["white_fraction_clean"] == 0
