"""p43: phase2.target_brighten -- a dark phase-2 TARGET is brightened, the input left as it is.

User (14/10/2026): "toi nghi do groundtruth chua manh, them vao anh train nao toi thi lam sang no len".

Pins: brighten_dark leaves a bright frame exactly as it is, never darkens, keeps the order of every pixel's values,
lifts a dark frame towards the target by at most max_gain and never pushes the brightest channel's 99.5th percentile
past 0.97 when that frame did not already; in phase 2 (train and validation) a dark target is brightened while the input,
and every phase-1 and flare-stage item, stays exactly as before; a brightened item's tone label says "light changed"
and its stop map is the corruption's minus log2(gain); absent, nothing changes; p43 is p42 plus the key, with p40's
phase-1 hash and p42's flare-stage hash, p42's hashes unchanged; bad settings fail.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from qjepa.cli import _dataset, _train_dataset
from qjepa.config import load_config, serializable_config, validate_config
from qjepa.corruptions.light import LUMA_709, brighten_dark
from qjepa.training.checkpoints import configuration_hash
from test_bilateral import GRID
from test_exposure_standardize import RELIGHT
from test_flare_pairs import _smoke
from test_tone_gate import GATE

KEY = {"below": 0.9, "target": 0.95, "max_gain": 2.0}      # every smoke frame counts as dark


def test_brighten_dark_lifts_dark_frames_only_and_keeps_highlights():
    rng = np.random.default_rng(0)
    bright = rng.uniform(0.35, 0.9, (48, 48, 3)).astype(np.float32)
    same, gain = brighten_dark(bright, below=0.3, target=0.4, max_gain=3.0)
    assert gain == 1.0 and np.array_equal(same, bright)
    dark = rng.uniform(0.0, 0.2, (48, 48, 3)).astype(np.float32)
    out, gain = brighten_dark(dark, below=0.3, target=0.4, max_gain=3.0)
    assert 1.0 < gain <= 3.0
    assert (out @ LUMA_709).mean() > (dark @ LUMA_709).mean()
    assert np.all(out >= dark - 1e-6)                                         # never darker
    order = np.argsort(dark.ravel(), kind="stable")
    assert np.all(np.diff(out.ravel()[order]) >= -1e-6)                       # same order of values
    # A dark frame with one lamp: the gain stops before the lamp would clip.
    lamp = dark.copy()
    lamp[:4, :4] = 0.9
    out, gain = brighten_dark(lamp, below=0.3, target=0.4, max_gain=3.0)
    assert np.percentile(out.max(axis=-1), 99.5) <= 0.97 + 1e-4 and gain < 3.0


def _config(tmp_path, **phase2):
    config, manifest, _, _ = _smoke(tmp_path, phase1_batch=4, phase2_batch=4)
    config["phase2"].update(GRID, **GATE, **RELIGHT, **phase2)
    validate_config(config)
    return config, manifest


def test_phase2_targets_are_brightened_the_inputs_and_other_stages_are_not(tmp_path):
    plain, manifest = _config(tmp_path)
    bright, _ = _config(tmp_path, target_brighten=KEY)
    before, after = _train_dataset(plain, manifest, "phase2"), _train_dataset(bright, manifest, "phase2")
    brightened = 0
    for index in range(8):
        a, b = before[index], after[index]
        assert np.array_equal(a["image_noisy"], b["image_noisy"])            # the input stays
        assert bool((b["image_clean"] >= a["image_clean"] - 1e-6).all())       # the target only brightens
        gain = b["corruption"]["image"]["target_gain"]
        if gain > 1.0:
            brightened += 1
            assert float(b["image_tone_label"]) == 1.0
            assert np.allclose(b["image_stops"].numpy(), a["image_stops"].numpy() - np.log2(gain), atol=1e-5)
        else:
            assert np.array_equal(a["image_clean"], b["image_clean"])
    assert brightened > 0
    # Validation brightens too (phase 2 scores against the same targets it trains on).
    valid = _dataset(bright, manifest, "valid", fixed_realization=True)
    assert valid.target_brighten == KEY
    # Phase 1 and the flare stage keep the plain frame.
    one_a, one_b = _train_dataset(plain, manifest, "phase1"), _train_dataset(bright, manifest, "phase1")
    assert one_b.target_brighten is None
    for index in range(3):
        assert np.array_equal(one_a[index]["image_clean"], one_b[index]["image_clean"])
    assert _dataset(bright, manifest, "train", fixed_realization=False, target_brighten=False).target_brighten is None
    # Absent: no gain recorded, items as before.
    assert "target_gain" not in before[0]["corruption"]["image"]


def test_kaggle_brightgt_is_p42_plus_the_key_reusing_p40_phase1_and_p42_flare_stage():
    p40 = load_config("configs/kaggle_flarefocus.yaml")
    p42, p43 = load_config("configs/kaggle_flarestage.yaml"), load_config("configs/kaggle_brightgt.yaml")
    validate_config(p43)
    assert configuration_hash(p43, "phase1") == configuration_hash(p40, "phase1")
    assert configuration_hash(p43, "flare") == configuration_hash(p42, "flare")
    assert configuration_hash(p43, "phase2") != configuration_hash(p42, "phase2")
    assert configuration_hash(p42, "phase2").startswith("e90ef9e5e0f7")      # p42: 12000-update fp32 flare stage
    a, b = (serializable_config(copy.deepcopy(config)) for config in (p42, p43))
    assert b["runtime"].pop("output_dir").endswith("outputs/p43_brightgt")
    a["runtime"].pop("output_dir")
    assert b["phase2"].pop("target_brighten") == {"below": 0.30, "target": 0.40, "max_gain": 3.0}
    assert a == b


@pytest.mark.parametrize("value, message", [
    ({"below": 0.3, "target": 0.4}, "exactly"),
    ({"below": 0.5, "target": 0.4, "max_gain": 3.0}, "below <= target"),
    ({"below": 0.3, "target": 0.4, "max_gain": 1.0}, "max_gain"),
    ("yes", "exactly"),
])
def test_bad_settings_fail(value, message):
    config = load_config("configs/kaggle_brightgt.yaml")
    config["phase2"]["target_brighten"] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_the_preview_tool_draws_images_and_dataset_frames(tmp_path, capsys):
    import yaml
    from PIL import Image
    from tools.brighten_preview import main as preview
    folder = tmp_path / "anh"
    folder.mkdir()
    rng = np.random.default_rng(1)
    Image.fromarray((rng.uniform(0, 0.15, (40, 50, 3)) * 255).astype(np.uint8)).save(folder / "toi.png")
    Image.fromarray((rng.uniform(0.4, 0.9, (40, 50, 3)) * 255).astype(np.uint8)).save(folder / "sang.png")
    preview(["--images", str(folder), "--output", str(tmp_path / "a.png")])
    out = capsys.readouterr().out
    assert (tmp_path / "a.png").stat().st_size > 1000 and "1/2 ảnh được làm sáng" in out
    config, _, _, manifest_dir = _smoke(tmp_path, phase1_batch=4, phase2_batch=4)
    config["phase2"]["target_brighten"] = KEY
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(config))
    preview(["--config", str(path), "--manifest", str(manifest_dir), "--count", "3", "--stats", "8",
             "--output", str(tmp_path / "b.png")])
    out = capsys.readouterr().out
    assert (tmp_path / "b.png").stat().st_size > 1000 and "được làm sáng" in out and "Thống kê " in out
