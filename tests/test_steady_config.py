"""configs/kaggle_steady.yaml (p28_steady): p27_blur's corruption and architecture, a steadier phase 1 and a
phase 2 that learns harder and penalises blur a little more.

The user (p28): phase 1's loss fell then rose; phase 2 learns but does not restore sharp, "tang cac thong so hoc va
phat len cao 1 chut". Pins: p28 differs from p27 only in phase1.batch_size and the phase-2 learning rate, gradient
clip and sharpness weights (and its output directory), so a run of it is p27 with that recipe; the phase-1 batch
is a whole global batch that splits over two GPUs; every sharpness weight goes up by half and nothing else in the
loss moves -- the roughness penalty stays, so no stripes are bought; both hashes change, as phase 1 retrains.
"""

from __future__ import annotations

from qjepa.config import load_config, validate_config
from qjepa.training.checkpoints import configuration_hash

RECIPE = {
    ("phase1", "batch_size"): (8, 32),
    ("phase2", "learning_rate"): (2.0e-4, 3.0e-4),
    ("phase2", "gradient_clip_norm"): (5.0, 10.0),
    ("phase2", "reconstruction_detail_weight"): (2.0, 3.0),
    ("phase2", "detail_energy_weight"): (1.0, 1.5),
    ("phase2", "split_edge_weight"): (1.0, 1.5),
    ("phase2", "split_gradient_weight"): (1.0, 1.5),
    ("phase2", "split_edge_fft_weight"): (1.0, 1.5),
    ("phase2", "perceptual_weight"): (0.5, 0.75),
}
SHARPNESS = {"reconstruction_detail_weight", "detail_energy_weight", "split_edge_weight", "split_gradient_weight",
             "split_edge_fft_weight", "perceptual_weight"}


def test_p28_is_p27_with_the_recipe_changes_only():
    p27, p28 = load_config("configs/kaggle_blur.yaml"), load_config("configs/kaggle_steady.yaml")
    validate_config(p28)
    differ = {(section, key) for section in ("data", "model", "corruption", "phase1", "phase2", "monitor", "runtime")
              for key in {*p27.get(section, {}), *p28.get(section, {})}
              if p27.get(section, {}).get(key) != p28.get(section, {}).get(key)}
    assert differ == set(RECIPE) | {("runtime", "output_dir")}
    for (section, key), (old, new) in RECIPE.items():
        assert (p27[section][key], p28[section][key]) == (old, new), key
    for phase in ("phase1", "phase2"):
        assert configuration_hash(p27, phase) != configuration_hash(p28, phase)


def test_the_batch_is_global_and_the_sharpness_weights_rise_by_half():
    p27, p28 = load_config("configs/kaggle_blur.yaml"), load_config("configs/kaggle_steady.yaml")
    batch = p28["phase1"]["batch_size"]
    assert batch >= 8 and batch % 2 == 0                      # latent gate needs B >= 8; two GPUs share it
    for key in SHARPNESS:
        assert p28["phase2"][key] == 1.5 * p27["phase2"][key], key
    assert p28["phase2"]["split_edge_smooth_weight"] == p27["phase2"]["split_edge_smooth_weight"]
