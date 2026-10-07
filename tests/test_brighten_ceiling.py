"""corruption.image.illum_max_brighten_stops: the uneven light brightens thinly, never like a flash.

The user (p27): "giam loe sang cua anh lai, mong thoi, loe sang qua thi mat dac trung se khoi phuc" --
regions lit +1.5-3.5 stops and rolled off towards white wiped out what phase 2 should restore. Pins: a
config without the key renders and reports exactly as before (ceiling 3 stops) and setting it moves no
draw, only adds itself to the illumination parameters; no spot is brightened past the ceiling, while
every darkened spot and every spot under the ceiling keeps its old value; it counts as optional, so
older configs that spell the illum_* keys out still validate; bad values fail; p27 sets 0.75.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from qjepa.config import load_config, validate_config
from qjepa.corruptions.image import LowLightImageCorruptionConfig, LowLightImageCorruptor
from qjepa.corruptions.light import illumination_field

ILLUM = {key: value for key, value in load_config("configs/kaggle_local.yaml")["corruption"]["image"].items()
         if key.startswith("illum_")}


def _frame(size=96):
    rng = np.random.default_rng(5)
    return np.clip(0.4 + 0.3 * rng.standard_normal((size, size, 3)), 0.0, 1.0)


def _corruptor(**values):
    return LowLightImageCorruptor(LowLightImageCorruptionConfig(clean_probability=0.0, **values), 73128)


def _call(corruptor, image, index):
    return corruptor(image, split="train", realization=0, trajectory="t", timestamp=0.1 * index,
                     frame_index=index, mode="full")


def test_absent_key_renders_as_before_and_setting_it_moves_no_draw():
    image = _frame()
    old, capped = _corruptor(**ILLUM), _corruptor(**ILLUM, illum_max_brighten_stops=0.75)
    explicit = _corruptor(**ILLUM, illum_max_brighten_stops=3.0)
    changed = 0
    for index in range(12):
        (a, pa), (b, pb), (c, _) = (_call(x, image, index) for x in (old, capped, explicit))
        assert "max_brighten_stops" not in (pa.get("illumination_params") or {})
        assert np.array_equal(a, c)                                     # absent = the old ceiling of 3 stops
        if pa["illumination"]:
            drawn = dict(pb["illumination_params"])
            assert drawn.pop("max_brighten_stops") == 0.75 and drawn == pa["illumination_params"]
            changed += not np.array_equal(a, b)
        assert {key: value for key, value in pb.items() if key != "illumination_params"} == \
            {key: value for key, value in pa.items() if key != "illumination_params"}
    assert changed > 0


def test_no_spot_brightens_past_the_ceiling_and_the_dark_keeps_its_value():
    regions = {"gradient_stops": 2.5, "gradient_angle": 0.7, "smudges": [],
               "blobs": [{"y": 0.3, "x": 0.3, "sigma": 0.2, "stops": 3.5},
                         {"y": 0.7, "x": 0.7, "sigma": 0.2, "stops": -3.0}]}
    old = illumination_field(64, 64, regions)
    capped = illumination_field(64, 64, dict(regions, max_brighten_stops=0.75))
    assert np.log2(old).max() > 2.0                                     # the flash the user saw
    assert np.log2(capped).max() == pytest.approx(0.75, abs=1e-5)
    below = np.log2(old) <= 0.75
    assert below.mean() > 0.5 and np.array_equal(capped[below], old[below])   # darkened and dim spots unchanged


def test_older_configs_still_validate_and_bad_values_fail():
    for name in ("kaggle_local", "kaggle_ijepa"):
        validate_config(load_config(f"configs/{name}.yaml"))           # illum_* spelled out, no ceiling
    for bad in (0.0, -1.0, 3.5, True, "1"):
        with pytest.raises(ValueError, match="illum_max_brighten_stops"):
            LowLightImageCorruptionConfig(illum_max_brighten_stops=bad).validate()
        config = copy.deepcopy(load_config("configs/kaggle_blur.yaml"))
        config["corruption"]["image"]["illum_max_brighten_stops"] = bad
        with pytest.raises(ValueError, match="illum_max_brighten_stops"):
            validate_config(config)


def test_p27_lights_thinly():
    config = load_config("configs/kaggle_blur.yaml")
    assert config["corruption"]["image"]["illum_max_brighten_stops"] == 0.75
    assert "illum_max_brighten_stops" not in load_config("configs/kaggle_ijepa.yaml")["corruption"]["image"]
