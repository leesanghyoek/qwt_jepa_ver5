"""p50_noghost (configs/kaggle_noghost.yaml): p49_glint without the repo's procedural ghosts in the flare stage.

The user (29/10/2026, looking at the flare stage's training samples in Cell 5b) found the hexagonal ghosts "made by code
and unnatural". Pins:
- p50 is p49 with only lens_optics_probability 0, lens_wu_ghost_boost (0, 0) and the output directory changed; it
  validates, keeps p49's phase-1 hash (so Cell 4 reuses p49's phase 1), and its flare and phase-2 hashes differ;
- under p50 no flare layer of any source (Flare7K, Flare-R, Wu captured, Wu simulated) draws optics: no ghost chain, halo
  ring, veil, bloom or dirt, so the composite is the layer alone; under p49 every Wu layer drew a six-bladed chain.
"""

from __future__ import annotations

import copy

import numpy as np

from qjepa.config import load_config, serializable_config, validate_config
from qjepa.corruptions.lens_flare import WU_SOURCES, LensFlareBank, draw_lens_parameters, lens_flare_layers
from qjepa.data.flare_stage import lens_config
from qjepa.training.checkpoints import configuration_hash
from test_lens_flare import _write_flare7k
from test_wu_flares import _write_wu

P49 = load_config("configs/kaggle_glint.yaml")
P50 = load_config("configs/kaggle_noghost.yaml")


def test_p50_is_p49_without_procedural_optics_and_reuses_its_phase1():
    p49, p50 = copy.deepcopy(serializable_config(P49)), copy.deepcopy(serializable_config(P50))
    assert p50["runtime"]["output_dir"].endswith("/p50_noghost")
    assert p50["flare"]["lens"]["lens_optics_probability"] == 0.0
    assert p50["flare"]["lens"]["lens_wu_ghost_boost"] == [0.0, 0.0]
    assert p49["flare"]["lens"]["lens_optics_probability"] > 0
    assert p49["flare"]["lens"]["lens_wu_ghost_boost"][1] > 0
    for key in ("lens_optics_probability", "lens_wu_ghost_boost"):
        p50["flare"]["lens"][key] = p49["flare"]["lens"][key]
    p50["runtime"]["output_dir"] = p49["runtime"]["output_dir"]
    assert p50 == p49
    validate_config(P50)
    assert configuration_hash(P50, "phase1") == configuration_hash(P49, "phase1")
    for phase in ("flare", "phase2"):
        assert configuration_hash(P50, phase) != configuration_hash(P49, phase)


def test_p50_draws_no_ghost_on_any_layer_where_p49_drew_a_hexagonal_chain_on_every_wu_layer(tmp_path):
    bank = LensFlareBank(_write_flare7k(tmp_path / "flare7k"), holdout_fraction=0.1, wu_root=_write_wu(tmp_path / "wu"))
    old, new = lens_config(P49["flare"]["lens"]), lens_config(P50["flare"]["lens"])
    sources = set()
    for seed in range(300):
        drawn = draw_lens_parameters(np.random.default_rng(seed), new, bank, "train")
        sources.add(drawn["source"])
        assert drawn["optics"] is None
        before = draw_lens_parameters(np.random.default_rng(seed), old, bank, "train")
        if before["source"] in WU_SOURCES:
            assert before["optics"]["aperture"]["blades"] == 6 and before["optics"]["ghosts"]
    assert sources == {"Flare7K", "Flare-R", *WU_SOURCES}
    # Without optics the composite's layer is the flare photo alone: the same as drawing it with optics removed.
    drawn = draw_lens_parameters(np.random.default_rng(7), new, bank, "train")
    flare, light = lens_flare_layers(bank, drawn, 64, 64)
    again, _ = lens_flare_layers(bank, dict(drawn, optics=None), 64, 64)
    assert np.array_equal(flare, again)
