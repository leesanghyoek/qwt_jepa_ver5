"""p45_sunflare: Wu et al. 2021's daytime flare-only photos as flare layers, and the flare stage fine-tuned on from p44.

Pins: fetch_wu_flares lists the public bucket, shrinks each photo in linear light to a long side of ``side`` px and skips
what it already has; find_wu_flares finds a complete copy in Input; the bank adds "Wu-captured" (blocks of 10 frames)
and "Wu-simulated" (grouped by aperture: no aperture in two splits) beside Flare7K++, the light source is the near-
saturated core of the flare photo itself (the target keeps the sun), Wu layers take their own gain, and a config
without the keys draws exactly as before; every Wu layer gets a big ghost chain, hexagonal or round, and a halo ring (ghosts
as bright per pixel at any size), other layers keep their optics; a faint Wu layer is drawn again with its source, and
without lens_wu_min_flare draw_lens_flare is one draw as before; the flare stage's composites draw Wu layers of their own split; select_on
real_and_composite scores the mean PSNR gain of pairs and composites; train-flare with uformer_init stage starts from a
previous run's flare/best.pt (its sha256 recorded) and find_stage_checkpoint finds that file in Input or in a zip;
p45 is p44 plus its keys with p44's phase-1 hash, p44's hashes unchanged, the Wu folder out of every hash.
"""

from __future__ import annotations

import copy
import io
import json
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

from qjepa.config import load_config, serializable_config, validate_config
from qjepa.corruptions import lens_flare as lens_module
from qjepa.corruptions.lens_flare import (GHOST_REF_RADIUS, WU_DRAWS, LensFlareBank, WU_SOURCES, draw_lens_flare,
                                          draw_lens_parameters, lens_flare_layers, wu_light)
from qjepa.data import wu_flares
from qjepa.data.flare_stage import lens_config
from qjepa.models.uformer_flare import file_sha256, find_stage_checkpoint
from qjepa.training.checkpoints import configuration_hash, load_checkpoint
from qjepa.training.flare_stage import selection_score
from test_lens_flare import _write_flare7k

P44 = load_config("configs/kaggle_uformer.yaml")
P45 = load_config("configs/kaggle_sunflare.yaml")


def _flare_photo(index: int, shape=(30, 40)) -> np.ndarray:
    """A dim polygonal ghost plus, on even frames, a saturated light source (as a Wu capture: dark DC background)."""
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    photo = np.full(shape + (3,), 12, np.uint8)
    ghost = (rows - 20) ** 2 + (cols - 10) ** 2 <= 16
    photo[ghost] = (40, 60, 50)
    if index % 2 == 0:
        core = (rows - 8) ** 2 + (cols - 30) ** 2 <= 4
        photo[core] = 255
    return photo


def _write_wu(root, captured=60, apertures=12):
    (root / "captured").mkdir(parents=True)
    (root / "simulated").mkdir(parents=True)
    for index in range(captured):
        Image.fromarray(_flare_photo(index)).save(root / "captured" / f"frame_{index:04d}.png")
    for aperture in range(apertures):
        for blur in range(2):
            for crop in range(2):
                name = f"aperture{aperture:04d}_blur{blur:02d}_crop{crop:02d}.png"
                Image.fromarray(_flare_photo(aperture + blur)).save(root / "simulated" / name)
    return root


def test_the_light_source_is_the_saturated_core_and_dark_frames_have_none():
    bright, dark = _flare_photo(0), _flare_photo(1)
    light = wu_light(bright)
    assert light[8, 30].tolist() == [255, 255, 255]                  # the core kept
    assert light[20, 10].tolist() == [0, 0, 0] and light[0, 0].tolist() == [0, 0, 0]   # ghost and background gone
    assert int(wu_light(dark).max()) == 0                             # the source is outside this frame


def test_the_bank_adds_both_wu_sources_and_keeps_an_aperture_in_one_split(tmp_path):
    flare7k = _write_flare7k(tmp_path / "f7k")
    wu = _write_wu(tmp_path / "wu")
    bank = LensFlareBank(flare7k, holdout_fraction=0.2, wu_root=wu)
    assert bank.sources == ("Flare7K", "Flare-R") + WU_SOURCES and bank.has_wu()
    assert not LensFlareBank(flare7k, holdout_fraction=0.2).has_wu()
    where = {}
    for split in ("train", "valid", "test"):
        for name in bank.splits[split]["Wu-simulated"]:
            aperture = name.split("_")[0]
            assert where.setdefault(aperture, split) == split, aperture
        assert bank.splits[split]["Wu-captured"]
    flare, light = bank.load("Wu-captured", "frame_0000.png")
    assert flare.shape == light.shape == (40, 40, 3)                   # padded square as Flare-R
    assert int(light.max()) == 255 and int(flare.max()) == 255


def test_wu_draws_take_their_own_gain_and_a_config_without_the_keys_draws_as_before(tmp_path):
    bank = LensFlareBank(_write_flare7k(tmp_path / "f7k"), holdout_fraction=0.2, wu_root=_write_wu(tmp_path / "wu"))
    plain = lens_config(dict(P44["flare"]["lens"]))
    with_keys = lens_config(dict(P44["flare"]["lens"], lens_flare_wu_fraction=0.0))
    for seed in range(20):
        a = draw_lens_parameters(np.random.default_rng(seed), plain, bank, "train")
        b = draw_lens_parameters(np.random.default_rng(seed), with_keys, bank, "train")
        assert a == b and a["source"] not in WU_SOURCES
    every = lens_config(dict(P45["flare"]["lens"], lens_flare_wu_fraction=1.0))
    drawn = [draw_lens_parameters(np.random.default_rng(seed), every, bank, "train") for seed in range(200)]
    sources = [d["source"] for d in drawn]
    assert set(sources) == set(WU_SOURCES)
    assert sources.count("Wu-captured") / 200 == pytest.approx(0.7, abs=0.08)
    assert all(3.0 <= d["gain"] <= 10.0 and d["name"] in bank.splits["train"][d["source"]] for d in drawn)
    layer, light = lens_flare_layers(bank, drawn[0], 32, 32)
    assert np.isfinite(layer).all() and float(layer.max()) > 0 and (light <= layer + 1e-6).all()


def test_wu_layers_get_a_big_hexagonal_ghost_chain_and_other_layers_keep_their_optics(tmp_path):
    bank = LensFlareBank(_write_flare7k(tmp_path / "f7k"), holdout_fraction=0.2, wu_root=_write_wu(tmp_path / "wu"))
    lens = dict(P45["flare"]["lens"], lens_optics_probability=0.0)
    cfg = lens_config(lens)
    strength, boost, radii = lens["lens_ghost_strength"], lens["lens_wu_ghost_boost"], lens["lens_wu_ghost_radius"]
    seen, rounds = set(), []
    for seed in range(120):
        drawn = draw_lens_parameters(np.random.default_rng(seed), cfg, bank, "train")
        seen.add(drawn["source"] in WU_SOURCES)
        optics = drawn["optics"]
        if drawn["source"] not in WU_SOURCES:
            assert optics is None                                    # lens_optics_probability 0: none
            continue
        assert optics["aperture"]["blades"] == 6 and optics["halo"] is not None and optics["ghosts"]
        rounds += [ghost["round"] for ghost in optics["ghosts"]]
        for ghost in optics["ghosts"]:
            assert radii[0] <= ghost["radius"] <= radii[1]
            # As bright per pixel at any size: reflectance / area is lens_ghost_strength x lens_wu_ghost_boost.
            per_area = ghost["reflectance"] / (ghost["radius"] / GHOST_REF_RADIUS) ** 2
            assert strength[0] * boost[0] * (1 - 1e-9) <= per_area <= strength[1] * boost[1] * (1 + 1e-9)
    assert seen == {True, False}
    assert sum(rounds) / len(rounds) == pytest.approx(lens["lens_wu_ghost_round"], abs=0.1)
    # Boost off: a Wu layer takes the optics as any other layer, small ghosts, by lens_optics_probability.
    plain = lens_config(dict(lens, lens_wu_ghost_boost=[0.0, 0.0], lens_optics_probability=1.0))
    for seed in range(40):
        optics = draw_lens_parameters(np.random.default_rng(seed), plain, bank, "train")["optics"]
        assert all(0.015 <= ghost["radius"] <= 0.08 and "round" not in ghost for ghost in optics["ghosts"])
        assert all(ghost["reflectance"] <= strength[1] for ghost in optics["ghosts"])
    # A big ghost on a frame: its pixels are lit as brightly as a small one's (to the profile's edge).
    drawn = draw_lens_parameters(np.random.default_rng(3), lens_config(dict(lens, lens_flare_wu_fraction=1.0)), bank,
                                 "train")
    with_ghosts, _ = lens_flare_layers(bank, drawn, 64, 64)
    without, _ = lens_flare_layers(bank, dict(drawn, optics=None), 64, 64)
    assert float((with_ghosts - without).max()) > 0.05
    # A round ghost is drawn without the aperture's polygon.
    ghost = dict(drawn["optics"]["ghosts"][0], round=False, t=0.0, kind="disk")
    hexagon = dict(drawn, optics=dict(drawn["optics"], ghosts=[ghost], halo=None, veil=0.0, bloom=0.0, dirt=0.0))
    circle = dict(hexagon, optics=dict(hexagon["optics"], ghosts=[dict(ghost, round=True)]))
    assert not np.allclose(lens_flare_layers(bank, hexagon, 64, 64)[0], lens_flare_layers(bank, circle, 64, 64)[0])


def test_a_faint_wu_layer_is_drawn_again_with_its_source_and_nothing_else_changes(tmp_path, monkeypatch):
    bank = LensFlareBank(_write_flare7k(tmp_path / "f7k"), holdout_fraction=0.2, wu_root=_write_wu(tmp_path / "wu"))
    calls = []
    original = lens_module.draw_lens_parameters

    def counted(*args, **kwargs):
        calls.append(kwargs.get("keep_source"))
        return original(*args, **kwargs)

    monkeypatch.setattr(lens_module, "draw_lens_parameters", counted)
    floor = 0.03
    cfg = lens_config(dict(P45["flare"]["lens"], lens_wu_min_flare=floor, lens_optics_probability=0.0))
    redrawn = 0
    for seed in range(60):
        calls.clear()
        params, flare, light = draw_lens_flare(np.random.default_rng(seed), cfg, bank, "train", 32, 32)
        mean = float((flare - light).mean())
        if params["source"] not in WU_SOURCES:
            assert calls == [None]                                    # one draw, as before
            continue
        assert all(source == params["source"] for source in calls[1:])  # the source is kept
        assert mean >= floor or len(calls) == WU_DRAWS
        redrawn += len(calls) > 1
    assert redrawn > 0
    monkeypatch.setattr(lens_module, "draw_lens_parameters", original)
    # Without the floor (and for p44's lens) draw_lens_flare is draw_lens_parameters + lens_flare_layers.
    for lens in (dict(P45["flare"]["lens"], lens_wu_min_flare=0.0), dict(P44["flare"]["lens"])):
        cfg = lens_config(lens)
        for seed in range(10):
            params, flare, light = draw_lens_flare(np.random.default_rng(seed), cfg, bank, "train", 32, 32)
            expected = draw_lens_parameters(np.random.default_rng(seed), cfg, bank, "train")
            assert params == expected
            assert all(np.array_equal(a, b) for a, b in zip((flare, light), lens_flare_layers(bank, expected, 32, 32)))


def test_shrink_keeps_the_light_and_fetch_writes_each_photo_once(tmp_path, monkeypatch):
    photo = np.zeros((752, 1008, 3), np.uint8)
    photo[300:340, 500:540] = 255
    small = wu_flares.shrink(photo, 640)
    assert small.shape == (477, 640, 3)
    linear = lambda image: (image.astype(np.float64) / 255) ** wu_flares.GAMMA
    assert linear(small).sum() * (1008 / 640) ** 2 == pytest.approx(linear(photo).sum(), rel=0.05)
    names = {"captured": ["frame_0000.png", "frame_0001.png"], "simulated": ["aperture0000_blur00_crop00.png"]}
    buffer = io.BytesIO()
    Image.fromarray(photo).save(buffer, format="PNG")
    calls = []

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        if "storage/v1" in url:
            part = "captured" if "captured" in url else "simulated"
            return _Response(json.dumps({"items": [{"name": f"lens-flare/{part}/{n}"} for n in names[part]]}).encode())
        return _Response(buffer.getvalue())

    monkeypatch.setattr(wu_flares.urllib.request, "urlopen", fake_urlopen)
    root = wu_flares.fetch_wu_flares(tmp_path / "wu", side=640, workers=2)
    assert sorted(p.name for p in (root / "captured").iterdir()) == names["captured"]
    assert Image.open(root / "simulated" / names["simulated"][0]).size == (640, 477)
    downloads = len([c for c in calls if "storage/v1" not in c])
    wu_flares.fetch_wu_flares(tmp_path / "wu", side=640, workers=2)
    assert len([c for c in calls if "storage/v1" not in c]) == downloads == 3          # nothing fetched twice
    monkeypatch.setattr(wu_flares, "EXPECTED", {"captured": 2, "simulated": 1})
    assert wu_flares.find_wu_flares(tmp_path) == root
    monkeypatch.setattr(wu_flares, "EXPECTED", {"captured": 3, "simulated": 1})
    assert wu_flares.find_wu_flares(tmp_path) is None                                  # an incomplete copy is not one


def test_the_stage_composites_draw_wu_layers_of_their_split(tmp_path):
    from test_flare_stage import _stage
    from qjepa.cli import _dataset, flare_pair_bank
    from qjepa.data.flare_stage import FlareStageDataset
    config, manifest, _ = _stage(tmp_path)
    wu = _write_wu(tmp_path / "wu")
    stage = copy.deepcopy(config["flare"])
    stage["lens"].update(lens_flare_wu_fraction=1.0)
    bank = LensFlareBank(config["data"]["lens_flare_root"], holdout_fraction=0.1, wu_root=wu)
    for split in ("train", "valid"):
        data = FlareStageDataset(_dataset(config, manifest, split, fixed_realization=split != "train"), lens_bank=bank,
                                 flare_pairs=flare_pair_bank(config), section=dict(stage, kinds={
                                     "composite": 1.0, "pair": 0.0, "plain": 0.0}), split=split, length=12, seed=1)
        for index in range(12):
            info = data[index]["info"]
            assert info["source"] in WU_SOURCES and info["layer"] in bank.splits[split][info["source"]]


def test_select_on_real_and_composite_scores_the_mean_gain():
    report = {"real_pairs": 3, "real_psnr_in": 20.0, "real_psnr_out": 22.0, "composite_psnr_in": 18.0,
              "composite_psnr_out": 24.0}
    assert selection_score(report) == 22.0                                         # p44: the pairs' PSNR
    assert selection_score(report, "real_and_composite") == pytest.approx(4.0)     # (2 + 6) / 2
    assert selection_score({**report, "real_pairs": 0}, "real_and_composite") == pytest.approx(6.0)


def test_train_flare_starts_from_a_previous_best_and_it_is_found_in_input(tmp_path):
    from test_uformer_stage import _uformer_stage
    from test_flare_stage import _train_flare
    from qjepa.cli import main
    torch.set_num_threads(2)
    config, manifest_dir, _ = _uformer_stage(tmp_path)
    path, first = _train_flare(config, tmp_path, "p44")
    main(["train-flare", "--config", str(path), "--manifest", str(manifest_dir), "--output", str(first)])
    previous = first / "flare/best.pt"
    # Found in Input as a file under flare/, or inside a zip of the run's outputs.
    archive = tmp_path / "input/datasets/me/p44/p44_uformer_phase1.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(previous, "outputs/p44_uformer/flare/best.pt")
    found = find_stage_checkpoint(tmp_path / "input", tmp_path / "cache", prefer="p44_uformer")
    assert found is not None and file_sha256(found) == file_sha256(previous)
    assert find_stage_checkpoint(tmp_path / "input/datasets/me/nothing", tmp_path / "cache2") is None
    later = copy.deepcopy(config)
    later["flare"].update(uformer_init="stage", select_on="real_and_composite")
    later["flare"]["lens"].update({key: value for key, value in P45["flare"]["lens"].items() if "wu" in key})
    later["data"].update(uformer_checkpoint=str(found), wu_flare_root=str(_write_wu(tmp_path / "wu")))
    validate_config(later)
    path, second = _train_flare(later, tmp_path, "p45")
    main(["train-flare", "--config", str(path), "--manifest", str(manifest_dir), "--output", str(second)])
    last, best = load_checkpoint(second / "flare/last.pt"), load_checkpoint(second / "flare/best.pt")
    assert last["metadata"]["pretrained_sha256"] == file_sha256(previous)
    rows = [json.loads(line) for line in (second / "flare/train.jsonl").read_text().splitlines()]
    checks = {row["successful_updates"]: row["validation"] for row in rows if "validation" in row}
    scores = {u: selection_score(r, "real_and_composite") for u, r in checks.items()}
    assert best["metadata"]["selected_update"] == max(scores, key=lambda u: (scores[u], -u))
    if best["metadata"]["selected_update"] == 0:                     # p44's layer itself
        start = load_checkpoint(previous)["model"]
        assert all(torch.equal(best["model"][k], v) for k, v in start.items())
    wrong = copy.deepcopy(later)
    wrong["data"]["uformer_checkpoint"] = str(tmp_path / "f7k_release.pth")
    torch.save({"params": {}}, tmp_path / "f7k_release.pth")
    path, third = _train_flare(wrong, tmp_path, "wrong")
    with pytest.raises(SystemExit):                                  # a release file is not a flare stage
        main(["train-flare", "--config", str(path), "--manifest", str(manifest_dir), "--output", str(third)])


def test_kaggle_sunflare_is_p44_plus_its_keys_with_p44s_phase1_hash():
    validate_config(P45)
    assert configuration_hash(P45, "phase1") == configuration_hash(P44, "phase1")
    assert configuration_hash(P44, "phase2").startswith("75f72025b824")       # p44 as trained
    assert configuration_hash(P44, "flare").startswith("c9f0a56f901c")
    a, b = (serializable_config(copy.deepcopy(config)) for config in (P44, P45))
    assert b["runtime"].pop("output_dir").endswith("outputs/p45_sunflare")
    a["runtime"].pop("output_dir")
    assert b["flare"].pop("uformer_init") == "stage" and b["flare"].pop("select_on") == "real_and_composite"
    for key, value in {"lens_flare_wu_fraction": 0.5, "lens_flare_wu_captured_fraction": 0.7,
                       "lens_flare_wu_gain": [3.0, 10.0], "lens_wu_ghost_boost": [6.0, 20.0],
                       "lens_wu_ghost_radius": [0.04, 0.16], "lens_wu_ghost_blades": 6, "lens_wu_ghost_round": 0.4,
                       "lens_wu_min_flare": 0.01}.items():
        assert b["flare"]["lens"].pop(key) == value
    assert a == b
    moved = copy.deepcopy(P45)
    moved["data"]["wu_flare_root"] = "/elsewhere"
    for phase in ("phase1", "phase2", "flare"):
        assert configuration_hash(moved, phase) == configuration_hash(P45, phase)


@pytest.mark.parametrize("change, message", [
    (lambda c: c["flare"].update(uformer_init="p44"), "uformer_init"),
    (lambda c: c["flare"].update(select_on="composite"), "select_on"),
    (lambda c: c["flare"].update(select_best=False), "select_best"),
    (lambda c: c["flare"]["lens"].update(lens_flare_wu_fraction=1.5), "lens_flare_wu_fraction"),
    (lambda c: c["flare"]["lens"].update(lens_flare_wu_gain=[0.0, 10.0]), "lens_flare_wu_gain"),
    (lambda c: c["flare"]["lens"].pop("lens_flare_wu_gain"), "written out"),
    (lambda c: c["flare"]["lens"].pop("lens_wu_min_flare"), "written out"),
    (lambda c: c["flare"]["lens"].update(lens_wu_ghost_boost=[0.0, 10.0]), "lens_wu_ghost_boost"),
    (lambda c: c["flare"]["lens"].update(lens_wu_ghost_radius=[0.2, 0.6]), "lens_wu_ghost_radius"),
    (lambda c: c["flare"]["lens"].update(lens_wu_ghost_blades=2), "lens_wu_ghost_blades"),
    (lambda c: c["flare"]["lens"].update(lens_wu_ghost_round=1.5), "lens_wu_ghost_round"),
    (lambda c: c["flare"]["lens"].update(lens_wu_min_flare=-0.1), "lens_wu_min_flare"),
    (lambda c: c["corruption"]["image"].update(lens_wu_min_flare=0.01), "flare stage"),
])
def test_bad_settings_fail(change, message):
    config = copy.deepcopy(P45)
    change(config)
    with pytest.raises(ValueError, match=message):
        validate_config(config)
