"""p46_visdrone: real scene photos with a composited flare (data.scene_pairs_*, e.g. Kaggle visdrone-lens-flare-640) in
both phases and the flare stage, and samples without IMU (data.imu_absent_probability).

A scene pair (index.csv + gt/ + input/, split written in index.csv) becomes a sample whose target is a crop of its clean
photo and whose source is the flared photo of the same crop (like a FlareImage pair: "clear", the environment skipped)
or the clean photo (through the whole corruption, like a TartanAir frame). Its IMU window is borrowed from a TartanAir
sample. A sample without IMU has its clean and noisy window all zero.

Pins: the bank keeps index.csv's train / val split, shrinks each pair once in linear light and reads the cache after,
refuses a broken folder, and the notebook finds it however Kaggle mounted it (not the visdrone-day-2000 folder);
a scene item is the flared or clean crop in and the clean crop out with its donor's IMU, turned on the training split
only, flared at its share; every training batch holds its FlareImage and scene slots at fixed places, resume and DDP
ranks rebuilding them, and one group gives FlarePairSlots' batches; TartanAir and FlareImage items are bit-identical
with and without scene pairs and with the absent-IMU key at 0; samples without IMU come at their share with an all-zero
window and the same image; configs/kaggle_visdrone.yaml is p45 plus the keys, changes the hashes, keeps
data.scene_pairs_root out of them, and bad settings fail; the flare stage's "scene" kind and scene backgrounds build
their input, target and layer, its validation scores the valid scene pairs and real_composite_scene picks best.pt on
them, and a section without the keys draws as before; both phases train through the CLI with scene pairs and absent
IMU and log the "VisDrone valid" and "khong IMU" validation lines; infer runs without --imu and evaluate --imu-absent
scores samples without IMU; two DDP processes train what one does.
"""

from __future__ import annotations

import copy
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from qjepa.cli import (_dataset, _manifest, _train_dataset, _training_batch_stream, flare_pair_bank, main,
                       scene_pair_bank)
from qjepa.config import load_config, serializable_config, validate_config
from qjepa.corruptions.image import frame_kind
from qjepa.data.dataset import SCENE_DRAW_BASE
from qjepa.data.flare_stage import FlareStageDataset, held_out_pairs
from qjepa.data.sampler import ExtraSlots, FlarePairSlots
from qjepa.data.scene_pairs import ScenePairBank, find_scene_pairs_root
from qjepa.training.checkpoints import configuration_hash, load_checkpoint
from qjepa.training.flare_stage import selection_score
from test_flare_pairs import _smoke
from test_sharp_cli import _run_until_done

SHORT_SIDE = 40
SCENE = {"scene_pairs_fraction": 0.25, "scene_pairs_short_side": SHORT_SIDE, "scene_pairs_flare_probability": 0.5}


def _write_scene_pairs(root, train=24, valid=8, size=(48, 64)):
    """A visdrone-lens-flare folder: index.csv (file, split, scene, scene_split, flare, ...), gt/, input/ = gt + blob."""
    (root / "gt").mkdir(parents=True)
    (root / "input").mkdir()
    height, width = size
    rows = []
    for index in range(train + valid):
        name = f"{index:05d}.png"
        gt = np.zeros((height, width, 3), dtype=np.uint8)
        gt[..., 0] = np.linspace(30, 200, width, dtype=np.uint8)[None, :]
        gt[..., 1] = np.linspace(20, 160, height, dtype=np.uint8)[:, None]
        gt[..., 2] = 50 + (index % 40)
        flared = gt.astype(int)
        flared[height // 4:height // 4 + 10, width // 3:width // 3 + 10] += 110
        Image.fromarray(gt).save(root / "gt" / name)
        Image.fromarray(np.clip(flared, 0, 255).astype(np.uint8)).save(root / "input" / name)
        split = "train" if index < train else "val"
        flare = ("flarex/0050.png", "captured/frame_0450.png", "simulated/aperture0013_blur01_crop02.png")[index % 3]
        rows.append({"file": name, "split": split, "scene": f"000{index:04d}_00001_d_0000001.jpg",
                     "scene_split": "test-dev" if split == "val" else "train", "flare": flare, "seed": index,
                     "flare_mean": 0.05, "flare_area": 0.1, "saturated": 0.0, "params": "{}"})
    with (root / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def _scene_smoke(tmp_path, *, phase1_batch=4, phase2_batch=4, **data):
    config, manifest, _, manifest_dir = _smoke(tmp_path, phase1_batch=phase1_batch, phase2_batch=phase2_batch)
    root = tmp_path / "visdrone"
    if not root.exists():
        _write_scene_pairs(root)
    config["data"].update(SCENE, scene_pairs_root=str(root), flare_pairs_fraction=0.25)
    config["data"].update(data)
    validate_config(config)
    return config, manifest, manifest_dir


def test_the_bank_keeps_index_splits_shrinks_once_and_is_found_however_mounted(tmp_path):
    root = _write_scene_pairs(tmp_path / "a/datasets/buidinhkhoi/visdrone-lens-flare-640")
    bank = ScenePairBank(root, short_side=SHORT_SIDE, cache_dir=tmp_path / "cache")
    assert [len(bank.pairs(split)) for split in ("train", "valid")] == [24, 8]
    assert {pair.source for pair in bank.pairs("train")} == {"flarex", "captured", "simulated"}
    with pytest.raises(ValueError, match="split"):
        bank.pairs("test")
    first = bank.pairs("valid")[0]
    gt, flared = bank.load(first)
    assert gt.shape == flared.shape == (40, 53, 3)                     # short side 48 -> 40, in linear light
    assert (flared.astype(int) - gt).max() > 50                       # the blob survives the shrink
    # The second read comes from the cache: the originals can go.
    (root / "gt" / first.name).rename(tmp_path / "moved.png")
    again = bank.load(first)
    assert np.array_equal(again[0], gt) and np.array_equal(again[1], flared)
    (tmp_path / "moved.png").rename(root / "gt" / first.name)
    # The notebook finds it however Kaggle mounted it, and not visdrone-day-2000 (index.csv + images/).
    day = tmp_path / "a/datasets/buidinhkhoi/visdrone-day-2000"
    (day / "images").mkdir(parents=True)
    (day / "index.csv").write_text("file,source_split,sequence\n")
    assert find_scene_pairs_root(tmp_path / "a") == root
    with pytest.raises(FileNotFoundError, match="visdrone-lens-flare"):
        find_scene_pairs_root(tmp_path / "empty")
    _write_scene_pairs(tmp_path / "a/datasets/other/copy", train=2, valid=1)
    with pytest.raises(ValueError, match="hon mot lan"):
        find_scene_pairs_root(tmp_path / "a")
    # A broken folder fails with its reason.
    broken = _write_scene_pairs(tmp_path / "broken", train=2, valid=1)
    (broken / "input" / "00000.png").unlink()
    with pytest.raises(FileNotFoundError, match="thieu anh"):
        ScenePairBank(broken, short_side=SHORT_SIDE)
    text = (root / "index.csv").read_text().replace(",val,", ",test,")
    other = _write_scene_pairs(tmp_path / "other", train=2, valid=1)
    (other / "index.csv").write_text(text)
    with pytest.raises(ValueError, match="train / val"):
        ScenePairBank(other, short_side=SHORT_SIDE)


def _crop_places(whole, crop):
    """(top, left, view) where a dihedral view of ``crop`` sits in ``whole``."""
    size = crop.shape[0]
    found = []
    for turns in range(4):
        for flip in (False, True):
            view = np.rot90(crop[:, ::-1] if flip else crop, -turns)          # undo rot90(k) then flip
            for top in range(whole.shape[0] - size + 1):
                for left in range(whole.shape[1] - size + 1):
                    if np.array_equal(whole[top:top + size, left:left + size], view):
                        found.append((top, left, turns, flip))
    return found


def test_a_scene_item_is_the_flared_or_clean_crop_in_the_clean_crop_out_with_a_donors_imu(tmp_path):
    config, manifest, _ = _scene_smoke(tmp_path)
    bank = scene_pair_bank(config)
    plain = _dataset(config, manifest, "train", fixed_realization=False, image_mode="clean")
    dataset = _dataset(config, manifest, "train", fixed_realization=False, image_mode="clean", scene_pairs=bank,
                       flare_layer=True)
    first = len(dataset.samples) + SCENE_DRAW_BASE
    by_id = {sample.sample_id: index for index, sample in enumerate(dataset.samples)}
    flared_count = 0
    for draw in range(len(bank.pairs("train"))):
        item = dataset[first + draw]
        name = item["sample_id"].split("/")[1].split("#")[0]
        pair = next(p for p in bank.pairs("train") if p.name == name)
        gt, photo = (np.asarray(image, dtype=np.float32) / 255 for image in bank.load(pair))
        clean, noisy = (item[key].numpy().transpose(1, 2, 0) for key in ("image_clean", "image_noisy"))
        params = item["corruption"]["image"]
        places = _crop_places(gt, clean)
        assert places, draw
        if params.get("flare_pair"):
            flared_count += 1
            assert params["flare_pair"]["clear"] and frame_kind(params) == "flare"
            assert any(np.array_equal(np.rot90(photo[t:t + 32, l:l + 32], k)[:, ::-1] if f else
                                      np.rot90(photo[t:t + 32, l:l + 32], k), noisy) for t, l, k, f in places)
            assert float(item["image_flare_target"].abs().max()) > 0.1
            assert item["trajectory_key"] == f"scenepair/{pair.source}"
        else:
            assert np.array_equal(noisy, clean) and item["trajectory_key"] == "scenepair/clean"
            assert float(item["image_flare_target"].abs().max()) == 0.0
        donor = plain[by_id[item["corruption"]["imu_donor"]]]
        for key in ("imu_clean_phys", "imu_noisy_phys", "imu_times", "image_time", "imu_start"):
            assert torch.equal(item[key], donor[key]), key
    assert 0 < flared_count < len(bank.pairs("train"))               # data.scene_pairs_flare_probability 0.5
    # The valid split: no turn, and evaluation forces the flared photo.
    dataset.scene_pairs_split = "valid"
    for draw in range(4):
        item = dataset._scene_item(draw, flared=True)
        info = item["corruption"]["image"]["flare_pair"]
        assert "rot90" not in info and info["name"] in {p.name for p in bank.pairs("valid")}
    shares = [dataset._scene_item(draw)["corruption"]["image"].get("flare_pair") is not None for draw in range(400)]
    assert np.mean(shares) == pytest.approx(0.5, abs=0.08)


class _Mixed(torch.utils.data.Dataset):
    """Ids only: 'tartan:<index>', 'flare:<draw>' or 'scene:<draw>', each with the realization it was read at."""

    def __init__(self, length, flare=0.25, scene=0.25):
        self.samples = [SimpleNamespace(trajectory_key=f"t{i % 3}") for i in range(length)]
        self.flare, self.scene, self.realization = flare, scene, 0

    def flare_pairs_per_batch(self, size):
        return max(1, round(self.flare * size)) if self.flare else 0

    def scene_pairs_per_batch(self, size):
        return max(1, round(self.scene * size)) if self.scene else 0

    def set_realization(self, realization):
        self.realization = realization

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if index >= len(self.samples) + SCENE_DRAW_BASE:
            name = f"scene:{index - len(self.samples) - SCENE_DRAW_BASE}"
        elif index >= len(self.samples):
            name = f"flare:{index - len(self.samples)}"
        else:
            name = f"tartan:{index}"
        return {"image_clean": torch.zeros(3, 2, 2), "image_noisy": torch.zeros(3, 2, 2),
                "imu_clean_phys": torch.zeros(2, 6), "imu_noisy_phys": torch.zeros(2, 6),
                "image_time": torch.tensor(0.0, dtype=torch.float64), "imu_times": torch.zeros(2, dtype=torch.float64),
                "imu_start": torch.tensor(0), "sample_id": f"{self.realization}/{name}", "trajectory_key": "t",
                "corruption": {}}


def test_every_batch_holds_its_flareimage_and_scene_slots_and_resume_and_ranks_rebuild_them():
    config = load_config("configs/smoke.yaml")
    for size, length, flare, scene in ((8, 12, 2, 2), (32, 48, 8, 8), (4, 6, 1, 1), (8, 12, 0, 2)):
        mixed = lambda: _Mixed(length, flare=flare / size, scene=scene / size)
        stream = _training_batch_stream(config, mixed(), size, start_microbatch=0, namespace="phase1")
        per_epoch = length // (size - flare - scene)
        batches = [next(stream)["sample_id"] for _ in range(2 * per_epoch + 1)]
        for number, ids in enumerate(batches):
            assert len(ids) == size
            epoch, within = divmod(number, per_epoch)
            for kind, count in (("flare", flare), ("scene", scene)):
                draws = [int(value.split(":")[1]) for value in ids if f"/{kind}:" in value]
                assert draws == list(range(within * count, (within + 1) * count)), (size, kind, ids)
            # The extra slots end the batch and sit evenly; the kinds take turns over them.
            extra = [k for k, value in enumerate(ids) if "tartan:" not in value]
            assert extra[-1] == size - 1 and len(extra) == flare + scene
            assert all(value.startswith(f"{epoch}/") for value in ids)
            tartan = [value for value in ids if "tartan:" in value]
            assert len(set(tartan)) == len(tartan) == size - flare - scene
        resumed = _training_batch_stream(config, mixed(), size, start_microbatch=3, namespace="phase1")
        assert [next(resumed)["sample_id"] for _ in range(len(batches) - 3)] == batches[3:]
        if size % 2 == 0 and flare and scene:
            halves = [_training_batch_stream(config, mixed(), size, start_microbatch=0, namespace="phase1",
                                             rank=rank, world=2) for rank in (0, 1)]
            for ids in batches[:3]:
                first, second = next(halves[0])["sample_id"], next(halves[1])["sample_id"]
                assert first + second == ids
                if flare >= 2 and scene >= 2:                       # each rank loads some of each kind
                    for half in (first, second):
                        assert any("flare:" in v for v in half) and any("scene:" in v for v in half)
    # One group: FlarePairSlots' very batches.
    inner = [[0, 1, 2], [3, 4, 5]]
    assert list(ExtraSlots(inner, [(10, 2), (99, 0)])) == list(FlarePairSlots(inner, 10, 2))


def test_tartanair_and_flareimage_items_are_bit_identical_with_and_without_scene_pairs(tmp_path):
    config, manifest, _ = _scene_smoke(tmp_path)
    without = copy.deepcopy(config)
    without["data"] = {key: value for key, value in config["data"].items() if not key.startswith("scene_pairs_")}
    for phase in ("phase1", "phase2"):
        before = _train_dataset(without, manifest, phase)
        after = _train_dataset(dict(config, data=dict(config["data"], imu_absent_probability=0.0)), manifest, phase)
        assert before.scene_bank is None and after.scene_bank is not None
        for realization in (0, 2):
            before.set_realization(realization)
            after.set_realization(realization)
            for index in (0, 3, len(before.samples), len(before.samples) + 5):      # TartanAir, then FlareImage
                a, b = before[index], after[index]
                assert "imu_present" not in b and a["sample_id"] == b["sample_id"]
                for key, value in a.items():
                    if isinstance(value, torch.Tensor):
                        assert torch.equal(value, b[key]), (phase, index, key)


def test_samples_without_imu_come_at_their_share_with_an_all_zero_window_and_the_same_image(tmp_path):
    config, manifest, _ = _scene_smoke(tmp_path, imu_absent_probability=0.25)
    validate_config(config)
    reference = _train_dataset(dict(config, data=dict(config["data"], imu_absent_probability=0.0)), manifest, "phase2")
    dataset = _train_dataset(config, manifest, "phase2")
    absent = []
    indices = list(range(len(dataset.samples))) + [len(dataset.samples) + d for d in range(10)] + \
        [len(dataset.samples) + SCENE_DRAW_BASE + d for d in range(10)]
    for index in indices:
        item, plain = dataset[index], reference[index]
        missing = item["imu_present"].item() == 0.0
        absent.append(missing)
        assert torch.equal(item["image_noisy"], plain["image_noisy"]) and torch.equal(
            item["image_clean"], plain["image_clean"])
        if missing:
            assert float(item["imu_clean_phys"].abs().max()) == float(item["imu_noisy_phys"].abs().max()) == 0.0
            assert item["corruption"]["imu"]["absent"] is True
        else:
            assert torch.equal(item["imu_noisy_phys"], plain["imu_noisy_phys"])
            assert float(plain["imu_clean_phys"].abs().max()) > 0.1     # a real window is never all zero
    assert np.mean(absent) == pytest.approx(0.25, abs=0.12)
    # evaluate --imu-absent: every sample.
    every = _dataset(config, manifest, "valid", fixed_realization=True, imu_absent_probability=1.0)
    assert all(every[k]["imu_present"].item() == 0.0 for k in range(4))


def test_kaggle_visdrone_is_p45_plus_the_keys_changes_the_hashes_and_bad_settings_fail():
    p45 = serializable_config(load_config("configs/kaggle_sunflare.yaml"))
    p46 = serializable_config(load_config("configs/kaggle_visdrone.yaml"))
    validate_config(p46)
    assert p46["data"]["scene_pairs_fraction"] == 0.25 and p46["data"]["imu_absent_probability"] == 0.25
    assert p46["data"]["flare_pairs_fraction"] == 0.25 and p45["data"]["flare_pairs_fraction"] == 0.5
    assert p46["flare"]["kinds"]["scene"] == 0.2 and p46["flare"]["select_on"] == "real_composite_scene"
    rebuilt = copy.deepcopy(p45)
    rebuilt["runtime"]["output_dir"] = p46["runtime"]["output_dir"]
    rebuilt["data"].update({key: p46["data"][key] for key in (
        "flare_pairs_fraction", "scene_pairs_root", "scene_pairs_fraction", "scene_pairs_short_side",
        "scene_pairs_flare_probability", "imu_absent_probability")})
    rebuilt["flare"].update(kinds=p46["flare"]["kinds"], background_scene_fraction=0.3,
                            select_on="real_composite_scene")
    assert rebuilt == p46
    for phase in ("phase1", "phase2", "flare"):
        assert configuration_hash(p46, phase) != configuration_hash(p45, phase)
        moved = copy.deepcopy(p46)
        moved["data"]["scene_pairs_root"] = "/elsewhere"
        assert configuration_hash(moved, phase) == configuration_hash(p46, phase)    # a path, out of the hash
    # The root alone, with no fraction, leaves every hash of p45 as it was.
    rooted = copy.deepcopy(p45)
    rooted["data"]["scene_pairs_root"] = "/x"
    assert all(configuration_hash(rooted, phase) == configuration_hash(p45, phase) for phase in ("phase1", "flare"))

    def fails(change, message):
        bad = copy.deepcopy(p46)
        change(bad)
        with pytest.raises(ValueError, match=message):
            validate_config(bad)

    fails(lambda c: c["data"].pop("scene_pairs_short_side"), "missing: scene_pairs_short_side")
    fails(lambda c: c["data"].update(scene_pairs_fraction=0.6), r"scene_pairs_fraction must be a number in \[0, 0.5\]")
    fails(lambda c: c["data"].update(scene_pairs_fraction=0.5, flare_pairs_fraction=0.5), "minimum_trajectories")
    fails(lambda c: c["data"].update(scene_pairs_flare_probability=1.5), "flare_probability")
    fails(lambda c: c["data"].update(scene_pairs_short_side=16), "short_side")
    fails(lambda c: c["data"].update(scene_pairs_typo=1), "unknown keys")
    fails(lambda c: c["data"].update(imu_absent_probability=1.0), "imu_absent_probability")
    fails(lambda c: c["flare"].update(select_on="all"), "select_on")
    fails(lambda c: c["flare"].update(background_scene_fraction=2), "background_scene_fraction")
    fails(lambda c: c["flare"]["kinds"].update(other=0.1), "flare.kinds")
    without = lambda c: c["data"].update(scene_pairs_fraction=0.0)
    fails(without, "scene pairs")


def _flare_stage(tmp_path, **change):
    from test_flare_stage import P42
    from test_lens_flare import _write_flare7k
    config, manifest, manifest_dir = _scene_smoke(tmp_path)
    root = tmp_path / "f7k"
    if not root.exists():
        _write_flare7k(root)
    config["data"]["lens_flare_root"] = str(root)
    stage = copy.deepcopy(P42["flare"])
    stage.update(widths=[8, 8, 8], blocks=1, batch_size=4, max_successful_updates=4, warmup_updates=1,
                 validation_every_updates=2, validation_composites=6, checkpoint_every_updates=2, precision="fp32",
                 kinds={"composite": 0.4, "pair": 0.2, "plain": 0.2, "scene": 0.2}, background_scene_fraction=0.5)
    stage.update(change)
    config["flare"] = stage
    validate_config(config)
    return config, manifest, manifest_dir


def _stage_dataset(config, manifest, split="train", length=60, scenes=True, **section):
    from qjepa.corruptions.lens_flare import LensFlareBank
    stage = dict(config["flare"], **section)
    bank = LensFlareBank(config["data"]["lens_flare_root"], holdout_fraction=0.1)
    tartan = _dataset(config, manifest, split, fixed_realization=split != "train")
    return FlareStageDataset(tartan, lens_bank=bank, flare_pairs=flare_pair_bank(config), section=stage, split=split,
                             length=length, seed=int(stage["seed"]),
                             scene_pairs=scene_pair_bank(config) if scenes else None)


def test_the_flare_stage_scene_kind_and_scene_backgrounds_build_their_input_target_and_layer(tmp_path):
    config, manifest, _ = _flare_stage(tmp_path)
    data = _stage_dataset(config, manifest, grain_probability=0.0)
    seen, backgrounds = set(), set()
    held = {pair.name for pair in scene_pair_bank(config).pairs("train")}
    for index in range(60):
        item = data[index]
        kind = item["info"]["kind"]
        seen.add(kind)
        assert torch.allclose(item["image_noisy"], item["image_target"] + item["flare_target"], atol=1e-6)
        if kind == "scene":
            assert item["info"]["pair"] in held and float(item["flare_target"].abs().max()) > 0.1
            assert int(item["kind"]) == 3
        if kind == "composite":
            backgrounds.add(item["info"]["frame"].split("/")[0])
    assert seen == {"composite", "pair", "plain", "scene"} and "scene" in backgrounds
    kinds = [data.kind_of(index) for index in range(600)]
    assert kinds.count("scene") / 600 == pytest.approx(0.2, abs=0.05)
    # A section without the p46 keys draws exactly as before: same kinds, same samples.
    old = copy.deepcopy(config["flare"])
    old["kinds"] = {"composite": 0.6, "pair": 0.2, "plain": 0.2}
    old.pop("background_scene_fraction")
    with_bank = _stage_dataset(dict(config, flare=old), manifest, length=30)
    no_bank = _stage_dataset(dict(config, flare=old), manifest, length=30, scenes=False)
    assert with_bank.kinds == no_bank.kinds == ("composite", "pair", "plain")
    for index in range(30):
        a, b = with_bank[index], no_bank[index]
        assert a["info"] == b["info"] and torch.equal(a["image_noisy"], b["image_noisy"])


def test_train_flare_scores_the_valid_scene_pairs_and_picks_best_on_them(tmp_path):
    torch.set_num_threads(1)
    config, _, manifest_dir = _flare_stage(tmp_path, model="unet", select_best=True,
                                           select_on="real_composite_scene")
    path = tmp_path / "stage.yaml"
    path.write_text(yaml.safe_dump(serializable_config(config)))
    output = tmp_path / "run"
    main(["train-flare", "--config", str(path), "--manifest", str(manifest_dir), "--output", str(output)])
    rows = [json.loads(line) for line in (output / "flare/train.jsonl").read_text().splitlines()]
    checks = [row["validation"] for row in rows if "validation" in row]
    assert checks and all(check["scene_pairs"] == 8 for check in checks)
    assert all(np.isfinite(check["scene_psnr_in"]) and np.isfinite(check["scene_psnr_out"]) for check in checks)
    best = load_checkpoint(output / "flare/best.pt")
    assert best["metadata"]["selection_score"] == pytest.approx(
        max(selection_score(check, "real_composite_scene") for check in checks))
    # The score is the mean of the three gains; without scene pairs it is real_and_composite's.
    report = {"real_psnr_in": 20.0, "real_psnr_out": 22.0, "real_pairs": 3, "composite_psnr_in": 25.0,
              "composite_psnr_out": 26.0, "scene_psnr_in": 18.0, "scene_psnr_out": 21.0, "scene_pairs": 8}
    assert selection_score(report, "real_composite_scene") == pytest.approx(2.0)
    assert selection_score(report, "real_and_composite") == pytest.approx(1.5)
    assert len(held_out_pairs(scene_pair_bank(config), "valid", (32, 32))) == 8


def _train_tiny(tmp_path, config, manifest_dir, name):
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(serializable_config(config)))
    output = tmp_path / name
    common = ["--config", str(path), "--manifest", str(manifest_dir), "--output", str(output)]
    _run_until_done(["train-phase1", *common], output / "phase1/last.pt")
    _run_until_done(["train-phase2", *common, "--backbone-checkpoint", str(output / "phase1/last.pt")],
                    output / "phase2/last.pt")
    return output


def test_both_phases_train_with_scene_pairs_and_absent_imu_and_infer_and_evaluate_without_imu(tmp_path, capsys):
    torch.set_num_threads(1)
    config, _, manifest_dir = _scene_smoke(tmp_path, imu_absent_probability=0.25)
    output = _train_tiny(tmp_path, config, manifest_dir, "run")
    assert load_checkpoint(output / "phase2/last.pt")["successful_updates"] == 3
    rows = [json.loads(line) for line in (output / "phase2/train.jsonl").read_text().splitlines()]
    losses = [row["loss"] for row in rows if "loss" in row]
    assert losses and all(np.isfinite(losses))
    checks = [row for row in rows if "validation_image_psnr_db" in row]
    assert checks and all(row["scene_pairs_validation_count"] == 8 for row in checks)
    assert all(np.isfinite(row["scene_pairs_validation_psnr_out"]) for row in checks)
    assert all(np.isfinite(row["imu_absent_validation_image_psnr_db"]) for row in checks)
    printed = capsys.readouterr().out
    assert "VisDrone valid n=8" in printed and "khong IMU: PSNR" in printed
    # tools/training_report.py (Cell 13) prints both lines at the last checkpoint.
    import subprocess
    import sys
    report = subprocess.run([sys.executable, "tools/training_report.py", "--run", str(output)], capture_output=True,
                            text=True)
    assert report.returncode == 0, report.stderr[-2000:]
    assert "VisDrone valid" in report.stdout and "không IMU" in report.stdout and "FlareImage valid" in report.stdout
    # infer with an image alone, then with an IMU window: both write a restored image.
    image = tmp_path / "frame.png"
    Image.fromarray(np.full((32, 32, 3), 90, dtype=np.uint8)).save(image)
    main(["infer", "--checkpoint", str(output / "phase2/last.pt"), "--image", str(image), "--output",
          str(tmp_path / "alone"), "--device", "cpu"])
    assert (tmp_path / "alone/image_restored.png").is_file()
    assert json.loads((tmp_path / "alone/metadata.json").read_text())["imu"] is None
    # evaluate --imu-absent: the run records it and scores the same frames.
    main(["evaluate", "--checkpoint", str(output / "phase2/last.pt"), "--manifest", str(manifest_dir), "--split",
          "valid", "--max-batches", "1", "--panels", "0", "--device", "cpu", "--output", str(tmp_path / "eval"),
          "--imu-absent"])
    assert json.loads((tmp_path / "eval/evaluation_config.json").read_text())["imu_absent"] is True
    metrics = json.loads((tmp_path / "eval/metrics.json").read_text())
    assert np.isfinite(metrics["requested"]["image_psnr_db"])


def _records(run, phase, key):
    lines = (run / phase / "train.jsonl").read_text().splitlines()
    return [record[key] for record in map(json.loads, lines) if key in record and "loss" in record]


def test_two_processes_train_with_scene_pairs_and_absent_imu_what_one_does(tmp_path):
    """Kaggle trains on two T4s (DDP): each rank loads its share of every batch (TartanAir, FlareImage and scene
    slots), and the gathered batch must be the one-process batch, through restarts too."""
    torch.set_num_threads(1)
    config, _, manifest_dir = _scene_smoke(tmp_path, imu_absent_probability=0.25)
    config["phase1"].update(max_successful_updates=3, augment_hflip=False)
    config["phase2"].update(gradient_accumulation=1, max_successful_updates=3, augment_hflip=False)
    config["runtime"]["restart_above_rss_gib"] = None
    single = tmp_path / "single.yaml"
    single.write_text(yaml.safe_dump(serializable_config(config)))
    config["runtime"].update(parallel="ddp", gpu_count=2, restart_above_rss_gib=1e-6)
    ddp = tmp_path / "ddp.yaml"
    ddp.write_text(yaml.safe_dump(serializable_config(config)))
    for name, path in (("single", single), ("ddp", ddp)):
        output = tmp_path / name
        common = ["--config", str(path), "--manifest", str(manifest_dir), "--output", str(output)]
        _run_until_done(["train-phase1", *common], output / "phase1/last.pt")
        _run_until_done(["train-phase2", *common, "--backbone-checkpoint", str(output / "phase1/last.pt")],
                        output / "phase2/last.pt")
    for phase in ("phase1", "phase2"):
        for key in ("loss", "gradient_norm"):
            single_values, ddp_values = (_records(tmp_path / name, phase, key) for name in ("single", "ddp"))
            assert len(single_values) == len(ddp_values) == 3, (phase, key)
            assert ddp_values == pytest.approx(single_values, rel=2e-3), (phase, key)


def test_ijepa_phase1_and_phase2_train_with_scene_pairs_and_absent_imu(tmp_path):
    """The Kaggle recipe's phase 1 is I-JEPA on a ViT: it trains with scene pairs and samples without IMU too."""
    from test_ijepa import _ijepa_dict
    from test_kaggle_workflow import _write_dataset
    torch.set_num_threads(1)
    root, manifest, output = tmp_path / "dataset", tmp_path / "manifest", tmp_path / "run"
    _write_dataset(root)
    scenes = _write_scene_pairs(tmp_path / "visdrone", size=(72, 96))
    config = _ijepa_dict(max_successful_updates=2, batch_size=4)
    config["phase2"].update(max_successful_updates=2, batch_size=4)
    config["data"].update(scene_pairs_root=str(scenes), scene_pairs_fraction=0.25, scene_pairs_short_side=64,
                          scene_pairs_flare_probability=0.5, imu_absent_probability=0.5)
    validate_config(config)
    path = tmp_path / "ijepa.yaml"
    path.write_text(yaml.safe_dump(config))
    common = ["--config", str(path), "--manifest", str(manifest), "--output", str(output)]
    main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest)])
    main(["train-phase1", *common])
    phase1 = load_checkpoint(output / "phase1/last.pt")
    assert phase1["successful_updates"] == 2 and phase1["metadata"]["phase1_objective"] == "ijepa"
    losses = [json.loads(line).get("loss") for line in (output / "phase1/train.jsonl").read_text().splitlines()]
    assert [value for value in losses if value is not None] and all(
        np.isfinite(value) for value in losses if value is not None)
    main(["train-phase2", *common, "--backbone-checkpoint", str(output / "phase1/last.pt")])
    rows = [json.loads(line) for line in (output / "phase2/train.jsonl").read_text().splitlines()]
    assert all(np.isfinite(row["loss"]) for row in rows if "loss" in row)
    assert any("imu_absent_validation_image_psnr_db" in row for row in rows)


def test_every_imu_loss_keeps_a_finite_gradient_on_a_window_without_imu():
    """An IMU window without IMU is all zero in and out, and a residual decoder starts by returning its input: the
    detail bands of output and target are then exactly 0. detail_energy_gap took sqrt(0) there (infinite slope, NaN
    gradient: phase 2 skipped every update on the I-JEPA recipe); it now holds the energy at 1e-12, which any real band
    is far above -- its value on real coefficients is bit-identical to the old formula."""
    from qjepa.training.losses import (detail_band_l1, detail_bands, detail_energy_gap, excess_jitter,
                                       first_difference_l1, imu_increment_loss)
    zero = torch.zeros(2, 6 * 4, 8, requires_grad=True)       # [B, C, L] Haar coefficients of an all-zero window
    target = torch.zeros(2, 6 * 4, 8)
    for loss in (detail_energy_gap(zero, target, dim=1), detail_band_l1(zero, target, dim=1)):
        (grad,) = torch.autograd.grad(loss, zero)
        assert torch.isfinite(loss) and torch.isfinite(grad).all()
    signal = torch.zeros(2, 6, 32, requires_grad=True)
    for loss in (first_difference_l1(signal, torch.zeros(2, 6, 32)), excess_jitter(signal, torch.zeros(2, 6, 32)),
                 imu_increment_loss(signal, torch.zeros(2, 6, 32), (8, 32))):
        (grad,) = torch.autograd.grad(loss, signal)
        assert torch.isfinite(grad).all()
    generator = torch.Generator().manual_seed(0)
    got, want = (torch.randn(3, 24, 16, generator=generator) * scale for scale in (0.05, 0.2))
    axes = tuple(range(2, detail_bands(got, 1).ndim))
    old = (detail_bands(got, 1).square().mean(axes).sqrt() - detail_bands(want, 1).square().mean(axes).sqrt()).abs().mean()
    assert torch.equal(detail_energy_gap(got, want, dim=1), old)
