"""data.flare_pairs_*: FlareImage's real flared photos as their own samples in every training batch (p36_flareimage).

FlareImage (Kaggle buidinhkhoi/flareimage) holds aligned pairs -- train/<name> flared, gt/<name> clean -- and no IMU.
A pair becomes a sample: the flared crop is the input, the clean crop of the same place the target, and the IMU window
(clean and noisy) is borrowed from a TartanAir sample.

Pins: every training batch holds exactly max(1, round(fraction * batch)) pairs at evenly spaced slots, in both phases,
with resume rebuilding the very batches the run would have seen and each DDP rank loading its own share; a pair item is
the flared crop in and the clean crop of the same place out, with the IMU window of its TartanAir donor; TartanAir
items are bit-identical with and without the pairs; the corruption draws on a pair are the ones it would be without
the pair, a pair never gets a HALO layer on top, and with "clear" it skips the environment (no darkness, uneven
light, lamps, fog) while blur and grain stay; a pair is a "flare" frame (tone label 1) even when the corruption draws
clean; the bank splits whole groups per source (HALO by scene and effect, others by blocks of 10 source images) into
train / valid / test, shrinks each pair once to the short side in linear light with the light kept and reads the
cache after; the notebook finds FlareImage however Kaggle mounted it; configs/kaggle_flareimage is p35_tonegate plus
the flare keys and HALO off, changes both hashes, keeps data.flare_pairs_root out of them, must spell every key out
and refuses a batch with no TartanAir sample left; both phases train through the CLI with the pairs, two DDP
processes train what one does (through restarts), and tools/flare_pairs_probe.py scores input and output PSNR on the
held-out pairs (trained with them or not).
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

from qjepa.cli import _dataset, _manifest, _train_dataset, _training_batch_stream, flare_pair_bank, main
from qjepa.config import FLARE_PAIR_KEYS, load_config, serializable_config, validate_config
from qjepa.corruptions.image import (LowLightImageCorruptionConfig, LowLightImageCorruptor, frame_kind)
from qjepa.corruptions.light import srgb_to_linear
from qjepa.data.flare_pairs import FlarePairBank, find_flare_pairs_root, pair_group, shrink_pair
from qjepa.training.checkpoints import configuration_hash, load_checkpoint
from test_kaggle_workflow import _write_dataset
from test_sharp_cli import _run_until_done, _sharp_smoke

FLARE = {key: value for key, value in load_config("configs/kaggle_flareimage.yaml")["data"].items()
         if key in FLARE_PAIR_KEYS and key != "flare_pairs_root"}
SHORT_SIDE = 40          # tiny pairs: the 80x96 ones shrink by 2, the others are already at most 40 on the short side
SIZES = {"flare_removal": (96, 80), "halo": (36, 64), "flare7k_real": (40, 40), "flare7k_synthetic": (40, 40)}


def _write_flareimage(root, per_source=30):
    """A FlareImage folder: metadata.csv, gt/<name>, train/<name> = gt + a bright blob; four sources as on Kaggle."""
    (root / "gt").mkdir(parents=True)
    (root / "train").mkdir()
    rows, index = [], 0
    for source, (height, width) in SIZES.items():
        for k in range(per_source if source != "halo" else 24):
            index += 1
            name = f"{index:06d}.png"
            if source == "halo":
                scene, effect, camera = k // 6, (k // 2) % 3, k % 2
                src = (f"/kaggle/input/datasets/k/halo/halo_reflective_1280/Scene{scene:03d}/"
                       f"Scene{scene:03d}_Reflective{effect:03d}_camera{camera:02d}_{k:04d}.gt.png")
            elif source == "flare_removal":
                src = f"/kaggle/input/datasets/a/flare-removal/train_gt_2k/train_gt_2k/{100 + k:06d}.png"
            else:
                src = f"/kaggle/input/datasets/k/flare7k/test_data/{source.split('_')[1]}/gt/gt_{k:06d}.png"
            gt = np.zeros((height, width, 3), dtype=np.uint8)
            gt[..., 0] = np.linspace(20, 180, width, dtype=np.uint8)[None, :]
            gt[..., 1] = np.linspace(30, 150, height, dtype=np.uint8)[:, None]
            gt[..., 2] = 40 + (index % 50)
            flared = gt.astype(int)
            flared[height // 3:height // 3 + 8, width // 2:width // 2 + 8] += 120
            Image.fromarray(gt).save(root / "gt" / name)
            Image.fromarray(np.clip(flared, 0, 255).astype(np.uint8)).save(root / "train" / name)
            rows.append({"name": name, "source": source, "width": width, "height": height, "gt_src": src,
                         "train_src": src.replace("gt", "input")})
    with (root / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def _bank(root, tmp_path):
    return FlarePairBank(root, short_side=SHORT_SIDE, holdout_fraction=0.1, cache_dir=tmp_path / "cache")


def _smoke(tmp_path, *, phase1_batch=4, phase2_batch=2):
    """Smoke recipe with the FlareImage keys; manifest of test_kaggle_workflow's tiny TartanAir; (config, manifest)."""
    root, manifest_dir = tmp_path / "dataset", tmp_path / "manifest"
    if not root.exists():
        _write_dataset(root)
    flare_root = tmp_path / "flare_dataset"
    if not flare_root.exists():
        _write_flareimage(flare_root)
    config = _sharp_smoke()
    config["data"].update(FLARE, flare_pairs_root=str(flare_root), flare_pairs_short_side=SHORT_SIDE)
    config["phase1"]["batch_size"], config["phase2"]["batch_size"] = phase1_batch, phase2_batch
    path = tmp_path / "flare.yaml"
    path.write_text(yaml.safe_dump(config))
    if not manifest_dir.exists():
        main(["build-manifest", "--config", str(path), "--data-root", str(root), "--output", str(manifest_dir)])
    manifest, _ = _manifest(config, str(manifest_dir))
    return config, manifest, path, manifest_dir


class _Mixed(torch.utils.data.Dataset):
    """Ids only: 'tartan:<index>' or 'flare:<draw>', each with the realization it was read at."""

    def __init__(self, length, fraction=0.25):
        self.samples = [SimpleNamespace(trajectory_key=f"t{i % 3}") for i in range(length)]
        self.fraction, self.realization = fraction, 0

    def flare_pairs_per_batch(self, size):
        return max(1, round(self.fraction * size))

    def set_realization(self, realization):
        self.realization = realization

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name = f"flare:{index - len(self.samples)}" if index >= len(self.samples) else f"tartan:{index}"
        return {"image_clean": torch.zeros(3, 2, 2), "image_noisy": torch.zeros(3, 2, 2),
                "imu_clean_phys": torch.zeros(2, 6), "imu_noisy_phys": torch.zeros(2, 6),
                "image_time": torch.tensor(0.0, dtype=torch.float64), "imu_times": torch.zeros(2, dtype=torch.float64),
                "imu_start": torch.tensor(0), "sample_id": f"{self.realization}/{name}", "trajectory_key": "t",
                "corruption": {}}


def test_every_batch_holds_its_flareimage_pairs_at_fixed_slots_and_resume_and_ranks_rebuild_them():
    config = load_config("configs/smoke.yaml")
    for size, length, slots in ((8, 12, [3, 7]), (32, 48, [3, 7, 11, 15, 19, 23, 27, 31]), (4, 6, [3])):
        stream = _training_batch_stream(config, _Mixed(length), size, start_microbatch=0, namespace="phase1")
        per_epoch = length // (size - len(slots))
        batches = [next(stream)["sample_id"] for _ in range(2 * per_epoch + 1)]
        for number, ids in enumerate(batches):
            assert len(ids) == size
            positions = [k for k, value in enumerate(ids) if "flare:" in value]
            assert positions == slots, (size, number, ids)
            epoch, within = divmod(number, per_epoch)
            draws = [int(ids[k].split(":")[1]) for k in positions]
            assert draws == list(range(within * len(slots), (within + 1) * len(slots)))   # fresh draws each batch
            assert all(value.startswith(f"{epoch}/") for value in ids)                   # realization per epoch
            tartan = [value for value in ids if "tartan:" in value]
            assert len(set(tartan)) == len(tartan) == size - len(slots)
        # Resume at microbatch 3 rebuilds batches 3, 4, ... without loading the ones it skips.
        resumed = _training_batch_stream(config, _Mixed(length), size, start_microbatch=3, namespace="phase1")
        assert [next(resumed)["sample_id"] for _ in range(len(batches) - 3)] == batches[3:]
        # DDP: each rank loads its contiguous half; together they are the batch.
        if size % 2 == 0:
            halves = [_training_batch_stream(config, _Mixed(length), size, start_microbatch=0, namespace="phase1",
                                             rank=rank, world=2) for rank in (0, 1)]
            for ids in batches[:3]:
                assert next(halves[0])["sample_id"] + next(halves[1])["sample_id"] == ids


def test_a_pair_is_the_flared_crop_in_the_clean_crop_out_with_its_donors_imu(tmp_path):
    config, manifest, _, _ = _smoke(tmp_path)
    bank = flare_pair_bank(config)
    plain = _dataset(config, manifest, "train", fixed_realization=False, image_mode="clean")
    dataset = _dataset(config, manifest, "train", fixed_realization=False, image_mode="clean", flare_pairs=bank,
                       tone_label=True)
    first = len(dataset.samples)
    by_id = {sample.sample_id: index for index, sample in enumerate(dataset.samples)}
    seen = set()
    for draw in range(len(bank.pairs("train"))):
        item = dataset[first + draw]
        pair = item["corruption"]["image"]["flare_pair"]
        seen.add(pair["name"])
        gt, flared = (np.asarray(image, dtype=np.float32) / 255 for image in bank.load(
            next(p for p in bank.pairs("train") if p.name == pair["name"])))
        clean, noisy = (item[key].numpy().transpose(1, 2, 0) for key in ("image_clean", "image_noisy"))
        size = clean.shape[0]
        places = [(top, left) for top in range(gt.shape[0] - size + 1) for left in range(gt.shape[1] - size + 1)
                  if np.array_equal(gt[top:top + size, left:left + size], clean)]
        assert any(np.array_equal(flared[top:top + size, left:left + size], noisy) for top, left in places)
        # The IMU window is its donor's: a TartanAir training sample, clean and noisy alike.
        donor = plain[by_id[item["corruption"]["imu_donor"]]]
        for key in ("imu_clean_phys", "imu_noisy_phys", "imu_times", "image_time", "imu_start"):
            assert torch.equal(item[key], donor[key]), key
        assert json.dumps(item["corruption"]["imu"], sort_keys=True, default=str) == json.dumps(
            donor["corruption"]["imu"], sort_keys=True, default=str)
        # A real flared photo is a "flare" frame, even with the corruption drawn clean.
        assert item["corruption"]["image"]["clean"] and frame_kind(item["corruption"]["image"]) == "flare"
        assert item["image_tone_label"].item() == 1.0
        assert item["sample_id"].startswith(f"flareimage/{pair['name']}#0:") and item["trajectory_key"].startswith(
            "flareimage/")
    assert seen == {pair.name for pair in bank.pairs("train")}        # one cycle: every training pair once


def test_tartanair_items_are_bit_identical_with_and_without_the_pairs(tmp_path):
    config, manifest, _, _ = _smoke(tmp_path)
    for phase in ("phase1", "phase2"):
        without = _train_dataset(dict(config, data={k: v for k, v in config["data"].items()
                                                    if not k.startswith("flare_pairs_")}), manifest, phase)
        with_pairs = _train_dataset(config, manifest, phase)
        assert without.flare_bank is None and with_pairs.flare_bank is not None
        for realization in (0, 3):
            without.set_realization(realization)
            with_pairs.set_realization(realization)
            for index in range(len(without)):
                a, b = without[index], with_pairs[index]
                assert a.keys() == b.keys()
                for key, value in a.items():
                    if torch.is_tensor(value):
                        assert torch.equal(value, b[key]), key
                assert json.dumps(a["corruption"], sort_keys=True, default=str) == json.dumps(
                    b["corruption"], sort_keys=True, default=str)


def test_a_pairs_draws_are_unchanged_it_never_gets_halo_and_clear_skips_the_environment(tmp_path):
    from test_halo_flare import HALO, _bank as halo_bank, _write_halo
    halo = halo_bank(_write_halo(tmp_path / "halo"))
    values = dict(HALO, halo_probability=1.0, halo_clear_probability=0.0, clean_probability=0.0,
                  light_probability=0.5, illum_probability=0.5, fog_probability=0.5, env_clear_probability=0.0)
    corruptor = LowLightImageCorruptor(LowLightImageCorruptionConfig(**values), 73128, halo_bank=halo)
    lit = 0
    for index in range(40):
        ids = ("train", 0, f"flareimage/x#0:{index}", 0.0, "full")
        alone = corruptor._parameters(*ids)
        assert alone["halo"]                                           # every frame here draws the HALO layer ...
        kept = corruptor._parameters(*ids, {"name": "x", "source": "s", "clear": False})
        cleared = corruptor._parameters(*ids, {"name": "x", "source": "s", "clear": True})
        assert not kept["halo"] and kept["halo_params"] is None       # ... but never on a real flared photo
        assert not cleared["halo"] and cleared["halo_params"] is None
        same = {key: value for key, value in alone.items() if key not in ("halo", "halo_params")}
        assert {key: value for key, value in kept.items() if key not in ("halo", "halo_params", "flare_pair")} == same
        lit += bool(kept["low_light"] or kept.get("light") or kept.get("illumination") or kept.get("fog"))
        # clear: no darkness, uneven light, lamps or fog; blur and sensor noise stay as drawn.
        assert not cleared["low_light"] and not cleared.get("light") and not cleared.get("illumination")
        assert not cleared.get("fog")
        for key in ("defocus", "defocus_sigma", "motion", "motion_length", "sensor_noise", "photon_count",
                    "read_noise_std", "jpeg", "exposure_gain"):
            assert cleared[key] == alone[key], key
        assert frame_kind(cleared) == frame_kind(kept) == "flare"
    assert lit > 0                                                     # without clear the environment does run


def test_the_bank_splits_whole_groups_per_source_and_shrinks_once_in_linear_light(tmp_path, monkeypatch):
    root = _write_flareimage(tmp_path / "flare_dataset")
    bank = _bank(root, tmp_path)
    rows = list(csv.DictReader((root / "metadata.csv").open()))
    group = {row["name"]: pair_group(row["source"], row["gt_src"]) for row in rows}
    assert pair_group("halo", rows[[r["source"] for r in rows].index("halo")]["gt_src"]).startswith("halo:Scene000_")
    assert pair_group("flare_removal", "/x/train_gt_2k/000117.png") == "flare_removal:11"
    homes = {}
    for split in ("train", "valid", "test"):
        for pair in bank.pairs(split):
            homes.setdefault(group[pair.name], set()).add(split)
    assert all(len(splits) == 1 for splits in homes.values())        # a group never straddles splits
    described = bank.describe()
    for source in SIZES:                                              # every source in every split
        assert all(described["sources"][split].get(source, 0) > 0 for split in ("train", "valid", "test")), source
        total = sum(row["source"] == source for row in rows)
        assert all(described["sources"][split][source] >= 0.1 * total for split in ("valid", "test"))
    # Shrink: short side to SHORT_SIDE, BOX in linear light -> at an integer ratio the mean light is kept; smaller pairs
    # are left as they are.
    big = next(p for p in bank.pairs("train") if p.source == "flare_removal")
    with Image.open(root / "train" / big.name) as image:
        original = np.asarray(image.convert("RGB"))
    opened = []
    real_open = Image.open
    monkeypatch.setattr(Image, "open", lambda path, *a, **k: (opened.append(str(path)), real_open(path, *a, **k))[1])
    gt, flared = bank.load(big)
    assert gt.shape == flared.shape == (96 * SHORT_SIDE // 80, SHORT_SIDE, 3)
    light = lambda image: srgb_to_linear(image.astype(np.float32) / 255).mean()
    assert light(flared) == pytest.approx(light(original), rel=5e-3)   # exact in float; 8-bit sRGB rounds ~0.3%
    decoded = sum(str(root) in path for path in opened)
    again = bank.load(big)
    assert all(np.array_equal(a, b) for a, b in zip(again, (gt, flared)))
    assert sum(str(root) in path for path in opened) == decoded == 2   # originals decoded once; then the cache
    small = next(p for p in bank.pairs("train") if p.source == "halo")
    with Image.open(root / "gt" / small.name) as image:
        assert np.array_equal(bank.load(small)[0], np.asarray(image.convert("RGB")))
    assert shrink_pair(original, original, 100)[0] is original
    # The bank refuses a folder without metadata or with pairs missing on disk.
    with pytest.raises(FileNotFoundError, match="metadata.csv"):
        FlarePairBank(tmp_path, short_side=SHORT_SIDE, holdout_fraction=0.1)
    (root / "train" / big.name).unlink()
    with pytest.raises(FileNotFoundError, match="thieu anh"):
        _bank(root, tmp_path)


def test_the_notebook_finds_flareimage_however_kaggle_mounted_it(tmp_path):
    api = _write_flareimage(tmp_path / "a/datasets/buidinhkhoi/flareimage/flare_dataset", per_source=12)
    assert find_flare_pairs_root(tmp_path / "a") == api
    (tmp_path / "a/datasets/buidinhkhoi/tartanairshard0/build_meta").mkdir(parents=True)
    (tmp_path / "a/datasets/buidinhkhoi/tartanairshard0/build_meta/metadata.csv").write_text("x\n")  # no gt/, train/
    assert find_flare_pairs_root(tmp_path / "a") == api
    with pytest.raises(FileNotFoundError, match="FlareImage"):
        find_flare_pairs_root(tmp_path / "empty")
    _write_flareimage(tmp_path / "a/datasets/other/flare-copy/flare_dataset", per_source=12)
    with pytest.raises(ValueError, match="hon mot lan"):
        find_flare_pairs_root(tmp_path / "a")


def test_kaggle_flareimage_is_p35_plus_the_flare_keys_with_halo_off_and_spells_them_out():
    tonegate = serializable_config(load_config("configs/kaggle_tonegate.yaml"))
    flare = serializable_config(load_config("configs/kaggle_flareimage.yaml"))
    assert set(FLARE) == set(FLARE_PAIR_KEYS) - {"flare_pairs_root"} and FLARE["flare_pairs_fraction"] == 0.25
    for config in (tonegate, flare):
        config.pop("_config_path", None)
        config["runtime"].pop("output_dir")
    for key in FLARE_PAIR_KEYS:
        flare["data"].pop(key)
    assert flare["corruption"]["image"].pop("halo_probability") == 0.0
    tonegate["corruption"]["image"].pop("halo_probability")
    assert flare == tonegate
    full_tonegate, full_flare = load_config("configs/kaggle_tonegate.yaml"), load_config("configs/kaggle_flareimage.yaml")
    validate_config(full_flare)
    moved = copy.deepcopy(full_flare)
    moved["data"]["flare_pairs_root"] = "/elsewhere/flare_dataset"
    for phase in ("phase1", "phase2"):
        assert configuration_hash(full_tonegate, phase) != configuration_hash(full_flare, phase)
        assert configuration_hash(moved, phase) == configuration_hash(full_flare, phase)   # a path, not the recipe
    # p35 and the configs before it hash exactly as before the flare keys existed.
    assert configuration_hash(full_tonegate, "phase1").startswith("5600ba61e9ca")
    assert configuration_hash(full_tonegate, "phase2").startswith("87dcbd919b1b")
    assert configuration_hash(load_config("configs/kaggle_relight.yaml"), "phase1").startswith("d2e4c470cb9c")
    for key in FLARE_PAIR_KEYS:
        missing = copy.deepcopy(full_flare)
        del missing["data"][key]
        if key == "flare_pairs_fraction":
            validate_config(missing)                  # no fraction: the pairs are off, as before the keys existed
            continue
        with pytest.raises(ValueError, match=key):
            validate_config(missing)
    for key, value in (("flare_pairs_fraction", 0.6), ("flare_pairs_fraction", -0.1), ("flare_pairs_short_side", 128),
                       ("flare_pairs_short_side", 384.0), ("flare_pairs_holdout_fraction", 0.0),
                       ("flare_pairs_holdout_fraction", 0.5), ("flare_pairs_clear_probability", 1.5),
                       ("flare_pairs_root", ""), ("flare_pairs_typo", 1)):
        bad = copy.deepcopy(full_flare)
        bad["data"][key] = value
        with pytest.raises(ValueError):
            validate_config(bad)
    crowded = copy.deepcopy(full_flare)               # one FlareImage slot of a phase-2 batch of 2 leaves 1 < 2
    crowded["phase2"]["batch_size"] = 2
    with pytest.raises(ValueError, match="minimum_trajectories_per_batch"):
        validate_config(crowded)


def _train_tiny(tmp_path, name, with_pairs):
    config, _, path, manifest_dir = _smoke(tmp_path)
    if not with_pairs:
        config["data"] = {key: value for key, value in config["data"].items() if not key.startswith("flare_pairs_")}
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / name
    common = ["--config", str(path), "--manifest", str(manifest_dir), "--output", str(output)]
    _run_until_done(["train-phase1", *common], output / "phase1/last.pt")
    last = output / "phase2/last.pt"
    _run_until_done(["train-phase2", *common, "--backbone-checkpoint", str(output / "phase1/last.pt")], last)
    return last, manifest_dir


def test_both_phases_train_with_the_pairs_and_the_probe_scores_the_held_out_pairs(tmp_path):
    import subprocess
    import sys
    torch.set_num_threads(1)
    last, manifest = _train_tiny(tmp_path, "flare", True)
    assert load_checkpoint(last)["successful_updates"] == 3
    plain, _ = _train_tiny(tmp_path, "plain", False)                    # trained without pairs: needs the root
    flare_root = tmp_path / "flare_dataset"
    held_out = {pair.name for pair in FlarePairBank(flare_root, short_side=SHORT_SIDE, holdout_fraction=0.1,
                                                    cache_dir=tmp_path / "cache").pairs("valid")}
    for checkpoint, extra, trained in ((last, [], True), (plain, ["--flare-pairs-root", str(flare_root)], False)):
        report = tmp_path / f"probe_{trained}.json"
        result = subprocess.run([sys.executable, "tools/flare_pairs_probe.py", "--checkpoint", str(checkpoint),
                                 "--manifest", str(manifest), "--samples", "6", "--batch", "4", "--device", "cpu",
                                 "--output", str(report), *extra], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr[-3000:]
        table = json.loads(report.read_text())
        assert table["trained_with_flare_pairs"] is trained and table["split"] == "valid"
        for label, mode in (("chi loe", "clean"), ("nhu luc train", "full")):
            whole = table["modes"][label]["all"]
            assert table["modes"][label]["image_mode"] == mode and whole["count"] == 6
            assert np.isfinite(whole["input_psnr_db"]) and np.isfinite(whole["restored_psnr_db"])
            names = {row["name"] for row in table["modes"][label]["pairs"]}
            assert len(names) == 6 and names <= held_out                # only pairs held out of training
        assert "chi loe" in result.stdout and "nhu luc train" in result.stdout
    no_root = subprocess.run([sys.executable, "tools/flare_pairs_probe.py", "--checkpoint", str(plain), "--manifest",
                              str(manifest), "--samples", "2", "--device", "cpu"], capture_output=True, text=True)
    assert no_root.returncode != 0 and "--flare-pairs-root" in no_root.stderr + no_root.stdout


def _records(run, phase, key):
    lines = (run / phase / "train.jsonl").read_text().splitlines()
    return [record[key] for record in map(json.loads, lines) if key in record and "loss" in record]


def test_two_processes_train_with_the_pairs_what_one_does(tmp_path):
    """Kaggle trains on two T4s (DDP): each rank loads its contiguous share of the batch -- here one TartanAir sample
    and one FlareImage pair -- and the gathered batch must be the one-process batch, through restarts too."""
    torch.set_num_threads(1)
    config, _, _, manifest = _smoke(tmp_path, phase1_batch=2, phase2_batch=2)
    config["phase1"].update(max_successful_updates=3, augment_hflip=False)
    config["phase2"].update(gradient_accumulation=2, max_successful_updates=3, augment_hflip=False)
    config["runtime"]["restart_above_rss_gib"] = None
    single = tmp_path / "single.yaml"
    single.write_text(yaml.safe_dump(config))
    config["runtime"].update(parallel="ddp", gpu_count=2, restart_above_rss_gib=1e-6)
    ddp = tmp_path / "ddp.yaml"
    ddp.write_text(yaml.safe_dump(config))
    for name, path in (("single", single), ("ddp", ddp)):
        output = tmp_path / name
        common = ["--config", str(path), "--manifest", str(manifest), "--output", str(output)]
        restarts = _run_until_done(["train-phase1", *common], output / "phase1/last.pt")
        restarts += _run_until_done(["train-phase2", *common, "--backbone-checkpoint", str(output / "phase1/last.pt")],
                                    output / "phase2/last.pt")
        assert restarts == (4 if name == "ddp" else 0)
    assert json.loads((tmp_path / "ddp/phase2/execution.json").read_text())["backend"] == "ddp"
    for phase in ("phase1", "phase2"):
        for key in ("loss", "gradient_norm"):
            single_values, ddp_values = (_records(tmp_path / name, phase, key) for name in ("single", "ddp"))
            assert len(single_values) == len(ddp_values) == 3, (phase, key)
            assert ddp_values == pytest.approx(single_values, rel=2e-3), (phase, key)
            assert ddp_values[0] == pytest.approx(single_values[0], rel=1e-4), (phase, key)
