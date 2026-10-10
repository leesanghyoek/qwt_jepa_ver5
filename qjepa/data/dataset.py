"""Paired clean/corrupted RGB and IMU dataset."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..corruptions import LowLightImageCorruptor, TrajectoryImuCorruptor
from ..corruptions.image import brightness_stops, degradation_vector, frame_kind
from ..corruptions.rng import derive_seed
from .flare_pairs import FlarePairBank
from .manifest import PairedSample
from .tartanair import Trajectory

# A horizontal flip mirrors the world across the camera's forward-down plane.
# lcam_front is the IMU body frame (x forward, y right, z down; see
# corruptions/motion.py), so the polar accel flips y and the axial gyro flips
# every component but y: [ax, -ay, az, -gx, gy, -gz].
IMU_MIRROR_SIGNS = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)


def mirror_imu(imu: np.ndarray) -> np.ndarray:
    """The IMU window [L, 6] of the horizontally mirrored scene."""
    return imu * np.asarray(IMU_MIRROR_SIGNS, dtype=imu.dtype)


def mirror_draw(seed: int, sample_id: str, realization: int, probability: float) -> bool:
    """Whether this sample is mirrored in this realization: reproducible, per sample."""
    if probability <= 0.0:
        return False
    rng = np.random.default_rng(derive_seed(seed, "hflip", sample_id, realization))
    return bool(rng.random() < probability)


def load_rgb(path: str | Path, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        target_height, target_width = size
        if image.size != (target_width, target_height):
            scale = max(target_width / image.width, target_height / image.height)
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
            )
            left = (image.width - target_width) // 2
            top = (image.height - target_height) // 2
            image = image.crop((left, top, left + target_width, top + target_height))
        return np.asarray(image, dtype=np.float32) / 255.0


class _TrajectoryCache:
    def __init__(self, size: int = 4):
        self.size = size
        self.values: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, sample: PairedSample) -> tuple[np.ndarray, np.ndarray]:
        key = sample.trajectory_key
        if key not in self.values:
            trajectory = Trajectory(
                path=Path(sample.trajectory_path),
                environment=sample.environment,
                difficulty=sample.difficulty,
                trajectory_id=sample.trajectory_id,
            )
            self.values[key] = trajectory.load_imu()
            if len(self.values) > self.size:
                self.values.popitem(last=False)
        else:
            self.values.move_to_end(key)
        return self.values[key]


# Moi worker giu IMU sach va IMU da lam hong cua toi da bay nhieu quy dao. Quy dao ~4000
# mau IMU ton ~0,4 MB cho ca hai ban, nen 2048 quy dao la tran ~0,8 GB moi worker.
MAX_CACHED_TRAJECTORIES = 2048


class PairedCameraImuDataset(Dataset):
    def __init__(
        self,
        samples: list[PairedSample],
        *,
        image_corruptor: LowLightImageCorruptor | None = None,
        imu_corruptor: TrajectoryImuCorruptor | None = None,
        image_size: tuple[int, int] = (256, 256),
        realization: int = 0,
        image_mode: str = "full",
        imu_mode: str = "full",
        scenarios: list[dict[str, object]] | None = None,
        scenario_seed: int = 0,
        cache_size: int = 4,
        sensor_reference: bool = False,
        hflip_probability: float = 0.0,
        source_size: tuple[int, int] | None = None,
        full_frame: bool = False,
        brightness_target: bool = False,
        tone_label: bool = False,
        flare_pairs: FlarePairBank | None = None,
        flare_pairs_fraction: float = 0.0,
        flare_pairs_clear_probability: float = 1.0,
    ) -> None:
        if not samples:
            raise ValueError("Dataset cannot be empty")
        if not 0.0 <= hflip_probability <= 1.0:
            raise ValueError("hflip_probability must be in [0, 1]")
        # data.flare_pairs_*: FlareImage pairs fill flare_pairs_per_batch() slots of every training batch
        # (sampler.FlarePairSlots asks for them with indices past the TartanAir samples). None: as before.
        if flare_pairs is not None and not 0.0 < flare_pairs_fraction < 1.0:
            raise ValueError("flare_pairs_fraction must lie in (0, 1) when FlareImage pairs are given")
        self.flare_bank = flare_pairs
        self.flare_pairs_split = "train"   # tools/flare_pairs_probe.py reads the held-out pairs instead
        self.flare_pairs_fraction = float(flare_pairs_fraction)
        self.flare_pairs_clear_probability = float(flare_pairs_clear_probability)
        # The frame without sensor grain, for the phase-1 Jacobian's noise direction.
        self.sensor_reference = sensor_reference
        # phase2.split_relight: the stops the corruption moved each pixel's light by ("image_stops").
        self.brightness_target = bool(brightness_target)
        # phase2.split_tone_gate: 1 when the corruption changed the frame's light or added a flare, else 0.
        self.tone_label = bool(tone_label)
        self.hflip_probability = hflip_probability
        self.samples = samples
        self.image_corruptor = image_corruptor or LowLightImageCorruptor()
        self.imu_corruptor = imu_corruptor or TrajectoryImuCorruptor(cache_size=cache_size)
        self.image_size = image_size
        # data.source_size: the frame is read at this size (TartanAir 640: its own pixels, no
        # downscale) and a crop of image_size goes through the corruption -- at a place drawn
        # per (sample, realization), so every epoch sees another crop and the fixed validation
        # realization always the same one. full_frame: the whole source frame (inference and
        # full-resolution evaluation; the model is fully convolutional). None: read at
        # image_size, as before the key existed.
        if source_size is not None and (source_size[0] < image_size[0] or source_size[1] < image_size[1]):
            raise ValueError("source_size must be at least image_size")
        self.source_size = None if source_size is None else tuple(int(v) for v in source_size)
        self.full_frame = bool(full_frame) and self.source_size is not None
        self.realization = realization
        self.image_mode = image_mode
        self.imu_mode = imu_mode
        self.scenarios = scenarios
        self.scenario_seed = scenario_seed
        # Ca hai cache giu MOI quy dao cua split (toi da MAX_CACHED_TRAJECTORIES). Nhieu IMU duoc
        # dung cho ca quy dao, theo seed (split, realization, quy dao, mode), roi moi cat cua so;
        # voi cache 4-8 quy dao ma batch lay ngau nhien tu hang tram quy dao, gan nhu mau nao cung
        # dung lai tu dau (~10 ms/mau, 15-22% thoi gian CPU cua mot mau). Ket qua giong het tung bit.
        keys = {getattr(sample, "trajectory_key", None) for sample in samples} - {None}
        trajectories = min(len(keys), MAX_CACHED_TRAJECTORIES)
        self.cache = _TrajectoryCache(max(cache_size, trajectories))
        self.imu_corruptor.cache_size = max(self.imu_corruptor.cache_size, trajectories)

    def set_realization(self, realization: int) -> None:
        self.realization = int(realization)

    def __len__(self) -> int:
        return len(self.samples)

    def _scenario_modes(self, sample: PairedSample) -> tuple[str, str]:
        return self._modes(sample.sample_id)

    def _modes(self, sample_id: str) -> tuple[str, str]:
        image_mode, imu_mode = self.image_mode, self.imu_mode
        if self.scenarios:
            rng = np.random.default_rng(derive_seed(
                self.scenario_seed, "phase2_scenario", sample_id, self.realization,
            ))
            weights = np.asarray([float(item["weight"]) for item in self.scenarios], dtype=np.float64)
            choice = self.scenarios[int(rng.choice(len(weights), p=weights / weights.sum()))]
            image_mode, imu_mode = str(choice["image_mode"]), str(choice["imu_mode"])
        return image_mode, imu_mode

    def crop_origin(self, sample: PairedSample) -> tuple[int, int]:
        """Top-left corner of this sample's crop in its source frame (deterministic)."""
        rng = np.random.default_rng(derive_seed(self.scenario_seed, "source_crop", sample.sample_id, self.realization))
        return (int(rng.integers(0, self.source_size[0] - self.image_size[0] + 1)),
                int(rng.integers(0, self.source_size[1] - self.image_size[1] + 1)))

    def _clean_frame(self, sample: PairedSample) -> np.ndarray:
        if self.source_size is None:
            return load_rgb(sample.image_path, self.image_size)
        frame = load_rgb(sample.image_path, self.source_size)
        if self.full_frame:
            return frame
        top, left = self.crop_origin(sample)
        return np.ascontiguousarray(frame[top:top + self.image_size[0], left:left + self.image_size[1]])

    def flare_pairs_per_batch(self, batch_size: int) -> int:
        """How many slots of a training batch of ``batch_size`` hold a FlareImage pair (0 without data.flare_pairs_*)."""
        if self.flare_bank is None:
            return 0
        return max(1, round(self.flare_pairs_fraction * batch_size))

    def __getitem__(self, index: int) -> dict[str, object]:
        if index >= len(self.samples):
            # Past the TartanAir samples: the draw-th FlareImage pair of this realization (sampler.FlarePairSlots).
            return self._flare_item(index - len(self.samples))
        sample = self.samples[index]
        image_mode, imu_mode = self._scenario_modes(sample)
        clean_image = self._clean_frame(sample)
        return self._item(sample, sample.sample_id, sample.trajectory_key, clean_image, clean_image,
                          image_mode, imu_mode, trajectory=sample.trajectory_key, timestamp=sample.image_time,
                          frame_index=sample.image_index)

    def _flare_item(self, draw: int) -> dict[str, object]:
        """A FlareImage pair: its flared photo is the input, its clean photo the target, both the same crop.

        The pair has no IMU, so it borrows the clean and noisy IMU window of a TartanAir sample drawn here: every
        loss still compares an IMU window with itself, only image and IMU are not one motion. Draw ``draw`` of a
        realization walks a fresh permutation of the training pairs per cycle, so each pair comes once per cycle.
        """
        pairs = self.flare_bank.pairs(self.flare_pairs_split)
        cycle, slot = divmod(int(draw), len(pairs))
        order = np.random.default_rng(derive_seed(
            self.scenario_seed, "flare_pair_order", self.realization, cycle)).permutation(len(pairs))
        pair = pairs[int(order[slot])]
        rng = np.random.default_rng(derive_seed(self.scenario_seed, "flare_pair", self.realization, draw))
        host = self.samples[int(rng.integers(len(self.samples)))]
        clear = bool(rng.random() < self.flare_pairs_clear_probability)
        gt, flared = self.flare_bank.load(pair)
        height, width = self.image_size
        if gt.shape[0] < height or gt.shape[1] < width:
            raise ValueError(f"FlareImage {pair.name}: {gt.shape[1]}x{gt.shape[0]} is smaller than the crop "
                             f"{width}x{height}; raise data.flare_pairs_short_side or use larger pairs")
        top = int(rng.integers(0, gt.shape[0] - height + 1))
        left = int(rng.integers(0, gt.shape[1] - width + 1))
        crop = lambda image: np.ascontiguousarray(image[top:top + height, left:left + width], dtype=np.float32) / 255.0
        sample_id = f"flareimage/{pair.name}#{self.realization}:{draw}"
        image_mode, imu_mode = self._modes(sample_id)
        return self._item(host, sample_id, f"flareimage/{pair.source}", crop(gt), crop(flared), image_mode,
                          imu_mode, trajectory=sample_id, timestamp=0.0, frame_index=0,
                          flare_pair={"name": pair.name, "source": pair.source, "clear": clear},
                          imu_donor=host.sample_id)

    def _item(self, host: PairedSample, sample_id: str, trajectory_key: str, clean_image: np.ndarray,
              source_image: np.ndarray, image_mode: str, imu_mode: str, *, trajectory: str, timestamp: float,
              frame_index: int, flare_pair: dict[str, object] | None = None,
              imu_donor: str | None = None) -> dict[str, object]:
        """One training/validation item: ``source_image`` through the corruption is the input, ``clean_image`` the
        target, and the IMU window is ``host``'s. For a TartanAir sample source and target are its own frame."""
        imu_all, imu_times_all = self.cache.get(host)
        clean_imu = np.asarray(imu_all[host.imu_start : host.imu_end], dtype=np.float32)
        imu_times = np.asarray(imu_times_all[host.imu_start : host.imu_end], dtype=np.float64)
        corrupt = dict(
            split=host.split,
            realization=self.realization,
            trajectory=trajectory,
            timestamp=timestamp,
            frame_index=frame_index,
            mode=image_mode,
            # The CLEAN gyro drives the blur: the true motion is what smears the
            # frame, while the IMU branch only ever sees a noisy measurement of
            # it. That gap is the task the decoder has to close.
            gyro=clean_imu[:, 3:6].astype(np.float64),
            imu_times=imu_times,
        )
        if flare_pair is not None:
            corrupt["flare_pair"] = flare_pair
        noise_free = None
        if self.sensor_reference:
            noisy_image, noise_free, image_parameters = self.image_corruptor.render_with_sensor_reference(
                source_image, **corrupt)
        else:
            noisy_image, image_parameters = self.image_corruptor(source_image, **corrupt)
        noisy_imu, imu_parameters = self.imu_corruptor.window(
            imu_all,
            imu_times_all,
            host.imu_start,
            host.imu_end,
            split=host.split,
            realization=self.realization,
            trajectory=host.trajectory_key,
            mode=imu_mode,
        )
        stops = brightness_stops(image_parameters, *noisy_image.shape[:2]) if self.brightness_target else None
        if mirror_draw(self.scenario_seed, sample_id, self.realization, self.hflip_probability):
            # After corruption: every corruption draw is mirror-symmetric in distribution,
            # and the seeds stay those of the unmirrored sample.
            clean_image, noisy_image = clean_image[:, ::-1], noisy_image[:, ::-1]
            stops = None if stops is None else stops[:, ::-1]
            noise_free = None if noise_free is None else noise_free[:, ::-1]
            clean_imu, noisy_imu = mirror_imu(clean_imu), mirror_imu(noisy_imu)
        chw = lambda value: torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1))).float()
        item = {
            "image_clean": chw(clean_image),
            "image_noisy": chw(noisy_image),
            "imu_clean_phys": torch.from_numpy(np.ascontiguousarray(clean_imu)).float(),
            "imu_noisy_phys": torch.from_numpy(np.ascontiguousarray(noisy_imu)).float(),
            "image_time": torch.tensor(host.image_time, dtype=torch.float64),
            "imu_times": torch.from_numpy(imu_times),
            "imu_start": torch.tensor(host.imu_start),
            "image_degradation": torch.from_numpy(degradation_vector(image_parameters)),
            "sample_id": sample_id,
            "trajectory_key": trajectory_key,
            "corruption": {"image": image_parameters, "imu": imu_parameters},
        }
        if imu_donor is not None:
            item["corruption"]["imu_donor"] = imu_donor
        if noise_free is not None:
            item["image_noise_free"] = chw(noise_free)
        if stops is not None:
            item["image_stops"] = torch.from_numpy(np.ascontiguousarray(stops))[None]
        if self.tone_label:
            item["image_tone_label"] = torch.tensor([float(frame_kind(image_parameters) != "plain")])
        return item


def collate_paired(batch: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key in ("image_clean", "image_noisy", "image_time", "imu_times", "imu_start"):
        output[key] = torch.stack([item[key] for item in batch])  # type: ignore[list-item]
    for key in ("image_degradation", "image_noise_free", "image_stops", "image_tone_label"):
        if key in batch[0]:
            output[key] = torch.stack([item[key] for item in batch])  # type: ignore[list-item]
    for key in ("imu_clean_phys", "imu_noisy_phys"):
        stacked = torch.stack([item[key] for item in batch])  # type: ignore[list-item]
        output[key] = stacked.transpose(1, 2).contiguous()
    # Keep absolute timestamps in float64; float32 loses 100 Hz spacing at Unix epochs.
    output["sample_id"] = [item["sample_id"] for item in batch]
    output["trajectory_key"] = [item["trajectory_key"] for item in batch]
    output["corruption"] = [item["corruption"] for item in batch]
    return output
