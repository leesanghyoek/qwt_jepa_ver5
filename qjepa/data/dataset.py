"""Paired clean/corrupted RGB and IMU dataset."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..corruptions import LowLightImageCorruptor, TrajectoryImuCorruptor
from ..corruptions.image import brightness_stops, degradation_vector
from ..corruptions.rng import derive_seed
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
    ) -> None:
        if not samples:
            raise ValueError("Dataset cannot be empty")
        if not 0.0 <= hflip_probability <= 1.0:
            raise ValueError("hflip_probability must be in [0, 1]")
        # The frame without sensor grain, for the phase-1 Jacobian's noise direction.
        self.sensor_reference = sensor_reference
        # phase2.split_relight: the stops the corruption moved each pixel's light by ("image_stops").
        self.brightness_target = bool(brightness_target)
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
        image_mode, imu_mode = self.image_mode, self.imu_mode
        if self.scenarios:
            rng = np.random.default_rng(derive_seed(
                self.scenario_seed, "phase2_scenario", sample.sample_id, self.realization,
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

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image_mode, imu_mode = self._scenario_modes(sample)
        imu_all, imu_times_all = self.cache.get(sample)
        clean_image = self._clean_frame(sample)
        clean_imu = np.asarray(imu_all[sample.imu_start : sample.imu_end], dtype=np.float32)
        imu_times = np.asarray(imu_times_all[sample.imu_start : sample.imu_end], dtype=np.float64)
        corrupt = dict(
            split=sample.split,
            realization=self.realization,
            trajectory=sample.trajectory_key,
            timestamp=sample.image_time,
            frame_index=sample.image_index,
            mode=image_mode,
            # The CLEAN gyro drives the blur: the true motion is what smears the
            # frame, while the IMU branch only ever sees a noisy measurement of
            # it. That gap is the task the decoder has to close.
            gyro=clean_imu[:, 3:6].astype(np.float64),
            imu_times=imu_times,
        )
        noise_free = None
        if self.sensor_reference:
            noisy_image, noise_free, image_parameters = self.image_corruptor.render_with_sensor_reference(
                clean_image, **corrupt)
        else:
            noisy_image, image_parameters = self.image_corruptor(clean_image, **corrupt)
        noisy_imu, imu_parameters = self.imu_corruptor.window(
            imu_all,
            imu_times_all,
            sample.imu_start,
            sample.imu_end,
            split=sample.split,
            realization=self.realization,
            trajectory=sample.trajectory_key,
            mode=imu_mode,
        )
        stops = brightness_stops(image_parameters, *noisy_image.shape[:2]) if self.brightness_target else None
        if mirror_draw(self.scenario_seed, sample.sample_id, self.realization, self.hflip_probability):
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
            "image_time": torch.tensor(sample.image_time, dtype=torch.float64),
            "imu_times": torch.from_numpy(imu_times),
            "imu_start": torch.tensor(sample.imu_start),
            "image_degradation": torch.from_numpy(degradation_vector(image_parameters)),
            "sample_id": sample.sample_id,
            "trajectory_key": sample.trajectory_key,
            "corruption": {"image": image_parameters, "imu": imu_parameters},
        }
        if noise_free is not None:
            item["image_noise_free"] = chw(noise_free)
        if stops is not None:
            item["image_stops"] = torch.from_numpy(np.ascontiguousarray(stops))[None]
        return item


def collate_paired(batch: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key in ("image_clean", "image_noisy", "image_time", "imu_times", "imu_start"):
        output[key] = torch.stack([item[key] for item in batch])  # type: ignore[list-item]
    for key in ("image_degradation", "image_noise_free", "image_stops"):
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
