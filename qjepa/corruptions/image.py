"""Camera corruption focused on blurred, noisy, low-light captures.

The order follows a simple image-formation model: optical blur and resolution
loss happen before exposure reduction; shot/read noise happens after exposure;
quantization/JPEG happens last. Camera parameters are stable within a short
time segment while random sensor noise remains frame-specific.
"""

from __future__ import annotations

import io
import math
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from .motion import imu_blur_kernel
from .rng import generator

# "blur_low_light" is optics plus exposure and nothing else. "blur_only" leaves
# the frame at full brightness and "full" buries the blur under sensor grain, so
# neither shows what blur looks like on a dark frame -- which is the case this
# project is actually built for.
IMAGE_MODES = ("full", "clean", "low_light_only", "blur_only", "sensor_noise_only",
               "blur_low_light")


@dataclass(frozen=True)
class LowLightImageCorruptionConfig:
    clean_probability: float = 0.02
    segment_seconds: float = 0.05
    exposure_gain: tuple[float, float] = (0.21, 0.63)
    tone_gamma: tuple[float, float] = (0.46, 0.79)
    white_balance_gain: tuple[float, float] = (0.82, 1.18)
    black_level: tuple[float, float] = (-0.01, 0.01)
    vignette_strength: tuple[float, float] = (0.0, 0.45)
    defocus_probability: float = 0.68
    defocus_sigma_px: tuple[float, float] = (0.30, 1.45)
    motion_probability: float = 0.55
    motion_length_px: tuple[int, int] = (3, 9)
    # Motion blur integrated from the IMU the model also sees; see motion.py.
    # Off by default, matching configs/pipeline_v3.yaml: this project treats image
    # blur as the camera's doing and IMU noise as the environment's, two unrelated
    # causes. Turn it on to couple them, and `motion_probability`/
    # `motion_length_px`/`motion_angle` stop applying.
    motion_from_imu: bool = False
    exposure_seconds: tuple[float, float] = (0.008, 0.030)
    # A darker frame means the shutter stayed open longer, so low light and heavy
    # blur arrive together instead of being drawn independently.
    exposure_tracks_darkness: bool = True
    # TartanAir V2 lcam_front: 640 px at 90 deg FOV -> f = 320 px, and load_rgb
    # rescales to 256 px, so f = 128 px. Override if the crop or camera changes.
    focal_length_px: float = 128.0
    angular_gain: float = 1.0
    motion_path_samples: int = 25
    motion_max_radius_px: float = 16.0
    downsample_probability: float = 0.32
    downsample_scale: tuple[float, float] = (0.72, 0.96)
    photon_count: tuple[float, float] = (550.0, 4000.0)
    read_noise_std: tuple[float, float] = (0.7 / 255.0, 3.2 / 255.0)
    row_noise_std: tuple[float, float] = (0.0, 1.0 / 255.0)
    hot_pixel_probability: tuple[float, float] = (0.0, 1.5e-4)
    quantization_bits: tuple[int, int] = (6, 8)
    jpeg_probability: float = 0.35
    jpeg_quality: tuple[int, int] = (35, 80)
    # Per-frame variety in mode "full": a frame may skip the low-light stage (bright,
    # sensor noise only) or the sensor stage (dark, no grain); the rest get both.
    # Drawn LAST in `_parameters`, so every earlier draw -- and a config without
    # these keys -- is bit-identical to before they existed.
    noise_only_probability: float = 0.0
    low_light_only_probability: float = 0.0

    def validate(self) -> None:
        if not 0 <= self.clean_probability <= 1:
            raise ValueError("clean_probability must be in [0,1]")
        if (not 0 <= self.noise_only_probability <= 1 or not 0 <= self.low_light_only_probability <= 1
                or self.noise_only_probability + self.low_light_only_probability > 1):
            raise ValueError("noise_only_probability and low_light_only_probability must be in [0,1] "
                             "and sum to at most 1")
        if self.segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive")
        if self.exposure_gain[0] <= 0 or self.exposure_gain[1] > 1:
            raise ValueError("exposure_gain must stay in (0,1]")
        if self.photon_count[0] <= 0:
            raise ValueError("photon_count must be positive")
        low, high = self.exposure_seconds
        if not 0 < low <= high:
            raise ValueError("exposure_seconds must satisfy 0 < low <= high")
        if high >= 0.1:
            raise ValueError("exposure_seconds must stay below the 10 Hz frame period")
        if self.focal_length_px <= 0:
            raise ValueError("focal_length_px must be positive")
        if self.angular_gain <= 0:
            raise ValueError("angular_gain must be positive")
        if self.motion_max_radius_px <= 0:
            raise ValueError("motion_max_radius_px must be positive")
        if self.motion_path_samples < 3:
            raise ValueError("motion_path_samples must be at least 3")


def _motion_kernel(length: int, angle_radians: float) -> np.ndarray:
    length = max(1, int(length))
    size = length if length % 2 else length + 1
    kernel = np.zeros((size, size), dtype=np.float64)
    centre = (size - 1) / 2.0
    samples = max(2 * size + 1, 5)
    offsets = np.linspace(-(length - 1) / 2.0, (length - 1) / 2.0, samples)
    for offset in offsets:
        y = centre + offset * math.sin(angle_radians)
        x = centre + offset * math.cos(angle_radians)
        y0, x0 = int(math.floor(y)), int(math.floor(x))
        for yy, wy in ((y0, 1.0 - (y - y0)), (y0 + 1, y - y0)):
            for xx, wx in ((x0, 1.0 - (x - x0)), (x0 + 1, x - x0)):
                if 0 <= yy < size and 0 <= xx < size:
                    kernel[yy, xx] += wy * wx
    total = kernel.sum()
    return kernel / total if total > 0 else np.array([[1.0]])


def _resize_roundtrip(image: np.ndarray, scale: float) -> np.ndarray:
    height, width = image.shape[:2]
    small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    pil = Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255.0), mode="RGB")
    pil = pil.resize(small_size, Image.Resampling.BILINEAR)
    pil = pil.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.float64) / 255.0


def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255.0), mode="RGB").save(
        buffer, format="JPEG", quality=int(quality), subsampling=2
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.float64) / 255.0


def _vignette(height: int, width: int, strength: float) -> np.ndarray:
    yy = np.linspace(-1.0, 1.0, height)[:, None]
    xx = np.linspace(-1.0, 1.0, width)[None, :]
    radius2 = np.clip((xx * xx + yy * yy) / 2.0, 0.0, 1.0)
    return np.clip(1.0 - strength * radius2, 0.05, 1.0)[..., None]


class LowLightImageCorruptor:
    def __init__(self, config: LowLightImageCorruptionConfig | None = None, master_seed: int = 73128):
        self.config = config or LowLightImageCorruptionConfig()
        self.config.validate()
        self.master_seed = master_seed

    def _parameters(
        self, split: str, realization: int, trajectory: str, timestamp: float, mode: str
    ) -> dict[str, object]:
        if mode not in IMAGE_MODES:
            raise ValueError(f"Unsupported image corruption mode {mode!r}")
        segment = math.floor(timestamp / self.config.segment_seconds)
        rng = generator(self.master_seed, "image_parameters", split, realization, trajectory, segment)
        cfg = self.config
        # Draw unconditionally. Written as `mode == "full" and rng.random() < p`
        # the call short-circuits away for every other mode, which shifts the
        # whole parameter stream by one and leaves each named scenario looking at
        # a different camera than "full" did on the same frame -- measured as
        # exposure_gain 0.62 vs 0.27. Isolating one degradation only means
        # something while the other parameters stay put.
        # "full" is unaffected by this change, so training data is bit-identical.
        clean_draw = rng.random()
        clean = mode == "clean" or (mode == "full" and clean_draw < cfg.clean_probability)
        exposure_gain = float(rng.uniform(*cfg.exposure_gain))
        exposure_seconds = float(rng.uniform(*cfg.exposure_seconds))
        if cfg.exposure_tracks_darkness:
            gain_low, gain_high = cfg.exposure_gain
            span = gain_high - gain_low
            # brightness 0 = darkest draw -> longest shutter.
            brightness = (exposure_gain - gain_low) / span if span > 0 else 0.5
            low, high = cfg.exposure_seconds
            exposure_seconds = float(high - brightness * (high - low))
        parameters: dict[str, object] = {
            "mode": mode,
            "clean": bool(clean),
            "segment": segment,
            "exposure_gain": exposure_gain,
            "exposure_seconds": exposure_seconds,
            "motion_from_imu": bool(cfg.motion_from_imu),
            "tone_gamma": float(rng.uniform(*cfg.tone_gamma)),
            "white_balance": rng.uniform(*cfg.white_balance_gain, size=3).tolist(),
            "black_level": float(rng.uniform(*cfg.black_level)),
            "vignette_strength": float(rng.uniform(*cfg.vignette_strength)),
            "defocus": bool(rng.random() < cfg.defocus_probability),
            "defocus_sigma": float(rng.uniform(*cfg.defocus_sigma_px)),
            "motion": bool(rng.random() < cfg.motion_probability),
            "motion_length": int(rng.integers(cfg.motion_length_px[0], cfg.motion_length_px[1] + 1)),
            "motion_angle": float(rng.uniform(0.0, math.pi)),
            "downsample": bool(rng.random() < cfg.downsample_probability),
            "downsample_scale": float(rng.uniform(*cfg.downsample_scale)),
            "photon_count": float(np.exp(rng.uniform(np.log(cfg.photon_count[0]), np.log(cfg.photon_count[1])))),
            "read_noise_std": float(rng.uniform(*cfg.read_noise_std)),
            "row_noise_std": float(rng.uniform(*cfg.row_noise_std)),
            "hot_pixel_probability": float(rng.uniform(*cfg.hot_pixel_probability)),
            "quantization_bits": int(rng.integers(cfg.quantization_bits[0], cfg.quantization_bits[1] + 1)),
            "jpeg": bool(rng.random() < cfg.jpeg_probability),
            "jpeg_quality": int(rng.integers(cfg.jpeg_quality[0], cfg.jpeg_quality[1] + 1)),
        }
        variant = float(rng.random())
        noise_only = variant < cfg.noise_only_probability
        low_light_only = (cfg.noise_only_probability <= variant
                          < cfg.noise_only_probability + cfg.low_light_only_probability)
        parameters.update(variant_draw=variant, low_light=not noise_only, sensor_noise=not low_light_only)
        return parameters

    def __call__(
        self,
        image_clean: np.ndarray,
        *,
        split: str,
        realization: int,
        trajectory: str,
        timestamp: float,
        frame_index: int,
        mode: str = "full",
        gyro: np.ndarray | None = None,
        imu_times: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if image_clean.ndim != 3 or image_clean.shape[-1] != 3:
            raise ValueError(f"Expected image [H,W,3], got {image_clean.shape}")
        if not np.isfinite(image_clean).all() or image_clean.min() < 0 or image_clean.max() > 1:
            raise ValueError("Clean image must be finite and in [0,1]")
        if self.config.motion_from_imu and (gyro is None or imu_times is None):
            # Falling back to a random draw here would quietly undo the coupling
            # the whole design rests on, so refuse instead.
            raise ValueError(
                "motion_from_imu is enabled but no gyro/imu_times were supplied; "
                "pass the clean gyro window or set motion_from_imu: false"
            )
        params = self._parameters(split, realization, trajectory, timestamp, mode)
        if params["clean"]:
            return image_clean.astype(np.float32, copy=True), params

        optical = mode in ("full", "blur_only", "blur_low_light")
        # The named scenarios keep their meaning; only "full" draws the per-frame variant.
        low_light = mode in ("low_light_only", "blur_low_light") or (mode == "full" and params["low_light"])
        sensor_noise = mode == "sensor_noise_only" or (mode == "full" and params["sensor_noise"])
        image = image_clean.astype(np.float64, copy=True)

        if optical and params["defocus"]:
            image = ndimage.gaussian_filter(
                image, sigma=(params["defocus_sigma"], params["defocus_sigma"], 0), mode="reflect"
            )
        kernel = None
        if optical and self.config.motion_from_imu:
            kernel, report = imu_blur_kernel(
                np.asarray(gyro, dtype=np.float64),
                np.asarray(imu_times, dtype=np.float64),
                float(timestamp),
                float(params["exposure_seconds"]),
                focal_length_px=self.config.focal_length_px,
                angular_gain=self.config.angular_gain,
                samples=self.config.motion_path_samples,
                max_radius_px=self.config.motion_max_radius_px,
            )
            params.update(report)
            # A still camera yields a 1x1 kernel; skip the convolution rather than
            # spend it on an identity.
            params["motion"] = bool(kernel.shape[0] > 1)
            if kernel.shape[0] == 1:
                kernel = None
        elif optical and params["motion"]:
            kernel = _motion_kernel(int(params["motion_length"]), float(params["motion_angle"]))
        if kernel is not None:
            image = np.stack(
                [ndimage.convolve(image[..., channel], kernel, mode="reflect") for channel in range(3)],
                axis=-1,
            )
        if optical and params["downsample"]:
            image = _resize_roundtrip(image, float(params["downsample_scale"]))

        if low_light:
            image *= np.asarray(params["white_balance"], dtype=np.float64)[None, None, :]
            image *= _vignette(*image.shape[:2], float(params["vignette_strength"]))
            image = np.clip(image * float(params["exposure_gain"]), 0.0, None)
            image = np.power(image, float(params["tone_gamma"]))
            image += float(params["black_level"])

        if sensor_noise:
            rng = generator(
                self.master_seed, "image_sensor", split, realization, trajectory, frame_index
            )
            photons = float(params["photon_count"])
            image = rng.poisson(np.clip(image, 0.0, 1.0) * photons) / photons
            image += rng.normal(0.0, float(params["read_noise_std"]), image.shape)
            row_noise = rng.normal(0.0, float(params["row_noise_std"]), (image.shape[0], 1, 1))
            image += row_noise
            hot = rng.random(image.shape[:2]) < float(params["hot_pixel_probability"])
            if hot.any():
                image[hot] = rng.integers(0, 2, size=(int(hot.sum()), 1))

        image = np.clip(image, 0.0, 1.0)
        if sensor_noise:
            levels = float(2 ** int(params["quantization_bits"]) - 1)
            image = np.round(image * levels) / levels
            if params["jpeg"]:
                image = _jpeg(image, int(params["jpeg_quality"]))
        return np.clip(image, 0.0, 1.0).astype(np.float32), params

    def metadata(self) -> dict[str, object]:
        return {"type": type(self).__name__, "master_seed": self.master_seed, "config": asdict(self.config)}

