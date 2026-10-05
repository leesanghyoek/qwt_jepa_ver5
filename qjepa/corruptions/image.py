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

from .light import apply_light, draw_light_parameters, resize_channels
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
    # Lamps and glare (light.py), before the optics: small bright spots become many
    # times brighter than the rest, with bloom, starburst and ghosts, so a lamp stays
    # bright after the exposure drop and smears with the blur. Mode "full" only. Drawn from
    # its own stream ("image_light"), and at 0 nothing is drawn at all: a config
    # without these keys renders, and reports, exactly what it did before them.
    light_probability: float = 0.0
    light_threshold: tuple[float, float] = (0.5, 0.8)         # linear luminance where a light source starts
    light_width: tuple[float, float] = (0.1, 0.2)
    light_gain: tuple[float, float] = (5.0, 40.0)             # small bright spots (lamps): log-uniform
    light_wide_gain: tuple[float, float] = (0.0, 1.5)         # wide bright areas (sky)
    light_shape: tuple[float, float] = (1.0, 3.0)
    light_knee: tuple[float, float] = (1.0, 2.5)              # only light above this glares
    light_bloom_strength: tuple[float, float] = (0.15, 1.5)   # log-uniform
    light_bloom_sigma_px: tuple[tuple[float, float], ...] = ((1.5, 4.0), (6.0, 16.0), (20.0, 60.0))
    light_warmth: tuple[float, float] = (-0.4, 1.0)           # glare colour: <0 cold LED, >0 warm sodium
    light_star_probability: float = 0.6
    light_star_spikes: tuple[int, ...] = (4, 6, 8, 10)
    light_star_length: tuple[float, float] = (0.04, 0.18)     # fraction of the frame width
    light_star_strength: tuple[float, float] = (0.05, 0.6)    # log-uniform
    light_ghost_count: tuple[int, int] = (0, 3)
    light_ghost_strength: tuple[float, float] = (0.01, 0.08)

    def _validate_light(self) -> None:
        def bounds(name: str, low_limit: float, strict: bool = False) -> None:
            value = getattr(self, name)
            if len(value) != 2 or not value[0] <= value[1] or value[0] < low_limit or (strict and value[0] <= low_limit):
                sign = ">" if strict else ">="
                raise ValueError(f"{name} must be [low, high] with low <= high and low {sign} {low_limit:g}")

        for name in ("light_probability", "light_star_probability"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        for name in ("light_threshold", "light_wide_gain", "light_ghost_count", "light_ghost_strength"):
            bounds(name, 0.0)
        for name in ("light_width", "light_gain", "light_shape", "light_knee", "light_bloom_strength",
                     "light_star_length", "light_star_strength"):
            bounds(name, 0.0, strict=True)
        bounds("light_warmth", -1.0)
        if self.light_warmth[1] > 1:
            raise ValueError("light_warmth must stay in [-1, 1]")
        if not self.light_bloom_sigma_px or any(len(b) != 2 or not 0 < b[0] <= b[1] for b in self.light_bloom_sigma_px):
            raise ValueError("light_bloom_sigma_px must be a list of [low, high] with 0 < low <= high")
        if not self.light_star_spikes or any(int(n) < 2 for n in self.light_star_spikes):
            raise ValueError("light_star_spikes must list spike counts >= 2")
        if self.light_ghost_count[1] > 8:
            raise ValueError("light_ghost_count must stay at most 8")

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
        self._validate_light()


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


def _resize_roundtrip_hdr(image: np.ndarray, scale: float) -> np.ndarray:
    """_resize_roundtrip in float: the uint8 round trip would clip every lamp to 1."""
    height, width = image.shape[:2]
    small = resize_channels(image, (max(1, round(height * scale)), max(1, round(width * scale))),
                            Image.Resampling.BILINEAR)
    return resize_channels(small, (height, width), Image.Resampling.BILINEAR)


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


def active_stages(params: dict[str, object]) -> tuple[bool, bool, bool]:
    """Which stages (optical, low light, sensor) ran for these drawn parameters."""
    mode = params["mode"]
    optical = mode in ("full", "blur_only", "blur_low_light")
    # The named scenarios keep their meaning; only "full" draws the per-frame variant.
    low_light = mode in ("low_light_only", "blur_low_light") or (mode == "full" and bool(params.get("low_light", True)))
    sensor_noise = mode == "sensor_noise_only" or (mode == "full" and bool(params.get("sensor_noise", True)))
    return optical, low_light, sensor_noise


# What the phase-1 degradation head regresses: the corruption of this frame, each
# entry scaled to roughly [0, 1] over the configured ranges and 0 where the stage
# did not run, so a clean frame is the zero vector.
DEGRADATION_FEATURES = ("defocus_sigma", "motion_length", "downsample", "darkness", "gamma",
                        "shot_noise", "read_noise", "jpeg")
_DEFOCUS_SCALE_PX = 1.5
_MOTION_SCALE_PX = 10.0
_DOWNSAMPLE_SCALE = 0.3
_SHOT_NOISE_SCALE = 30.0           # 1/sqrt(photons): 2500 photons -> 0.6
_READ_NOISE_SCALE = 255.0 / 5.0    # 5/255 -> 1


def degradation_vector(params: dict[str, object]) -> np.ndarray:
    """The drawn corruption of one frame as DEGRADATION_FEATURES, float32."""
    vector = np.zeros(len(DEGRADATION_FEATURES), dtype=np.float32)
    if params.get("clean"):
        return vector
    optical, low_light, sensor_noise = active_stages(params)
    if optical:
        if params.get("defocus"):
            vector[0] = float(params["defocus_sigma"]) / _DEFOCUS_SCALE_PX
        if params.get("motion"):
            # Coupled to the IMU, the blur is the gyro path, not a drawn length.
            length = params.get("path_span_px", 0.0) if params.get("motion_from_imu") else params["motion_length"]
            vector[1] = float(length) / _MOTION_SCALE_PX
        if params.get("downsample"):
            vector[2] = (1.0 - float(params["downsample_scale"])) / _DOWNSAMPLE_SCALE
    if low_light:
        vector[3] = 1.0 - float(params["exposure_gain"])
        vector[4] = 1.0 - float(params["tone_gamma"])
    if sensor_noise:
        vector[5] = _SHOT_NOISE_SCALE / math.sqrt(float(params["photon_count"]))
        vector[6] = float(params["read_noise_std"]) * _READ_NOISE_SCALE
        if params.get("jpeg"):
            vector[7] = (100.0 - float(params["jpeg_quality"])) / 100.0
    return vector


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
        if cfg.light_probability > 0:
            light_rng = generator(self.master_seed, "image_light", split, realization, trajectory, segment)
            light = mode == "full" and float(light_rng.random()) < cfg.light_probability
            drawn = draw_light_parameters(light_rng, cfg)
            parameters.update(light=bool(light), light_params=drawn if light else None)
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
        noisy, _, params = self._render(
            image_clean, split=split, realization=realization, trajectory=trajectory, timestamp=timestamp,
            frame_index=frame_index, mode=mode, gyro=gyro, imu_times=imu_times, reference=False,
        )
        return noisy, params

    def render_with_sensor_reference(
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
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        """(noisy, the same frame without sensor grain, parameters).

        The reference goes through every stage the noisy frame did -- blur,
        exposure, quantization, JPEG -- except the shot/read/row/hot-pixel noise,
        so (noisy - reference) is the grain alone. The phase-1 Jacobian term uses
        it as its noise direction; (noisy - clean) on a blurred frame is mostly the
        missing detail, and penalising that taught the encoder to ignore edges.
        """
        return self._render(
            image_clean, split=split, realization=realization, trajectory=trajectory, timestamp=timestamp,
            frame_index=frame_index, mode=mode, gyro=gyro, imu_times=imu_times, reference=True,
        )

    def _render(
        self,
        image_clean: np.ndarray,
        *,
        split: str,
        realization: int,
        trajectory: str,
        timestamp: float,
        frame_index: int,
        mode: str,
        gyro: np.ndarray | None,
        imu_times: np.ndarray | None,
        reference: bool,
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, object]]:
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
            clean = image_clean.astype(np.float32, copy=True)
            return clean, (clean.copy() if reference else None), params

        optical, low_light, sensor_noise = active_stages(params)
        image = image_clean.astype(np.float64, copy=True)
        # Lamps and glare first: the blur below smears them, the exposure drop below
        # leaves them blown out. From here on values above 1 are real light.
        hdr = bool(params.get("light"))
        if hdr:
            image = apply_light(image, params["light_params"])

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
            image = (_resize_roundtrip_hdr if hdr else _resize_roundtrip)(image, float(params["downsample_scale"]))

        if low_light:
            image *= np.asarray(params["white_balance"], dtype=np.float64)[None, None, :]
            image *= _vignette(*image.shape[:2], float(params["vignette_strength"]))
            image = np.clip(image * float(params["exposure_gain"]), 0.0, None)
            image = np.power(image, float(params["tone_gamma"]))
            image += float(params["black_level"])

        before_grain = image
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

        noisy = self._readout(image, params, sensor_noise)
        if not reference:
            return noisy, None, params
        # Without grain the reference is the noisy frame; with it, the same readout of the
        # frame before the grain, so quantization and JPEG cancel out of the difference.
        without_grain = self._readout(before_grain, params, sensor_noise) if sensor_noise else noisy.copy()
        return noisy, without_grain, params

    @staticmethod
    def _readout(image: np.ndarray, params: dict[str, object], sensor_noise: bool) -> np.ndarray:
        """Clip, then the sensor's quantization and JPEG when the sensor stage ran."""
        image = np.clip(image, 0.0, 1.0)
        if sensor_noise:
            levels = float(2 ** int(params["quantization_bits"]) - 1)
            image = np.round(image * levels) / levels
            if params["jpeg"]:
                image = _jpeg(image, int(params["jpeg_quality"]))
        return np.clip(image, 0.0, 1.0).astype(np.float32)

    def metadata(self) -> dict[str, object]:
        return {"type": type(self).__name__, "master_seed": self.master_seed, "config": asdict(self.config)}

