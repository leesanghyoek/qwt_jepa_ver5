"""Den va loe sang: vung sang chay hon vung toi, den loe quang, tia sao, bong ma.

Anh sach cua TartanAir la anh da nen tone (LDR): mot bong den va mot buc tuong trang deu dung
o ~1.0. Canh that thi den sang gap hang chuc lan, va ong kinh lam anh sang do tran ra xung
quanh. Buoc nay dung lai dieu do tren anh sang TUYEN TINH, truoc buoc mo quang hoc (de den bi
keo thanh vet khi mo) va truoc buoc thieu sang (de den van chay trang khi anh bi toi di):

  1. Canh HDR   mat na vung sang (smoothstep tren do sang); dom sang NHO (den, cua so xa) nhan
                ``gain`` lon, vung sang RONG (troi) chi nhan ``wide_gain`` nho.
  2. Loe sang   phan sang vuot ``knee`` loe ra: quang Gauss ba tang co mau (am nhu den sodium
                toi lanh nhu LED), tia sao tu khe hoi ong kinh, bong ma doi xung qua tam anh.

Ket qua tra ve o thang sRGB MO RONG (gia tri > 1 duoc giu): cac buoc sau cua
qjepa/corruptions/image.py lam viec tren sRGB, va den chi bi cat o buoc doc cam bien.
Moi tham so duoc boc trong ``draw_light_parameters``; ``apply_light`` la ham thuan cua
(anh, tham so), nen tools/light_corruption_preview.py ve lai dung tung buoc.

Ve toc do (data loader chay tren CPU): quang lon, tia sao va bong ma tinh o do phan giai
thap (deu la tin hieu tron), nen mot anh 256x256 ton vai mili giay.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image
from scipy import ndimage, signal

LUMA_709 = np.array([0.2126, 0.7152, 0.0722])   # do sang tren anh sang tuyen tinh (sRGB/BT.709)
GHOSTS_MAX = 8


# float32: hai phep doi nay la phan dat nhat cua buoc (data loader chay tren CPU).
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, None)
    out = np.power((x + np.float32(0.055)) / np.float32(1.055), np.float32(2.4))
    low = x <= 0.04045
    out[low] = x[low] / np.float32(12.92)
    return out


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """sRGB mo rong: tren 1 van theo cung duong cong (lien tuc tai 1), khong cat."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, None)
    out = np.float32(1.055) * np.power(x, np.float32(1 / 2.4)) - np.float32(0.055)
    low = x <= 0.0031308
    out[low] = x[low] * np.float32(12.92)
    return out


def _smoothstep(low: float, high: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - low) / max(high - low, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def resize_channels(image: np.ndarray, size: tuple[int, int], method: int) -> np.ndarray:
    """Doi kich thuoc tung kenh bang PIL so thuc (mode "F"): khong cat o 1 nhu duong uint8."""
    height, width = size
    planes = image[..., None] if image.ndim == 2 else image
    out = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(planes[..., c], dtype=np.float32), mode="F")
                               .resize((width, height), method), dtype=np.float64)
                    for c in range(planes.shape[-1])], axis=-1)
    return out[..., 0] if image.ndim == 2 else out


def _blur_at(image: np.ndarray, sigma: float, factor: int) -> np.ndarray:
    """Gauss ``sigma`` (px cua anh goc), tinh o do phan giai 1/``factor`` roi phong lai."""
    height, width = image.shape[:2]
    if factor <= 1:
        return ndimage.gaussian_filter(image, (sigma, sigma) + (0,) * (image.ndim - 2), mode="reflect")
    small = resize_channels(image, (max(1, height // factor), max(1, width // factor)), Image.Resampling.BOX)
    small = ndimage.gaussian_filter(small, (sigma / factor,) * 2 + (0,) * (image.ndim - 2), mode="reflect")
    return resize_channels(small, (height, width), Image.Resampling.BILINEAR)


def _factor(sigma: float) -> int:
    """Giam do phan giai bao nhieu ma Gauss van con >= 1,5 px (toi da 8 lan)."""
    return int(min(8, max(1, 2 ** math.floor(math.log2(max(sigma / 1.5, 1.0))))))


def star_kernel(radius: int, spikes: int, angle: float, length: float) -> np.ndarray:
    """``spikes`` tia deu nhau qua tam, tat dan theo khoang cach; tong bang 1, tam bang 0."""
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float64)
    kernel = np.zeros_like(xx)
    lines = max(1, spikes // 2)                         # moi duong thang cho hai tia
    for index in range(lines):
        theta = angle + math.pi * index / lines
        across = np.abs(xx * math.sin(theta) - yy * math.cos(theta))
        along = np.abs(xx * math.cos(theta) + yy * math.sin(theta))
        kernel += np.exp(-(across / 0.8) ** 2) * np.exp(-along / max(length, 1e-3))
    kernel[radius, radius] = 0.0                        # nguon sang da co san trong anh
    total = kernel.sum()
    return kernel / total if total > 0 else kernel


def _tint(warmth: float) -> list[float]:
    """Mau anh den: warmth > 0 am (vang cam), < 0 lanh (trang xanh); kenh lon nhat bang 1."""
    if warmth >= 0:
        tint = np.array([1.0, 1.0 - 0.25 * warmth, 1.0 - 0.6 * warmth])
    else:
        tint = np.array([1.0 + 0.5 * warmth, 1.0 + 0.2 * warmth, 1.0])
    return (tint / tint.max()).tolist()


def _log_uniform(rng: np.random.Generator, bounds) -> float:
    low, high = (float(v) for v in bounds)
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def draw_light_parameters(rng: np.random.Generator, cfg) -> dict[str, object]:
    """Tham so loe sang cua mot frame tu cac khoa ``light_*`` cua LowLightImageCorruptionConfig.

    So lan boc co dinh (khong phu thuoc cac lan boc truoc), nen doi mot khoang khong lam
    lech cac tham so khac."""
    sigmas = [float(rng.uniform(*bounds)) for bounds in cfg.light_bloom_sigma_px]
    weights = rng.dirichlet(np.linspace(3.0, 1.0, len(sigmas))).tolist()   # quang hep manh hon quang rong
    star_draw = float(rng.random())
    ghost_count = int(rng.integers(cfg.light_ghost_count[0], cfg.light_ghost_count[1] + 1))
    ghosts = [{"scale": float(rng.uniform(-1.4, -0.3)),          # am: doi xung qua tam, nhu phan xa trong ong kinh
               "strength": float(rng.uniform(*cfg.light_ghost_strength)),
               "sigma_px": float(rng.uniform(3.0, 10.0)),
               "hue": rng.uniform(0.3, 1.0, 3).tolist()} for _ in range(GHOSTS_MAX)][:ghost_count]
    return {
        "threshold": float(rng.uniform(*cfg.light_threshold)),
        "width": float(rng.uniform(*cfg.light_width)),
        "gain": _log_uniform(rng, cfg.light_gain),
        "lamp_area": float(cfg.light_lamp_area),
        "wide_gain": float(rng.uniform(*cfg.light_wide_gain)),
        "shape": float(rng.uniform(*cfg.light_shape)),
        "knee": float(rng.uniform(*cfg.light_knee)),
        "bloom_strength": _log_uniform(rng, cfg.light_bloom_strength),
        "bloom_sigmas_px": sigmas,
        "bloom_weights": weights,
        "tint": _tint(float(rng.uniform(*cfg.light_warmth))),
        "star": star_draw < cfg.light_star_probability,
        "star_spikes": int(rng.choice(np.asarray(cfg.light_star_spikes, dtype=np.int64))),
        "star_angle": float(rng.uniform(0.0, math.pi)),
        "star_length": float(rng.uniform(*cfg.light_star_length)),  # phan cua canh anh
        "star_strength": _log_uniform(rng, cfg.light_star_strength),
        "ghosts": ghosts,
    }


def apply_light(image: np.ndarray, params: dict[str, object], stages: bool = False):
    """Canh HDR + loe sang o sRGB mo rong (> 1 duoc giu), float64 [H,W,3].

    ``image`` sRGB [H,W,3] trong [0,1]. Anh khong co vung sang nao tra ve nguyen ven. Voi
    ``stages=True`` tra ve (canh HDR, canh HDR + loe sang) de ve tung buoc."""
    height, width = image.shape[:2]
    linear = srgb_to_linear(image)

    # 1. Canh HDR: vung sang that ra sang hon nhieu, dom nho (den) hon vung rong (troi).
    mask = _smoothstep(float(params["threshold"]), float(params["threshold"]) + float(params["width"]),
                       linear @ LUMA_709.astype(np.float32))
    if not mask.any():
        out = np.asarray(image, dtype=np.float64).copy()
        return (out, out.copy()) if stages else out
    # "Dom nho": it diem sang quanh no (Gauss sigma = canh/16). Den ban kinh <~10 px va dai sang
    # mong (ong den) -> ~1; mep va long cua so, troi -> 0. Nguong cung, khong chuan hoa theo max:
    # anh khong co den nao thi khong co gi bi coi la den.
    nearby = _blur_at(mask, width / 16.0, _factor(width / 16.0))
    compact = mask * _smoothstep(0.5, 0.85, 1.0 - nearby)
    # Ngan sach dien tich den: vuot qua thi do la van sang (la cay co nang, mat bong), khong phai
    # hang tram bong den -- he so giam theo ti le de tong anh sang den giu nguyen muc ngan sach.
    area = float(compact.mean())
    lamp_gain = float(params["gain"]) * min(1.0, float(params["lamp_area"]) / area) if area > 0 else 0.0
    lamp_gain = max(lamp_gain, float(params["wide_gain"]))
    gain = lamp_gain * compact + float(params["wide_gain"]) * (1.0 - compact)
    scene = linear * (1.0 + gain * mask ** float(params["shape"]))[..., None]

    # 2. Loe sang: chi DEN loe (phan vuot nguong cua dom nho). Cua so va troi chi sang them,
    # khong phu suong len vung toi.
    source = np.maximum(scene - float(params["knee"]), 0.0) * compact[..., None]
    if not source.any():
        out = linear_to_srgb(scene).astype(np.float64)
        return (out, out.copy()) if stages else out
    tint = np.asarray(params["tint"], dtype=np.float64)
    glare = np.zeros_like(scene)
    for sigma, weight in zip(params["bloom_sigmas_px"], params["bloom_weights"]):
        glare += float(weight) * _blur_at(source, float(sigma), _factor(float(sigma)))
    glare *= float(params["bloom_strength"])
    if params["star"]:
        half = resize_channels(source, (max(1, height // 2), max(1, width // 2)), Image.Resampling.BOX)
        length = float(params["star_length"]) * width / 2.0
        radius = int(max(2, min(3.0 * length, min(half.shape[:2]) / 2.0)))
        kernel = star_kernel(radius, int(params["star_spikes"]), float(params["star_angle"]), length)
        star = np.stack([signal.fftconvolve(half[..., c], kernel, mode="same") for c in range(3)], axis=-1)
        glare += float(params["star_strength"]) * resize_channels(np.maximum(star, 0.0), (height, width),
                                                                  Image.Resampling.BILINEAR)
    if params["ghosts"]:
        factor = 4
        small = resize_channels(source, (max(1, height // factor), max(1, width // factor)), Image.Resampling.BOX)
        centre = (np.array(small.shape[:2], dtype=np.float64) - 1.0) / 2.0
        ghosts = np.zeros_like(small)
        for ghost in params["ghosts"]:
            matrix = np.eye(2) / float(ghost["scale"])
            offset = centre - matrix @ centre
            image_ghost = np.stack([ndimage.affine_transform(small[..., c], matrix, offset=offset, order=1,
                                                             mode="constant") for c in range(3)], axis=-1)
            sigma = float(ghost["sigma_px"]) / factor
            image_ghost = ndimage.gaussian_filter(image_ghost, (sigma, sigma, 0), mode="constant")
            ghosts += float(ghost["strength"]) * image_ghost * np.asarray(ghost["hue"], dtype=np.float64)
        glare += resize_channels(ghosts, (height, width), Image.Resampling.BILINEAR)
    lit = linear_to_srgb(scene + np.maximum(glare, 0.0) * tint).astype(np.float64)
    return (linear_to_srgb(scene).astype(np.float64), lit) if stages else lit
