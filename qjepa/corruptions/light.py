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
ILLUM_BLOBS_MAX = 8
SMUDGES_MAX = 6


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
    weights = rng.dirichlet(np.linspace(2.0, 1.0, len(sigmas))).tolist()
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


def draw_illumination_parameters(rng: np.random.Generator, cfg) -> dict[str, object]:
    """Anh sang khong deu cua mot frame tu cac khoa ``illum_*``: vung sang/toi, dai sang, vet nhoe toi.

    So lan boc co dinh (boc du ILLUM_BLOBS_MAX / SMUDGES_MAX roi cat)."""
    blob_count = int(rng.integers(cfg.illum_blobs[0], cfg.illum_blobs[1] + 1))
    blobs = [{"y": float(rng.random()), "x": float(rng.random()),
              "sigma": float(rng.uniform(*cfg.illum_blob_size)),
              "stops": float(rng.choice([-1.0, 1.0]) * rng.uniform(*cfg.illum_strength))}
             for _ in range(ILLUM_BLOBS_MAX)][:blob_count]
    smudge_draw = float(rng.random())
    smudge_count = int(rng.integers(cfg.illum_smudge_count[0], cfg.illum_smudge_count[1] + 1))
    smudges = [{"y": float(rng.random()), "x": float(rng.random()),
                "size": float(rng.uniform(*cfg.illum_smudge_size)),
                "elongation": float(rng.uniform(*cfg.illum_smudge_elongation)),
                "angle": float(rng.uniform(0.0, math.pi)),
                "depth": float(rng.uniform(*cfg.illum_smudge_depth))} for _ in range(SMUDGES_MAX)][:smudge_count]
    drawn = {"gradient_stops": float(rng.uniform(*cfg.illum_gradient)),
             "gradient_angle": float(rng.uniform(0.0, 2.0 * math.pi)),
             "blobs": blobs, "smudges": smudges if smudge_draw < cfg.illum_smudge_probability else []}
    if getattr(cfg, "illum_highlight_rolloff", False):    # chi khi bat: tham so cu giu nguyen
        drawn["rolloff"] = True
    return drawn


def illumination_field(height: int, width: int, params: dict[str, object]) -> np.ndarray:
    """He so nhan [H,W] tren anh sang tuyen tinh: anh sang chieu khong deu len canh.

    2^(dai sang + tong vung sang/toi Gauss, tru trung binh) -- do sang trung binh (theo stop)
    giu nguyen, chi phan bo thay doi -- nhan tiep cac vet nhoe toi: Gauss det, mep mem, keo
    dai theo mot huong, lam toi toi ``depth`` o tam. Toa do theo phan cua canh anh."""
    yy, xx = np.meshgrid((np.arange(height, dtype=np.float32) + 0.5) / height,
                         (np.arange(width, dtype=np.float32) + 0.5) / width, indexing="ij")
    angle = float(params["gradient_angle"])
    log2 = float(params["gradient_stops"]) * ((xx - 0.5) * math.cos(angle) + (yy - 0.5) * math.sin(angle))
    for blob in params["blobs"]:
        distance = (yy - float(blob["y"])) ** 2 + (xx - float(blob["x"])) ** 2
        log2 = log2 + float(blob["stops"]) * np.exp(-distance / (2.0 * float(blob["sigma"]) ** 2))
    field = np.exp2(np.clip(log2 - log2.mean(), -4.0, 3.0))
    for smudge in params["smudges"]:
        dy, dx = yy - float(smudge["y"]), xx - float(smudge["x"])
        c, s = math.cos(float(smudge["angle"])), math.sin(float(smudge["angle"]))
        along, across = dx * c + dy * s, -dx * s + dy * c
        size = float(smudge["size"])
        shape = np.exp(-(along / (size * float(smudge["elongation"]))) ** 2 / 2.0 - (across / size) ** 2 / 2.0)
        field = field * (1.0 - float(smudge["depth"]) * shape)
    return field.astype(np.float32)


def draw_fog_parameters(rng: np.random.Generator, cfg) -> dict[str, object]:
    """Suong mu / mu khoi cua mot frame tu cac khoa ``fog_*``. So lan boc co dinh."""
    return {"density": _log_uniform(rng, cfg.fog_density),
            "airlight": float(rng.uniform(*cfg.fog_airlight)),
            "coolness": float(rng.uniform(0.0, 1.0)),              # 0 trang xam, 1 hoi xanh
            "horizon": float(rng.uniform(*cfg.fog_horizon)),
            "tilt": float(rng.uniform(-0.3, 0.3)),
            "patchiness": float(rng.uniform(*cfg.fog_patchiness)),
            "patch_seed": int(rng.integers(0, 2 ** 31)),
            "scatter_px": float(rng.uniform(*cfg.fog_scatter_px))}


def fog_transmission(height: int, width: int, params: dict[str, object]) -> np.ndarray:
    """t [H,W] = exp(-density * do sau gia), do sau gia trong [0,1]: xa dan ve duong chan troi.

    Dataset khong co ban do do sau. Tren duong chan troi (o do cao ``horizon``, nghieng ``tilt``)
    xa nhat (1); duoi no gan dan ve day anh. Mang suong day/mong: nhan mat do voi 2^(nhieu muot)
    bien do ``patchiness`` stop."""
    yy, xx = np.meshgrid((np.arange(height, dtype=np.float32) + 0.5) / height,
                         (np.arange(width, dtype=np.float32) + 0.5) / width, indexing="ij")
    horizon = float(params["horizon"]) + float(params["tilt"]) * (xx - 0.5)
    depth = np.where(yy <= horizon, 1.0, 1.0 - (yy - horizon) / np.maximum(1.0 - horizon, 1e-3))
    depth = 0.15 + 0.85 * np.clip(depth, 0.0, 1.0)                 # khong co gi sat ong kinh
    noise = np.random.default_rng(int(params["patch_seed"])).standard_normal((6, 6)).astype(np.float32)
    patches = ndimage.zoom(noise, (height / 6.0, width / 6.0), order=3)[:height, :width]
    patches = patches / max(float(np.abs(patches).max()), 1e-6)
    density = float(params["density"]) * np.exp2(float(params["patchiness"]) * patches)
    return np.exp(-density * depth).astype(np.float32)


def apply_fog(linear: np.ndarray, params: dict[str, object]) -> np.ndarray:
    """Anh sang tuyen tinh qua suong: I = J*t + A*(1 - t), J duoc tan xa thuan lam mem them o noi suong day.

    Suong duoc chieu boi chinh anh sang cua canh: A = ``airlight`` x do sang o phan vi 90 cua canh
    (den nho khong keo duoc phan vi nay), nen dem toi thi suong toi, chi quanh den sang."""
    height, width = linear.shape[:2]
    reference = float(np.clip(np.percentile(linear @ LUMA_709.astype(np.float32), 90) * 1.2, 0.02, 1.0))
    t = fog_transmission(height, width, params)[..., None]
    sigma = float(params["scatter_px"])
    if sigma > 0:
        linear = t * linear + (1.0 - t) * _blur_at(linear, sigma, _factor(sigma))
    tint = np.array([1.0 - 0.06 * float(params["coolness"]), 1.0, 1.0 + 0.06 * float(params["coolness"])])
    airlight = float(params["airlight"]) * reference * tint / tint.max()
    return linear * t + airlight * (1.0 - t)


def highlight_rolloff(linear: np.ndarray, knee: float = 0.8) -> np.ndarray:
    """Nen mem vung sang ve phia 1 thay vi cat: x duoi ``knee`` giu nguyen, tren do tien dan toi 1.

    Vai dang phan thuc (Reinhard): knee + (1 - knee) * t / (1 + t), t = (x - knee) / (1 - knee). Lien tuc,
    dao ham 1 tai ``knee``, tang ngat va tien ve 1 CHAM (x = 1,5 / 3 / 11 -> 0,956 / 0,983 / 0,996: van
    phan biet duoc o 8 bit; vai mu exp da bang 1 tu x ~ 1,5) -- nhu duong cong vung sang cua may anh:
    cho duoc chieu sang hon van sang len ma giu chi tiet."""
    t = np.maximum(linear - knee, 0.0) / (1.0 - knee)
    return np.where(linear > knee, knee + (1.0 - knee) * t / (1.0 + t), linear)


def apply_light(image: np.ndarray, params: dict[str, object] | None, stages: bool = False,
                illumination: dict[str, object] | None = None, fog: dict[str, object] | None = None):
    """Canh HDR + anh sang khong deu + loe sang o sRGB mo rong (> 1 duoc giu), float64 [H,W,3].

    ``image`` sRGB [H,W,3] trong [0,1]. ``params`` (loe sang) hoac ``illumination`` co the None.
    Khong co anh sang khong deu va khong co vung sang nao thi tra ve nguyen ven. Voi
    ``stages=True`` tra ve (canh, canh + loe sang) de ve tung buoc."""
    height, width = image.shape[:2]
    linear = srgb_to_linear(image)
    field = None if illumination is None else illumination_field(height, width, illumination)[..., None]
    if params is None:                                  # khong loe: chi anh sang khong deu va/hoac suong
        scene = linear * (1.0 if field is None else field)
        if illumination is not None and illumination.get("rolloff"):
            scene = highlight_rolloff(scene)
        if fog is not None:
            scene = apply_fog(scene, fog)
        out = linear_to_srgb(scene).astype(np.float64)
        return (out, out.copy()) if stages else out

    # 1. Canh HDR: vung sang that ra sang hon nhieu, dom nho (den) hon vung rong (troi).
    mask = _smoothstep(float(params["threshold"]), float(params["threshold"]) + float(params["width"]),
                       linear @ LUMA_709.astype(np.float32))
    if not mask.any() and field is None and fog is None:
        out = np.asarray(image, dtype=np.float64).copy()
        return (out, out.copy()) if stages else out
    # "Dom nho" = mat na tru ban lam mo cua no (Gauss sigma = canh/16), chuan hoa theo max: den
    # va mep vung sang -> ~1, long vung sang rong (troi) -> ~0.
    compact = np.clip(mask - _blur_at(mask, width / 16.0, _factor(width / 16.0)), 0.0, 1.0)
    peak = float(compact.max())
    compact = compact / peak if peak > 1e-6 else compact
    gain = float(params["gain"]) * compact + float(params["wide_gain"]) * (1.0 - compact)
    scene = linear * (1.0 + gain * mask ** float(params["shape"]))[..., None]
    if field is not None:
        # Anh sang khong deu chieu len ca canh (ca den): cho duoc chieu sang hon loe manh hon.
        scene = scene * field
        if illumination.get("rolloff"):
            scene = highlight_rolloff(scene)
    if fog is not None:
        # Suong truoc loe: den xa bi suong lam mo, va loe tinh tren canh da qua suong.
        scene = apply_fog(scene, fog)

    # 2. Loe sang tu MOI phan vuot nguong (den, cua so, troi rat sang): quang co the phu len ca
    # vung toi -- nguoi dung chon giu (05/10/2026) de model hoc go.
    source = np.maximum(scene - float(params["knee"]), 0.0)
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
    glare = glare * tint                                # quang va tia sao mang mau den; bong ma co mau rieng
    if params["ghosts"]:
        glare += resize_channels(ghosts, (height, width), Image.Resampling.BILINEAR)
    lit = linear_to_srgb(scene + np.maximum(glare, 0.0)).astype(np.float64)
    return (linear_to_srgb(scene).astype(np.float64), lit) if stages else lit
