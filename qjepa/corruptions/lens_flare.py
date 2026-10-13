"""Loe ong kinh tren anh TartanAir: lop loe tan xa cua Flare7K++ va cac loi quang hoc dung lai, cong tren anh sang tuyen tinh.

Nguoi dung (13/10/2026): "ap dung loe va loi quang hoc ong kinh cho cac anh shard0 va shard1, anh tu FlareImage san co
loe nen k can chinh". Hai phan, cung mot nguon sang:

1. Flare7K++ (Kaggle ``khoidb/flare7k``, ban 640 cua flare7kpp-builder.ipynb; ban goc ``daiyuekun/flare7kpp`` cung
   doc duoc). Moi lop loe la anh CHI CO LOE tren nen den, kem anh NGUON SANG cung khung:

       <root>/Flare7K/Scattering_Flare/Compound_Flare/<ten>   5000 loe tan xa tong hop (tia, quang, chung lap lanh)
       <root>/Flare7K/Scattering_Flare/Light_Source/<ten>     nguon sang cua tung lop
       <root>/Flare-R/Compound_Flare/<ten>                    962 loe CHUP THAT trong phong toi
       <root>/Flare-R/Light_Source/<ten>

   Bien doi nhu loader cua Flare7K++: tuyen tinh hoa bang gamma U(1,8; 2,2), ``remove_background``, affine ngau nhien
   (xoay 0-360 do, ti le, truot, dich), lat, lam mo lop loe (khong lam mo nguon), lech DC, nhan do sang. Anh vao =
   canh + lop loe (da gom nguon); ANH DICH = canh + NGUON SANG: model hoc xoa loe ma giu den (Flare7K++, Wu va cs.
   2021 -- dich khong co nguon sang day model xoa luon den).

   Khung hinh: Flare7K++ train tren o 512 cat giua anh loe goc (1440 voi Flare7K, 1008x756 voi Flare-R), nen khung
   thay ``FRAME_FRACTION`` canh dai cua anh loe o ti le 1. Giu dung ti le do o moi co anh va moi ban dataset (640 hay
   goc): loe trong khung 256 to bang loe trong khung 512 cua Flare7K++ so voi khung.

2. Loi quang hoc ong kinh (dung lai theo tools/lens_flare_demo.py, cung vi tri nguon sang cua lop Flare7K++):
   choi che (tan xa trong than ong kinh: nang muc den, giam tuong phan), quang (bloom), bong ma (anh cua la khau qua
   phan xa giua cac mat thau kinh, nam tren duong qua nguon va TAM anh, moi bong mot mau lop phu va lech mau do tan
   sac), vanh halo, bui va vet lau tren mat kinh truoc sang len gan nguon. Nguon ngoai khung van gay bong ma va choi.
   Tia sao nhieu xa khong dung lai: lop Flare7K++ da co tia.

Chia theo KHOI 10 so thu tu lien tiep trong ten file (cac lop gan nhau thuong cung mot ong kinh / mot lan chup), tung nguon rieng:
valid va test moi ben ~holdout_fraction, nen PSNR valid/test do tren lop loe model chua thay luc train.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .halo import split_scenes
from .rng import generator

LENS_SOURCES = ("Flare7K", "Flare-R")
LAYOUT = {"Flare7K": ("Flare7K/Scattering_Flare/Compound_Flare", "Flare7K/Scattering_Flare/Light_Source"),
          "Flare-R": ("Flare-R/Compound_Flare", "Flare-R/Light_Source")}
# Phan canh dai cua anh loe ma o 512 cua Flare7K++ thay o ti le 1 (CenterCrop 512 tren anh 1440 / 1008x756).
FRAME_FRACTION = {"Flare7K": 512.0 / 1440.0, "Flare-R": 512.0 / 1008.0}
SPLITS = ("train", "valid", "test")
SUFFIXES = (".png", ".jpg", ".jpeg")
GROUP = 10
# Mau lop phu chong phan xa cua bong ma (lens_flare_demo.COATINGS): luc, tim hong, ho phach, lam...
COATINGS = ((0.35, 1.0, 0.45), (1.0, 0.35, 0.85), (1.0, 0.65, 0.25), (0.35, 0.6, 1.0), (0.85, 1.0, 0.3),
            (0.6, 0.4, 1.0))
GHOST_KINDS = ("disk", "ring", "blob")
REACH = 0.6     # nguon ngoai khung tat dan sau REACH x canh khung


class LensFlareBank:
    """Cac cap (lop loe, nguon sang) cua Flare7K++ trong mot thu muc, chia train/valid/test theo khoi 10 anh."""

    def __init__(self, root: str | Path, *, holdout_fraction: float, sources=LENS_SOURCES):
        self.root = Path(root)
        self.sources = tuple(sources)
        self.holdout_fraction = float(holdout_fraction)
        self.splits: dict[str, dict[str, tuple[str, ...]]] = {split: {} for split in SPLITS}
        for source in self.sources:
            flare_dir, light_dir = (self.root / part for part in LAYOUT[source])
            if not flare_dir.is_dir() or not light_dir.is_dir():
                raise FileNotFoundError(
                    f"data.lens_flare_root={self.root}: thieu {LAYOUT[source][0]}/ hoac {LAYOUT[source][1]}/. Gan "
                    "dataset Flare7K++ (khoidb/flare7k) va tro data.lens_flare_root toi thu muc chua Flare7K/ va Flare-R/.")
            names = sorted(name for name in os.listdir(flare_dir) if name.lower().endswith(SUFFIXES))
            lights = set(os.listdir(light_dir))
            unpaired = [name for name in names if name not in lights]
            if not names or unpaired:
                raise FileNotFoundError(f"{flare_dir}: {len(names)} lop loe, {len(unpaired)} khong co anh nguon sang "
                                        f"cung ten trong {light_dir.name}/ ({unpaired[:3]})")
            group = {name: f"{source}:{_number(name, index) // GROUP}" for index, name in enumerate(names)}
            counts: dict[str, int] = {}
            for key in group.values():
                counts[key] = counts.get(key, 0) + 1
            assignment = split_scenes(counts, self.holdout_fraction)
            for split in SPLITS:
                self.splits[split][source] = tuple(name for name in names if assignment[group[name]] == split)
            empty = [split for split in SPLITS if not self.splits[split][source]]
            if empty:
                raise ValueError(f"{source} o {self.root}: khong con lop nao cho {empty} ({len(names)} lop, "
                                 f"holdout_fraction {holdout_fraction})")

    def pick(self, split: str, source: str, rng: np.random.Generator) -> str:
        names = self.splits[split][source]
        return names[int(rng.integers(len(names)))]

    def load(self, source: str, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(lop loe, nguon sang) uint8 [S,S,3] vuong: anh khong vuong duoc dem hai ben bang gia tri nho nhat cua tung
        kenh (nen cua lop loe; remove_background keo no ve 0), nhu flare7kpp-builder."""
        out = []
        for part in LAYOUT[source]:
            with Image.open(self.root / part / name) as image:
                pixels = np.asarray(image.convert("RGB"))
            height, width = pixels.shape[:2]
            if height != width:
                side = max(height, width)
                square = np.empty((side, side, 3), np.uint8)
                square[:] = pixels.min(axis=(0, 1))
                top, left = (side - height) // 2, (side - width) // 2
                square[top:top + height, left:left + width] = pixels
                pixels = square
            out.append(pixels)
        if out[0].shape != out[1].shape:
            raise ValueError(f"{source}/{name}: lop loe {out[0].shape} va nguon sang {out[1].shape} khac co")
        return out[0], out[1]

    def describe(self) -> dict[str, object]:
        return {"root": str(self.root), "holdout_fraction": self.holdout_fraction,
                "samples": {split: {source: len(names) for source, names in by_source.items()}
                            for split, by_source in self.splits.items()}}


def _number(name: str, index: int) -> int:
    """So thu tu cua lop trong ten file (000137.png -> 137); khong co so thi vi tri trong danh sach."""
    digits = re.findall(r"\d+", Path(name).stem)
    return int(digits[-1]) if digits else index


def find_lens_flare_root(base: str | Path = "/kaggle/input", depth: int = 6, sources=LENS_SOURCES) -> Path:
    """Thu muc Flare7K++ (co ``LAYOUT`` cua moi nguon) trong ``base``, Kaggle gan theo kieu nao cung duoc."""
    base = Path(base)
    first = LAYOUT[sources[0]][0]
    found = sorted({path.parents[len(Path(first).parts) - 1] for level in range(0, depth)
                    for path in base.glob("/".join(["*"] * level + [first])) if path.is_dir()})
    found = [root for root in found if all((root / part).is_dir() for source in sources for part in LAYOUT[source])]
    if not found:
        mounted = sorted(str(path.relative_to(base)) for level in (1, 2, 3)
                         for path in base.glob("/".join(["*"] * level)) if path.is_dir())[:30]
        raise FileNotFoundError(f"Khong thay {', '.join(LAYOUT[s][0] for s in sources)} trong {depth} tang cua {base}: "
                                f"gan dataset Flare7K++ (khoidb/flare7k). Dang gan: {mounted}")
    if len(found) > 1:
        raise ValueError(f"Flare7K++ duoc gan hon mot lan: {[str(path) for path in found]}; giu mot ban")
    return found[0]


def _uniform(rng: np.random.Generator, bounds) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _log_uniform(rng: np.random.Generator, bounds) -> float:
    return float(math.exp(rng.uniform(math.log(float(bounds[0])), math.log(float(bounds[1])))))


def draw_lens_parameters(rng: np.random.Generator, cfg, bank: LensFlareBank, split: str) -> dict[str, object]:
    """Moi tham so ngau nhien cua mot lan loe: lop Flare7K++ va phep bien doi cua no, roi (neu co) cac loi quang hoc."""
    source = "Flare-R" if float(rng.random()) < cfg.lens_flare_real_fraction else "Flare7K"
    shift = float(cfg.lens_flare_translate)
    drawn: dict[str, object] = {
        "source": source,
        "name": bank.pick(split, source, rng),
        "gamma": _uniform(rng, cfg.lens_flare_gamma),
        "gain": _uniform(rng, cfg.lens_flare_gain),
        "rotation_deg": float(rng.uniform(0.0, 360.0)),
        "scale": _uniform(rng, cfg.lens_flare_scale),
        "shear_deg": float(rng.uniform(-cfg.lens_flare_shear_deg, cfg.lens_flare_shear_deg)),
        "translate": [float(rng.uniform(-shift, shift)), float(rng.uniform(-shift, shift))],
        "flip_x": bool(rng.random() < 0.5),
        "flip_y": bool(rng.random() < 0.5),
        "blur_sigma_px": _uniform(rng, cfg.lens_flare_blur_sigma_px),
        "dc_offset": float(rng.uniform(-cfg.lens_flare_dc_offset, cfg.lens_flare_dc_offset)),
    }
    optics = float(rng.random()) < cfg.lens_optics_probability
    drawn["optics"] = _draw_optics(rng, cfg) if optics else None
    return drawn


def _draw_optics(rng: np.random.Generator, cfg) -> dict[str, object]:
    """Mot ong kinh (theo lens_flare_demo.draw_flare_parameters): khau do, chuoi bong ma, vanh halo, bui. Do dai tinh
    theo canh khung (radius, soft) de cung mot ong kinh ra cung hinh o moi co anh."""
    aperture = {"blades": int(rng.integers(5, 10)), "rotation": float(rng.uniform(0.0, 2.0 * math.pi)),
                "roundness": float(rng.uniform(0.0, 0.3))}
    ghosts = []
    for _ in range(int(rng.integers(cfg.lens_ghost_count[0], cfg.lens_ghost_count[1] + 1))):
        kind = str(rng.choice(GHOST_KINDS, p=(0.7, 0.1, 0.2)))       # da so la hinh la khau dac
        radius = _log_uniform(rng, (0.015, 0.08))
        color = np.asarray(COATINGS[int(rng.integers(len(COATINGS)))]) * rng.uniform(0.85, 1.0, 3)
        ghosts.append({
            "kind": kind,
            "t": float(rng.uniform(-1.5, 0.8)),           # 1 = nguon, 0 = tam anh, < 0 = phia doi dien
            "radius": radius,                             # phan cua canh khung
            "soft": float(rng.uniform(0.004, max(0.006, 0.06 * radius)) if kind != "ring" else rng.uniform(0.002, 0.005)),
            "dispersion": float(rng.uniform(0.003, 0.012) if kind == "disk" else rng.uniform(0.006, 0.02)),
            "reflectance": _log_uniform(rng, cfg.lens_ghost_strength),
            "color": color.tolist(),
        })
    halo_ring = bool(rng.random() < 0.5)
    halo = {"kind": "ring", "t": float(rng.uniform(-0.6, -0.1)), "radius": float(rng.uniform(0.3, 0.45)),
            "soft": 0.012, "dispersion": 0.05, "reflectance": 3e-4 * _log_uniform(rng, (0.5, 2.0)),
            "color": [1.0, 0.9, 1.0]}
    return {"aperture": aperture, "ghosts": ghosts, "halo": halo if halo_ring else None,
            "energy": _log_uniform(rng, cfg.lens_optics_energy),
            "veil": _uniform(rng, cfg.lens_veil), "bloom": _uniform(rng, cfg.lens_bloom),
            "dirt": _uniform(rng, cfg.lens_dirt), "dirt_seed": int(rng.integers(2 ** 31))}


def _background_table(pixels: np.ndarray, gamma: float) -> np.ndarray:
    """Bang tra [3, 256]: gia tri 8 bit -> anh sang tuyen tinh (^gamma) da qua remove_background cua Flare7K++ (keo gia
    tri nho nhat cua tung kenh ve 0, giu gia tri lon nhat). x^gamma tang dan nen min/max tinh ngay tren uint8."""
    curve = np.power(np.arange(256, dtype=np.float64) / 255.0, gamma)
    planes = np.ascontiguousarray(pixels.reshape(-1, 3).T)          # kenh lien nhau: min/max nhanh ~20 lan
    low, high = curve[planes.min(axis=1)], curve[planes.max(axis=1)]
    table = (curve[None, :] - low[:, None]) * high[:, None] / (high - low + 1e-7)[:, None]
    return np.maximum(table, 0.0).astype(np.float32)


def _affine(plane: np.ndarray, inverse: tuple[float, ...], size: tuple[int, int]) -> np.ndarray:
    height, width = size
    image = Image.fromarray(np.ascontiguousarray(plane, dtype=np.float32), mode="F")
    return np.asarray(image.transform((width, height), Image.Transform.AFFINE, inverse,
                                      resample=Image.Resampling.BILINEAR, fillcolor=0.0), dtype=np.float32)


def flare7k_layers(bank: LensFlareBank, params: dict[str, object], height: int, width: int):
    """(lop loe gom nguon, nguon sang) [H,W,3] anh sang tuyen tinh, vi tri nguon (x, y) va mau nguon [3].

    Toa do lien tuc, pixel i phu [i, i+1) (cung quy uoc voi affine cua PIL). Vi tri nguon la trong tam anh sang cua
    anh nguon sang sau phep bien doi; co the nam ngoai khung. Chi phan anh loe ma khung nhin thay duoc tuyen tinh hoa
    va lay mau (nhanh ~4 lan so voi ca anh 640); khi khung thu nho anh loe (zoom < 0,9) phan do duoc thu nho truoc
    bang BOX tren anh sang tuyen tinh toi dung ti le (giu tong anh sang, khong rang cua o tia manh), roi affine ~1:1."""
    flare_u8, light_u8 = bank.load(str(params["source"]), str(params["name"]))
    gamma = float(params["gamma"])
    tables = [_background_table(pixels, gamma) for pixels in (flare_u8, light_u8)]
    side = flare_u8.shape[0]
    frame = float(max(height, width))
    fraction = FRAME_FRACTION[str(params["source"])]
    zoom = frame / (fraction * side) * float(params["scale"])          # pixel khung tren moi pixel anh loe
    theta, shear = math.radians(float(params["rotation_deg"])), math.radians(float(params["shear_deg"]))
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    forward = zoom * rotation @ np.array([[1.0, math.tan(shear)], [0.0, 1.0]])
    inverse = np.linalg.inv(forward)
    translate = np.asarray(params["translate"], dtype=np.float64) / fraction * frame   # pixel khung
    centre_out = np.array([width / 2.0, height / 2.0]) + translate
    centre_src = np.array([side / 2.0, side / 2.0])
    offset = centre_src - inverse @ centre_out
    # Hop bao (trong anh loe) cua bon goc khung, them le cho noi suy.
    corners = np.array([[0.0, 0.0], [width, 0.0], [0.0, height], [width, height]])
    reach = corners @ inverse.T + offset
    left, top = (max(0, int(math.floor(v)) - 2) for v in reach.min(axis=0))
    right, bottom = (min(side, int(math.ceil(v)) + 2) for v in reach.max(axis=0))
    layers = []
    if left < right and top < bottom:
        factor = zoom if zoom < 0.9 else 1.0
        size = (max(1, round((bottom - top) * factor)), max(1, round((right - left) * factor)))
        scale = np.array([size[1] / (right - left), size[0] / (bottom - top)])
        local_inverse = scale[:, None] * inverse
        local_offset = scale * (offset - np.array([left, top]))
        coefficients = (local_inverse[0, 0], local_inverse[0, 1], local_offset[0],
                        local_inverse[1, 0], local_inverse[1, 1], local_offset[1])
        for pixels, table in zip((flare_u8, light_u8), tables):
            crop = pixels[top:bottom, left:right]
            planes = []
            for c in range(3):
                plane = table[c][crop[..., c]]
                if size != plane.shape:
                    plane = _box_resize(plane, size)
                planes.append(_affine(plane, coefficients, (height, width)))
            layers.append(np.stack(planes, axis=-1))
    else:                                                    # khung khong cham anh loe nao
        layers = [np.zeros((height, width, 3), np.float32) for _ in range(2)]
    # Trong tam anh sang cua nguon tu tong theo khoi 4x4 (giu tong anh sang, vi tri sai < 0,1 px voi mot dom sang).
    step = 4
    coarse = _block_sum(light_u8, tables[1], step)
    if float(coarse.sum()) <= 1e-6:                         # anh nguon den: lay vung sang nhat cua lop loe
        coarse = _block_sum(flare_u8, tables[0], step).sum(axis=-1, keepdims=True) ** 4
    weight = coarse.sum(axis=-1)
    rows, cols = np.mgrid[0:weight.shape[0], 0:weight.shape[1]] * step + step / 2.0
    total = max(float(weight.sum()), 1e-12)
    centroid = np.array([float((weight * cols).sum()) / total, float((weight * rows).sum()) / total])
    x, y = centre_out + forward @ (centroid - centre_src)
    if params["flip_y"]:
        layers = [layer[::-1] for layer in layers]
        y = height - y
    if params["flip_x"]:
        layers = [layer[:, ::-1] for layer in layers]
        x = width - x
    flare_layer, light_layer = (np.ascontiguousarray(layer) for layer in layers)
    sigma = float(params["blur_sigma_px"])
    if sigma > 0:
        flare_layer = ndimage.gaussian_filter(flare_layer, sigma=(sigma, sigma, 0), mode="constant")
    gain = np.float32(params["gain"])
    flare_layer = np.maximum(flare_layer + np.float32(params["dc_offset"]), 0.0) * gain
    # Mau nguon: tong anh sang tung kenh cua anh nguon (lop loe neu anh nguon den), kenh lon nhat = 1.
    colour = coarse.sum(axis=(0, 1)) if coarse.shape[-1] == 3 else _block_sum(flare_u8, tables[0], step).sum(axis=(0, 1))
    colour = np.asarray(colour, dtype=np.float64)
    colour = colour / max(float(colour.max()), 1e-12)
    return flare_layer.astype(np.float32), (light_layer * gain).astype(np.float32), (float(x), float(y)), colour


def _block_sum(pixels: np.ndarray, table: np.ndarray, step: int) -> np.ndarray:
    """Tong anh sang tuyen tinh (qua ``table`` [3, 256]) cua tung khoi step x step, [S/step, S/step, 3]."""
    side = pixels.shape[0] - pixels.shape[0] % step
    out = np.empty((side // step, side // step, 3))
    for c in range(3):
        out[..., c] = table[c][pixels[:side, :side, c]].reshape(side // step, step, side // step, step).sum(axis=(1, 3))
    return out


def _box_resize(plane: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.ascontiguousarray(plane, dtype=np.float32), mode="F")
    return np.asarray(image.resize((size[1], size[0]), Image.Resampling.BOX), dtype=np.float32)


def moffat(xx: np.ndarray, yy: np.ndarray, x: float, y: float, r0: float, beta: float) -> np.ndarray:
    """(1 + r^2/r0^2)^-beta chuan hoa de tich phan tren ca mat phang = 1."""
    r2 = (xx - x) ** 2 + (yy - y) ** 2
    return (1.0 + r2 / r0 ** 2) ** -beta * (beta - 1.0) / (math.pi * r0 ** 2)


def polygon_radius(theta: np.ndarray, blades: int, rotation: float, roundness: float) -> np.ndarray:
    """Ban kinh mep khau do da giac deu (dinh cach tam 1) theo huong ``theta``; roundness 1 = tron."""
    sector = 2.0 * math.pi / blades
    phi = np.mod(theta - rotation, sector) - sector / 2.0
    return (1.0 - roundness) * math.cos(math.pi / blades) / np.cos(phi) + roundness


def _ghost_profile(dx: np.ndarray, dy: np.ndarray, kind: str, radius: float, soft: float,
                   aperture: dict | None) -> np.ndarray:
    shape = 1.0 if aperture is None else polygon_radius(np.arctan2(dy, dx), aperture["blades"], aperture["rotation"],
                                                         aperture["roundness"])
    rho = np.hypot(dx, dy) / (radius * shape)                          # 1 = dung mep
    if kind == "disk":                                                 # dia dac, mep sang hon long mot chut
        return np.clip((1.0 - rho) * radius / soft + 0.5, 0.0, 1.0) * (0.6 + 0.4 * np.minimum(rho, 1.0) ** 4)
    if kind == "ring":
        return np.exp(-((rho - 1.0) * radius / soft) ** 2)
    return np.exp(-3.0 * rho ** 2)


def _add_ghost(canvas: np.ndarray, centre: np.ndarray, axis: np.ndarray, ghost: dict, energy: np.ndarray,
               aperture: dict | None, size: float) -> None:
    """Cong mot bong ma vao ``canvas`` [H,W,3]: tam o centre + t*axis, moi kenh lech t va ban kinh theo dispersion.

    Chuan hoa tren luoi rieng du rong (khong phai tren khung): bong ra mot nua ngoai khung thi phan trong khung khong
    sang len. Chi tinh trong hop bao cua bong."""
    height, width = canvas.shape[:2]
    for channel in range(3):
        shift = ghost["dispersion"] * (1 - channel)                    # R lech ra ngoai, B lech vao trong
        gx, gy = centre + ghost["t"] * (1.0 + shift) * axis
        radius = ghost["radius"] * size * (1.0 + 0.5 * shift)
        soft = ghost["soft"] * size
        half = int(math.ceil(1.6 * radius + 3 * soft)) + 2
        local = np.arange(-half, half + 1, dtype=np.float64)
        total = float(_ghost_profile(local[None, :], local[:, None], ghost["kind"], radius, soft, aperture).sum())
        left, right = max(0, int(gx) - half), min(width, int(gx) + half + 1)
        top, bottom = max(0, int(gy) - half), min(height, int(gy) + half + 1)
        if total <= 0 or left >= right or top >= bottom:
            continue
        dx = np.arange(left, right, dtype=np.float64)[None, :] + 0.5 - gx
        dy = np.arange(top, bottom, dtype=np.float64)[:, None] + 0.5 - gy
        value = _ghost_profile(dx, dy, ghost["kind"], radius, soft, aperture) / total
        canvas[top:bottom, left:right, channel] += value * ghost["reflectance"] * energy[channel] * ghost["color"][channel]


def dirt_map(height: int, width: int, seed: int) -> np.ndarray:
    """Bui tren mat kinh truoc [H,W] trong [0,1]: dia mo mep (bui nam ngoai vung net) va 0-2 vet lau soi.

    Dung o nua do phan giai roi phong to: bui nam xa mat phang net nen da mo, va nhanh gap ~4 lan."""
    full_height, full_width = height, width
    height, width = max(8, (height + 1) // 2), max(8, (width + 1) // 2)
    rng = generator(int(seed), "lens_dirt")
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64) + 0.5
    scale = width / 512.0
    dirt = np.zeros((height, width))
    for _ in range(int(rng.integers(10, 41))):
        x, y = rng.uniform(0, width), rng.uniform(0, height)
        radius = math.exp(rng.uniform(math.log(1.5), math.log(14.0))) * scale
        soft = max(0.5, 0.35 * radius)
        strength = rng.uniform(0.15, 1.0)
        half = int(math.ceil(radius + soft)) + 1
        top, bottom = max(0, int(y) - half), min(height, int(y) + half + 1)
        left, right = max(0, int(x) - half), min(width, int(x) + half + 1)
        if top < bottom and left < right:
            distance = np.hypot(xx[top:bottom, left:right] - x, yy[top:bottom, left:right] - y)
            dirt[top:bottom, left:right] += strength * np.clip((radius - distance) / soft + 0.5, 0.0, 1.0)
    for _ in range(int(rng.integers(0, 3))):
        x, y = rng.uniform(0, width), rng.uniform(0, height)
        angle = rng.uniform(0.0, 180.0)
        fibres = ndimage.gaussian_filter(rng.standard_normal((height, width)), (max(0.5, 0.8 * scale), 30.0 * scale))
        fibres = ndimage.rotate(fibres, angle, reshape=False, mode="reflect", order=1)
        fibres = np.clip(fibres / (3.0 * fibres.std() + 1e-12) + 0.5, 0.0, 1.0)
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        along, across = (xx - x) * c - (yy - y) * s, (xx - x) * s + (yy - y) * c
        length, thickness = rng.uniform(0.12, 0.3) * width, rng.uniform(0.03, 0.07) * width
        dirt += 0.25 * np.exp(-0.5 * (along / length) ** 2 - 0.5 * (across / thickness) ** 2) * fibres
    dirt = np.clip(dirt, 0.0, 1.0).astype(np.float32)
    image = Image.fromarray(dirt, mode="F").resize((full_width, full_height), Image.Resampling.BILINEAR)
    return np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)


def optics_layers(height: int, width: int, x: float, y: float, optics: dict[str, object],
                  colour: np.ndarray) -> dict[str, np.ndarray]:
    """Cac lop loi quang hoc [H,W,3] (anh sang tuyen tinh, 1 = trang cua canh) cho nguon o (x, y).

    Nang luong nguon E = energy x H x W ("pixel trang"), mau ``colour``. Moi lop la mot phan cua E trai theo mot hinh da
    chuan hoa (choi che: 0,4 deu + 0,6 Moffat rong; quang: Moffat hep; bong ma: reflectance; bui: ban do bui nhan Moffat
    quanh nguon), nen nguon manh gap doi thi moi lop sang gap doi. Nguon cang xa ngoai khung cang it lot vao ong kinh."""
    size = float(width)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64) + 0.5
    outside = math.hypot(max(0.0, -x, x - width), max(0.0, -y, y - height))
    reach = math.exp(-(outside / (REACH * size)) ** 2)
    energy = float(optics["energy"]) * height * width * reach * np.asarray(colour, dtype=np.float64)
    veil = float(optics["veil"]) * (0.4 / (height * width) + 0.6 * moffat(xx, yy, x, y, 0.3 * size, 1.3))
    bloom = float(optics["bloom"]) * moffat(xx, yy, x, y, 0.025 * size, 1.6)
    ghosts = np.zeros((height, width, 3))
    centre = np.array([width / 2.0, height / 2.0])
    axis = np.array([x, y]) - centre
    for ghost in optics["ghosts"]:
        _add_ghost(ghosts, centre, axis, ghost, energy, optics["aperture"], size)
    if optics["halo"] is not None:
        _add_ghost(ghosts, centre, axis, optics["halo"], energy, None, size)
    layers = {"veil": veil[..., None] * energy, "bloom": bloom[..., None] * energy, "ghosts": ghosts}
    if float(optics["dirt"]) > 0:
        dirt = float(optics["dirt"]) * dirt_map(height, width, int(optics["dirt_seed"])) \
            * moffat(xx, yy, x, y, 0.2 * size, 1.4)
        layers["dirt"] = dirt[..., None] * energy
    return {key: value.astype(np.float32) for key, value in layers.items()}


def lens_flare_layers(bank: LensFlareBank, params: dict[str, object], height: int, width: int):
    """(toan bo anh sang loe cong vao anh vao, nguon sang cong vao anh dich), [H,W,3] anh sang tuyen tinh."""
    flare, light, (x, y), colour = flare7k_layers(bank, params, height, width)
    if params.get("optics"):
        for layer in optics_layers(height, width, x, y, params["optics"], colour).values():
            flare = flare + layer
    return flare, light
