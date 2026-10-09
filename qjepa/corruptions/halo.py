"""Loe sang that cua HALO: bong ma phan xa giua cac mat thau kinh, cong len anh TartanAir.

HALO (jdzhang0929/halo-flare-dataset, di kem bai UniSER, CVPR 2026) render bang Blender bo ba
(canh sach, canh co loe, CHI lop loe). Dataset Kaggle ``halo-reflective-1280`` (dung bang
``halo-flare-builder.ipynb``) giu mau Reflective o 1280x720:

    <root>/halo_index.csv              uid, scene, effect_type, ... (moi mau mot dong)
    <root>/halo_build.json             revision HF, effects, max_side
    <root>/<scene>/<uid>.separate.png  chi lop loe, sRGB 8 bit

TartanAir khong co lop loe; HALO khong co IMU. Nen HALO khong thanh sample rieng: lop loe cua no
la mot buoc nhieu tren anh TartanAir, va IMU van la IMU that cua chinh frame do. Moi lan dung:

  1. Cat GIUA theo ti le cua anh dich (anh vuong: 720x720 o giua): tam anh la truc quang, va bong
     ma luon nam tren duong qua nguon sang va tam anh. Cat lech tam se pha tinh chat do.
  2. Doi sang anh sang tuyen tinh roi thu nho bang BOX (trung binh dien tich), nen tong anh sang
     cua lop loe giu nguyen va khong co vong rung am cua Lanczos.
  3. Lat ngang / doc ngau nhien: duong qua tam van qua tam.
  4. Nhan ``gain`` (log-uniform) va CONG vao canh tren anh sang tuyen tinh (anh sang cong lai,
     nhu loe that). Do tren mot mau HALO: cong tuyen tinh lech 1,8/255 so voi anh ``flare`` goc.

Nguon sang sinh ra bong ma khong co trong anh TartanAir: tuong duong nguon sang ngay ngoai
khung, van sinh bong ma va choi nhu ngoai doi.

Chia theo SCENE cua HALO: valid va test moi ben giu ``holdout_fraction`` so mau, gom ca scene,
nen PSNR valid/test do tren nhung lop loe model chua thay luc train.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .light import linear_to_srgb, resize_channels, srgb_to_linear
from .rng import derive_seed

HALO_EFFECTS = ("Streak", "Reflective", "Glare", "Shimmer")
SPLITS = ("train", "valid", "test")
# Lop loe 256x256 float32 ~0,8 MB; vai lop la du cho anh lap lai trong mot segment va cho test.
_CACHE_SIZE = 8
# sRGB 8 bit -> tuyen tinh bang bang tra: dung bang srgb_to_linear, nhanh hon np.power ~10 lan.
_SRGB_TO_LINEAR = srgb_to_linear(np.arange(256, dtype=np.float32) / np.float32(255))


def split_scenes(counts: dict[str, int], holdout_fraction: float) -> dict[str, str]:
    """{scene: split}. Scene xep theo mot hash co dinh; test roi valid lay tron scene cho toi khi moi ben
    co >= holdout_fraction so mau; con lai la train. Cung tap scene -> cung phep chia."""
    order = sorted(counts, key=lambda scene: (derive_seed(0, "halo_split", scene), scene))
    total = sum(counts.values())
    assignment: dict[str, str] = {}
    filled = {"test": 0, "valid": 0}
    for scene in order:
        target = next((split for split in ("test", "valid") if filled[split] < holdout_fraction * total), "train")
        assignment[scene] = target
        if target != "train":
            filled[target] += counts[scene]
    return assignment


class HaloFlareBank:
    """Cac lop loe HALO cua mot thu muc dataset, da chia train/valid/test theo scene."""

    def __init__(self, root: str | Path, *, revision: str, effects, holdout_fraction: float):
        self.root = Path(root)
        index, build = self.root / "halo_index.csv", self.root / "halo_build.json"
        if not index.is_file() or not build.is_file():
            raise FileNotFoundError(
                f"data.halo_root={self.root}: thieu halo_index.csv / halo_build.json. Gan dataset halo-reflective-1280 "
                "(halo-flare-builder.ipynb) va tro data.halo_root toi thu muc chua halo_index.csv.")
        info = json.loads(build.read_text())
        self.effects = tuple(effects)
        if info.get("revision") != revision:
            raise ValueError(f"{build}: HALO revision {info.get('revision')!r}, config doi {revision!r}")
        if not set(self.effects) <= set(info.get("effects", ())):
            raise ValueError(f"{build}: dataset co {info.get('effects')}, config doi {list(self.effects)}")
        with index.open(newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row["effect_type"] in self.effects]
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["scene"]] = counts.get(row["scene"], 0) + 1
        self.scene_split = split_scenes(counts, holdout_fraction)
        self.paths = {row["uid"]: self.root / row["scene"] / f"{row['uid']}.separate.png" for row in rows}
        missing = [str(path) for path in self.paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} lop loe HALO khong co tren dia (Kaggle chua giai nen tar?): "
                                    f"{missing[:3]}")
        self.splits = {split: sorted(row["uid"] for row in rows if self.scene_split[row["scene"]] == split)
                       for split in SPLITS}
        empty = [split for split in SPLITS if not self.splits[split]]
        if empty:
            raise ValueError(f"HALO o {self.root}: khong con mau nao cho {empty} ({len(counts)} scene, "
                             f"holdout_fraction {holdout_fraction}); can them scene")
        self.revision = revision
        self.holdout_fraction = float(holdout_fraction)
        self._cache: dict[tuple[str, int, int], np.ndarray] = {}

    def pick(self, split: str, rng: np.random.Generator) -> str:
        if split not in self.splits:
            raise ValueError(f"HALO khong co split {split!r}")
        uids = self.splits[split]
        return uids[int(rng.integers(len(uids)))]

    def layer(self, uid: str, height: int, width: int) -> np.ndarray:
        """Lop loe ``uid`` o anh sang tuyen tinh, float32 [height, width, 3]: cat giua, BOX."""
        key = (uid, int(height), int(width))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        with Image.open(self.paths[uid]) as image:
            pixels = np.asarray(image.convert("RGB"))
        rows, cols = pixels.shape[:2]
        # Khung lon nhat co ti le height:width nam giua anh nguon.
        crop_h, crop_w = (rows, round(rows * width / height)) if cols * height >= rows * width \
            else (round(cols * height / width), cols)
        top, left = (rows - crop_h) // 2, (cols - crop_w) // 2
        linear = _SRGB_TO_LINEAR[pixels[top:top + crop_h, left:left + crop_w]]
        out = resize_channels(linear, (int(height), int(width)), Image.Resampling.BOX).astype(np.float32)
        if len(self._cache) >= _CACHE_SIZE:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = out
        return out

    def describe(self) -> dict[str, object]:
        return {"revision": self.revision, "effects": list(self.effects), "holdout_fraction": self.holdout_fraction,
                "samples": {split: len(uids) for split, uids in self.splits.items()},
                "scenes": {split: sorted(scene for scene, name in self.scene_split.items() if name == split)
                           for split in SPLITS}}


def draw_halo_parameters(rng: np.random.Generator, cfg, bank: HaloFlareBank, split: str) -> dict[str, object]:
    low, high = (float(value) for value in cfg.halo_gain)
    return {"uid": bank.pick(split, rng),
            "gain": float(np.exp(rng.uniform(np.log(low), np.log(high)))),
            "flip_x": bool(rng.random() < 0.5),
            "flip_y": bool(rng.random() < 0.5)}


def apply_halo(image: np.ndarray, bank: HaloFlareBank, params: dict[str, object]) -> np.ndarray:
    """``image`` sRGB mo rong [H,W,3] (> 1 duoc giu) + lop loe tren anh sang tuyen tinh; tra ve sRGB mo rong."""
    layer = bank.layer(str(params["uid"]), image.shape[0], image.shape[1])
    if params["flip_y"]:
        layer = layer[::-1]
    if params["flip_x"]:
        layer = layer[:, ::-1]
    return linear_to_srgb(srgb_to_linear(image) + np.float32(params["gain"]) * layer).astype(np.float64)
