"""Cap anh loe THAT cua FlareImage: anh co loe vao, anh sach la dich, IMU muon cua mot mau TartanAir.

FlareImage (Kaggle ``buidinhkhoi/flareimage``, thu muc ``flare_dataset/``) gom cac cap da can chinh, khong co IMU:

    <root>/metadata.csv    name, source, width, height, gt_src, train_src (moi cap mot dong)
    <root>/gt/<name>       anh sach
    <root>/train/<name>    cung khung hinh, co loe

Ban ngay 2026-10-10 co 1053 cap: flare_removal 600 (1440x1920 va 887x1920, chup that), halo 253 (1280x720, render
Blender, 4 scene), flare7k_real 100 va flare7k_synthetic 100 (512x512, bo test cua Flare7K++).

Model doc anh va IMU cung luc ma cap anh khong co IMU. Mau FlareImage (``PairedCameraImuDataset._flare_item``) muon
cua so IMU sach va nhieu cua mot mau TartanAir: moi loss van dung nghia (IMU sach va IMU nhieu cua cung mot cua so),
chi rieng mau nay anh va IMU khong cung mot chuyen dong.

Chia theo NHOM va theo tung nguon, de valid/test moi ben giu ~holdout_fraction cap cua MOI nguon:
- halo: theo ``<scene>_<effect>`` (vd Scene011_Reflective038): cac camera cua cung mot dung canh di cung nhau;
- nguon con lai: khoi 10 anh lien tiep theo so thu tu cua anh goc (anh gan nhau trong mot bo chup thuong cung canh).

Anh goc lon (PNG 1440x1920 ~3 MB): moi cap duoc thu nho MOT lan ve canh ngan ``short_side`` tren anh sang tuyen tinh
bang BOX (trung binh cac pixel goc, khong co vong rung o loi sang nhu Lanczos), cung mot phep cho gt va train, roi ghi
vao cache tren dia; cac lan sau chi doc ban nho. Anh da nho hon short_side giu nguyen. BOX cua PIL lay trong so tai tam
pixel goc: tong anh sang giu dung khi ti le thu nho la so nguyen; khi khong, do tren mot dom sang 8x8 trong anh 72x48
(ti le 1,2, truong hop xau) lech 0,9%.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..corruptions.halo import split_scenes
from ..corruptions.light import linear_to_srgb, resize_channels, srgb_to_linear

FLARE_PAIR_SPLITS = ("train", "valid", "test")
REQUIRED_COLUMNS = ("name", "source", "gt_src", "train_src")
# sRGB 8 bit -> tuyen tinh bang bang tra (cung bang srgb_to_linear).
_SRGB_TO_LINEAR = srgb_to_linear(np.arange(256, dtype=np.float32) / np.float32(255))


@dataclass(frozen=True)
class FlarePair:
    name: str      # ten file trong gt/ va train/
    source: str    # flare_removal, halo, flare7k_real, flare7k_synthetic, ...
    group: str     # cac cap cung nhom luon nam cung mot split


def pair_group(source: str, gt_src: str) -> str:
    """Nhom chia split cua mot cap: scene + hieu ung voi HALO, khoi 10 anh lien tiep voi nguon khac."""
    path = Path(gt_src)
    scene = next((part for part in path.parts if part.startswith("Scene")), None)
    if scene is not None:
        effect = path.name.split("_")[1] if path.name.count("_") >= 2 else ""
        return f"{source}:{scene}_{effect}"
    digits = re.findall(r"\d+", path.stem)
    return f"{source}:{int(digits[-1]) // 10}" if digits else f"{source}:{path.stem}"


def shrink_pair(gt: np.ndarray, flared: np.ndarray, short_side: int) -> tuple[np.ndarray, np.ndarray]:
    """Hai anh uint8 [H,W,3] cung co -> canh ngan short_side (BOX tren anh sang tuyen tinh); nho hon thi giu nguyen."""
    if gt.shape != flared.shape:
        raise ValueError(f"gt {gt.shape} va train {flared.shape} khac co")
    height, width = gt.shape[:2]
    scale = short_side / min(height, width)
    if scale >= 1:
        return gt, flared
    size = (max(short_side, round(height * scale)), max(short_side, round(width * scale)))
    out = []
    for image in (gt, flared):
        small = resize_channels(_SRGB_TO_LINEAR[image], size, Image.Resampling.BOX)
        out.append(np.round(np.clip(linear_to_srgb(small), 0.0, 1.0) * 255.0).astype(np.uint8))
    return out[0], out[1]


class FlarePairBank:
    """Cac cap cua mot thu muc FlareImage, da chia train/valid/test, doc qua cache da thu nho."""

    def __init__(self, root: str | Path, *, short_side: int, holdout_fraction: float,
                 cache_dir: str | Path | None = None):
        self.root = Path(root)
        index = self.root / "metadata.csv"
        if not index.is_file() or not (self.root / "gt").is_dir() or not (self.root / "train").is_dir():
            raise FileNotFoundError(
                f"data.flare_pairs_root={self.root}: can metadata.csv, gt/ va train/ (dataset FlareImage, "
                "thu muc flare_dataset/). Gan dataset roi tro data.flare_pairs_root toi thu muc do.")
        raw = index.read_bytes()
        rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
        missing_columns = [name for name in REQUIRED_COLUMNS if not rows or name not in rows[0]]
        if missing_columns:
            raise ValueError(f"{index}: thieu cot {missing_columns}")
        names = [row["name"] for row in rows]
        if len(set(names)) != len(names):
            raise ValueError(f"{index}: ten cap bi trung")
        on_disk = set(os.listdir(self.root / "gt")) & set(os.listdir(self.root / "train"))
        absent = [name for name in names if name not in on_disk]
        if absent:
            raise FileNotFoundError(f"{len(absent)} cap trong metadata.csv thieu anh gt/ hoac train/: {absent[:3]}")
        pairs = [FlarePair(row["name"], row["source"], pair_group(row["source"], row["gt_src"])) for row in rows]
        assignment: dict[str, str] = {}
        for source in sorted({pair.source for pair in pairs}):
            counts: dict[str, int] = {}
            for pair in pairs:
                if pair.source == source:
                    counts[pair.group] = counts.get(pair.group, 0) + 1
            assignment.update(split_scenes(counts, holdout_fraction))
        self.splits = {split: tuple(sorted((pair for pair in pairs if assignment[pair.group] == split),
                                           key=lambda pair: pair.name)) for split in FLARE_PAIR_SPLITS}
        empty = [split for split in FLARE_PAIR_SPLITS if not self.splits[split]]
        if empty:
            raise ValueError(f"FlareImage o {self.root}: khong con cap nao cho {empty} ({len(pairs)} cap, "
                             f"holdout_fraction {holdout_fraction})")
        self.short_side = int(short_side)
        self.holdout_fraction = float(holdout_fraction)
        self.digest = hashlib.sha256(raw).hexdigest()[:16]
        base = Path(cache_dir) if cache_dir is not None else Path(tempfile.gettempdir()) / "qjepa_flare_pairs"
        # Ban thu nho la ham cua (metadata, short_side): doi dataset hay short_side thi sang thu muc khac.
        self.cache_dir = base / f"{self.digest}_{self.short_side}"

    def pairs(self, split: str) -> tuple[FlarePair, ...]:
        if split not in self.splits:
            raise ValueError(f"FlareImage khong co split {split!r}")
        return self.splits[split]

    def load(self, pair: FlarePair) -> tuple[np.ndarray, np.ndarray]:
        """(gt, train) uint8 [H,W,3], canh ngan short_side. Lan dau giai ma anh goc va ghi cache; sau do doc cache.

        LUON tra ve ban doc tu cache (ke ca lan dau): ket qua khong phu thuoc cache da co hay chua."""
        stem = Path(pair.name).stem
        targets = (self.cache_dir / f"{stem}.gt.png", self.cache_dir / f"{stem}.train.png")
        if not all(path.is_file() for path in targets):
            with Image.open(self.root / "gt" / pair.name) as image:
                gt = np.asarray(image.convert("RGB"))
            with Image.open(self.root / "train" / pair.name) as image:
                flared = np.asarray(image.convert("RGB"))
            if gt.shape != flared.shape:
                raise ValueError(f"FlareImage {pair.name}: gt {gt.shape[1::-1]} va train {flared.shape[1::-1]} khac co")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for array, target in zip(shrink_pair(gt, flared, self.short_side), targets):
                partial = target.with_name(f"{target.name}.{os.getpid()}.part")   # nhieu worker/rank ghi cung luc
                Image.fromarray(array).save(partial, "PNG")
                os.replace(partial, target)
        loaded = []
        for path in targets:
            with Image.open(path) as image:
                loaded.append(np.asarray(image.convert("RGB")))
        return loaded[0], loaded[1]

    def describe(self) -> dict[str, object]:
        return {"root": str(self.root), "metadata_sha256": self.digest, "short_side": self.short_side,
                "holdout_fraction": self.holdout_fraction,
                "pairs": {split: len(pairs) for split, pairs in self.splits.items()},
                "sources": {split: {source: sum(pair.source == source for pair in pairs)
                                    for source in sorted({pair.source for pair in pairs})}
                            for split, pairs in self.splits.items()}}


def find_flare_pairs_root(base: str | Path = "/kaggle/input", depth: int = 6) -> Path:
    """Thu muc FlareImage (co metadata.csv, gt/ va train/) trong ``base``, Kaggle gan theo kieu nao cung duoc."""
    base = Path(base)
    found = sorted({path.parent for level in range(1, depth + 1)
                    for path in base.glob("/".join(["*"] * level) + "/metadata.csv")
                    if (path.parent / "gt").is_dir() and (path.parent / "train").is_dir()})
    if not found:
        mounted = sorted(str(path.relative_to(base)) for level in (1, 2, 3)
                         for path in base.glob("/".join(["*"] * level)) if path.is_dir())[:30]
        raise FileNotFoundError(f"Khong thay metadata.csv + gt/ + train/ trong {depth} tang cua {base}: gan dataset "
                                f"FlareImage (buidinhkhoi/flareimage). Dang gan: {mounted}")
    if len(found) > 1:
        raise ValueError(f"FlareImage duoc gan hon mot lan: {[str(path) for path in found]}; giu mot ban")
    return found[0]
