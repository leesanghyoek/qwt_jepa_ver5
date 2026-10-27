"""Cap anh canh da ghep loe san (p46, ``data.scene_pairs_*``): anh sach la dich, anh co loe la mot nguon vao.

Bo dau tien la "Lens flare on VisDrone, 640x640" (Kaggle ``buidinhkhoi/visdrone-lens-flare-640``, notebook
``wu-visdrone-composite.ipynb``): 2 000 anh drone giao thong ban ngay that cua VisDrone2019-DET, moi anh cong mot lop
loe Wu et al. (bong ma da lam sang) hoac FlareX tren anh sang tuyen tinh. Bo cuc:

    <root>/index.csv   file, split (train / val), scene, scene_split, flare, seed, ... (moi cap mot dong)
    <root>/gt/<file>     anh sach 640x640 (giu loi nguon sang, nhu dich cua repo)
    <root>/input/<file>  cung khung, co loe

Khac FlareImage, split da chia san trong index.csv (valid = anh VisDrone test-dev, khong chung chuoi voi train, ghep voi
lop loe nhom valid), nen bank giu nguyen: ``train`` va ``valid``, khong co ``test``.

p49 (``data.scene_pairs_holdout_fraction`` f): bank chia lai ``train`` / ``valid`` / ``test``, moi nguon anh nen (drone,
Flickr) ~f cap cho test va ~f cho valid, con lai train. Held-out chi lay tu cap ``val`` cua index.csv (da tach anh nen
va lop loe khoi train luc dung dataset), theo NHOM anh nen (cung chuoi VisDrone / cung anh Flickr di cung nhau). Nhom
co cap ``train`` trong index.csv, hay co cap dung mot lop loe (cot flare) ma cap ``train`` cua index.csv cung dung,
khong bao gio vao held-out. Test lay truoc, roi valid, cho toi khi moi ben du f cap cua nguon do. Cap ``val`` con thua
ve train, tru cap dung CUNG lop loe voi mot cap held-out: bo han. Vay train khong thay anh nen hay lop loe nao cua
valid/test, va moi cap ``train`` cua index.csv deu o lai train (wu-visdrone-composite da tach lop loe train/val). Thieu khoa: nhu cu. Moi cap duoc thu nho MOT lan ve
canh ngan ``short_side`` (BOX tren anh sang tuyen tinh, chung ham voi FlareImage), cache tren dia.

PairedCameraImuDataset doc moi cap theo hai cach (``data.scene_pairs_flare_probability``): anh SACH lam nguon (qua
du buoc nhieu nhu mot khung TartanAir: toi, mo, hat -- day khu toi va lam net tren anh chup that) hoac anh CO LOE lam
nguon (nhu mot cap FlareImage: dich la anh sach). Anh khong co IMU: muon cua so IMU cua mot mau TartanAir.
"""

from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .flare_pairs import shrink_pair

SCENE_PAIR_SPLITS = ("train", "valid")
HELDOUT_SPLITS = ("test", "valid")          # p49: thu tu lay nhom khi chia lai
REQUIRED_COLUMNS = ("file", "split")
# index.csv ghi "val"; trong repo split la "valid".
_INDEX_SPLIT = {"train": "train", "val": "valid", "valid": "valid"}


@dataclass(frozen=True)
class ScenePair:
    name: str      # ten file trong gt/ va input/
    source: str    # nguon lop loe: "flarex", "captured", "simulated" (cot flare cua index.csv), "" neu khong ghi
    scene: str     # anh sach goc (cot scene), de truy nguoc


class ScenePairBank:
    """Cac cap cua mot thu muc dang index.csv + gt/ + input/, split doc tu index.csv, doc qua cache da thu nho."""

    def __init__(self, root: str | Path, *, short_side: int, cache_dir: str | Path | None = None,
                 holdout_fraction: float | None = None):
        self.root = Path(root)
        index = self.root / "index.csv"
        if not index.is_file() or not (self.root / "gt").is_dir() or not (self.root / "input").is_dir():
            raise FileNotFoundError(
                f"data.scene_pairs_root={self.root}: can index.csv, gt/ va input/ (dataset visdrone-lens-flare-640, "
                "tao bang wu-visdrone-composite.ipynb). Gan dataset roi tro data.scene_pairs_root toi thu muc do.")
        raw = index.read_bytes()
        rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
        missing_columns = [name for name in REQUIRED_COLUMNS if not rows or name not in rows[0]]
        if missing_columns:
            raise ValueError(f"{index}: thieu cot {missing_columns}")
        names = [row["file"] for row in rows]
        if len(set(names)) != len(names):
            raise ValueError(f"{index}: ten file bi trung")
        unknown = sorted({row["split"] for row in rows} - set(_INDEX_SPLIT))
        if unknown:
            raise ValueError(f"{index}: split la {unknown}; chi nhan train / val")
        on_disk = set(os.listdir(self.root / "gt")) & set(os.listdir(self.root / "input"))
        absent = [name for name in names if name not in on_disk]
        if absent:
            raise FileNotFoundError(f"{len(absent)} cap trong index.csv thieu anh gt/ hoac input/: {absent[:3]}")
        assignment = {row["file"]: _INDEX_SPLIT[row["split"]] for row in rows}
        self.holdout_fraction = None if holdout_fraction is None else float(holdout_fraction)
        self.dropped = 0
        if self.holdout_fraction is not None:
            assignment = resplit(rows, self.holdout_fraction)
            self.dropped = sum(split is None for split in assignment.values())
        names_by_split = (*SCENE_PAIR_SPLITS, "test") if self.holdout_fraction is not None else SCENE_PAIR_SPLITS
        pairs = {split: [] for split in names_by_split}
        for row in rows:
            source = (row.get("flare") or "").split("/")[0]
            split = assignment[row["file"]]
            if split is not None:
                pairs[split].append(ScenePair(row["file"], source, row.get("scene") or ""))
        self.splits = {split: tuple(sorted(items, key=lambda pair: pair.name)) for split, items in pairs.items()}
        empty = [split for split in names_by_split if not self.splits[split]]
        if empty and self.holdout_fraction is not None:
            raise ValueError(f"Cap canh o {self.root}: chia lai (scene_pairs_holdout_fraction) khong co cap nao cho {empty}: "
                             "held-out chi lay nhom anh nen chi co cap val, khong dung lop loe nao cua cap train")
        if empty:
            raise ValueError(f"Cap canh o {self.root}: khong co cap nao cho {empty} ({len(rows)} cap)")
        self.short_side = int(short_side)
        self.digest = hashlib.sha256(raw).hexdigest()[:16]
        base = Path(cache_dir) if cache_dir is not None else Path(tempfile.gettempdir()) / "qjepa_scene_pairs"
        # Ban thu nho la ham cua (index, short_side): doi dataset hay short_side thi sang thu muc khac.
        self.cache_dir = base / f"{self.digest}_{self.short_side}"

    def pairs(self, split: str) -> tuple[ScenePair, ...]:
        if split not in self.splits:
            raise ValueError(f"Cap canh khong co split {split!r} (chi {list(self.splits)})")
        return self.splits[split]

    def load(self, pair: ScenePair) -> tuple[np.ndarray, np.ndarray]:
        """(gt, anh loe) uint8 [H,W,3], canh ngan short_side. Lan dau giai ma anh goc va ghi cache; sau do doc cache.

        LUON tra ve ban doc tu cache (ke ca lan dau): ket qua khong phu thuoc cache da co hay chua."""
        stem = Path(pair.name).stem
        targets = (self.cache_dir / f"{stem}.gt.png", self.cache_dir / f"{stem}.input.png")
        if not all(path.is_file() for path in targets):
            with Image.open(self.root / "gt" / pair.name) as image:
                gt = np.asarray(image.convert("RGB"))
            with Image.open(self.root / "input" / pair.name) as image:
                flared = np.asarray(image.convert("RGB"))
            if gt.shape != flared.shape:
                raise ValueError(f"Cap canh {pair.name}: gt {gt.shape[1::-1]} va input {flared.shape[1::-1]} khac co")
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
        extra = ({"holdout_fraction": self.holdout_fraction, "dropped": self.dropped}
                 if self.holdout_fraction is not None else {})
        return {"root": str(self.root), "index_sha256": self.digest, "short_side": self.short_side, **extra,
                "pairs": {split: len(pairs) for split, pairs in self.splits.items()},
                "sources": {split: {source or "?": sum(pair.source == source for pair in pairs)
                                    for source in sorted({pair.source for pair in pairs})}
                            for split, pairs in self.splits.items()}}


def scene_origin(row: dict[str, str]) -> str:
    """Nguon anh nen cua mot cap: "flickr" (scene_split flickr24k cua wu-visdrone-composite) hay "drone"."""
    return "flickr" if (row.get("scene_split") or "").lower().startswith("flickr") else "drone"


def scene_group(row: dict[str, str]) -> str:
    """Nhom anh nen: cung chuoi VisDrone (phan truoc dau _ cua ten khung, vd 0000011) hoac cung anh Flickr."""
    origin, scene = scene_origin(row), row.get("scene") or row["file"]
    return f"{origin}:{scene if origin == 'flickr' else scene.split('_')[0]}"


def resplit(rows: list[dict[str, str]], fraction: float) -> dict[str, str | None]:
    """{file: train / valid / test, hoac None = bo}: p49 data.scene_pairs_holdout_fraction (xem docstring cua module)."""
    if not 0 < fraction <= 0.25:
        raise ValueError("scene_pairs_holdout_fraction must be in (0, 0.25]")
    split_of = {row["file"]: _INDEX_SPLIT[row["split"]] for row in rows}
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(scene_group(row), []).append(row)
    # Nhom nao co cap train trong index.csv, hay dung lop loe cua mot cap train, thi o train: model da thay no.
    train_flares = {row.get("flare") for row in rows if split_of[row["file"]] == "train" and row.get("flare")}
    candidates = {key: members for key, members in groups.items()
                  if all(split_of[row["file"]] == "valid" and row.get("flare") not in train_flares for row in members)}
    totals: dict[str, int] = {}
    for row in rows:
        totals[scene_origin(row)] = totals.get(scene_origin(row), 0) + 1
    filled = {(origin, split): 0 for origin in totals for split in HELDOUT_SPLITS}
    assignment: dict[str, str | None] = {row["file"]: "train" for row in rows}
    order = sorted(candidates, key=lambda key: (hashlib.sha256(key.encode()).hexdigest(), key))
    for key in order:
        members = candidates[key]
        origin = scene_origin(members[0])
        target = next((split for split in HELDOUT_SPLITS if filled[(origin, split)] < fraction * totals[origin]),
                      "train")
        for row in members:
            assignment[row["file"]] = target
        if target != "train":
            filled[(origin, target)] += len(members)
    held_flares = {row.get("flare") for row in rows if assignment[row["file"]] != "train" and row.get("flare")}
    for row in rows:
        if split_of[row["file"]] == "valid" and assignment[row["file"]] == "train" and row.get("flare") in held_flares:
            assignment[row["file"]] = None
    return assignment


def is_scene_pairs_root(path: Path, kind: str = "") -> bool:
    """``path`` co index.csv (cot file, split, scene_split), gt/ va input/: dau ra cua wu-visdrone-composite.ipynb.

    p47: ``kind`` la cot ``kind`` cua index.csv (dong dau): "" cho bo khong co cot do (VisDrone), "lowlight" cho cap
    anh toi that (lowlight-pairs-builder.ipynb). Hai bo cung bo cuc nen phai phan biet o day."""
    index = path / "index.csv"
    if not index.is_file() or not (path / "gt").is_dir() or not (path / "input").is_dir():
        return False
    with index.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        first = handle.readline().strip().split(",")
    if not all(column in header for column in ("file", "split", "scene_split")):
        return False
    found = first[header.index("kind")] if "kind" in header and len(first) == len(header) else ""
    return found == kind


def find_scene_pairs_root(base: str | Path = "/kaggle/input", depth: int = 6, skip=(), kind: str = "") -> Path:
    """Thu muc cap canh (index.csv + gt/ + input/) trong ``base``, Kaggle gan theo kieu nao cung duoc. Bo qua ``skip``.
    ``kind`` nhu is_scene_pairs_root: "lowlight" tim bo anh toi that (p47)."""
    base, skip = Path(base), {str(path) for path in skip}
    found = sorted({path.parent for level in range(1, depth + 1)
                    for path in base.glob("/".join(["*"] * level) + "/index.csv")
                    if not any(str(path).startswith(s) for s in skip) and is_scene_pairs_root(path.parent, kind)})
    if not found:
        wanted = ("lowlight-pairs (lowlight-pairs-builder.ipynb)" if kind == "lowlight"
                  else "visdrone-lens-flare-640 (wu-visdrone-composite.ipynb)")
        raise FileNotFoundError(f"Khong thay index.csv + gt/ + input/ trong {depth} tang cua {base}: gan dataset "
                                f"{wanted}")
    if len(found) > 1:
        raise ValueError(f"Cap canh duoc gan hon mot lan: {[str(path) for path in found]}; giu mot ban")
    return found[0]
