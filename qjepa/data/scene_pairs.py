"""Cap anh canh da ghep loe san (p46, ``data.scene_pairs_*``): anh sach la dich, anh co loe la mot nguon vao.

Bo dau tien la "Lens flare on VisDrone, 640x640" (Kaggle ``buidinhkhoi/visdrone-lens-flare-640``, notebook
``wu-visdrone-composite.ipynb``): 2 000 anh drone giao thong ban ngay that cua VisDrone2019-DET, moi anh cong mot lop
loe Wu et al. (bong ma da lam sang) hoac FlareX tren anh sang tuyen tinh. Bo cuc:

    <root>/index.csv   file, split (train / val), scene, scene_split, flare, seed, ... (moi cap mot dong)
    <root>/gt/<file>     anh sach 640x640 (giu loi nguon sang, nhu dich cua repo)
    <root>/input/<file>  cung khung, co loe

Khac FlareImage, split da chia san trong index.csv (valid = anh VisDrone test-dev, khong chung chuoi voi train, ghep voi
lop loe nhom valid), nen bank giu nguyen: ``train`` va ``valid``, khong co ``test``. Moi cap duoc thu nho MOT lan ve
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

    def __init__(self, root: str | Path, *, short_side: int, cache_dir: str | Path | None = None):
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
        pairs = {split: [] for split in SCENE_PAIR_SPLITS}
        for row in rows:
            source = (row.get("flare") or "").split("/")[0]
            pairs[_INDEX_SPLIT[row["split"]]].append(ScenePair(row["file"], source, row.get("scene") or ""))
        self.splits = {split: tuple(sorted(items, key=lambda pair: pair.name)) for split, items in pairs.items()}
        empty = [split for split in SCENE_PAIR_SPLITS if not self.splits[split]]
        if empty:
            raise ValueError(f"Cap canh o {self.root}: khong co cap nao cho {empty} ({len(rows)} cap)")
        self.short_side = int(short_side)
        self.digest = hashlib.sha256(raw).hexdigest()[:16]
        base = Path(cache_dir) if cache_dir is not None else Path(tempfile.gettempdir()) / "qjepa_scene_pairs"
        # Ban thu nho la ham cua (index, short_side): doi dataset hay short_side thi sang thu muc khac.
        self.cache_dir = base / f"{self.digest}_{self.short_side}"

    def pairs(self, split: str) -> tuple[ScenePair, ...]:
        if split not in self.splits:
            raise ValueError(f"Cap canh khong co split {split!r} (chi {list(SCENE_PAIR_SPLITS)})")
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
        return {"root": str(self.root), "index_sha256": self.digest, "short_side": self.short_side,
                "pairs": {split: len(pairs) for split, pairs in self.splits.items()},
                "sources": {split: {source or "?": sum(pair.source == source for pair in pairs)
                                    for source in sorted({pair.source for pair in pairs})}
                            for split, pairs in self.splits.items()}}


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
