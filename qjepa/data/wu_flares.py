"""Anh CHI CO LOE cua Wu va cs. (ICCV 2021, "How to Train Neural Networks for Flare Removal", Google Research): loe ban
ngay kieu mat troi hat vao ong kinh -- quang choi, chuoi bong ma hinh la khau (da giac), tia sao.

Bucket cong khai ``gs://gresearch/lens-flare`` (giay phep CC BY 4.0; ghi nguon: Wu, He, Xue, Garg, Chen, Veeraraghavan,
Barron 2021), 5 001 anh 1008x752 RGB 8 bit tren nen toi, ~2,3 GiB:

    captured/frame_0000.png ... frame_2000.png     2 001 anh CHUP THAT trong phong lab: mot nguon diem rat sang qua
                                                   ong kinh that, nguon di chuyen giua cac khung -> bong ma, quang
    simulated/aperture0000_blur00_crop00.png ...   3 000 anh MO PHONG quang hoc song: 189 hinh la khau x 4 muc mo x
                                                   4 o cat -> tia sao nhieu xa

Khong co anh nguon sang rieng (Flare7K++ co): corruptions/lens_flare.py tach loi sang bao hoa cua chinh anh loe lam
nguon sang de anh DICH giu mat troi.

``fetch_wu_flares`` tai qua HTTPS (khong can dang nhap), thu canh dai ve ``side`` px BOX tren anh sang tuyen tinh (giu
tong anh sang, nhu flare7kpp-builder), ghi PNG 8 bit vao <dich>/{captured,simulated}/; chay lai thi bo qua file da co.
``find_wu_flares`` tim ban da dung san trong Input (vd Output cua phien truoc tao thanh dataset).
"""

from __future__ import annotations

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

BUCKET = "https://storage.googleapis.com/gresearch"
LISTING = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
PARTS = ("captured", "simulated")
EXPECTED = {"captured": 2001, "simulated": 3000}
GAMMA = 2.2                     # chi de thu nho tren anh sang tuyen tinh; giai doan loe tu rut gamma rieng


def list_wu_flares(part: str) -> list[str]:
    """Ten cac anh cua ``part`` trong bucket (API JSON cua Cloud Storage, phan trang)."""
    names, token = [], None
    while True:
        url = f"{LISTING}?prefix=lens-flare/{part}/&fields=items(name),nextPageToken&maxResults=1000"
        if token:
            url += f"&pageToken={token}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.load(response)
        names += [item["name"].rsplit("/", 1)[-1] for item in data.get("items", []) if item["name"].endswith(".png")]
        token = data.get("nextPageToken")
        if not token:
            return sorted(names)


def shrink(pixels: np.ndarray, side: int) -> np.ndarray:
    """Canh dai ve ``side`` px, BOX tren anh sang tuyen tinh (gamma 2,2), tra ve uint8; nho hon thi giu nguyen."""
    height, width = pixels.shape[:2]
    if max(height, width) <= side:
        return pixels
    scale = side / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    linear = np.power(pixels.astype(np.float32) / 255.0, GAMMA)
    planes = [np.asarray(Image.fromarray(np.ascontiguousarray(linear[..., c]), mode="F").resize(
        size, Image.Resampling.BOX)) for c in range(3)]
    out = np.power(np.clip(np.stack(planes, axis=-1), 0.0, 1.0), 1.0 / GAMMA)
    return np.round(out * 255.0).astype(np.uint8)


def _fetch_one(part: str, name: str, target: Path, side: int) -> None:
    path = target / part / name
    if path.is_file() and path.stat().st_size > 0:
        return
    with urllib.request.urlopen(f"{BUCKET}/lens-flare/{part}/{name}", timeout=120) as response:
        data = response.read()
    with Image.open(BytesIO(data)) as image:
        pixels = np.asarray(image.convert("RGB"))
    temporary = path.with_suffix(".tmp.png")
    Image.fromarray(shrink(pixels, side)).save(temporary)
    os.replace(temporary, path)


def fetch_wu_flares(target, side: int = 640, workers: int = 16, parts=PARTS) -> Path:
    """Tai (hoac hoan tat) ban ``side`` px cua bo anh loe Wu va cs. vao ``target``; tra ve ``target``."""
    target = Path(target)
    for part in parts:
        (target / part).mkdir(parents=True, exist_ok=True)
        names = list_wu_flares(part)
        missing = [name for name in names if not (target / part / name).is_file()]
        print(f"Wu et al. {part}: {len(names)} anh, cần tải {len(missing)}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda name: _fetch_one(part, name, target, side), missing))
    (target / "SOURCE.txt").write_text(
        "Wu, He, Xue, Garg, Chen, Veeraraghavan, Barron. How to Train Neural Networks for Flare Removal. ICCV 2021.\n"
        "gs://gresearch/lens-flare (CC BY 4.0), shrunk to a long side of "
        f"{side} px in linear light by qjepa/data/wu_flares.py.\n", encoding="utf-8")
    return target


def is_wu_root(path: Path) -> bool:
    """``path`` chua captured/ va simulated/ voi du so anh."""
    return all((path / part).is_dir() and sum(1 for name in os.listdir(path / part) if name.endswith(".png"))
               >= EXPECTED[part] for part in PARTS)


def find_wu_flares(base, depth: int = 6, skip=()) -> Path | None:
    """Ban da dung san trong ``base`` (Kaggle Input): thu muc co captured/ va simulated/ du anh. Bo qua ``skip``."""
    base, skip = Path(base), {str(path) for path in skip}
    for level in range(depth):
        for captured in sorted(base.glob("/".join(["*"] * level + ["captured"]))):
            root = captured.parent
            if captured.is_dir() and not any(str(root).startswith(s) for s in skip) and is_wu_root(root):
                return root
    return None
