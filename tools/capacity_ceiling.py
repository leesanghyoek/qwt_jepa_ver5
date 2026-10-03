"""Tran cua mot ma tuyen tinh 128 chieu tren chinh cac mang anh sach.

latent_probe.py tra ve mot con so nhu "rut duoc 53%", nhung khong co moc nao de
biet 53% la tot hay te. O latent co 128 chieu con mang anh co 16x16x3 = 768 so,
nen 100% la KHONG THE dat; cau hoi that su la tran cua viec nen 6 lan nam o dau.

PCA la ma tuyen tinh toi uu o so chieu cho truoc. Chay dung ham ridge_probe tren
dac trung PCA, cung du lieu va cung cach chia, cho ra tran do. Khoang cach giua
latent va tran noi phase 1 con du dia hay da cham nguong:

  latent ~ tran   -> dung dung luong; muon hon phai NOI TRAN (ZI 16x16 -> 32x32).
  latent << tran  -> latent dang phi cho; neo va lich train con cho de sua.

Khong nap model nao — chi can manifest va anh sach.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qjepa.cli import _dataset, _loader
from qjepa.config import load_config
from qjepa.data import read_manifest
from qjepa.models.color_edge import luminance
from latent_probe import ridge_probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline_v3.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="valid", choices=("valid", "test"))
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--cells-per-image", type=int, default=8)
    parser.add_argument("--latent-grid", type=int, default=16, help="ZI la [C, grid, grid]")
    parser.add_argument("--dims", type=int, default=128, help="so chieu cua mot o latent")
    parser.add_argument("--luminance", action="store_true",
                        help="mang kenh sang Y thay vi RGB: tran de so voi dong 'Y theo o' cua latent_probe.py")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = read_manifest(args.manifest)
    dataset = _dataset(config, manifest, args.split, fixed_realization=True,
                       image_mode="clean", imu_mode="clean")
    loader = _loader(config, dataset, config["phase2"]["batch_size"], train=False)

    rng = np.random.default_rng(0)
    patches: list[np.ndarray] = []
    seen = 0
    for raw in loader:
        if seen >= args.samples:
            break
        image = raw["image_clean"]
        if args.luminance:
            image = luminance(image)
        count, channels, height, width = image.shape
        rows = columns = args.latent_grid
        tile_h, tile_w = height // rows, width // columns
        tiles = image.unfold(2, tile_h, tile_h).unfold(3, tile_w, tile_w)
        tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(count, rows * columns, -1)
        picked = rng.choice(rows * columns, size=min(args.cells_per_image, rows * columns),
                            replace=False)
        patches.append(tiles[:, picked].reshape(-1, tiles.shape[-1]).numpy())
        seen += count

    data = np.concatenate(patches).astype(np.float64)
    print(f"mau {seen} anh | {len(data)} mang | moi mang {data.shape[1]}d"
          f" ({args.latent_grid}x{args.latent_grid} o, mang {256 // args.latent_grid}px)")

    # PCA chi tren phan train, dung ty le 0.7 giong ridge_probe — neu khop PCA tren
    # ca tap thi phan held-out bi ro ri va tran se bi thoi phong.
    split = int(len(data) * 0.7)
    centre = data[:split].mean(0, keepdims=True)
    _, _, components = np.linalg.svd(data[:split] - centre, full_matrices=False)

    print(f"\n{'so chieu':>10}{'probe':>10}{'doan TB':>10}{'rut duoc':>11}")
    dims = sorted({args.dims, args.dims // 2, args.dims * 2, 32})
    if len(data) <= max(dims) * 2:
        print(f"CANH BAO: chi {len(data)} hang cho toi da {max(dims)} chieu"
              f" — tang --samples, tran se bi thoi phong")
    for k in dims:
        if k > components.shape[0]:
            print(f"{k:>10}   bo qua: chi co {components.shape[0]} thanh phan")
            continue
        features = (data - centre) @ components[:k].T
        probe, baseline = ridge_probe(features, data)
        share = 100.0 * (1.0 - probe / baseline)
        mark = "   <- so chieu cua mot o latent" if k == args.dims else ""
        print(f"{k:>10}{probe:>10.4f}{baseline:>10.4f}{share:>10.0f}%{mark}")

    print(f"\nDoc ket qua: dong danh dau la TRAN cua mot ma tuyen tinh {args.dims} chieu.")
    print(f"So sanh voi '{'Y' if args.luminance else 'ANH'} theo o' cua latent_probe.py:")
    print("  latent gan tran  -> dung du dung luong; phai noi tran moi hon duoc.")
    print("  latent kem xa    -> con du dia o phase 1 (neo, lich train, trong so bang).")


if __name__ == "__main__":
    main()
