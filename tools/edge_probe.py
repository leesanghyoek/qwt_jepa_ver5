"""D2 — Latent có biết thêm đường nét nào ngoài những gì ảnh hỏng đã cho? Không train.

Với mỗi ô của lưới latent ZI (16×16 px ở ảnh 256), một hồi quy ridge đoán CHI TIẾT
của ảnh sạch tại ô đó từ từng bộ đặc trưng:

  ảnh hỏng            kênh sáng Y của ảnh hỏng quanh ô (thêm lề 4 px mỗi phía)
  ZI                  vector latent của ô, tính từ ảnh hỏng
  ZI sạch             latent của chính ảnh sạch: lưới 16×16 của encoder này giữ được gì
  ảnh hỏng + ZI       câu hỏi chính
  ảnh hỏng + ZI sạch  trần: latent của chính ảnh sạch qua cùng encoder
  ảnh hỏng + tầng 1/8, ảnh hỏng + tầng 1/4
                      các tầng mịn hơn của encoder JEPA (tính từ ảnh hỏng)

Bốn đích, trên Y của ảnh sạch:
  mịn 2–4 px          Y − G(σ=1)·Y        (vật nhỏ / xa), có dấu
  đường nét 4–16 px   G(σ=1)·Y − G(σ=4)·Y, có dấu
  năng lượng mịn / năng lượng nét
                      căn bậc hai trung bình bình phương của băng đó trên từng khối 4×4:
                      cạnh ở ĐÂU và MẠNH bao nhiêu, không cần dấu. Đặc trưng sau ReLU/norm
                      hay mã hoá độ mạnh (như modulus QWT) chứ không giữ dấu; một probe
                      tuyến tính đoán đích có dấu từ chúng ra gần 0 dù thông tin có ở đó.

Điểm "% rút được" = 1 − MAE probe / MAE đoán trung bình, như tools/latent_probe.py.
Hồi quy tuyến tính là cận DƯỚI của thông tin có trong đặc trưng.

Mỗi khối đặc trưng có MỨC PHẠT RIÊNG, chọn trên validation tách từ phần train (kể cả
"bỏ khối"), rồi train lại trên cả phần train và chấm trên phần test. Bản trước dùng một
mức phạt chung: ghép thêm khối nào cũng làm điểm tụt theo số chiều, kể cả khối nhiễu
thuần (đo trên 7200 ô TartanAir: +128 chiều nhiễu −3,4 điểm mịn, +1024 chiều −19,1), nên
"ảnh hỏng + X" thấp hơn "ảnh hỏng" không nói gì về X. Giờ "ảnh hỏng + X" ≥ "ảnh hỏng"
trừ sai số chọn mức phạt; chênh lệch dương mới là thông tin X mang thêm.

Đọc kết quả:
  "ảnh hỏng + ZI" ≈ "ảnh hỏng"                      -> ZI không mang thêm đường nét nào.
  "+ ZI sạch" hơn rõ, "+ ZI" thì không                -> latent 16×16 chứa được, nhưng JEPA
                                                        chưa đưa thông tin đó vào ZI ảnh hỏng.
  cả "+ ZI sạch" cũng không hơn                       -> lưới 16×16 của encoder này không giữ
                                                        đường nét; phải dùng tầng mịn hơn.
  "+ tầng 1/8" hoặc "+ tầng 1/4" hơn rõ               -> thông tin có ở tầng mịn: đưa tầng đó
                                                        vào decoder / làm đích JEPA.

    python3 tools/edge_probe.py --checkpoint <phase1 last.pt hoặc phase2 best.pt> \\
        --manifest <manifest> --output outputs/<run>/diagnostics/edge_probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from qjepa.cli import _dataset, _loader, _system_from_phase2, _to_device  # noqa: E402
from qjepa.config import build_normalizer, build_phase1_model  # noqa: E402
from qjepa.data import read_manifest  # noqa: E402
from qjepa.training.checkpoints import load_checkpoint  # noqa: E402
from tools.image_blur_audit import SCENARIOS, _spread_indices  # noqa: E402

TARGETS = ("fine", "edges", "fine_energy", "edges_energy")
TARGET_LABELS = {"fine": "mịn 2–4 px", "edges": "đường nét 4–16 px",
                 "fine_energy": "năng lượng mịn", "edges_energy": "năng lượng nét"}
ENERGY_BLOCK = 4                                  # px: cạnh ở đâu, ở độ phân giải này
# Mức phạt của mỗi khối (đặc trưng đã chuẩn hoá). 1e12 = bỏ khối: trọng số về ~0.
PENALTIES = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e12)


def luma(image: torch.Tensor) -> torch.Tensor:
    return image[:, :1] * 0.299 + image[:, 1:2] * 0.587 + image[:, 2:3] * 0.114


def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian, reflect padding, [B, 1, H, W]."""
    radius = max(1, int(math.ceil(3 * sigma)))
    offsets = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / kernel.sum()
    padded = F.pad(image, (radius, radius, 0, 0), mode="reflect")
    image = F.conv2d(padded, kernel.view(1, 1, 1, -1))
    padded = F.pad(image, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel.view(1, 1, -1, 1))


def detail_targets(clean_y: torch.Tensor) -> dict[str, torch.Tensor]:
    """The two detail bands of the clean luminance, [B, 1, H, W] each."""
    smooth1 = gaussian_blur(clean_y, 1.0)
    return {"fine": clean_y - smooth1, "edges": smooth1 - gaussian_blur(clean_y, 4.0)}


def energy_targets(bands: dict[str, torch.Tensor], block: int = ENERGY_BLOCK) -> dict[str, torch.Tensor]:
    """Root mean square of each band over block x block tiles, [B, 1, H/block, W/block]."""
    return {f"{name}_energy": F.avg_pool2d(band.square(), block).sqrt() for name, band in bands.items()}


def block_ridge_probe(blocks: list[np.ndarray], targets: dict[str, np.ndarray], ratio: float = 0.7,
                      validation: float = 0.2) -> dict[str, tuple[float, float]]:
    """Held-out MAE per target of a ridge with one penalty PER BLOCK, and the mean-guess MAE.

    Rows [0, ratio) train, the rest test. The last ``validation`` of the train rows picks
    each block's penalty (PENALTIES, the largest of which drops the block); the model is
    then refit on all train rows. Features are standardised and targets centred on the
    train rows, so no intercept is needed or penalised.
    """
    rows = len(blocks[0])
    split = int(rows * ratio)
    fit = int(split * (1.0 - validation))
    standardised = []
    for block in blocks:
        mean, deviation = block[:split].mean(0), block[:split].std(0) + 1e-8
        standardised.append((block - mean) / deviation)
    x = np.concatenate(standardised, axis=1)
    widths = [block.shape[1] for block in blocks]
    names = list(targets)
    y_all = np.concatenate([targets[name] for name in names], axis=1)
    columns = np.cumsum([0] + [targets[name].shape[1] for name in names])

    def solve(train_rows: slice, penalties: tuple[float, ...]) -> np.ndarray:
        xt, yt = x[train_rows], y_all[train_rows]
        diagonal = np.concatenate([np.full(width, penalty) for width, penalty in zip(widths, penalties)])
        return np.linalg.solve(xt.T @ xt + np.diag(diagonal), xt.T @ (yt - yt.mean(0)))

    combos = list(np.array(np.meshgrid(*[PENALTIES] * len(blocks))).T.reshape(-1, len(blocks)))
    best: dict[str, tuple[float, tuple[float, ...]]] = {name: (np.inf, ()) for name in names}
    centre = y_all[:fit].mean(0)
    for penalties in map(tuple, combos):
        prediction = x[fit:split] @ solve(slice(0, fit), penalties) + centre
        for index, name in enumerate(names):
            part = slice(columns[index], columns[index + 1])
            error = float(np.abs(prediction[:, part] - y_all[fit:split, part]).mean())
            if error < best[name][0]:
                best[name] = (error, penalties)
    results = {}
    centre = y_all[:split].mean(0)
    refits: dict[tuple[float, ...], np.ndarray] = {}
    for index, name in enumerate(names):
        penalties = best[name][1]
        if penalties not in refits:
            refits[penalties] = x[split:] @ solve(slice(0, split), penalties) + centre
        part = slice(columns[index], columns[index + 1])
        probe = float(np.abs(refits[penalties][:, part] - y_all[split:, part]).mean())
        baseline = float(np.abs(centre[part] - y_all[split:, part]).mean())
        results[name] = (probe, baseline)
    return results


def cells(map_: torch.Tensor, grid: tuple[int, int], margin: int = 0) -> torch.Tensor:
    """Split [B, C, H, W] into the latent grid's cells: [B, cells, C·(tile+2·margin)²].

    Cell (i, j) covers rows i·tile … (i+1)·tile of the map, widened by ``margin``
    on every side (reflect padding at the frame border).
    """
    rows, columns = grid
    height, width = map_.shape[-2:]
    if height % rows or width % columns:
        raise ValueError(f"map {height}x{width} does not divide into a {rows}x{columns} grid")
    tile_h, tile_w = height // rows, width // columns
    if margin:
        map_ = F.pad(map_, (margin,) * 4, mode="reflect")
    patches = F.unfold(map_, (tile_h + 2 * margin, tile_w + 2 * margin), stride=(tile_h, tile_w))
    return patches.transpose(1, 2)


def load_backbone(checkpoint: str, manifest: dict, device: torch.device):
    """Backbone + normalizer from either phase's checkpoint (phase 1 needs no decoder)."""
    payload = load_checkpoint(checkpoint, device)
    if payload.get("metadata", {}).get("phase") == "latent_pretrain":
        config = payload["config"]
        model = build_phase1_model(config, build_normalizer(manifest["meta"])).to(device)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        return model.backbone, model.normalizer, config
    system, config = _system_from_phase2(checkpoint, device)
    return system.backbone, system.normalizer, config


def collect(backbone, normalizer, loader, device, cells_per_image: int, margin: int,
            rng: np.random.Generator) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {}
    for raw in loader:
        batch = _to_device(raw, device)
        with torch.no_grad():
            imu = normalizer.normalize(batch["imu_noisy_phys"])
            latent = backbone.encode_online(batch["image_noisy"], imu, batch["image_time"],
                                            batch["imu_times"], with_skips=True)
            latent_clean = backbone.encode_online(batch["image_clean"], imu, batch["image_time"],
                                                  batch["imu_times"], with_skips=False)
            grid = tuple(latent.ZI.shape[-2:])
            parts = {
                "input": cells(luma(batch["image_noisy"].clamp(0, 1)), grid, margin),
                "zi": cells(latent.ZI, grid),
                "zi_clean": cells(latent_clean.ZI, grid),
            }
            # image_skips run coarse to fine: 1/8, 1/4, 1/2 of the frame.
            for name, index in (("stage8", 0), ("stage4", 1)):
                if latent.image_skips is not None and len(latent.image_skips) > index:
                    parts[name] = cells(latent.image_skips[index], grid)
            bands = detail_targets(luma(batch["image_clean"].clamp(0, 1)))
            for name, band in {**bands, **energy_targets(bands)}.items():
                parts[f"target_{name}"] = cells(band, grid)
        total = parts["zi"].shape[1]
        for sample in range(parts["zi"].shape[0]):
            picked = torch.as_tensor(rng.choice(total, size=min(cells_per_image, total), replace=False),
                                     device=parts["zi"].device)
            for name, value in parts.items():
                rows.setdefault(name, []).append(value[sample, picked].double().cpu().numpy())
    return {name: np.concatenate(values) for name, values in rows.items()}


FEATURE_SETS = (
    ("input", ("input",), "ảnh hỏng"),
    ("zi", ("zi",), "ZI"),
    ("zi_clean", ("zi_clean",), "ZI sạch"),
    ("input+zi", ("input", "zi"), "ảnh hỏng + ZI"),
    ("input+zi_clean", ("input", "zi_clean"), "ảnh hỏng + ZI sạch (trần)"),
    ("input+stage8", ("input", "stage8"), "ảnh hỏng + tầng 1/8"),
    ("input+stage4", ("input", "stage4"), "ảnh hỏng + tầng 1/4"),
)


def probe_all(data: dict[str, np.ndarray]) -> dict[str, dict]:
    results = {}
    for key, parts, label in FEATURE_SETS:
        if any(part not in data for part in parts):
            continue
        blocks = [data[part] for part in parts]
        entry = {"label": label, "dims": int(sum(block.shape[1] for block in blocks)), "rows": int(len(blocks[0]))}
        scores = block_ridge_probe(blocks, {target: data[f"target_{target}"] for target in TARGETS})
        for target, (probe, baseline) in scores.items():
            entry[target] = {"probe_mae": probe, "mean_guess_mae": baseline,
                             "recovered_pct": 100.0 * (1.0 - probe / baseline) if baseline > 0 else float("nan")}
        results[key] = entry
    return results


def edge_probe(checkpoint: str | Path, manifest_path: str | Path, *, output: str | Path,
               split: str = "valid", samples: int = 600, cells_per_image: int = 12, margin: int = 4,
               scenarios: tuple[str, ...] = ("blur_only", "full_full"), device: str = "cuda",
               seed: int = 0) -> dict:
    device_obj = torch.device(device)
    manifest = read_manifest(manifest_path)
    backbone, normalizer, config = load_backbone(str(checkpoint), manifest, device_obj)
    config["data"] = dict(config["data"], num_workers=0, pin_memory=False)
    indices = _spread_indices(len(manifest["samples"][split]), samples)
    report = {}
    for scenario in scenarios:
        image_mode, imu_mode = SCENARIOS[scenario]
        dataset = _dataset(config, manifest, split, fixed_realization=True,
                           image_mode=image_mode, imu_mode=imu_mode)
        loader = _loader(config, Subset(dataset, indices), config["phase2"]["batch_size"], train=False)
        data = collect(backbone, normalizer, loader, device_obj, cells_per_image, margin,
                       np.random.default_rng(seed))
        report[scenario] = probe_all(data)

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"checkpoint": str(checkpoint), "split": split, "spread_samples": len(indices),
                                "cells_per_image": cells_per_image, "margin_px": margin,
                                "scenarios": report}, indent=2), encoding="utf-8")
    print_report(report)
    print(f"Đã lưu: {path}")
    return report


def print_report(report: dict) -> None:
    print("D2 — probe đường nét · ridge tuyến tính, 70% train / 30% test · % rút được (cao hơn = biết nhiều hơn)")
    for scenario, results in report.items():
        rows = next(iter(results.values()))["rows"]
        print(f"\n{scenario}: {rows} ô")
        print(f"  {'đặc trưng':<30}{'chiều':>7}" + "".join(f"{TARGET_LABELS[t]:>20}" for t in TARGETS))
        for entry in results.values():
            warn = "  (ít hàng so với số chiều)" if entry["rows"] <= 2 * entry["dims"] else ""
            print(f"  {entry['label']:<30}{entry['dims']:>7}"
                  + "".join(f"{entry[t]['recovered_pct']:>19.1f}%" for t in TARGETS) + warn)
        base = results["input"]
        for key, label in (("input+zi", "ZI"), ("input+zi_clean", "ZI sạch"),
                           ("input+stage8", "tầng 1/8"), ("input+stage4", "tầng 1/4")):
            if key in results:
                gains = " · ".join(f"{TARGET_LABELS[t]} {results[key][t]['recovered_pct'] - base[t]['recovered_pct']:+.1f} điểm"
                                   for t in TARGETS)
                print(f"  thêm {label} vào ảnh hỏng: {gains}")
    print("\nĐọc: xem docstring đầu file (python3 tools/edge_probe.py --help).")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="checkpoint phase 1 hoặc phase 2")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="file JSON kết quả")
    parser.add_argument("--split", default="valid", choices=("valid", "test"))
    parser.add_argument("--samples", type=int, default=600, help="số frame, rải đều trên cả split")
    parser.add_argument("--cells-per-image", type=int, default=12)
    parser.add_argument("--margin", type=int, default=4, help="lề px quanh ô cho đặc trưng ảnh hỏng")
    parser.add_argument("--scenarios", nargs="+", default=["blur_only", "full_full"], choices=sorted(SCENARIOS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    edge_probe(args.checkpoint, args.manifest, output=args.output, split=args.split, samples=args.samples,
               cells_per_image=args.cells_per_image, margin=args.margin, scenarios=tuple(args.scenarios),
               device=args.device, seed=args.seed)


if __name__ == "__main__":
    main()
