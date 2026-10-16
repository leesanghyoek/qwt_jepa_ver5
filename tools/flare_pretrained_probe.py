"""Nhanh khu loe dang dung vs Uformer DA TRAIN SAN cua Flare7K++, tren cung cac cap anh loe that (khong train).

Do de quyet dinh p44: thay nhanh loe tu train (giai doan loe, OUT/flare/last.pt, U-Net train tu dau ~12 000 buoc) bang
Uformer Flare7K++ (qjepa/models/uformer_flare.py, trong so cong bo, train ~4 ngay tren chinh loai du lieu nay).
Ca hai chay DUNG MOT MINH, tren anh loe chup that vao thang (khong mo, khong hat), so voi anh gt GOC (khong lam sang
nhu phase2.target_brighten): chi do khu loe.

Cap lay tu split cua FlareImage (mac dinh test: khong cap nao cung canh voi train/valid cua repo). Uformer Flare7K++
cung chua thay cac cap nay: flare7k_real/synthetic cua FlareImage la bo TEST cua Flare7K++; flare_removal (anh chup
that) va halo khong thuoc Flare7K++. Moi cap thu nho nhu luc train (canh ngan data.flare_pairs_short_side), roi:
  --frame full (mac dinh): ca khung, moi canh cat (o giua) ve boi so cua 16; --frame crop: o data.image_size o giua.
Bang tung cap va theo nguon: PSNR ca anh, VUNG LOE (pixel anh vao lech gt hon 0,05, nhu tools/flare_image_test.py) va
NGOAI LOE, cho anh vao / nhanh hien tai / Uformer. Ve --show cap dau: vao | nhanh hien tai | Uformer | sach.

    python3 tools/flare_pretrained_probe.py --uformer <net_g_last.pth> --flare-stage OUT/flare/last.pt \\
        --output OUT/flare_pretrained_probe/<gio> [--flare-pairs-root ...] [--split test] [--frame full] [--show 8]
Khong co --flare-stage thi can --config (de biet cach chia FlareImage) va chi do Uformer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from qjepa.config import load_config  # noqa: E402
from qjepa.data.flare_pairs import FLARE_PAIR_SPLITS, FlarePairBank  # noqa: E402
from qjepa.models.uformer_flare import load_flare7kpp, remove_flare  # noqa: E402
from qjepa.training.checkpoints import load_checkpoint  # noqa: E402
from qjepa.training.flare_stage import build_flare_layer  # noqa: E402
from tools.flare_image_test import FLARE_THRESHOLD, frame_of, psnr  # noqa: E402

METHODS = ("vao", "nhanh", "uformer")
LABELS = {"vao": "vào", "nhanh": "nhánh hiện tại", "uformer": "Uformer F7K++"}


def scores(image: torch.Tensor, clean: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    return {"ca": psnr(image, clean), "loe": psnr(image, clean, mask), "ngoai": psnr(image, clean, ~mask)}


def summarize(rows: list[dict], methods: tuple[str, ...]) -> dict:
    summary: dict = {"count": len(rows)}
    for method in methods:
        for part in ("ca", "loe", "ngoai"):
            values = np.array([row[method][part] for row in rows], dtype=float)
            summary[f"{method}_{part}"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    if "uformer" in methods and "nhanh" in methods:
        summary["uformer_hon_nhanh_loe"] = float(np.mean([r["uformer"]["loe"] > r["nhanh"]["loe"] for r in rows
                                                          if np.isfinite(r["uformer"]["loe"])]))
    return summary


def panel(path: Path, images: dict[str, torch.Tensor], clean: torch.Tensor, row: dict) -> None:
    shown = [(f"{LABELS[m]} {row[m]['ca']:.2f} dB (lóe {row[m]['loe']:.1f})", images[m]) for m in images]
    shown.append(("sạch (gt)", clean))
    figure, axes = plt.subplots(1, len(shown), figsize=(3.6 * len(shown), 3.6 * clean.shape[1] / clean.shape[2] + 0.7))
    for axis, (title, picture) in zip(axes, shown):
        axis.imshow(picture.permute(1, 2, 0).clamp(0, 1).numpy())
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    figure.suptitle(f"{row['source']} {row['name']} ({row['width']}x{row['height']}) | vùng lóe "
                    f"{100 * row['flare_fraction']:.0f}% khung", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=100)
    plt.close(figure)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uformer", required=True, help="checkpoint Flare7K++ (net_g_last.pth)")
    parser.add_argument("--flare-stage", help="OUT/flare/last.pt: nhanh loe hien tai (khong co: chi do Uformer)")
    parser.add_argument("--config", help="config da resolve, khi khong co --flare-stage")
    parser.add_argument("--output", required=True, help="thu muc moi: hinh va report.json")
    parser.add_argument("--flare-pairs-root", help="thu muc FlareImage (mac dinh: data.flare_pairs_root cua config)")
    parser.add_argument("--split", default="test", choices=FLARE_PAIR_SPLITS)
    parser.add_argument("--frame", default="full", choices=("full", "crop"))
    parser.add_argument("--max-pairs", type=int, default=0, help="0 = moi cap cua split")
    parser.add_argument("--show", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.flare_stage is None and args.config is None:
        parser.error("can --flare-stage (lay config cua no) hoac --config")
    device = torch.device(args.device)
    stage = None
    if args.flare_stage:
        payload = load_checkpoint(args.flare_stage, "cpu")
        if payload.get("metadata", {}).get("phase") != "flare_stage":
            raise ValueError(f"{args.flare_stage} khong phai checkpoint giai doan loe (OUT/flare/last.pt)")
        config = payload["config"]
        stage = build_flare_layer(config["flare"])
        stage.load_state_dict(payload["model"], strict=True)
        stage = stage.to(device).eval()
        del payload
    else:
        config = load_config(args.config)
    data = config["data"]
    root = args.flare_pairs_root or data.get("flare_pairs_root")
    if not root:
        raise ValueError("can --flare-pairs-root (config khong co data.flare_pairs_root)")
    bank = FlarePairBank(root, short_side=int(data.get("flare_pairs_short_side", 384)),
                         holdout_fraction=float(data.get("flare_pairs_holdout_fraction", 0.1)))
    pairs = list(bank.pairs(args.split))
    if args.max_pairs > 0:
        # Rai deu qua danh sach (cac nguon nam lien nhau theo ten), khong lay dau danh sach.
        pairs = [pairs[int(k)] for k in np.linspace(0, len(pairs) - 1, min(args.max_pairs, len(pairs))).round()]
    uformer = load_flare7kpp(args.uformer, device)
    methods = METHODS if stage is not None else ("vao", "uformer")
    size = tuple(int(side) for side in data["image_size"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    print(f"FlareImage {args.split}: {len(pairs)} cặp | khung {args.frame} | {bank.root}")
    print(f"  so với gt GỐC (không làm sáng đích); vùng lóe = pixel vào lệch gt > {FLARE_THRESHOLD}")
    header = " | ".join(f"{part}: " + "/".join(LABELS[m].split()[0] for m in methods)
                        for part in ("cả ảnh", "vùng lóe", "ngoài lóe"))
    print(f"  {'nguồn':18s} {'cặp':11s} {'cỡ':>9s} {'lóe%':>5s}  {header}")
    rows = []
    for number, pair in enumerate(pairs):
        gt, flared = bank.load(pair)
        gt, flared = (frame_of(array, args.frame, size).astype(np.float32) / 255.0 for array in (gt, flared))
        clean = torch.from_numpy(gt).permute(2, 0, 1)
        noisy = torch.from_numpy(flared).permute(2, 0, 1)
        mask = (noisy - clean).abs().mean(0) > FLARE_THRESHOLD
        images = {"vao": noisy}
        if stage is not None:
            with torch.no_grad():
                images["nhanh"] = stage(None, noisy[None].to(device))[0][0].float().clamp(0, 1).cpu()
        images["uformer"] = torch.from_numpy(remove_flare(uformer, flared)).permute(2, 0, 1)
        row = {"name": pair.name, "source": pair.source, "width": int(clean.shape[-1]), "height": int(clean.shape[-2]),
               "flare_fraction": float(mask.float().mean()),
               **{method: scores(images[method], clean, mask) for method in methods}}
        rows.append(row)
        if number < args.show:
            row["panel"] = f"pair_{number:02d}.png"
            panel(output / row["panel"], images, clean, row)
        cells = " | ".join("/".join(f"{row[m][part]:5.2f}" for m in methods) for part in ("ca", "loe", "ngoai"))
        print(f"  {pair.source:18s} {pair.name:11s} {str(row['width']) + 'x' + str(row['height']):>9s}"
              f" {100 * row['flare_fraction']:4.0f}%  {cells}", flush=True)
    sources = sorted({row["source"] for row in rows})
    summary = {"tat ca": summarize(rows, methods),
               **{source: summarize([r for r in rows if r["source"] == source], methods) for source in sources}}
    names = "/".join(LABELS[m].split()[0] for m in methods)
    print(f"\n  {'nhóm':18s} {'số':>4s}  " + "  ".join(f"{part + ' ' + names:>30s}"
                                                  for part in ("cả ảnh", "vùng lóe", "ngoài lóe")))
    for name, values in summary.items():
        cells = "  ".join(f"{' / '.join(format(values[f'{m}_{part}'], '6.2f') for m in methods):>30s}"
                          for part in ("ca", "loe", "ngoai"))
        extra = (f"  Uformer hơn nhánh ở vùng lóe: {100 * values['uformer_hon_nhanh_loe']:.0f}% cặp"
                 if "uformer_hon_nhanh_loe" in values else "")
        print(f"  {name:18s} {values['count']:>4d}  {cells}{extra}")
    report = {"uformer": str(args.uformer), "flare_stage": args.flare_stage, "split": args.split, "frame": args.frame,
              "threshold": FLARE_THRESHOLD, "rows": rows, "summary": summary}
    (output / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã lưu {output}: {min(args.show, len(rows))} hình pair_XX.png, report.json")
    return report


if __name__ == "__main__":
    main()
