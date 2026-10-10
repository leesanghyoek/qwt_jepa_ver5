"""Model phase 2 go duoc bao nhieu phan loe HALO? Cung mot frame, co loe va khong loe, moi nhieu khac giu nguyen.

Moi frame cua split (mac dinh valid: lop loe cua cac scene HALO danh rieng cho split do, model chua thay luc
train) duoc lam nhieu HAI lan voi cung bo tham so -- mo, hat, phoi sang, ke ca cac buoc moi truong da bo vi co
loe (halo_clear_probability) -- mot lan CO lop loe HALO, mot lan BO lop loe. Model chay tren ca hai ban:
  * "loe lam mat (dB)" = PSNR khong loe - PSNR co loe, cua anh vao va cua anh khoi phuc. Khoi phuc 0 dB = model
    go het thiet hai do loe; bang anh vao = model khong lam gi voi loe.
  * "anh sang loe con lai" = tong (khoi phuc co loe - khoi phuc khong loe) / tong (vao co loe - vao khong loe),
    tren anh sang tuyen tinh: trong vung loe (khoi 8x8 px ma loe them > 0,005) va ca khung. 0% = go het,
    100% = de nguyen, > 100% = lam loe manh them, < 0 = tru qua tay (toi hon ca ban khong loe).
  * "nhe" / "manh": nua it anh sang loe hon / nhieu hon (theo tong anh sang loe them vao anh vao).
Checkpoint KHONG train voi HALO (vd p32) cung do duoc: lop loe lay theo cac khoa halo_* cua
configs/kaggle_halo.yaml, can --halo-root (thu muc co halo_index.csv). So p32 voi p33 cho biet train voi HALO
giup go loe bao nhieu. PSNR tinh tung anh tren sRGB [0, 1] roi lay trung binh.

Bang thu hai: PSNR qua tung tang cua decoder, tren ca hai ban -- ban khong loe chi con mo nhe va hat, tuc la anh
TOT, model nen de gan nguyen. "decoder de nguyen" = dau ra cua decoder luc chua hoc gi (update 0): p33 mat mau mong
o luoi 1/color_scale, p34 (split_chroma_detail) bang dung anh vao. "sau ban do stop" = anh vao chia 2^S (dung lai tu
image_stops, nhu RelightStops), "sau luoi" = anh J sau ca ban do stop va luoi song phuong. Moi tang co PSNR ca anh,
tan thap (trung binh khoi 8x8 px, muc tang tone duoc giam sat) va tan cao (phan con lai sau khi tru trung binh khoi,
dinh 1), va Y (rieng kenh do sang: Y cao ma RGB thap -> loi o mau).

    python3 tools/halo_probe.py --checkpoint outputs/p33_halo/phase2/best_joint_validation.pt \\
        --manifest manifests/kaggle [--split valid] [--samples 256] [--amp] [--halo-root <thu muc HALO>] \\
        [--output halo_probe.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from qjepa.cli import _dataset, _manifest, _system_from_phase2  # noqa: E402
from qjepa.config import load_config  # noqa: E402
from qjepa.corruptions.halo import HaloFlareBank  # noqa: E402
from qjepa.corruptions.image import LowLightImageCorruptor  # noqa: E402
from qjepa.data import collate_paired  # noqa: E402
from qjepa.execution import RestorationForward  # noqa: E402
from qjepa.models.color_edge import compose, luminance, split_targets  # noqa: E402
from qjepa.models.decoders import RELIGHT_STOP_RANGE, linear_to_srgb, srgb_to_linear  # noqa: E402

REGION = 0.005   # anh sang tuyen tinh ma loe them (trung binh khoi 8x8 px) de tinh la "vung loe"
BLOCK = 8


class WithoutFlare(LowLightImageCorruptor):
    """Cung tham so voi ban co loe (ca buoc moi truong da bo vi loe), chi bo lop loe."""

    def _parameters(self, *args, **kwargs):
        params = super()._parameters(*args, **kwargs)
        if params.get("halo"):
            params = dict(params, halo=False, halo_params=None, halo_dropped=True)
        return params


def linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x.clamp_min(0) + 0.055) / 1.055) ** 2.4)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(10 * torch.log10(1.0 / (a - b).square().mean().clamp_min(1e-10)))


def corruptors(dataset, config: dict, halo_root: str | None):
    """(co loe tren MOI frame, ban sinh doi khong loe), cung bank lop loe va cung seed voi dataset."""
    base = dataset.image_corruptor
    image = config["corruption"]["image"]
    source = image if float(image.get("halo_probability", 0)) > 0 else \
        load_config(REPO / "configs/kaggle_halo.yaml")["corruption"]["image"]
    halo = {key: value for key, value in source.items() if key.startswith("halo_")}
    settings = replace(base.config, **{**halo, "halo_probability": 1.0})
    bank = base.halo_bank
    if bank is None:
        root = halo_root or config["data"].get("halo_root")
        if not root:
            raise SystemExit("Checkpoint nay khong train voi HALO: them --halo-root <thu muc co halo_index.csv>")
        bank = HaloFlareBank(root, revision=settings.halo_revision, effects=settings.halo_effects,
                             holdout_fraction=settings.halo_holdout_fraction)
    return (LowLightImageCorruptor(settings, base.master_seed, halo_bank=bank),
            WithoutFlare(settings, base.master_seed, halo_bank=bank))


STAGES = (("input", "vao"), ("identity", "decoder de nguyen"), ("relit", "sau ban do stop"),
          ("tone", "sau luoi (J)"), ("restored", "khoi phuc"))


def bands(x: torch.Tensor, clean: torch.Tensor) -> dict[str, float]:
    """PSNR on the whole frame, on 8x8 block means (what the tone stage is trained on) and on the rest, and of Y."""
    low_x, low_c = F.avg_pool2d(x[None], BLOCK), F.avg_pool2d(clean[None], BLOCK)
    high_x = x[None] - F.interpolate(low_x, scale_factor=BLOCK, mode="nearest")
    high_c = clean[None] - F.interpolate(low_c, scale_factor=BLOCK, mode="nearest")
    return {"full": psnr(x, clean), "low": psnr(low_x, low_c), "high": psnr(high_x, high_c),
            "y": psnr(luminance(x[None]), luminance(clean[None]))}


def relit_from_stops(image: torch.Tensor, stops: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] divided by 2^S in linear light, S [B,1,h,w] upsampled: what decoders.RelightStops returns."""
    gain = torch.exp2(-F.interpolate(stops.clamp(*RELIGHT_STOP_RANGE), size=image.shape[-2:], mode="bilinear",
                                     align_corners=False))
    return linear_to_srgb(srgb_to_linear(image) * gain).clamp(0.0, 1.0)


def stages(noisy: torch.Tensor, tone: torch.Tensor | None, stops: torch.Tensor | None, restored: torch.Tensor,
           clean: torch.Tensor, config: dict) -> dict[str, dict[str, float] | None]:
    """Each decoder stage scored against the clean frame (bands); a stage the checkpoint lacks is None."""
    phase2 = config["phase2"]
    identity = relit = None
    if phase2.get("image_decoder") == "split_color_edge":
        identity = noisy if phase2.get("split_chroma_detail", False) else compose(
            *split_targets(noisy[None], int(phase2.get("split_color_scale", 2)),
                           int(phase2.get("split_illumination_scale", 8))))[0].clamp(0, 1)
    if stops is not None:
        relit = relit_from_stops(noisy[None], stops[None])[0]
    images = {"input": noisy, "identity": identity, "relit": relit, "tone": tone, "restored": restored}
    return {name: None if image is None else bands(image, clean) for name, image in images.items()}


def summarize(rows: list[dict]) -> dict:
    def ratio(numerator: str, denominator: str) -> float:
        total = sum(row[denominator] for row in rows)
        return sum(row[numerator] for row in rows) / total if total > 0 else float("nan")

    mean = {key: float(np.mean([row[key] for row in rows])) for key in ("in_with", "in_without", "out_with",
                                                                         "out_without")}
    lost_in, lost_out = mean["in_without"] - mean["in_with"], mean["out_without"] - mean["out_with"]
    return {"count": len(rows), "psnr": mean, "flare_cost_db": {"input": lost_in, "restored": lost_out},
            "flare_cost_kept": lost_out / lost_in if lost_in > 0 else float("nan"),
            "flare_light_left_region": ratio("flare_out_region", "flare_in_region"),
            "flare_light_left_frame": ratio("flare_out", "flare_in"),
            "region_fraction": float(np.mean([row["region_fraction"] for row in rows]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="checkpoint phase 2")
    parser.add_argument("--manifest")
    parser.add_argument("--split", default="valid", choices=("valid", "test"))
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="fp16 autocast tren CUDA, nhu phase 2 da train")
    parser.add_argument("--halo-root", help="thu muc co halo_index.csv (mac dinh: data.halo_root cua checkpoint)")
    parser.add_argument("--output", help="JSON ket qua")
    args = parser.parse_args()
    device = torch.device(args.device)
    system, config = _system_from_phase2(args.checkpoint, device)
    if args.halo_root:
        config["data"]["halo_root"] = str(args.halo_root)
    forward = RestorationForward(system)
    forward.amp = bool(args.amp) and device.type == "cuda"
    manifest, _ = _manifest(config, args.manifest)
    dataset = _dataset(config, manifest, args.split, fixed_realization=True)
    with_flare, without_flare = corruptors(dataset, config, args.halo_root)
    trained_with_halo = float(config["corruption"]["image"].get("halo_probability", 0)) > 0
    indices = np.random.default_rng(0).choice(len(dataset), min(args.samples, len(dataset)), replace=False)
    rows: list[dict] = []
    with torch.no_grad():
        for start in range(0, len(indices), args.batch):
            chosen = [int(i) for i in indices[start:start + args.batch]]
            renders = []
            for corruptor in (with_flare, without_flare):
                dataset.image_corruptor = corruptor
                samples = [dataset[i] for i in chosen]
                collated = collate_paired(samples)
                batch = {key: collated[key].to(device) for key in ("image_noisy", "imu_noisy_phys", "image_time",
                                                                    "imu_times")}
                outputs = forward(batch["image_noisy"], batch["imu_noisy_phys"], batch["image_time"],
                                  batch["imu_times"])
                tone = outputs["image_light"].float().clamp(0, 1).cpu() if "image_light" in outputs else None
                stops = outputs["image_stops"].float().cpu() if "image_stops" in outputs else None
                renders.append((samples, collated, outputs["image"].float().clamp(0, 1).cpu(), (tone, stops)))
            (samples, collated, out_with, tone_with), (_, twin, out_without, tone_without) = renders
            clean, in_with, in_without = (x["image_" + key].float() for x, key in
                                          ((collated, "clean"), (collated, "noisy"), (twin, "noisy")))
            for k, sample in enumerate(samples):
                halo = sample["corruption"]["image"]["halo_params"]
                added = (linear(in_with[k]) - linear(in_without[k])).mean(0)          # [H, W] anh sang loe them
                left = (linear(out_with[k]) - linear(out_without[k])).mean(0)
                region = F.avg_pool2d(added[None, None], BLOCK)[0, 0] > REGION
                mask = region.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
                stage = {version: stages(noisy[k], None if tone[0] is None else tone[0][k],
                                         None if tone[1] is None else tone[1][k], out[k], clean[k], config)
                         for version, noisy, tone, out in (("with", in_with, tone_with, out_with),
                                                           ("without", in_without, tone_without, out_without))}
                rows.append({"uid": halo["uid"], "gain": halo["gain"], "stage": stage,
                             "in_with": psnr(in_with[k], clean[k]), "in_without": psnr(in_without[k], clean[k]),
                             "out_with": psnr(out_with[k], clean[k]), "out_without": psnr(out_without[k], clean[k]),
                             "flare_in_region": float(added[mask].sum()), "flare_out_region": float(left[mask].sum()),
                             "flare_in": float(added.sum()), "flare_out": float(left.sum()),
                             "region_fraction": float(mask.float().mean())})
    median = float(np.median([row["flare_in"] for row in rows]))
    groups = {"tat ca": rows, "nhe": [r for r in rows if r["flare_in"] <= median],
              "manh": [r for r in rows if r["flare_in"] > median]}
    table = {name: summarize(group) for name, group in groups.items() if group}
    print(f"{len(rows)} anh {args.split}, moi anh co loe HALO (scene HALO cua split {args.split}); ban sinh doi chi bo lop "
          f"loe. Checkpoint {'CO' if trained_with_halo else 'KHONG'} train voi HALO.")
    print(f"\n{'nhom':<8}{'so anh':>7}{'PSNR vao':>18}{'PSNR khoi phuc':>22}{'loe lam mat (dB)':>22}"
          f"{'anh sang loe con lai':>26}")
    print(f"{'':<8}{'':>7}{'co loe / khong':>18}{'co loe / khong':>22}{'vao -> khoi phuc':>22}"
          f"{'vung loe / ca khung':>26}")
    for name, row in table.items():
        psnrs = row["psnr"]
        print(f"{name:<8}{row['count']:>7}{psnrs['in_with']:>10.2f} / {psnrs['in_without']:<5.2f}"
              f"{psnrs['out_with']:>14.2f} / {psnrs['out_without']:<5.2f}"
              f"{row['flare_cost_db']['input']:>12.2f} -> {row['flare_cost_db']['restored']:<6.2f}"
              f"{100 * row['flare_light_left_region']:>16.0f}% / {100 * row['flare_light_left_frame']:.0f}%")
    stage_table = {version: {name: {band: float(np.mean([row["stage"][version][name][band] for row in rows]))
                                    for band in ("full", "low", "high", "y")}
                             for name, _ in STAGES if rows[0]["stage"][version][name] is not None}
                   for version in ("without", "with")}
    print(f"\nPSNR qua tung tang (dB, cao hon la tot). Ban khong loe = anh TOT (chi mo nhe + hat): model nen de gan nguyen.")
    print(f"{'ban':<11}{'tang':<20}{'ca anh':>9}{'tan thap 8x8':>14}{'tan cao':>10}{'Y':>9}")
    for version, label in (("without", "khong loe"), ("with", "co loe")):
        for position, (name, title) in enumerate(STAGES):
            row = stage_table[version].get(name)
            if row is None:
                continue
            print(f"{label if position == 0 else '':<11}{title:<20}{row['full']:>9.2f}{row['low']:>14.2f}"
                  f"{row['high']:>10.2f}{row['y']:>9.2f}")
    print("Doc: 'decoder de nguyen' < 'vao' -> kien truc mat truoc khi hoc gi (p33: mau o luoi 1/color_scale).\n"
          "     Tang nao tut nhieu nhat so voi tang truoc no la tang lam hong anh tot. Tut o 'tan thap' = doi do\n"
          "     sang / mau theo vung; tut o 'tan cao' = them hoac xoa chi tiet (vet, quang, soc). Y cao ma RGB thap -> loi o mau.")
    print("\nDoc: 'anh sang loe con lai' la so do chinh: gan 0% -> model go duoc loe; gan 100% -> de nguyen loe, can mot\n"
          "     buoc go loe rieng. 'loe lam mat' cua khoi phuc nho ma anh sang loe con lai cao -> loi khac (do sang, chi\n"
          "     tiet) lan at loe trong PSNR, loe van con. 'manh' con lai nhieu hon 'nhe' -> model chi go duoc loe nhat.")
    if args.output:
        Path(args.output).write_text(json.dumps({"split": args.split, "trained_with_halo": trained_with_halo,
                                                 "groups": table, "stages": stage_table}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
