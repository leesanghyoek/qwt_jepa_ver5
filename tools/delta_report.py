"""Do truc tiep Delta = C_out - C_in cua decoder residual, tach theo bang.

Metric tong hop chi noi anh ra "tot hon" hay "te hon"; no khong noi decoder dang
LAM GI. Voi C_out = C_in + Delta, cau hoi that su la Delta co dau nao tren bang
chi tiet: cong them duong net, hay tru bot di.

Tru bot la nghiem ma L1 thuong. O mot vi tri co canh, xoa phan tan so cao giam
sai so chac chan, con doan them canh thi rui ro — nen trung vi co dieu kien chon
lam phang. Khi do anh ra MO HON chinh anh vao, du moi metric deu thang baseline.

Chay bang subprocess nen luon nap code moi nhat tu dia.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qjepa.cli import _dataset, _loader, _system_from_phase2, _to_device
from qjepa.data import read_manifest

BANDS = ("LL  (do sang, bo cuc)", "LH  (net ngang)", "HL  (net doc)", "HH  (net cheo)")


def rms(x: torch.Tensor) -> float:
    return float(x.float().square().mean().sqrt())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="valid", choices=("valid", "test"))
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--ablate-latent",
        action="store_true",
        help="Chay them mot luot voi ZI/ZU bang 0 de tach cong cua latent khoi skip",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    manifest = read_manifest(args.manifest)
    system, config = _system_from_phase2(args.checkpoint, device)
    if not config["phase2"].get("input_coefficient_residual", False):
        raise SystemExit("Checkpoint nay khong dung residual; Delta khong dinh nghia duoc.")
    print("phase2:",
          "beta", config["phase2"]["smooth_l1_beta"],
          "| detail", config["phase2"]["reconstruction_detail_weight"],
          "| variation", config["phase2"].get("imu_variation_weight"))

    dataset = _dataset(config, manifest, args.split, fixed_realization=True)
    loader = _loader(config, dataset, config["phase2"]["batch_size"], train=False)

    # Tong binh phuong tich luy, chia ra o cuoi — tranh giu tensor cua moi batch.
    image_in = np.zeros(4)
    image_clean_rms = np.zeros(4)   # nang luong cua anh SACH: dich that su
    image_delta = np.zeros(4)
    image_gap_in = np.zeros(4)      # |C_in - C_clean| : sai so cua viec khong lam gi
    image_gap_out = np.zeros(4)     # |C_out - C_clean| : sai so sau khi sua
    imu_in = imu_clean_rms = imu_delta = imu_gap_in = imu_gap_out = 0.0
    seen = 0

    for index, raw in enumerate(loader):
        if index >= args.batches:
            break
        batch = _to_device(raw, device)
        with torch.no_grad():
            latent = system.encode(batch["image_noisy"], batch["imu_noisy_phys"],
                                   batch["image_time"], batch["imu_times"])
            restored = system.decode(latent, system.motion_for(batch))
            clean_image, _ = system.backbone.image_transform.analysis(batch["image_clean"])
            clean_imu_signal = system.normalizer.normalize(batch["imu_clean_phys"])
            clean_imu, _ = system.backbone.imu_transform.analysis(clean_imu_signal)

        c_in, c_out = latent.image_coefficients, restored.image_coefficients
        shape = (c_in.shape[0], c_in.shape[1] // 16, 4, 4) + c_in.shape[2:]
        for band in range(4):
            a = c_in.reshape(shape)[:, :, band]
            b = c_out.reshape(shape)[:, :, band]
            t = clean_image.reshape(shape)[:, :, band]
            image_in[band] += rms(a) ** 2
            image_clean_rms[band] += rms(t) ** 2
            image_delta[band] += rms(b - a) ** 2
            image_gap_in[band] += rms(a - t) ** 2
            image_gap_out[band] += rms(b - t) ** 2

        half = latent.imu_coefficients.shape[1] // 2
        u_in = latent.imu_coefficients[:, half:]
        u_out = restored.imu_coefficients[:, half:]
        u_target = clean_imu[:, half:]
        imu_in += rms(u_in) ** 2
        imu_clean_rms += rms(u_target) ** 2
        imu_delta += rms(u_out - u_in) ** 2
        imu_gap_in += rms(u_in - u_target) ** 2
        imu_gap_out += rms(u_out - u_target) ** 2
        seen += 1

    if not seen:
        raise SystemExit("Loader khong tra ve batch nao.")
    root = lambda total: np.sqrt(np.asarray(total) / seen)
    image_in, image_delta = root(image_in), root(image_delta)
    image_clean_rms = root(image_clean_rms)
    image_gap_in, image_gap_out = root(image_gap_in), root(image_gap_out)

    print(f"\nmau: {seen} batch x {config['phase2']['batch_size']}\n")
    print("ANH — he so QWT theo bang")
    print(f"  {'bang':<24}{'RMS(C_in)':>11}{'RMS(sach)':>11}{'RMS(delta)':>12}"
          f"{'delta/in':>10}{'|C_in-sach|':>13}{'|C_out-sach|':>14}{'':>6}")
    for band in range(4):
        share = image_delta[band] / max(image_in[band], 1e-12)
        better = image_gap_out[band] < image_gap_in[band]
        print(f"  {BANDS[band]:<24}{image_in[band]:>11.5f}{image_clean_rms[band]:>11.5f}"
              f"{image_delta[band]:>12.5f}{100 * share:>9.1f}%"
              f"{image_gap_in[band]:>13.5f}{image_gap_out[band]:>14.5f}"
              f"{'   tot hon' if better else '   TE HON'}")
    # RMS(sach) > RMS(C_in) o bang chi tiet nghia la mo da XOA nang luong duong
    # net, tuc huong dung la TANG. Neu model dang co nho, no di nguoc chieu.
    thieu = image_clean_rms[1:] > image_in[1:]
    if thieu.any():
        print(f"\n  Anh mo THIEU nang luong duong net o {int(thieu.sum())}/3 bang"
              f" -> huong dung la TANG, khong phai co nho.")

    detail_out_gap = np.sqrt(np.square(image_gap_out[1:]).mean())
    detail_in_gap = np.sqrt(np.square(image_gap_in[1:]).mean())
    print(f"\n  Bang chi tiet gop lai: |C_in-sach| {detail_in_gap:.5f}"
          f" -> |C_out-sach| {detail_out_gap:.5f}"
          f"  {'tot hon' if detail_out_gap < detail_in_gap else 'TE HON'}")

    imu_in, imu_delta = np.sqrt(imu_in / seen), np.sqrt(imu_delta / seen)
    imu_clean_rms = np.sqrt(imu_clean_rms / seen)
    imu_gap_in, imu_gap_out = np.sqrt(imu_gap_in / seen), np.sqrt(imu_gap_out / seen)
    print("\nIMU — nua bang detail cua Haar")
    print(f"  RMS(C_in) {imu_in:.5f} | RMS(sach) {imu_clean_rms:.5f}"
          f" | RMS(delta) {imu_delta:.5f}"
          f" | delta/in {100 * imu_delta / max(imu_in, 1e-12):.1f}%")
    print(f"  |C_in-sach| {imu_gap_in:.5f} -> |C_out-sach| {imu_gap_out:.5f}"
          f"  {'tot hon' if imu_gap_out < imu_gap_in else 'TE HON'}")
    # Nguoc chieu voi anh: nhieu THEM nang luong tan so cao vao IMU, nen huong dung
    # la GIAM. Cung mot so hang khop nang luong lo ca hai chieu, vi no la phep khop
    # chu khong phai phep toi da hoa.
    huong = "GIAM" if imu_clean_rms < imu_in else "TANG"
    print(f"  -> IMU can {huong} nang luong tan so cao"
          f" ({'nhieu da them vao' if huong == 'GIAM' else 'tin hieu sach dong hon'})"
          f" — NGUOC chieu voi anh.")

    print("\nDoc ket qua:")
    print("  delta/in gan 0%      -> decoder khong lam gi, dau ra = dau vao.")
    print("  |C_out-sach| TE HON  -> decoder dang lam hong chinh bang do.")
    print("  Bang chi tiet TE HON nhung LL tot hon -> chinh phoi sang, xoa duong net.")

    if args.ablate_latent:
        _ablate(system, loader, device, args.batches)


@torch.no_grad()
def _ablate(system, loader, device, batches: int) -> None:
    """Chay lai voi ZI/ZU = 0: phan con lai chi con den tu skip va he so input.

    Voi encoder_skips bat, mot decoder co the hoc cach chep skip roi bo mac latent.
    Khi do anh van net — nhung net vi no copy dau vao, khong phai vi khoi phuc, va
    luan diem latent-first mat sach. Day la phep do duy nhat tach duoc hai truong hop.
    """
    # The pixel ResNet always has a path around the latent: the blurry image.
    if not system.decoders.uses_skips and system.decoders.image_decoder == "qwt_coefficients":
        print("\n(--ablate-latent: decoder khong dung skip, phep do nay khong co y nghia)")
        return
    full = zeroed = count = 0.0
    for index, raw in enumerate(loader):
        if index >= batches:
            break
        batch = _to_device(raw, device)
        latent = system.encode(batch["image_noisy"], batch["imu_noisy_phys"],
                               batch["image_time"], batch["imu_times"])
        clean = batch["image_clean"]
        motion = system.motion_for(batch)          # p47: the gyro path stays; only the latent goes
        full += float((system.decode(latent, motion).image - clean).abs().mean())
        latent.ZI = torch.zeros_like(latent.ZI)
        latent.ZU = torch.zeros_like(latent.ZU)
        zeroed += float((system.decode(latent, motion).image - clean).abs().mean())
        count += 1

    full, zeroed = full / count, zeroed / count
    share = 100.0 * (zeroed - full) / max(zeroed, 1e-12)
    print("\n=== ABLATE LATENT: dat ZI/ZU = 0 ===")
    print(f"  MAE anh binh thuong : {full:.5f}")
    print(f"  MAE khi ZI/ZU = 0   : {zeroed:.5f}")
    print(f"  latent dong gop     : {share:.1f}% sai so")
    if share < 5:
        print("  -> Decoder gan nhu BO MAC latent: net den tu skip/anh mo, khong phai tu JEPA.")
    elif share < 20:
        print("  -> Latent co dong gop nhung skip dang gan het viec.")
    else:
        print("  -> Latent that su dang dieu khien dau ra.")


if __name__ == "__main__":
    main()
