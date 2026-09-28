"""Time the phase-2 decoders piece by piece, in fp32 and fp16 (and with cuDNN benchmark, on request).

p16 ran phase 2 with fp16 autocast and got 5x SLOWER than p14 in fp32 (7.5 s vs
1.46 s per update on Kaggle T4 x2). This finds out which block is slow in which
setting, on one GPU, with random weights and a synthetic batch -- no dataset, no
checkpoint, no download (VGG weights are random; timing does not depend on them).

    python3 tools/decoder_speed_probe.py                 # on Kaggle: about 2 minutes
    python3 tools/decoder_speed_probe.py --steps 20
    python3 tools/decoder_speed_probe.py --benchmark     # also cuDNN benchmark; crashed p16 on T4

Prints milliseconds per forward + backward for each block and for one whole
training step, and the speed-up of every setting over fp32 without benchmark.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qjepa.config import build_decoders, build_phase1_model, load_config  # noqa: E402
from qjepa.data import ImuNormalizer  # noqa: E402
from qjepa.execution import RestorationForward  # noqa: E402
from qjepa.models import RestorationSystem  # noqa: E402
from qjepa.models.color_edge import downsample  # noqa: E402
from qjepa.training.perceptual import PerceptualLoss, load_vgg16_features  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def _timed(fn, device: torch.device, steps: int, warmup: int) -> float:
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(steps):
        start = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - start)
    return float(np.median(times)) * 1000.0


def probe(config_path: Path, device: torch.device, batch: int, steps: int, warmup: int,
          benchmark: bool = False) -> list[dict]:
    config = load_config(config_path)
    torch.manual_seed(0)
    phase1 = build_phase1_model(config, ImuNormalizer())
    system = RestorationSystem(phase1.backbone, phase1.normalizer, build_decoders(config)).to(device)
    system.freeze_backbone()
    decoder = system.decoders.image
    height, width = config["data"]["image_size"]
    length = config["data"]["imu_window"]
    image = torch.rand(batch, 3, height, width, device=device)
    imu = torch.randn(batch, 6, length, device=device)
    imu_times = (torch.arange(length, device=device, dtype=torch.float64) * 0.01).repeat(batch, 1)
    image_time = imu_times.mean(dim=1)
    crop = int(config["phase2"].get("perceptual_crop", 0))
    vgg = PerceptualLoss(load_vgg16_features(pretrained=False), crop=crop).to(device)
    forward = RestorationForward(system)
    optimizer = torch.optim.AdamW(system.decoders.parameters(), lr=1e-4)
    with torch.no_grad():
        latent = system.encode(image, imu, image_time, imu_times)
    zi = latent.ZI.detach()
    small = downsample(image, int(config["phase2"].get("split_color_scale", 2)))
    edge_in = torch.rand(batch, 2, height, width, device=device)
    detail = torch.rand(batch, 1, height, width, device=device)

    precisions = ["fp32", "fp16"] if device.type == "cuda" else ["fp32"]
    settings = [(precision, on) for precision in precisions for on in ((False, True) if benchmark else (False,))]
    rows = []
    for precision, benchmark in settings:
        amp = precision == "fp16"
        torch.backends.cudnn.benchmark = benchmark
        forward.amp = amp
        scaler = torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch.amp, "GradScaler") \
            else torch.cuda.amp.GradScaler(enabled=amp)

        def autocast():
            return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp)

        def backbone():
            with torch.no_grad(), autocast():
                system.encode(image, imu, image_time, imu_times)

        def colour():
            with autocast():
                out = decoder.color(zi, small)
            out.float().square().mean().backward()

        def nafnet():
            with autocast():
                out, _ = decoder.edge.forward_with_aux(zi, edge_in, base=detail)
            out.float().square().mean().backward()

        def refiner():
            if decoder.refiner is None:
                return
            with autocast():
                out = decoder.refiner(detail, edge_in)
            out.float().square().mean().backward()

        def perceptual():
            predicted = image.clone().requires_grad_(True)
            with autocast():
                loss = vgg(predicted, image)
            loss.backward()

        def whole_step():
            optimizer.zero_grad(set_to_none=True)
            out = forward(image, imu, image_time, imu_times)
            loss = F.l1_loss(out["image"], image) + F.l1_loss(out["imu_physical"], imu)
            with autocast():
                loss = loss + 0.5 * vgg(out["image"], image)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(system.decoders.parameters()), 5.0)
            scaler.step(optimizer)
            scaler.update()

        row = {"setting": f"{precision}{' + benchmark' if benchmark else ''}"}
        for name, fn in (("backbone (no grad)", backbone), ("colour branch", colour), ("NAFNet edge", nafnet),
                         ("edge refiner", refiner), (f"VGG {'crop ' + str(crop) if crop else 'full'}", perceptual),
                         ("whole step", whole_step)):
            row[name] = _timed(fn, device, steps, warmup)
            system.decoders.zero_grad(set_to_none=True)
        rows.append(row)
        print(f"  measured {row['setting']}", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO / "configs/kaggle_tartanair_v2.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=4, help="per GPU; the recipe's 8 split over 2 GPUs")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--benchmark", action="store_true",
                        help="also time with cuDNN benchmark (it crashed p16's phase 2 on Kaggle T4)")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        print(f"GPU {torch.cuda.get_device_name(device)} · torch {torch.__version__} · "
              f"cuDNN {torch.backends.cudnn.version()}")
    rows = probe(args.config, device, args.batch, args.steps, args.warmup, args.benchmark)
    names = [key for key in rows[0] if key != "setting"]
    print(f"\nms per forward + backward, batch {args.batch}, median of {args.steps}")
    print(f"{'block':<22}" + "".join(f"{row['setting']:>20}" for row in rows))
    for name in names:
        print(f"{name:<22}" + "".join(f"{row[name]:>20.1f}" for row in rows))
    base = rows[0]["whole step"]
    print(f"\n{'whole step vs fp32':<22}" + "".join(f"{base / row['whole step']:>19.2f}x" for row in rows))
    best = min(rows, key=lambda row: row["whole step"])
    print(f"\nFastest: {best['setting']} ({best['whole step']:.0f} ms per step of {args.batch} images).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
