"""Xem truoc nhieu ANH SANG cua train (mac dinh configs/kaggle_local.yaml, run p25_local; so voi p24_env).

Do sang thay doi ngau nhien theo vung, vet nhoe toi, den sang gap nhieu lan va loe ra (quang, tia
sao, bong ma), moi gia tri ngau nhien. Cac buoc lay thang tu code train (qjepa/corruptions/light.py
va LowLightImageCorruptor), nen hinh la dung thu model se thay:

  Sach | Canh: sang/toi khong deu, den | + loe sang | Nhieu train cu (--baseline) | Nhieu train moi (--config)

"Nhieu train moi" = anh sang khong deu + loe roi chuoi nhieu cu cua repo (mo, thieu sang, hat,
JPEG). Trong train light_probability / illum_probability = 0.8; o day ep 1.0 de hang nao cung thay.

    python3 tools/light_corruption_preview.py                      # 4 anh co den, mo cua so hinh
                                                                   # (hoac bam Run trong VS Code)
    python3 tools/light_corruption_preview.py --count 6 --pick random --seed 7   # seed co dinh: lap lai dung hinh
    python3 tools/light_corruption_preview.py --no-show            # chi luu PNG

Chay duoc bang python3 cua he thong (khong can torch) lan ~/venv-torch/bin/python. Chi doc
dataset; hinh luu o outputs/light_corruption/ (gitignore). Khong mo duoc cua so (khong co
backend giao dien) thi mo file PNG bang trinh xem anh cua he thong.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image


def _find_repo() -> Path:
    """Thu muc chua qjepa/ -- ca khi chay trong cua so Interactive cua VS Code (co the khong co __file__)."""
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for folder in (start, *start.parents):
        if (folder / "qjepa").is_dir():
            return folder
    raise SystemExit("Khong tim thay repo (thu muc chua qjepa/)")


REPO = _find_repo()
sys.path.insert(0, str(REPO))

# Chi qjepa.corruptions (numpy, scipy, PIL): python3 cua he thong khong co torch van chay duoc.
from qjepa.corruptions.image import LowLightImageCorruptionConfig, LowLightImageCorruptor  # noqa: E402
from qjepa.corruptions.light import apply_light, linear_to_srgb, srgb_to_linear  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722])


def load_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    """Nhu qjepa.data.dataset.load_rgb (khong keo torch): phong cho phu khung roi cat giua, LANCZOS."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        height, width = size
        if image.size != (width, height):
            scale = max(width / image.width, height / image.height)
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
            left, top = (image.width - width) // 2, (image.height - height) // 2
            image = image.crop((left, top, left + width, top + height))
        return np.asarray(image, dtype=np.float32) / 255.0


def tonemap(srgb_hdr: np.ndarray) -> np.ndarray:
    """De XEM anh sRGB mo rong (> 1): Reinhard tren anh sang tuyen tinh, vung > 1 khong bi cat phang."""
    linear = srgb_to_linear(srgb_hdr)
    return np.clip(linear_to_srgb(linear / (1.0 + linear)), 0.0, 1.0)


def frames(root: Path, split: str) -> list[Path]:
    """Duong dan anh cua manifest.csv; tuong doi so voi thu muc split (symlink) hoac tartanair-v2."""
    rows = list(csv.DictReader((root / split / "manifest.csv").open(newline="", encoding="utf-8")))
    paths = []
    for row in rows:
        for base in (root / split, root.parent / "tartanair-v2"):
            path = base / row["image_path"]
            if path.is_file():
                paths.append(path)
                break
    if not paths:
        raise FileNotFoundError(f"Khong doc duoc anh nao tu {root / split / 'manifest.csv'}")
    return paths


def load_many(paths: list[Path], size: int, progress=None) -> list[np.ndarray]:
    """Doc song song (giai nen PNG nha GIL); giu dung thu tu ``paths``."""
    images: list[np.ndarray] = [None] * len(paths)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(load_rgb, path, (size, size)): index for index, path in enumerate(paths)}
        for done, future in enumerate(as_completed(futures), 1):
            images[futures[future]] = future.result()
            if progress is not None and (done % 25 == 0 or done == len(paths)):
                progress(done, len(paths))
    return images


def pick(paths: list[Path], count: int, pool: int, mode: str, rng: np.random.Generator,
         size: int, progress=None) -> list[tuple[Path, np.ndarray]]:
    chosen = [paths[i] for i in rng.choice(len(paths), size=min(pool, len(paths)), replace=False)]
    if mode == "random":
        return list(zip(chosen[:count], load_many(chosen[:count], size)))
    # "Co den": nen toi voi vai dom rat sang nho -- khong phai anh troi sang (dom sang rong).
    print(f"Dang chon anh co den trong {len(chosen)} anh ngau nhien...", flush=True)
    scored = []
    for path, image in zip(chosen, load_many(chosen, size, progress)):
        lum = image @ LUMA.astype(np.float32)
        bright = float((lum > 0.92).mean())
        if 0.0005 < bright < 0.08 and float(lum.mean()) < 0.45:
            scored.append((bright / (float(lum.mean()) + 0.05), path, image))
    scored.sort(key=lambda item: -item[0])
    by_environment: dict[str, list] = {}
    for _, path, image in scored:                         # trai deu qua cac moi truong
        by_environment.setdefault(path.parts[-5], []).append((path, image))
    picked = []
    while len(picked) < count and any(by_environment.values()):
        for group in by_environment.values():
            if group and len(picked) < count:
                picked.append(group.pop(0))
    if len(picked) < count:
        print(f"Chi tim duoc {len(picked)} anh co den trong {len(chosen)} anh; tang --pool hoac dung --pick random")
    return picked


def _config_chain(path: Path) -> dict:
    """configs/*.yaml voi ``extends`` da tron sau (nhu qjepa.config.load_config, khong keo torch)."""
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if not parent:
        return raw
    merged = _config_chain(path.parent / parent)

    def merge(base: dict, update: dict) -> dict:
        for key, value in update.items():
            base[key] = merge(base.get(key, {}), value) if isinstance(value, dict) and isinstance(base.get(key), dict) \
                else value
        return base
    return merge(merged, raw)


def corruptor_from(config_name: str, light_probability: float | None = None) -> LowLightImageCorruptor:
    """LowLightImageCorruptor dung nhu luc train voi configs/<config_name>."""
    config = _config_chain(REPO / "configs" / config_name)
    image = LowLightImageCorruptionConfig(**config["corruption"]["image"])
    if light_probability is not None:
        image = replace(image, light_probability=light_probability if image.light_probability > 0 else 0.0,
                        illum_probability=light_probability if image.illum_probability > 0 else 0.0,
                        fog_probability=light_probability if image.fog_probability > 0 else 0.0,
                        env_clear_probability=0.0)
    return LowLightImageCorruptor(image, config["data"]["corruption_seed"])


def describe(params: dict[str, object]) -> str:
    light = params.get("light_params") or {}
    uneven = params.get("illumination_params") or {}
    fog = params.get("fog_params") or {}
    lines = []
    if fog:
        lines.append(f"sương: mật độ {fog['density']:.1f} (xa còn {100 * math.exp(-fog['density']):.0f}%)")
    if uneven:
        stops = [blob["stops"] for blob in uneven["blobs"]]
        lines.append(f"{len(stops)} vùng sáng/tối ({min(stops, default=0):+.1f}…{max(stops, default=0):+.1f} stop)"
                     + (f" · {len(uneven['smudges'])} vết nhòe tối" if uneven["smudges"] else ""))
    if light:
        extras = [f"sao {light['star_spikes']} tia"] if light["star"] else []
        extras += [f"{len(light['ghosts'])} bóng ma"] if light["ghosts"] else []
        lines.append(f"đèn ×{light['gain']:.0f} · quầng {light['bloom_strength']:.2f}"
                     + (f" · {', '.join(extras)}" if extras else ""))
    lines.append(f"phơi sáng {params['exposure_gain']:.2f}" + (f" · mờ động {params['motion_length']}px"
                                                                if params.get("motion") else ""))
    return "\n".join(lines)


def bring_to_front(figure) -> None:
    """Cua so hinh phong to va noi len tren VS Code (Qt hoac Tk); loi thi bo qua."""
    window = getattr(figure.canvas.manager, "window", None)
    try:
        figure.canvas.manager.set_window_title("Nhiễu ánh sáng -- đóng cửa sổ để kết thúc")
        if hasattr(window, "showMaximized"):                       # Qt
            window.showMaximized()
            window.raise_()
            window.activateWindow()
        elif hasattr(window, "attributes"):                         # Tk
            window.attributes("-zoomed", True)
            window.lift()
            window.attributes("-topmost", True)
            window.after(800, lambda: window.attributes("-topmost", False))
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path.home() / "Datasets/tartanair-v2-jepa")
    parser.add_argument("--split", default="valid", choices=("train", "valid", "test"))
    parser.add_argument("--count", type=int, default=4, help="so anh ve ra")
    parser.add_argument("--pool", type=int, default=600, help="so anh ngau nhien de chon anh co den")
    parser.add_argument("--pick", default="bright", choices=("bright", "random"))
    parser.add_argument("--size", type=int, default=256, help="canh anh (256 nhu luc train)")
    parser.add_argument("--seed", type=int, default=None,
                        help="bo ngau nhien; mac dinh moi lan chay mot seed moi (in ra de chay lai dung hinh do)")
    parser.add_argument("--config", default="kaggle_local.yaml", help="config nhieu dang train (trong configs/)")
    parser.add_argument("--baseline", default="kaggle_env.yaml", help="config nhieu cu de so sanh")
    parser.add_argument("--output", type=Path, default=REPO / "outputs/light_corruption/preview.png")
    parser.add_argument("--no-show", action="store_true", help="chi luu PNG, khong mo cua so")
    # Trong Jupyter / cua so Interactive, sys.argv la cua kernel: dung mac dinh.
    args = parser.parse_args([] if "ipykernel" in sys.modules else None)
    if args.seed is None:                                           # moi lan bam Run: anh khac, nhieu khac
        args.seed = int(np.random.default_rng().integers(0, 100_000))
    print(f"seed {args.seed} (chay lai dung hinh nay: --seed {args.seed})", flush=True)

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    backend = matplotlib.get_backend().lower()
    has_window = any(name in backend for name in ("qt", "tk", "gtk", "wx", "macosx"))
    inline = "inline" in backend                                    # Jupyter: hinh hien ngay trong o ket qua
    live = has_window and not args.no_show

    # Mo cua so NGAY (truoc khi doc anh), roi dien dan tung hang khi tinh xong.
    # Config khong co den / loe (p25): bo cot "+ loe sang", cot canh khong nhac den.
    glares = float(_config_chain(REPO / "configs" / args.config)["corruption"]["image"].get("light_probability", 0)) > 0
    columns = [("Sạch", None), ("Cảnh: sáng/tối không đều,\nnhòe tối" + (", đèn" if glares else ""), "scene")]
    if glares:
        columns.append(("+ lóe sáng\n(quầng, sao, bóng ma)", "lit"))
    columns += [(f"Nhiễu train cũ\n({args.baseline})", "old"), (f"Nhiễu train mới\n({args.config})", "new")]
    cell = min(16.0 / len(columns), 9.0 / args.count)
    png_size = (cell * len(columns), cell * args.count + 1.2)
    # layout "constrained": tinh lai moi lan ve, nen cua so phong to / keo gian van khong de chu.
    figure, axes = plt.subplots(args.count, len(columns), figsize=png_size, squeeze=False, layout="constrained")
    for axis in axes.flat:
        axis.set_xticks([]), axis.set_yticks([])
    for col, (title, _) in enumerate(columns):
        axes[0, col].set_title(title, fontsize=9)
    status = figure.suptitle("", fontsize=11)

    def show(text: str) -> None:
        status.set_text(text)
        if live:                                                    # ve lai cua so ma khong keo no len lai
            figure.canvas.draw_idle()
            figure.canvas.flush_events()

    if live:
        plt.ion()
        plt.show(block=False)
        bring_to_front(figure)
    show("Đang đọc danh sách ảnh…")

    rng = np.random.default_rng(args.seed)
    samples = pick(frames(args.root, args.split), args.count, args.pool, args.pick, rng, args.size,
                   progress=lambda done, total: show(f"Đang chọn ảnh có đèn: đã đọc {done}/{total} ảnh…"))
    if not samples:
        print("Khong co anh nao de ve.")
        show("Không tìm được ảnh nào — thử --pick random hoặc tăng --pool")
        if live:
            plt.ioff()
            plt.show()
        return 1
    glare = corruptor_from(args.config, light_probability=1.0)
    baseline = corruptor_from(args.baseline)
    for row, (path, clean) in enumerate(samples):
        show(f"Đang làm nhiễu ảnh {row + 1}/{len(samples)}…")
        corrupt = dict(split=args.split, realization=0, trajectory=f"preview{args.seed}-{row}",
                       timestamp=0.1 * row, frame_index=row, mode="full")
        stages = {"old": baseline(clean, **corrupt)[0]}
        stages["new"], params = glare(clean, **corrupt)
        if params.get("light_params") or params.get("illumination_params") or params.get("fog_params"):
            stages["scene"], stages["lit"] = apply_light(clean, params.get("light_params"), stages=True,
                                                         illumination=params.get("illumination_params"),
                                                         fog=params.get("fog_params"))
        else:                                                       # frame "sach" (clean_probability)
            stages["scene"] = stages["lit"] = clean
        print(f"[{row}] {path.relative_to(args.root.parent) if args.root.parent in path.parents else path}")
        print("    " + describe(params).replace("\n", " | "))
        for col, (title, key) in enumerate(columns):
            image = clean if key is None else stages[key]
            shown = tonemap(image) if key in ("scene", "lit") else image
            axes[row, col].imshow(np.clip(shown, 0, 1))
        axes[row, 0].set_ylabel(path.parts[-5] if len(path.parts) >= 5 else path.stem, fontsize=8)
        axes[row, -1].set_xlabel(describe(params), fontsize=7)
    for row in range(len(samples), args.count):                    # it anh co den hon --count
        for axis in axes[row]:
            axis.set_visible(False)
    status.set_text(f"seed {args.seed} · split {args.split} · chạy lại để có ảnh và nhiễu khác")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    window_size = figure.get_size_inches().copy()                  # PNG co kich thuoc co dinh, khong theo cua so
    figure.set_size_inches(*png_size, forward=False)
    figure.savefig(args.output, dpi=150)
    figure.set_size_inches(*window_size, forward=False)
    print("Da luu:", args.output)
    if args.no_show:
        return 0
    if live or inline:
        if live:
            print("Cua so hinh dang mo; dong cua so de ket thuc.", flush=True)
            plt.ioff()
        plt.show()
    else:
        print(f"Matplotlib khong co cua so (backend {backend}); mo file PNG bang trinh xem anh.")
        subprocess.Popen(["xdg-open", str(args.output)])
    return 0


if __name__ == "__main__":
    code = main()
    if "ipykernel" not in sys.modules:                             # Jupyter: khong in canh bao SystemExit
        raise SystemExit(code)
