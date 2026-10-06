"""Chinh tung thong so nhieu bang thanh truot va xem ngay anh trong ra sao.

Dung DUNG code nhieu luc train (qjepa.corruptions: LowLightImageCorruptor), nen anh thay duoc la
anh model se gap. Moi thanh truot dat MOT gia tri co dinh cho thong so do (khoang [v, v] trong
config); vi tri cac vung sang/toi, mang suong, huong mo, hat nhieu van ngau nhien -- nut
"Doi ngau nhien" boc lai chung.

  Nut "Anh khac"          anh khac (tu dataset, hoac giua cac --image)
  Nut "Doi ngau nhien"    giu gia tri thanh truot, boc lai vi tri / hinh dang ngau nhien
  Nut "Boc nhu luc train" lay mot mau dung phan phoi train cua --config (co ca xac suat bat/tat
                          tung tang) va dat thanh truot theo no
  Nut "Luu"               luu PNG va doan YAML (corruption.image) vao outputs/noise_tuner/

    python3 tools/noise_tuner.py                                # anh ngau nhien tu dataset
    python3 tools/noise_tuner.py --image anh1.jpg anh2.png      # anh cua ban
    python3 tools/noise_tuner.py --config kaggle_illum.yaml     # khoang "Boc nhu luc train" cua run khac
    python3 tools/noise_tuner.py --no-show                      # chi luu mot hinh mau (khong mo cua so)

Suong mu da bo khoi recipe (nguoi dung, 06/10/2026) nen khong co thanh truot suong.

Chay duoc bang python3 cua he thong (khong can torch) lan venv / Anaconda; bam Run trong VS Code
cung duoc. Chi doc dataset; file luu o outputs/noise_tuner/ (gitignore).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from light_corruption_preview import REPO, _config_chain, bring_to_front, frames, load_rgb  # noqa: E402

from qjepa.corruptions.image import LowLightImageCorruptionConfig, LowLightImageCorruptor  # noqa: E402

# (khoa, nhan, min, max, buoc) -- thu tu = thu tu ve tren cua so
SLIDERS = [
    ("exposure", "Phơi sáng (1 = không tối)", 0.05, 1.0, None),
    ("gamma", "Gamma (1 = không đổi)", 0.4, 1.0, None),
    ("vignette", "Tối góc", 0.0, 0.8, None),
    ("illum", "Vùng sáng/tối (± stop)", 0.0, 3.5, None),
    ("gradient", "Dải sáng dần (stop)", 0.0, 3.0, None),
    ("smudge", "Vết nhòe tối (độ tối)", 0.0, 0.95, None),
    ("lamp", "Đèn sáng gấp (×, 1 = tắt)", 1.0, 40.0, None),
    ("bloom", "Lóe: quầng (0 = tắt)", 0.0, 1.5, None),
    ("knee", "Lóe: ngưỡng", 0.8, 4.0, None),
    ("defocus", "Mờ lệch tiêu cự (px)", 0.0, 2.5, None),
    ("motion", "Mờ chuyển động (px)", 0.0, 10.0, 1),
    ("photons", "Hạt nhiễu: log10 photon", 2.5, 4.5, None),
    ("jpeg", "JPEG chất lượng (100 = tắt)", 30, 100, 1),
]


def _mid(bounds, log=False) -> float:
    low, high = (float(v) for v in bounds)
    return math.sqrt(low * high) if log and low > 0 else (low + high) / 2.0


def initial_values(image: dict) -> dict[str, float]:
    """Giua cac khoang cua config: nhin thay ngay muc nhieu dien hinh luc train."""
    return {
        "exposure": _mid(image["exposure_gain"]), "gamma": _mid(image["tone_gamma"]),
        "vignette": _mid(image["vignette_strength"]),
        "illum": _mid(image.get("illum_strength", (1.0, 3.0))) if image.get("illum_probability", 0) > 0 else 0.0,
        "gradient": _mid(image.get("illum_gradient", (0.0, 2.5))) if image.get("illum_probability", 0) > 0 else 0.0,
        "smudge": _mid(image.get("illum_smudge_depth", (0.5, 0.95))) if image.get("illum_probability", 0) > 0 else 0.0,
        "lamp": _mid(image.get("light_gain", (5.0, 30.0)), log=True) if image.get("light_probability", 0) > 0 else 1.0,
        "bloom": _mid(image.get("light_bloom_strength", (0.08, 0.8)), log=True) if image.get("light_probability", 0) > 0
        else 0.0,
        "knee": _mid(image.get("light_knee", (1.5, 3.0))),
        "defocus": _mid(image["defocus_sigma_px"]), "motion": 0.0,
        "photons": math.log10(_mid(image["photon_count"], log=True)), "jpeg": 100.0,
    }


def corruption_values(base: dict, v: dict[str, float]) -> dict:
    """corruption.image cua config cong voi gia tri thanh truot (moi thong so: khoang [v, v])."""
    pair = lambda x: (float(x), float(x))
    values = dict(base)
    values.update(
        clean_probability=0.0, noise_only_probability=0.0, low_light_only_probability=0.0,
        env_clear_probability=0.0, motion_from_imu=False, downsample_probability=0.0,
        exposure_gain=pair(v["exposure"]), tone_gamma=pair(v["gamma"]), vignette_strength=pair(v["vignette"]),
        white_balance_gain=(1.0, 1.0), black_level=(0.0, 0.0),
        defocus_probability=1.0 if v["defocus"] > 0.05 else 0.0, defocus_sigma_px=pair(max(v["defocus"], 0.05)),
        motion_probability=1.0 if v["motion"] >= 2 else 0.0, motion_length_px=(int(v["motion"]), int(v["motion"])),
        photon_count=pair(10 ** v["photons"]),
        jpeg_probability=1.0 if v["jpeg"] < 100 else 0.0, jpeg_quality=(int(v["jpeg"]), int(v["jpeg"])),
        illum_probability=1.0 if v["illum"] > 0 or v["gradient"] > 0 or v["smudge"] > 0 else 0.0,
        illum_strength=pair(v["illum"]), illum_gradient=pair(v["gradient"]),
        illum_smudge_probability=1.0 if v["smudge"] > 0 else 0.0, illum_smudge_depth=pair(min(v["smudge"], 0.95)),
        fog_probability=0.0,
        light_probability=1.0 if v["bloom"] > 0 or v["lamp"] > 1.0 else 0.0,
        light_gain=pair(v["lamp"]), light_bloom_strength=pair(max(v["bloom"], 1e-4)), light_knee=pair(v["knee"]),
    )
    return values


def values_from_params(params: dict) -> dict[str, float]:
    """Thanh truot gan dung voi mot mau da boc luc train (de nguoi dung thay no la gi)."""
    uneven, light = (params.get(key) or {} for key in ("illumination_params", "light_params"))
    stops = [abs(blob["stops"]) for blob in uneven.get("blobs", [])]
    return {
        "exposure": params["exposure_gain"] if params.get("low_light", True) else 1.0,
        "gamma": params["tone_gamma"] if params.get("low_light", True) else 1.0,
        "vignette": params["vignette_strength"] if params.get("low_light", True) else 0.0,
        "illum": float(np.mean(stops)) if stops else 0.0, "gradient": uneven.get("gradient_stops", 0.0),
        "smudge": max((smudge["depth"] for smudge in uneven.get("smudges", [])), default=0.0),
        "lamp": light.get("gain", 1.0), "bloom": light.get("bloom_strength", 0.0), "knee": light.get("knee", 2.0),
        "defocus": params["defocus_sigma"] if params.get("defocus") else 0.0,
        "motion": float(params["motion_length"]) if params.get("motion") else 0.0,
        "photons": math.log10(params["photon_count"]) if params.get("sensor_noise", True) else 4.5,
        "jpeg": float(params["jpeg_quality"]) if params.get("jpeg") else 100.0,
    }


def yaml_snippet(v: dict[str, float]) -> str:
    """Doan corruption.image de dan vao config: moi gia tri thanh mot khoang hep quanh no."""
    around = lambda x, spread, low, high: [round(max(low, x * (1 - spread)), 3), round(min(high, x * (1 + spread)), 3)]
    lines = ["corruption:", "  image:",
             f"    exposure_gain: {around(v['exposure'], 0.3, 0.01, 1.0)}",
             f"    tone_gamma: {around(v['gamma'], 0.15, 0.3, 1.0)}",
             f"    vignette_strength: {around(v['vignette'], 0.5, 0.0, 1.0)}"]
    if v["illum"] > 0 or v["gradient"] > 0 or v["smudge"] > 0:
        lines += ["    illum_probability: 0.8", f"    illum_strength: {around(v['illum'], 0.5, 0.0, 4.0)}",
                  f"    illum_gradient: {around(v['gradient'], 0.5, 0.0, 4.0)}",
                  f"    illum_smudge_depth: {around(v['smudge'], 0.3, 0.0, 0.95)}"]
    if v["bloom"] > 0 or v["lamp"] > 1:
        lines += ["    light_probability: 0.8", f"    light_gain: {around(v['lamp'], 0.5, 1.0, 60.0)}",
                  f"    light_bloom_strength: {around(max(v['bloom'], 0.01), 0.6, 0.001, 2.0)}",
                  f"    light_knee: {around(v['knee'], 0.3, 0.5, 6.0)}"]
    lines += [f"    defocus_sigma_px: {around(max(v['defocus'], 0.05), 0.4, 0.05, 4.0)}",
              f"    photon_count: {around(10 ** v['photons'], 0.6, 50.0, 1e6)}"]
    return "\n".join(lines)


@dataclass
class State:
    images: list = field(default_factory=list)          # [(ten, anh)]
    image_index: int = 0
    variant: int = 0
    values: dict = field(default_factory=dict)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="kaggle_env.yaml", help="config trong configs/ (khoang cho 'Boc nhu luc train')")
    parser.add_argument("--image", nargs="*", type=Path, help="anh cua ban (bat ky dinh dang PIL doc duoc)")
    parser.add_argument("--root", type=Path, default=Path.home() / "Datasets/tartanair-v2-jepa")
    parser.add_argument("--split", default="valid", choices=("train", "valid", "test"))
    parser.add_argument("--count", type=int, default=30, help="so anh dataset nap san de bam 'Anh khac'")
    parser.add_argument("--size", type=int, default=256, help="canh anh (256 nhu luc train)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-show", action="store_true", help="chi luu mot hinh mau vao outputs/noise_tuner/")
    args = parser.parse_args([] if "ipykernel" in sys.modules else None)

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    config = _config_chain(REPO / "configs" / args.config)
    base = config["corruption"]["image"]
    master_seed = int(config["data"]["corruption_seed"])
    rng = np.random.default_rng(args.seed)
    state = State(values=initial_values(base))
    if args.image:
        state.images = [(path.name, load_rgb(path, (args.size, args.size))) for path in args.image]
    else:
        paths = frames(args.root, args.split)
        chosen = [paths[i] for i in rng.choice(len(paths), size=min(args.count, len(paths)), replace=False)]
        state.images = [(f"{path.parts[-5]}/{path.stem}", load_rgb(path, (args.size, args.size))) for path in chosen]
    output = REPO / "outputs/noise_tuner"

    def render(values: dict) -> tuple[np.ndarray, float]:
        corruptor = LowLightImageCorruptor(LowLightImageCorruptionConfig(**corruption_values(base, values)), master_seed)
        started = time.perf_counter()
        noisy, _ = corruptor(state.images[state.image_index][1], split="train", realization=0,
                             trajectory=f"tuner-{state.variant}", timestamp=0.0, frame_index=state.variant, mode="full")
        return np.asarray(noisy), time.perf_counter() - started

    def training_draw() -> tuple[np.ndarray, dict]:
        """Mot mau dung phan phoi train cua --config (ca xac suat bat/tat tung tang)."""
        image_config = LowLightImageCorruptionConfig(**{**base, "motion_from_imu": False})
        corruptor = LowLightImageCorruptor(image_config, master_seed)
        noisy, params = corruptor(state.images[state.image_index][1], split="train", realization=0,
                                  trajectory=f"tuner-train-{state.variant}", timestamp=0.0,
                                  frame_index=state.variant, mode="full")
        return np.asarray(noisy), params

    if args.no_show:
        figure, axes = plt.subplots(2, 3, figsize=(12, 8.4), layout="constrained")
        for row in range(2):
            state.image_index = row % len(state.images)
            noisy, _ = render(state.values)
            state.variant = row + 1
            trained, params = training_draw()
            for axis, (title, image) in zip(axes[row], (("Sạch", state.images[state.image_index][1]),
                                                        ("Thanh trượt mặc định", noisy),
                                                        ("Bốc như lúc train", trained))):
                axis.imshow(np.clip(image, 0, 1)); axis.set_title(title, fontsize=10); axis.axis("off")
        output.mkdir(parents=True, exist_ok=True)
        figure.savefig(output / "sample.png", dpi=110)
        print("Da luu:", output / "sample.png")
        print(yaml_snippet(state.values))
        return 0

    figure = plt.figure(figsize=(14, 9.5))
    try:
        figure.canvas.manager.set_window_title("Chỉnh thông số nhiễu")
    except Exception:
        pass
    clean_axis = figure.add_axes([0.03, 0.42, 0.45, 0.53])
    noisy_axis = figure.add_axes([0.52, 0.42, 0.45, 0.53])
    for axis in (clean_axis, noisy_axis):
        axis.set_xticks([]), axis.set_yticks([])
    clean_artist = clean_axis.imshow(state.images[0][1])
    noisy_artist = noisy_axis.imshow(state.images[0][1])
    sliders = {}
    for index, (key, label, low, high, step) in enumerate(SLIDERS):
        column, row = divmod(index, 8)
        axis = figure.add_axes([0.17 + 0.47 * column, 0.355 - 0.037 * row, 0.25, 0.024])
        sliders[key] = Slider(axis, label, low, high, valinit=float(np.clip(state.values[key], low, high)),
                              valstep=step)
        sliders[key].label.set_fontsize(9)
    buttons = {}
    for index, name in enumerate(("Ảnh khác", "Đổi ngẫu nhiên", "Bốc như lúc train", "Lưu")):
        buttons[name] = Button(figure.add_axes([0.1 + 0.21 * index, 0.015, 0.18, 0.045]), name)
    syncing = {"active": False}

    def redraw(_=None, image: np.ndarray | None = None, title: str | None = None) -> None:
        if syncing["active"]:
            return
        state.values = {key: float(slider.val) for key, slider in sliders.items()}
        name, clean = state.images[state.image_index]
        clean_artist.set_data(clean)
        clean_axis.set_title(f"Sạch · {name}", fontsize=10)
        if image is None:
            image, seconds = render(state.values)
            title = f"Nhiễu (thanh trượt) · biến thể {state.variant} · {seconds * 1000:.0f} ms"
        noisy_artist.set_data(np.clip(image, 0, 1))
        noisy_axis.set_title(title, fontsize=10)
        figure.canvas.draw_idle()

    def next_image(_):
        state.image_index = (state.image_index + 1) % len(state.images)
        redraw()

    def reshuffle(_):
        state.variant += 1
        redraw()

    def like_training(_):
        state.variant += 1
        image, params = training_draw()
        syncing["active"] = True
        for key, value in values_from_params(params).items():
            low, high = sliders[key].valmin, sliders[key].valmax
            sliders[key].set_val(float(np.clip(value, low, high)))
        syncing["active"] = False
        on = [name for name, flag in (("sáng/tối lệch", params.get("illumination")),
                                       ("lóe", params.get("light")), ("tối", params.get("low_light", True)))
              if flag]
        redraw(image=image, title="Bốc như lúc train · " + (", ".join(on) if on else "môi trường trong"))
        print("Mau train: " + (", ".join(on) if on else "moi truong trong (chi mo + hat)"))

    def save(_):
        output.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        figure.savefig(output / f"tuner_{stamp}.png", dpi=110)
        snippet = yaml_snippet(state.values)
        (output / f"tuner_{stamp}.yaml").write_text(snippet + "\n", encoding="utf-8")
        print(f"Da luu {output / f'tuner_{stamp}.png'} va .yaml:\n{snippet}")

    for slider in sliders.values():
        slider.on_changed(redraw)
    buttons["Ảnh khác"].on_clicked(next_image)
    buttons["Đổi ngẫu nhiên"].on_clicked(reshuffle)
    buttons["Bốc như lúc train"].on_clicked(like_training)
    buttons["Lưu"].on_clicked(save)
    redraw()
    bring_to_front(figure)
    print("Cua so dang mo: keo thanh truot de doi nhieu; dong cua so de ket thuc.", flush=True)
    plt.show()
    return 0


if __name__ == "__main__":
    code = main()
    if "ipykernel" not in sys.modules:
        raise SystemExit(code)
