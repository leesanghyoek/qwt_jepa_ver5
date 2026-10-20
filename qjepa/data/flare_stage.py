"""Giai doan loe (p42, config muc ``flare``): du lieu de train rieng nhanh khu loe truoc phase 2.

p36-p41 chi day loe bang ~840 cap FlareImage lap lai (~16 000 luot ca phase 2): model chi go duoc ~1/4-1/3 loe ca tren
chinh cap da thay. Flare7K++ train tren hang tram nghin anh, moi anh mot lop loe ghep moi. Day la cach do: moi mau
mot lan rut moi (seed theo chi so), ba loai:

* ``composite`` -- mot lop loe Flare7K++ (tong hop Flare7K hoac chup that Flare-R) va (co the) loi quang hoc dung lai
  tu p39 (corruptions/lens_flare.py), cong tren anh sang tuyen tinh len anh nen sach: anh TartanAir hoac anh gt cua
  FlareImage. Anh DICH = nen + nguon sang (giu den, chi xoa tia, quang, bong ma). Mo may anh (cung mot phep cho ca
  hai) va hat (chi anh vao) nhu anh vao cua phase 2;
* ``pair`` -- mot cap FlareImage that (anh loe / anh sach), o cat ngau nhien, xoay k x 90 do va lat;
* ``plain`` -- mot anh TartanAir qua DUNG buoc nhieu cua phase 2 (toi, mo, hat...), KHONG co loe: dich cua lop loe
  bang 0, de nhanh hoc de yen anh khong co loe (p40 lam hong ngoai vung loe 4-13 dB).

Dich cua nhanh: lop loe = anh vao (truoc hat) - anh dich, theo sRGB, dung phep tru nhanh se lam (anh ra = vao - lop).
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np
import torch
from scipy import ndimage

from ..corruptions.image import LowLightImageCorruptionConfig
from ..corruptions.lens_flare import LensFlareBank, draw_lens_flare
from ..corruptions.light import linear_to_srgb, srgb_to_linear
from ..corruptions.rng import derive_seed
from .flare_pairs import FlarePairBank

FLARE_STAGE_KINDS = ("composite", "pair", "plain")
LENS_FIELDS = tuple(field.name for field in fields(LowLightImageCorruptionConfig) if field.name.startswith("lens_"))
READ_NOISE = 2.0 / 255.0


def lens_config(lens: dict[str, Any]) -> LowLightImageCorruptionConfig:
    """Cac khoa lens_* cua muc flare.lens thanh mot cau hinh nhieu (draw_lens_parameters doc no)."""
    unknown = sorted(set(lens) - set(LENS_FIELDS))
    if unknown:
        raise ValueError(f"flare.lens: khoa khong phai lens_*: {unknown}")
    values = {key: tuple(value) if isinstance(value, list) else value for key, value in lens.items()}
    return LowLightImageCorruptionConfig(**values)


def _chw(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(np.asarray(image, dtype=np.float32).transpose(2, 0, 1)))


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    return ndimage.gaussian_filter(image, sigma=(sigma, sigma, 0), mode="reflect")


def _view(image: np.ndarray, turns: int, flip: bool) -> np.ndarray:
    turned = np.rot90(image, turns)
    return np.ascontiguousarray(turned[:, ::-1] if flip else turned)


class FlareStageDataset(torch.utils.data.Dataset):
    """Mau thu ``index`` = mot lan rut moi, chi phu thuoc (seed, split, index): resume doc lai dung cac mau do."""

    def __init__(self, tartan, *, lens_bank: LensFlareBank, flare_pairs: FlarePairBank | None,
                 section: dict[str, Any], split: str, length: int, seed: int) -> None:
        self.tartan = tartan                     # PairedCameraImuDataset cung split (anh nen, anh "plain")
        self.lens_bank = lens_bank
        self.flare_pairs = flare_pairs
        self.lens = lens_config(section["lens"])
        self.split = split
        self.length = int(length)
        self.seed = int(seed)
        kinds = section["kinds"]
        self.kinds = FLARE_STAGE_KINDS
        self.weights = np.asarray([float(kinds[kind]) for kind in FLARE_STAGE_KINDS])
        self.weights = self.weights / self.weights.sum()
        self.background_pair_fraction = float(section["background_pair_fraction"]) if flare_pairs else 0.0
        self.blur_probability = float(section["blur_probability"])
        self.blur_sigma = tuple(float(v) for v in section["blur_sigma_px"])
        self.grain_probability = float(section["grain_probability"])
        self.photon_count = tuple(float(v) for v in section["photon_count"])
        self.size = tuple(int(v) for v in tartan.image_size)
        if flare_pairs is None and self.weights[1] > 0:
            raise ValueError("flare.kinds.pair > 0 can cap FlareImage (data.flare_pairs_*)")

    def __len__(self) -> int:
        return self.length

    def kind_of(self, index: int) -> str:
        return self.kinds[int(self._rng(index).choice(len(self.kinds), p=self.weights))]

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(derive_seed(self.seed, "flare_stage", self.split, int(index)))

    def _pair_crop(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, str]:
        """(gt, anh loe) cua mot cap FlareImage cung split: o cat ngau nhien, xoay k x 90 do, co the lat."""
        pairs = self.flare_pairs.pairs(self.split)
        pair = pairs[int(rng.integers(len(pairs)))]
        gt, flared = self.flare_pairs.load(pair)
        height, width = self.size
        top = int(rng.integers(0, gt.shape[0] - height + 1))
        left = int(rng.integers(0, gt.shape[1] - width + 1))
        turns, flip = int(rng.integers(4)), bool(rng.random() < 0.5)
        crop = lambda image: _view(image[top:top + height, left:left + width].astype(np.float32) / 255.0, turns, flip)
        return crop(gt), crop(flared), pair.name

    def _background(self, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        if float(rng.random()) < self.background_pair_fraction:
            gt, _, name = self._pair_crop(rng)
            return gt, f"flareimage/{name}"
        sample = self.tartan.samples[int(rng.integers(len(self.tartan.samples)))]
        frame = np.asarray(self.tartan._clean_frame(sample), dtype=np.float32)
        return _view(frame, 0, bool(rng.random() < 0.5)), sample.sample_id

    def _camera(self, rng: np.random.Generator) -> tuple[float, float | None]:
        """(sigma mo, so photon cua hat hoac None): rut vo dieu kien de cac lan rut sau khong doi theo nhanh."""
        blurred, sigma = float(rng.random()) < self.blur_probability, float(rng.uniform(*self.blur_sigma))
        grainy, photons = float(rng.random()) < self.grain_probability, float(rng.uniform(*self.photon_count))
        return (sigma if blurred else 0.0), (photons if grainy else None)

    @staticmethod
    def _grain(image: np.ndarray, photons: float | None, rng: np.random.Generator) -> np.ndarray:
        if photons is None:
            return image
        shot = np.sqrt(np.clip(image, 0.0, None) / photons)
        noisy = image + rng.normal(size=image.shape) * shot + rng.normal(size=image.shape) * READ_NOISE
        return np.clip(noisy, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = self._rng(index)
        kind = self.kinds[int(rng.choice(len(self.kinds), p=self.weights))]
        info: dict[str, Any] = {"kind": kind}
        if kind == "plain":
            item = self.tartan[int(rng.integers(len(self.tartan.samples)))]
            image = item["image_noisy"].permute(1, 2, 0).numpy()
            info["frame"] = item["sample_id"]
            zero = np.zeros_like(image)
            return {"image_noisy": _chw(image), "image_target": _chw(image), "flare_target": _chw(zero),
                    "kind": torch.tensor(FLARE_STAGE_KINDS.index(kind)), "info": info}
        sigma, photons = self._camera(rng)
        if kind == "composite":
            background, info["frame"] = self._background(rng)
            params, flare, light = draw_lens_flare(rng, self.lens, self.lens_bank, self.split, *self.size)
            linear = srgb_to_linear(background)
            flared = linear_to_srgb(_blur(linear + flare, sigma))
            target = linear_to_srgb(_blur(linear + light, sigma))
            info.update(source=params["source"], layer=params["name"], gain=params["gain"],
                        optics=bool(params["optics"]))
        else:
            gt, photo, info["pair"] = self._pair_crop(rng)
            flared = linear_to_srgb(_blur(srgb_to_linear(photo), sigma))
            target = linear_to_srgb(_blur(srgb_to_linear(gt), sigma))
        flared = np.clip(flared, 0.0, 1.0).astype(np.float32)
        target = np.clip(target, 0.0, 1.0).astype(np.float32)
        info.update(blur_sigma_px=sigma, photons=photons)
        image = self._grain(flared, photons, rng)
        return {"image_noisy": _chw(image), "image_target": _chw(target), "flare_target": _chw(flared - target),
                "kind": torch.tensor(FLARE_STAGE_KINDS.index(kind)), "info": info}


def collate_flare_stage(batch: list[dict[str, Any]]) -> dict[str, Any]:
    output = {key: torch.stack([item[key] for item in batch]) for key in
              ("image_noisy", "image_target", "flare_target", "kind")}
    output["info"] = [item["info"] for item in batch]
    return output


def held_out_pairs(flare_pairs: FlarePairBank, split: str, size: tuple[int, int]) -> list[dict[str, Any]]:
    """Moi cap FlareImage cua split (model chua thay): o cat GIUA co ``size``, khong them gi -- anh chup that."""
    height, width = size
    items = []
    for pair in flare_pairs.pairs(split):
        gt, flared = flare_pairs.load(pair)
        if gt.shape[0] < height or gt.shape[1] < width:
            continue
        top, left = (gt.shape[0] - height) // 2, (gt.shape[1] - width) // 2
        crop = lambda image: image[top:top + height, left:left + width].astype(np.float32) / 255.0
        items.append({"name": pair.name, "source": pair.source, "image_noisy": _chw(crop(flared)),
                      "image_target": _chw(crop(gt))})
    return items
