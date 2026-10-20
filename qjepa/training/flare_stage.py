"""Giai doan loe (p42, ``train-flare``): train rieng nhanh khu loe (FlareLayer khong latent) truoc phase 2.

Mot GPU, fp16 tren CUDA. Loss: L1 giua lop loe nhanh doan va lop loe dich (anh vao truoc hat - anh dich, xem
data/flare_stage.py). Validation tren hai thu model chua thay: anh ghep loe cua cac lop loe valid len anh TartanAir
valid, va cac cap FlareImage valid (anh chup that) -- PSNR vung loe va ngoai vung loe, vao -> ra.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..models.decoders import FlareLayer
from .schedules import warmup_cosine_lr

FLARE_THRESHOLD = 0.05
NONFINITE_LIMIT = 50            # so batch NaN/inf lien tiep thi train-flare dung han (p42 Kaggle: NaN tu ~2 300 mai mai)          # nhu tools/flare_image_test.py: pixel vao lech anh sach hon 0,05 la vung loe


def build_flare_layer(stage: dict[str, Any]) -> nn.Module:
    """The flare section's layer: p42's U-Net (flare.model absent or "unet"), or p44's Uformer of Flare7K++
    (flare.model: uformer_flare7kpp; train-flare loads its released weights before the first update)."""
    if stage.get("model", "unet") == "uformer_flare7kpp":
        from ..models.uformer_flare import UformerFlareLayer
        layer = UformerFlareLayer(float(stage["uformer_scale"]))
        layer.net.set_checkpointing(True)          # fine-tuning at 512x512 on a 16 GB T4
        return layer
    return FlareLayer(widths=tuple(int(w) for w in stage["widths"]), blocks=int(stage["blocks"]), uses_latent=False)


def selection_score(report: dict[str, float], select_on: str = "real") -> float:
    """p44 (flare.select_best): the flare stage's checkpoint score, higher is better -- PSNR over the whole crop of
    the held-out FlareImage pairs (flare taken off AND the rest kept), or of the valid composites without pairs.
    p45 (flare.select_on: real_and_composite): the mean of the pairs' and the composites' PSNR gains (out - in)."""
    def finite(key):
        value = report.get(key)
        return float(value) if value is not None and math.isfinite(float(value)) else None

    if select_on == "real_and_composite":
        gains = [finite(f"{kind}_psnr_out") - finite(f"{kind}_psnr_in") for kind in ("real", "composite")
                 if finite(f"{kind}_psnr_out") is not None and finite(f"{kind}_psnr_in") is not None
                 and (kind != "real" or report.get("real_pairs"))]
        return sum(gains) / len(gains) if gains else float("-inf")
    value = finite("real_psnr_out") if report.get("real_pairs") else finite("composite_psnr_out")
    return value if value is not None else float("-inf")


def _grad_scaler(enabled: bool):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


class FlareStageTrainer:
    def __init__(self, model: FlareLayer, stage: dict[str, Any], device: torch.device) -> None:
        self.model = model.to(device)
        self.stage = stage
        self.device = device
        self.amp = stage["precision"] == "amp_fp16" and device.type == "cuda"
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(stage["learning_rate"]),
                                           weight_decay=float(stage["weight_decay"]))
        self.scaler = _grad_scaler(self.amp)
        self.successful_updates = 0
        # Training forward; distribute() swaps in DDP. Validation always runs the bare model (rank 0 alone).
        self.train_model = self.model
        self.world = 1

    def distribute(self) -> None:
        """Several ranks (one per GPU): each takes its share of every batch, DDP averages the gradients -- the
        global batch's mean loss, as on one GPU. Call after the weights are loaded: DDP starts from rank 0's."""
        from ..distributed import rank_and_world
        self.world = rank_and_world()[1]
        if self.world > 1:
            ids = [torch.cuda.current_device()] if self.device.type == "cuda" else None
            self.train_model = nn.parallel.DistributedDataParallel(self.model, device_ids=ids, broadcast_buffers=False)

    def _forward(self, image: torch.Tensor, model: nn.Module | None = None) -> torch.Tensor:
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
            _, flare = (model or self.model)(None, image)
        return flare.float()

    def step(self, batch: dict[str, Any]) -> dict[str, Any]:
        stage = self.stage
        lr = warmup_cosine_lr(self.successful_updates, int(stage["max_successful_updates"]),
                              int(stage["warmup_updates"]), float(stage["learning_rate"]), float(stage["minimum_lr"]))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.model.train(True)
        self.optimizer.zero_grad(set_to_none=True)
        image = batch["image_noisy"].to(self.device, non_blocking=True)
        target = batch["flare_target"].to(self.device, non_blocking=True)
        flare = self._forward(image, self.train_model)
        loss = F.l1_loss(flare, target)
        bad = not bool(torch.isfinite(loss))
        if self.world > 1:
            from ..distributed import any_rank
            bad = any_rank(bad)             # every rank skips together, or DDP waits for a backward that never comes
        if bad:
            # A non-finite batch leaves the weights as they are; train-flare stops after NONFINITE_LIMIT in a row.
            self.optimizer.zero_grad(set_to_none=True)
            self.successful_updates += 1
            return {"loss": float("nan"), "learning_rate": lr, "nonfinite": True,
                    "successful_updates": self.successful_updates}
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(stage["gradient_clip_norm"])))
        overflow = not math.isfinite(norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.successful_updates += 1
        with torch.no_grad():
            error = (flare - target).abs().mean(dim=(1, 2, 3))
            size = target.abs().mean(dim=(1, 2, 3))
            kinds = batch["kind"].to(self.device)
            report = {"loss": float(loss), "learning_rate": lr, "gradient_norm": norm, "amp_overflow": overflow,
                      "nonfinite": False, "successful_updates": self.successful_updates}
            for index, name in enumerate(("composite", "pair", "plain")):
                chosen = kinds == index
                if bool(chosen.any()):
                    report[f"flare_l1_{name}"] = float(error[chosen].mean())
                    report[f"flare_size_{name}"] = float(size[chosen].mean())     # = L1 cua "khong lam gi"
        return report

    def payload(self, metadata: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        return {"metadata": metadata, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(), "successful_updates": self.successful_updates, "config": config}


def _psnr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    error = (a.clamp(0, 1) - b.clamp(0, 1)).square()
    if mask is not None:
        if not bool(mask.any()):
            return float("nan")
        error = error[:, mask]
    return float(-10 * torch.log10(error.mean().clamp_min(1e-12)))


def _mean(values: list[float]) -> float:
    kept = [v for v in values if math.isfinite(v)]
    return sum(kept) / len(kept) if kept else float("nan")


@torch.no_grad()
def validate_flare_stage(trainer: FlareStageTrainer, composites, real_pairs: list[dict[str, Any]],
                         batch_size: int) -> dict[str, float]:
    """Loe ghep valid (lop loe valid tren anh TartanAir valid) va cap FlareImage valid: model chua thay ca hai."""
    trainer.model.eval()
    device = trainer.device
    left, size, plain = [], [], []
    psnr_in, psnr_out = [], []
    for batch in composites:
        image = batch["image_noisy"].to(device)
        target = batch["flare_target"].to(device)
        clean = batch["image_target"].to(device)
        flare = trainer._forward(image)
        for k in range(image.shape[0]):
            if int(batch["kind"][k]) == 2:
                plain.append(float(flare[k].abs().mean()))
                continue
            left.append(float((flare[k] - target[k]).abs().mean()))
            size.append(float(target[k].abs().mean()))
            psnr_in.append(_psnr(image[k], clean[k]))
            psnr_out.append(_psnr(image[k] - flare[k], clean[k]))
    report = {"composite_flare_left": sum(left) / max(sum(size), 1e-12),
              "composite_psnr_in": _mean(psnr_in), "composite_psnr_out": _mean(psnr_out),
              "plain_flare_mean": _mean(plain)}
    rows = {key: [] for key in ("psnr_in", "psnr_out", "flare_in", "flare_out", "rest_in", "rest_out")}
    for start in range(0, len(real_pairs), batch_size):
        chunk = real_pairs[start:start + batch_size]
        image = torch.stack([item["image_noisy"] for item in chunk]).to(device)
        clean = torch.stack([item["image_target"] for item in chunk]).to(device)
        out = image - trainer._forward(image)
        for k in range(image.shape[0]):
            mask = (image[k] - clean[k]).abs().mean(0) > FLARE_THRESHOLD
            rows["psnr_in"].append(_psnr(image[k], clean[k]))
            rows["psnr_out"].append(_psnr(out[k], clean[k]))
            rows["flare_in"].append(_psnr(image[k], clean[k], mask))
            rows["flare_out"].append(_psnr(out[k], clean[k], mask))
            rows["rest_in"].append(_psnr(image[k], clean[k], ~mask))
            rows["rest_out"].append(_psnr(out[k], clean[k], ~mask))
    report.update({f"real_{key}": _mean(values) for key, values in rows.items()})
    report["real_pairs"] = len(real_pairs)
    trainer.model.train(True)
    return report


def validation_text(report: dict[str, float]) -> str:
    text = (f"loe ghep: PSNR {report['composite_psnr_in']:.2f} -> {report['composite_psnr_out']:.2f}, "
            f"loe con {100 * report['composite_flare_left']:.0f}% | anh khong loe: lop |F| {report['plain_flare_mean']:.4f}")
    if report.get("real_pairs"):
        text += (f" | FlareImage valid n={report['real_pairs']}: ca anh {report['real_psnr_in']:.2f} -> "
                 f"{report['real_psnr_out']:.2f}, vung loe {report['real_flare_in']:.2f} -> {report['real_flare_out']:.2f}"
                 f", ngoai loe {report['real_rest_in']:.2f} -> {report['real_rest_out']:.2f}")
    return text
