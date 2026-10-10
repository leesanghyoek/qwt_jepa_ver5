"""Command-line entrypoints for data preparation, both training phases, and inference."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from .config import (
    build_backbone,
    build_corruptors,
    build_decoders,
    build_normalizer,
    build_phase1_model,
    load_config,
    phase2_backbone,
    phase2_latent_modules,
    resolve_device,
    seed_everything,
    serializable_config,
)
from .data import (
    ImuNormalizer,
    PairedCameraImuDataset,
    TrajectoryDiverseBatchSampler,
    build_manifest,
    collate_paired,
    read_manifest,
    write_manifest,
)
from .data.dataset import load_rgb
from .corruptions.image import FRAME_KINDS, brightness_stops, degradation_vector, frame_kind
from .corruptions.rng import derive_seed
from .evaluation import ImuOverlapMerger, image_metrics, latent_diagnostics
from .evaluation.reporting import (
    evaluation_summary, image_panel, plot_training, save_imu_result, write_csv, write_json,
)
from .models import LatentPretrainingModel, RestorationSystem
from .distributed import any_rank, rank0_section, rank_and_world, share_rank0_rng, spawn as spawn_ranks
from .execution import (
    IJEPAForward, Phase1Forward, RestorationForward, execution_metadata, parallel_forward, select_device_ids,
)
from .training.checkpoints import (
    atomic_torch_save,
    configuration_hash,
    load_checkpoint,
    require_phase1_checkpoint,
    restore_rng_state,
    state_dict_hash,
)
from .training.losses import jepa_fine_loss, jepa_latent_loss
from .training.ijepa import IJEPATrainer, ijepa_loss
from .training.phase1 import Phase1Trainer, _to_device, jepa_report_terms
from .training.phase2 import Phase2Trainer, latent_predictor_hash


def _config_path(value: str | None) -> Path:
    return Path(value or "configs/pipeline_v3.yaml")


def _configure_execution(config: dict[str, Any], args: argparse.Namespace, device: torch.device,
                         *, use_saved_setting: bool = True) -> dict[str, Any]:
    count = getattr(args, "gpus", None) or (config["runtime"].get("gpu_count", "auto") if use_saved_setting else "auto")
    config["runtime"]["gpu_count"] = count
    info = execution_metadata(device, select_device_ids(device, count), rank_and_world()[1])
    # Opt-in: letting cuDNN time its conv algorithms once can be faster for
    # fixed-shape batches; the heuristic choice has run every earlier recipe.
    if device.type == "cuda" and config["runtime"].get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
        info["cudnn_benchmark"] = True
    # torch 2.10 + cuDNN 9.10 on Kaggle T4 x2: fp32 phase 2 under DataParallel died
    # with "misaligned address" -- not on one GPU, not with cuDNN off, not with
    # CUDA_LAUNCH_BLOCKING=1, so a race between the two replica threads. Off means
    # PyTorch's own conv kernels; one GPU (gpu_count 1) keeps cuDNN.
    if device.type == "cuda" and not config["runtime"].get("cudnn_enabled", True):
        torch.backends.cudnn.enabled = False
        info["cudnn_enabled"] = False
    print("Execution: " + json.dumps(info))
    return info


def _ddp_world(config: dict[str, Any], args: argparse.Namespace, device: torch.device) -> int:
    """Phase-2 processes: one per GPU with runtime.parallel ddp, otherwise 1.

    On CPU, ``gpu_count: 2`` gives two gloo processes -- what the tests run.
    Only the launcher counts GPUs: a rank sees just its own one (qjepa.distributed
    sets CUDA_VISIBLE_DEVICES), launches nothing, and must not ask for two.
    """
    if config["runtime"].get("parallel", "data_parallel") != "ddp" or rank_and_world()[1] > 1:
        return 1
    count = str(getattr(args, "gpus", None) or config["runtime"].get("gpu_count", "auto"))
    if count == "auto":
        return min(2, torch.cuda.device_count()) if device.type == "cuda" else 1
    if device.type == "cuda" and int(count) > torch.cuda.device_count():
        raise ValueError(f"DDP asked for {count} GPUs; only {torch.cuda.device_count()} visible")
    return int(count)


def _manifest(config: dict[str, Any], override: str | None):
    location = override or config["data"].get("manifest_dir")
    if not location:
        raise ValueError("Set data.manifest_dir in YAML or pass --manifest")
    return read_manifest(location), str(Path(location).resolve())


def _dataset(
    config: dict[str, Any],
    manifest: dict[str, Any],
    split: str,
    *,
    fixed_realization: bool,
    image_mode: str = "full",
    imu_mode: str = "full",
    scenarios: list[dict[str, object]] | None = None,
    sensor_reference: bool = False,
    hflip_probability: float = 0.0,
    full_frame: bool = False,
    brightness_target: bool = False,
    tone_label: bool = False,
) -> PairedCameraImuDataset:
    image_corruptor, imu_corruptor = build_corruptors(config)
    if fixed_realization:
        image_corruptor.config = replace(image_corruptor.config, clean_probability=0.0)
        imu_corruptor.config = replace(imu_corruptor.config, clean_probability=0.0)
    samples = manifest["samples"][split]
    if not samples:
        raise ValueError(f"Manifest split {split!r} is empty")
    return PairedCameraImuDataset(
        samples,
        image_corruptor=image_corruptor,
        imu_corruptor=imu_corruptor,
        image_size=tuple(config["data"]["image_size"]),
        realization=config["data"]["validation_realization"] if fixed_realization else 0,
        image_mode=image_mode,
        imu_mode=imu_mode,
        scenarios=scenarios,
        scenario_seed=config["data"]["corruption_seed"],
        sensor_reference=sensor_reference,
        hflip_probability=hflip_probability,
        source_size=tuple(config["data"]["source_size"]) if config["data"].get("source_size") else None,
        full_frame=full_frame,
        brightness_target=brightness_target,
        tone_label=tone_label,
    )


# Mirror half the training pairs when a phase asks for it (phase{1,2}.augment_hflip).
HFLIP_PROBABILITY = 0.5


def _train_dataset(config: dict[str, Any], manifest: dict[str, Any], phase: str, **kwargs) -> PairedCameraImuDataset:
    """The training split of ``phase``, with that phase's augmentation and references."""
    section = config[phase]
    sensitivity = config["encoder_sensitivity"]
    # Only phase 1's Jacobian reads the frame without grain; rendering it costs a readout.
    sensor_reference = (phase == "phase1" and bool(sensitivity.get("enabled", False))
                        and sensitivity.get("noise_direction", "corruption") == "sensor_noise")
    # Only phase 2's relight loss reads the corruption's stop map (phase2.split_relight).
    brightness_target = phase == "phase2" and bool(config["phase2"].get("split_relight", False))
    # Only phase 2's tone gate reads whether the corruption changed the light (phase2.split_tone_gate).
    tone_label = phase == "phase2" and bool(config["phase2"].get("split_tone_gate", False))
    return _dataset(config, manifest, "train", fixed_realization=False, sensor_reference=sensor_reference,
                    hflip_probability=HFLIP_PROBABILITY if section.get("augment_hflip", False) else 0.0,
                    brightness_target=brightness_target, tone_label=tone_label, **kwargs)


def _loader(
    config: dict[str, Any],
    dataset: PairedCameraImuDataset,
    batch_size: int,
    *,
    train: bool,
    generator: torch.Generator | None = None,
    batch_sampler: TrajectoryDiverseBatchSampler | None = None,
) -> DataLoader:
    common = {
        "dataset": dataset,
        "num_workers": config["data"]["num_workers"],
        "pin_memory": config["data"]["pin_memory"] and torch.cuda.is_available(),
        "collate_fn": collate_paired,
        "persistent_workers": False,
        "generator": generator,
    }
    if batch_sampler is not None:
        return DataLoader(batch_sampler=batch_sampler, **common)
    return DataLoader(
        batch_size=batch_size,
        shuffle=train and config["data"]["shuffle_paired_samples"],
        drop_last=train,
        **common,
    )


def _training_batch_stream(
    config: dict[str, Any],
    dataset: PairedCameraImuDataset,
    batch_size: int,
    *,
    start_microbatch: int,
    namespace: str,
    rank: int = 0,
    world: int = 1,
) -> Iterator[dict[str, Any]]:
    batches_per_epoch = len(dataset) // batch_size
    if batches_per_epoch < 1:
        raise ValueError("Dataset has fewer samples than one full training batch")
    epoch, first_batch = divmod(start_microbatch, batches_per_epoch)
    while True:
        dataset.set_realization(epoch if config["data"].get("train_realization_per_epoch", True) else 0)
        epoch_seed = derive_seed(
            config["data"]["sampler_seed"], "sampler", namespace, epoch
        )
        trajectory_keys = [sample.trajectory_key for sample in dataset.samples]
        batch_sampler = TrajectoryDiverseBatchSampler(
            trajectory_keys,
            batch_size,
            config["data"]["minimum_trajectories_per_batch"],
            epoch_seed,
        )
        sampler = _SkipBatches(batch_sampler, first_batch) if first_batch else batch_sampler
        loader = _loader(
            config,
            dataset,
            batch_size,
            train=True,
            generator=torch.Generator().manual_seed(epoch_seed),
            batch_sampler=_RankShare(sampler, rank, world) if world > 1 else sampler,
        )
        produced = False
        for batch in loader:
            produced = True
            yield batch
        if not produced:
            raise ValueError("DataLoader produced no remaining full training batches")
        epoch += 1
        first_batch = 0


class _SkipBatches:
    """Bo qua N batch DAU TIEN o muc sampler, truoc khi du lieu duoc nap.

    Sampler chi sinh ra danh sach chi so, nen bo qua o day khong ton gi. Truoc
    day vong lap bo qua o muc DataLoader — moi batch bi bo van duoc giai nen PNG
    va chay het corruption roi moi bi vut di, nen resume o update 2500 phai nap
    20.000 anh truoc khi in duoc dong log dau tien.

    An toan ve ngu nghia: sampler cung seed cho cung thu tu batch, va corruption
    duoc seed theo TUNG MAU (derive_seed) chu khong theo thu tu, nen bo qua khong
    lam lech bat cu thu gi.
    """

    def __init__(self, inner, skip: int) -> None:
        self.inner, self.skip = inner, skip

    def __iter__(self):
        for index, batch in enumerate(self.inner):
            if index >= self.skip:
                yield batch

    def __len__(self) -> int:
        return max(0, len(self.inner) - self.skip)


class _RankShare:
    """Phan cua rank r trong moi batch: doan lien tiep [r*n/w, (r+1)*n/w).

    Gom cac phan theo thu tu rank thi ra dung batch theo thu tu sampler, nen
    loss (tinh tren ca batch) giong het mot tien trinh tu nap ca batch. Moi rank
    chi giai nen va lam nhieu phan cua minh.
    """

    def __init__(self, inner, rank: int, world: int) -> None:
        self.inner, self.rank, self.world = inner, rank, world

    def __iter__(self):
        for batch in self.inner:
            if len(batch) % self.world:
                raise ValueError(f"Batch of {len(batch)} does not split over {self.world} ranks")
            share = len(batch) // self.world
            yield batch[self.rank * share:(self.rank + 1) * share]

    def __len__(self) -> int:
        return len(self.inner)


class _Jsonl:
    def __init__(self, path: Path | None):
        # None: a DDP rank other than 0, which writes nothing.
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, payload: dict[str, Any]) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


# Exit status that means "checkpoint saved, restart me to free memory" (EX_TEMPFAIL).
RESTART_EXIT_CODE = 75


def _restart_if_memory_high(config: dict[str, Any], memory: dict[str, float], update: int, maximum: int) -> None:
    """After a checkpoint: exit with RESTART_EXIT_CODE once RSS passes the limit.

    On Kaggle the phase-2 process grew by ~6.7 MiB per update until the kernel
    killed it at update 4097 of 5000 -- the same code shows no growth on CPU, so
    the leak is in the GPU runtime path. A restart from the checkpoint just
    saved frees it, and resume is exact (weights, optimizer, RNG, data position),
    so the run ends where an uninterrupted one would. Off unless
    runtime.restart_above_rss_gib is set; runtime keys are outside the hash.
    """
    limit = config["runtime"].get("restart_above_rss_gib")
    over = limit is not None and update < maximum and memory["rss_mib"] > float(limit) * 1024
    # Under DDP every rank calls this after rank 0 has saved; all go when one must.
    if not any_rank(over):
        return
    print(f"  RSS {memory['rss_mib'] / 1024:.1f} GiB > {limit} GiB: checkpoint da luu o update {update};"
          f" thoat de khoi dong lai va resume (exit {RESTART_EXIT_CODE}).", flush=True)
    sys.exit(RESTART_EXIT_CODE)


def _progress(update: int, maximum: int) -> str:
    """Tien do dang 1340/10000 13%, canh phai de log thang cot khi cuon."""
    width = len(str(maximum))
    return f"{update:>{width}}/{maximum} {100.0 * update / max(1, maximum):>3.0f}%"


def _pace(started: float, first_update: int, update: int, waited: float | None = None) -> str:
    """Giay moi update tu luc tien trinh nay bat dau, tinh ca validation.

    ``waited``: tong giay vong train dung cho batch tiep theo (GPU ranh vi thieu du lieu).
    Gan bang s/update la CPU/DataLoader dang ghim toc do; gan 0 la GPU.
    """
    updates = max(1, update - first_update)
    pace = f"{(time.perf_counter() - started) / updates:.2f} s/update"
    return pace if waited is None else f"{pace} (cho du lieu {waited / updates:.2f})"


class _RecentMean:
    """Trung binh ``window`` update gan nhat cua vai so do, in kem dong log.

    Moi update la MOT batch (phase 2: 8 anh), moi anh mot kieu nhieu (co / khong loe HALO, toi / sang, mo), nen
    loss cua tung dong nhay theo batch (p33: 0,8 -> 2,5 trong cung mot doan); trung binh 100 update moi cho thay
    loss co giam hay khong. Sau khi resume, trung binh chi tinh tren nhung update da chay trong tien trinh nay.
    """

    def __init__(self, window: int = 100):
        self.window = int(window)
        self.values: dict[str, deque] = {}

    def add(self, metrics: dict[str, Any], keys: tuple[str, ...]) -> None:
        for key in keys:
            value = metrics.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                self.values.setdefault(key, deque(maxlen=self.window)).append(float(value))

    def text(self, keys: tuple[str, ...]) -> str:
        kept = {key: self.values[key] for key in keys if self.values.get(key)}
        if not kept:
            return ""
        count = max(len(values) for values in kept.values())
        return f"| TB {count} update: " + " ".join(f"{key}={sum(v) / len(v):.4f}" for key, v in kept.items())


KIND_LABELS = {"plain": "tot", "relit": "doi sang", "flare": "loe"}


def kind_summary(metrics: dict[str, Any], prefix: str = "") -> str:
    """"tot 36.71->27.85 (n=40, 98% te hon vao) | doi sang ... | loe ..." from _evaluate_with_overlap's keys."""
    parts = []
    for kind in FRAME_KINDS:
        if f"{prefix}{kind}_count" in metrics:
            parts.append(f"{KIND_LABELS[kind]} {metrics[f'{prefix}{kind}_baseline_image_psnr_db']:.2f}->"
                         f"{metrics[f'{prefix}{kind}_image_psnr_db']:.2f} (n={int(metrics[f'{prefix}{kind}_count'])}, "
                         f"{100 * metrics[f'{prefix}{kind}_worse_fraction']:.0f}% te hon vao)")
    return " | ".join(parts)


def _next_timed(batches: Iterator[Any]) -> tuple[Any, float]:
    """The next batch and the seconds spent blocked on it.

    Each update already reads its metrics back to the CPU, so the GPU is idle here:
    this is the time the DataLoader costs the run.
    """
    fetch = time.perf_counter()
    batch = next(batches)
    return batch, time.perf_counter() - fetch


def _memory_mib() -> dict[str, float]:
    """RSS hien tai cua tien trinh train va cac worker con.

    SIGKILL khong de lai traceback, nen khong co so nay thi mot lan OOM chi cho
    biet "het RAM" chu khong cho biet dang o dau va tang theo nhip nao.
    """

    def resident(pid: str) -> float:
        try:
            with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0
        except OSError:
            pass
        return 0.0

    def child_pids(pid: str) -> list[str]:
        found: list[str] = []
        try:
            for task in os.listdir(f"/proc/{pid}/task"):
                with open(f"/proc/{pid}/task/{task}/children", encoding="utf-8") as handle:
                    found += handle.read().split()
        except OSError:
            pass
        return found

    # Moi tien trinh con chau, khong chi con truc tiep (vd. worker cua mot forkserver).
    children, pending, seen = 0.0, child_pids("self"), set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        children += resident(pid)
        pending += child_pids(pid)
    return {"rss_mib": resident("self"), "children_rss_mib": children}


def _prepare_run(output: Path, resume: str | None, update: int) -> None:
    if not resume and any((output / name).exists() for name in ("last.pt", "train.jsonl")):
        raise ValueError(f"Existing training run at {output}; use --resume or choose a new --output")
    log = output / "train.jsonl"
    if resume and log.exists():
        # A crash may leave log entries newer than the last atomic checkpoint.
        records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        retained = [record for record in records if int(record.get("successful_updates", 0)) <= update]
        log.write_text("".join(json.dumps(record) + "\n" for record in retained), encoding="utf-8")


def _fixed_validation_bank(dataset: PairedCameraImuDataset, size: int) -> None:
    """Round-robin trajectories and spread frames across each trajectory's timeline."""
    groups: dict[str, list[Any]] = {}
    for sample in dataset.samples:
        groups.setdefault(sample.trajectory_key, []).append(sample)
    quotas = {key: 0 for key in sorted(groups)}
    remaining = min(size, len(dataset.samples))
    while remaining:
        for key in quotas:
            if remaining and quotas[key] < len(groups[key]):
                quotas[key] += 1
                remaining -= 1
    selected = []
    for key, count in quotas.items():
        rows = sorted(groups[key], key=lambda sample: sample.image_time)
        index = np.linspace(0, len(rows) - 1, count, dtype=int)
        selected.extend(rows[item] for item in index)
    dataset.samples = selected


def _every_nth_frame(dataset: PairedCameraImuDataset, every: int) -> None:
    """Keep every ``every``-th frame of each trajectory in time order, and its last frame:
    every trajectory and environment stays, frames 0.1 s apart (nearly the same picture) are
    thinned. While the kept frames are closer than an IMU window -- every <= 12 at 10 Hz for
    128 samples at 100 Hz -- the windows still overlap, and with the last frame kept the merged
    IMU covers each trajectory as the whole split does."""
    groups: dict[str, list[Any]] = {}
    for sample in dataset.samples:
        groups.setdefault(sample.trajectory_key, []).append(sample)
    selected = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda row: row.image_time)
        kept = rows[::every]
        if kept[-1] is not rows[-1]:
            kept.append(rows[-1])
        selected.extend(kept)
    dataset.samples = selected


@torch.no_grad()
def _validate_active_blur(
    system: RestorationSystem, loader: DataLoader, device: torch.device,
    forward_model: torch.nn.Module | None = None,
) -> dict[str, float | int]:
    """Evaluate only frames whose fixed blur corruption actually contains blur."""
    system.eval()
    pixel_input = pixel_restored = edge_input = edge_restored = 0.0
    pixel_count = edge_count = active_frames = 0
    for raw in loader:
        active = torch.tensor([
            any(item["image"][key] for key in ("defocus", "motion", "downsample"))
            for item in raw["corruption"]
        ], dtype=torch.bool, device=device)
        if not active.any():
            continue
        batch = _to_device(raw, device)
        inputs = (batch["image_noisy"], batch["imu_noisy_phys"], batch["image_time"], batch["imu_times"])
        restored = system(*inputs).image if forward_model is None else forward_model(*inputs)["image"]
        clean, noisy, output_image = (
            tensor[active].clamp(0, 1) for tensor in
            (batch["image_clean"], batch["image_noisy"], restored)
        )
        pixel_input += float((noisy - clean).abs().sum())
        pixel_restored += float((output_image - clean).abs().sum())
        pixel_count += clean.numel()
        active_frames += clean.shape[0]
        greys = [tensor[:, :1] * 0.299 + tensor[:, 1:2] * 0.587 + tensor[:, 2:3] * 0.114
                 for tensor in (clean, noisy, output_image)]
        gradients = [
            (grey[..., 1:] - grey[..., :-1], grey[..., 1:, :] - grey[..., :-1, :])
            for grey in greys
        ]
        clean_x, clean_y = gradients[0]
        magnitude = torch.nn.functional.pad(clean_x.abs(), (0, 1)) + torch.nn.functional.pad(clean_y.abs(), (0, 0, 0, 1))
        threshold = torch.quantile(magnitude.flatten(1), 0.9, dim=1).view(-1, 1, 1, 1)
        mask = magnitude > threshold
        edge_count += int(mask.sum())
        for gradient, name in ((gradients[1], "input"), (gradients[2], "restored")):
            gx, gy = gradient
            error = (torch.nn.functional.pad((gx - clean_x).abs(), (0, 1)) +
                     torch.nn.functional.pad((gy - clean_y).abs(), (0, 0, 0, 1)))
            if name == "input":
                edge_input += float((error * mask).sum())
            else:
                edge_restored += float((error * mask).sum())
    if active_frames == 0 or edge_count == 0:
        raise ValueError("Blur validation bank contains no actual blurred edges")
    return {
        "active_frames": active_frames,
        "image_mae_input": pixel_input / pixel_count,
        "image_mae_restored": pixel_restored / pixel_count,
        "strong_edge_gradient_mae_input": edge_input / edge_count,
        "strong_edge_gradient_mae_restored": edge_restored / edge_count,
    }


def _phase2_full_blur_guard(
    evaluation: dict[str, Any], blur: dict[str, Any], reference: dict[str, Any],
    limits: dict[str, float],
) -> tuple[bool, list[str]]:
    """Keep a blur improvement only if full-image and IMU quality remain near initialization."""
    old_full, old_blur = reference["full"], reference["blur"]
    checks = {
        "image_psnr_db": evaluation["image_psnr_db"] >= old_full["image_psnr_db"] - limits["max_psnr_drop_db"],
        "image_ssim": evaluation["image_ssim"] >= old_full["image_ssim"] - limits["max_ssim_drop"],
        "accel_rmse": evaluation["accel_rmse"] <= old_full["accel_rmse"] * limits["max_accel_rmse_ratio"],
        "gyro_rmse": evaluation["gyro_rmse"] <= old_full["gyro_rmse"] * limits["max_gyro_rmse_ratio"],
        "blur_image_mae": blur["image_mae_restored"] < old_blur["image_mae_restored"],
        "blur_edge_error": blur["strong_edge_gradient_mae_restored"] < old_blur["strong_edge_gradient_mae_restored"],
    }
    failures = [name for name, passed in checks.items() if not passed]
    return not failures, failures


def _write_resolved(config: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(serializable_config(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _phase1_trainer(model, config: dict[str, Any], device: torch.device, manifest_hash: str):
    """phase1.objective: the VICReg JEPA (absent key) or I-JEPA."""
    if config["phase1"].get("objective", "jepa") == "ijepa":
        return IJEPATrainer(model, config, device, manifest_hash)
    return Phase1Trainer(model, config, device, manifest_hash)


def _ijepa_validation_batch(model, runner, batch: dict[str, Any], index: int, device: torch.device):
    """I-JEPA's loss on a bank batch under masks fixed per batch, and the whole-view features."""
    image_grid, imu_grid = model.token_grids(tuple(batch["image_noisy"].shape), int(batch["imu_noisy_phys"].shape[-1]))
    masks = model.sample_masks(image_grid, imu_grid, int(batch["image_noisy"].shape[0]), "validation", index)
    outputs = runner(batch["image_noisy"], batch["imu_noisy_phys"], batch["image_clean"], batch["imu_clean_phys"],
                     batch["image_time"], batch["imu_times"], **{key: value.to(device) for key, value in masks.items()},
                     dense=True)
    total, image, imu = ijepa_loss(outputs)
    return outputs, {"jepa": float(total), "jepa_image": float(image), "jepa_imu": float(imu)}


@torch.no_grad()
def _validate_latent(
    model: LatentPretrainingModel,
    loader: DataLoader,
    device: torch.device,
    maximum_batches: int,
    forward_model: torch.nn.Module | None = None,
) -> dict[str, float]:
    model.eval()
    ijepa = getattr(model, "objective", "jepa") == "ijepa"
    runner = forward_model if forward_model is not None else (IJEPAForward(model) if ijepa else Phase1Forward(model))
    collected: dict[str, list[float]] = {}
    bank: dict[str, list[torch.Tensor]] = {}
    counts = []
    for index, raw in enumerate(loader):
        if index >= maximum_batches:
            break
        batch = _to_device(raw, device)
        counts.append(batch["image_clean"].shape[0])
        if ijepa:
            outputs, values = _ijepa_validation_batch(model, runner, batch, index, device)
            teachers = (("teacher_TI", outputs["teacher_i"]), ("teacher_TU", outputs["teacher_u"]))
        else:
            outputs, values, teachers = _jepa_validation_batch(runner, batch)
        features = (
            ("noisy_FI", outputs["FI"]),
            ("noisy_FU", outputs["FU"]),
            ("noisy_ZI", outputs["ZI"]),
            ("noisy_ZU", outputs["ZU"]),
            ("clean_FI", outputs["FI_clean"]),
            ("clean_FU", outputs["FU_clean"]),
            ("clean_ZI", outputs["ZI_clean"]),
            ("clean_ZU", outputs["ZU_clean"]),
            *teachers,
        )
        for name, feature in features:
            bank.setdefault(name, []).append(feature.cpu())
        for key, value in values.items():
            collected.setdefault(key, []).append(value)
    result = {f"validation_{key}": float(np.average(values, weights=counts)) for key, values in collected.items()}
    for name, features in bank.items():
        feature = torch.cat(features)
        if feature.shape[0] >= 2:
            for metric, value in latent_diagnostics(feature).items():
                result[f"validation_{name}_{metric}"] = value
    result["validation_bank_samples"] = sum(counts)
    return result


def _jepa_validation_batch(runner, batch: dict[str, Any]):
    """The VICReg JEPA's validation terms on one bank batch."""
    outputs = runner(
        batch["image_noisy"], batch["imu_noisy_phys"], batch["image_clean"],
        batch["imu_clean_phys"], batch["image_time"], batch["imu_times"],
    )
    total, image, imu = jepa_latent_loss(
        outputs["prediction_i"], outputs["prediction_u"], outputs["target_i"], outputs["target_u"]
    )
    # No mask here: validation JEPA stays comparable with runs that never masked.
    values = {"jepa": float(total), "jepa_image": float(image), "jepa_imu": float(imu),
              **jepa_report_terms(outputs)}
    if "prediction_i_fine" in outputs:
        values["jepa_image_fine"] = float(jepa_fine_loss(outputs["prediction_i_fine"],
                                                         outputs["target_i_fine"]))
    return outputs, values, (("teacher_TI", outputs["target_i"]), ("teacher_TU", outputs["target_u"]))


def _latent_gate(
    reference: dict[str, float], current: dict[str, float], monitor: dict[str, Any],
    absolute_scale: bool = False,
) -> tuple[bool, list[str]]:
    """PASS unless diversity fell or the raw feature scale left its band.

    ``absolute_scale`` (model.encoder_norm: centre): the band bounds the raw RMS itself
    instead of its ratio to the initialisation. A norm that only subtracts the mean
    starts the encoders tiny -- FI RMS 1.9e-4 at init against 0.86 with GroupNorm,
    measured on TartanAir -- so p20's trained FI of ~3.4 read as x18000 and failed
    the gate three times while being as large as a GroupNorm feature.
    """
    relative_floor = float(monitor["relative_rank_std_warning"])
    raw_low, raw_high = (float(value) for value in monitor["raw_scale_ratio_warning"])
    reasons: list[str] = []
    diagnostic_keys = [
        key
        for key in reference
        if key.endswith(("same_position_std", "pooled_effective_rank", "raw_rms"))
    ]
    if not diagnostic_keys:
        return False, ["validation bank did not provide diversity diagnostics (need batch >= 2)"]
    for key in diagnostic_keys:
        if key not in current or not np.isfinite(current[key]):
            reasons.append(f"{key} missing or non-finite")
            continue
        denominator = max(abs(reference[key]), 1e-12)
        ratio = current[key] / denominator
        if key.endswith("raw_rms") and absolute_scale:
            if current[key] < raw_low or current[key] > raw_high:
                reasons.append(f"{key} scale {current[key]:.4g} outside [{raw_low},{raw_high}]")
        elif key.endswith("raw_rms"):
            if ratio < raw_low or ratio > raw_high:
                reasons.append(f"{key} scale ratio {ratio:.4g} outside [{raw_low},{raw_high}]")
        elif ratio < relative_floor:
            reasons.append(f"{key} diversity ratio {ratio:.4g} below {relative_floor}")
    return not reasons, reasons


# evaluate --protocol: (image mode, IMU mode) per scenario, in the order they run.
PROTOCOL_SCENARIOS = {
    "clean_clean": ("clean", "clean"),
    "noisy_image_clean_imu": ("full", "clean"),
    "clean_image_noisy_imu": ("clean", "full"),
    "noisy_noisy": ("full", "full"),
    "low_light_only": ("low_light_only", "clean"),
    "blur_only": ("blur_only", "clean"),
    "sensor_noise_only": ("sensor_noise_only", "clean"),
    "imu_white_noise_only": ("clean", "white_noise_only"),
    "imu_bias_only": ("clean", "bias_only"),
    "imu_bandwidth_only": ("clean", "bandwidth_only"),
}


@torch.no_grad()
def _evaluate_with_overlap(
    system: RestorationSystem,
    loader: DataLoader,
    dataset: PairedCameraImuDataset,
    device: torch.device,
    maximum_batches: int,
    smooth_l1_beta: float = 1.0,
    output: Path | None = None,
    panels: int = 6,
    label: str = "evaluation",
    forward_model: torch.nn.Module | None = None,
    progress: bool = False,
) -> dict[str, float | list[float] | int]:
    """Final metrics: each physical IMU timestamp is counted once after merging.

    ``progress`` prints a line about every 10% of the batches: a whole test split
    takes long, and a silent cell looks hung."""
    system.eval()
    trajectory_lengths: dict[str, int] = {}
    for sample in dataset.samples:
        trajectory_lengths[sample.trajectory_key] = max(
            trajectory_lengths.get(sample.trajectory_key, 0), sample.imu_end
        )
    predicted_mergers = {
        key: ImuOverlapMerger(length) for key, length in trajectory_lengths.items()
    }
    clean_mergers = {key: ImuOverlapMerger(length) for key, length in trajectory_lengths.items()}
    noisy_mergers = {key: ImuOverlapMerger(length) for key, length in trajectory_lengths.items()}
    trajectory_times = {
        key: np.full(length, np.nan, dtype=np.float64) for key, length in trajectory_lengths.items()
    }
    image_values: dict[str, list[float]] = {}
    # PSNR in and out by what the corruption did to the light (corruptions.image.frame_kind): one mean over a
    # mixed bank hid that p33 improved dark and flared frames while ruining plain ones by 9 dB.
    kind_values: dict[str, list[tuple[float, float]]] = {}
    frame_rows = []
    frame_index = 0
    estimated_frames = min(len(dataset.samples), maximum_batches * getattr(loader, "batch_size", 1))
    panel_indices = set(np.linspace(0, max(0, estimated_frames - 1), min(panels, estimated_frames), dtype=int))
    total_batches = min(maximum_batches, len(loader))
    report_every = max(1, total_batches // 10)
    started = time.perf_counter()
    for batch_index, raw in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        if progress and batch_index and batch_index % report_every == 0:
            print(f"  {label}: {frame_index}/{estimated_frames} ảnh ({100 * batch_index // total_batches}%)"
                  f" · {time.perf_counter() - started:.0f} s", flush=True)
        batch = _to_device(raw, device)
        inputs = (batch["image_noisy"], batch["imu_noisy_phys"], batch["image_time"], batch["imu_times"])
        if forward_model is None:
            native = system(*inputs)
            restored = {"image": native.image, "imu_physical": native.imu_physical}
        else:
            restored = forward_model(*inputs)
        for row in range(restored["image"].shape[0]):
            metrics = image_metrics(restored["image"][row : row + 1], batch["image_clean"][row : row + 1])
            metrics.update({f"baseline_{key}": value for key, value in image_metrics(
                batch["image_noisy"][row : row + 1], batch["image_clean"][row : row + 1]
            ).items()})
            for key, value in metrics.items():
                image_values.setdefault(key, []).append(value)
            if "corruption" in raw:
                kind = frame_kind(raw["corruption"][row]["image"])
                kind_values.setdefault(kind, []).append((metrics["baseline_image_psnr_db"], metrics["image_psnr_db"]))
            trajectory = raw["trajectory_key"][row]
            if output is not None:
                sample_id = raw.get("sample_id", [str(frame_index)] * restored["image"].shape[0])[row]
                frame_rows.append({"sample_id": sample_id, "trajectory": trajectory, **metrics})
                if frame_index in panel_indices:
                    arrays = [tensor[row].detach().cpu().permute(1, 2, 0).numpy()
                              for tensor in (batch["image_clean"], batch["image_noisy"], restored["image"])]
                    image_panel(output / "images" / f"frame_{frame_index:06d}.png", *arrays,
                                f"{label} | {sample_id} | PSNR {metrics['baseline_image_psnr_db']:.2f} → {metrics['image_psnr_db']:.2f} dB")
            frame_index += 1
            start = int(raw["imu_start"][row])
            prediction = restored["imu_physical"][row].transpose(0, 1).cpu().numpy()
            clean = batch["imu_clean_phys"][row].transpose(0, 1).cpu().numpy()
            times = raw["imu_times"][row].cpu().numpy()
            predicted_mergers[trajectory].add(start, prediction)
            clean_mergers[trajectory].add(start, clean)
            noisy_mergers[trajectory].add(start, batch["imu_noisy_phys"][row].transpose(0, 1).cpu().numpy())
            end = start + len(times)
            existing = trajectory_times[trajectory][start:end]
            conflict = np.isfinite(existing) & ~np.isclose(existing, times, rtol=0, atol=1e-9)
            if conflict.any():
                raise ValueError(f"Inconsistent overlapping IMU timestamps in {trajectory}")
            trajectory_times[trajectory][start:end] = times

    errors, baseline_errors, variation_errors, baseline_variations = [], [], [], []
    imu_files = []
    covered_rows = 0
    for trajectory in trajectory_lengths:
        predicted, predicted_coverage = predicted_mergers[trajectory].result()
        clean, clean_coverage = clean_mergers[trajectory].result()
        noisy, noisy_coverage = noisy_mergers[trajectory].result()
        coverage = predicted_coverage & clean_coverage & noisy_coverage & np.isfinite(trajectory_times[trajectory])
        covered_rows += int(coverage.sum())
        errors.append(predicted[coverage] - clean[coverage])
        baseline_errors.append(noisy[coverage] - clean[coverage])
        if output is not None and coverage.any():
            stem = save_imu_result(output / "imu", trajectory, trajectory_times[trajectory],
                                   clean, noisy, predicted, coverage, make_plot=len(imu_files) < panels, label=label)
            imu_files.append({"trajectory": trajectory, "file": f"imu/{stem}.npz", "covered_rows": int(coverage.sum())})
        adjacent = coverage[:-1] & coverage[1:]
        if adjacent.any():
            dt = np.diff(trajectory_times[trajectory])[adjacent]
            if not np.isfinite(dt).all() or (dt <= 0).any():
                raise ValueError(f"Invalid IMU time differences in {trajectory}")
            predicted_rate = np.diff(predicted, axis=0)[adjacent] / dt[:, None]
            clean_rate = np.diff(clean, axis=0)[adjacent] / dt[:, None]
            variation_errors.append(predicted_rate - clean_rate)
            baseline_variations.append(np.diff(noisy, axis=0)[adjacent] / dt[:, None] - clean_rate)
    if not errors or covered_rows == 0:
        raise ValueError("Evaluation produced no covered IMU timestamps")
    error = np.concatenate(errors, axis=0)
    baseline_error = np.concatenate(baseline_errors, axis=0)
    rmse_axis = np.sqrt(np.mean(np.square(error), axis=0))
    mae_axis = np.mean(np.abs(error), axis=0)
    bias_axis = np.mean(error, axis=0)
    result: dict[str, float | list[float] | int] = {
        key: float(np.mean(value)) for key, value in image_values.items()
    }
    for kind, pairs in kind_values.items():
        before, after = np.array(pairs).T
        result.update({f"{kind}_baseline_image_psnr_db": float(before.mean()), f"{kind}_image_psnr_db": float(after.mean()),
                       f"{kind}_count": len(pairs), f"{kind}_worse_fraction": float((after < before).mean())})
    result.update(
        {
            "imu_covered_unique_rows": covered_rows,
            "image_count": frame_index,
            "imu_rmse_axis": rmse_axis.tolist(),
            "imu_mae_axis": mae_axis.tolist(),
            "imu_bias_axis": bias_axis.tolist(),
            "accel_rmse": float(np.sqrt(np.mean(np.square(error[:, :3])))),
            "gyro_rmse": float(np.sqrt(np.mean(np.square(error[:, 3:])))),
        }
    )
    result.update({
        "baseline_imu_rmse_axis": np.sqrt(np.square(baseline_error).mean(axis=0)).tolist(),
        "baseline_imu_mae_axis": np.abs(baseline_error).mean(axis=0).tolist(),
        "baseline_imu_bias_axis": baseline_error.mean(axis=0).tolist(),
        "baseline_accel_rmse": float(np.sqrt(np.square(baseline_error[:, :3]).mean())),
        "baseline_gyro_rmse": float(np.sqrt(np.square(baseline_error[:, 3:]).mean())),
    })
    scale = system.normalizer.scale.detach().cpu().numpy()
    normalized_absolute = np.abs(error / scale[None, :])
    smooth = np.where(
        normalized_absolute < smooth_l1_beta,
        0.5 * np.square(normalized_absolute) / smooth_l1_beta,
        normalized_absolute - 0.5 * smooth_l1_beta,
    )
    accel_smooth = float(smooth[:, :3].mean())
    gyro_smooth = float(smooth[:, 3:].mean())
    result["joint_validation_score"] = float(result["image_mae"]) + 0.5 * (
        accel_smooth + gyro_smooth
    )
    result["imu_accel_smooth_l1_normalized"] = accel_smooth
    result["imu_gyro_smooth_l1_normalized"] = gyro_smooth
    if variation_errors:
        variation = np.concatenate(variation_errors, axis=0)
        result["accel_variation_rmse"] = float(np.sqrt(np.mean(np.square(variation[:, :3]))))
        result["gyro_variation_rmse"] = float(np.sqrt(np.mean(np.square(variation[:, 3:]))))
        baseline_variation = np.concatenate(baseline_variations, axis=0)
        result["baseline_accel_variation_rmse"] = float(np.sqrt(np.square(baseline_variation[:, :3]).mean()))
        result["baseline_gyro_variation_rmse"] = float(np.sqrt(np.square(baseline_variation[:, 3:]).mean()))
    if output is not None:
        write_csv(output / "per_frame.csv", frame_rows)
        write_json(output / "imu_index.json", imu_files)
    return result


def command_build_manifest(args: argparse.Namespace) -> None:
    config = load_config(_config_path(args.config))
    root = args.data_root or config["data"].get("root")
    if not root:
        raise ValueError("Pass --data-root or set data.root")
    manifest = build_manifest(root, window=config["data"]["imu_window"], seed=config["data"]["corruption_seed"],
                              split_rule=config["data"].get("split_rule", "hash"))
    write_manifest(manifest, args.output)
    print(json.dumps(manifest["meta"]["samples_per_split"], indent=2))


def command_train_phase1(args: argparse.Namespace) -> None:
    config = load_config(_config_path(args.config))
    manifest, _ = _manifest(config, args.manifest)
    device = resolve_device(args.device or config["runtime"]["device"])
    rank, world = rank_and_world()
    ranks = _ddp_world(config, args, device)
    if ranks > 1 and world == 1:
        # runtime.parallel ddp: one process per GPU. The ranks gather their features,
        # so VICReg, the coding rate and InfoNCE still see the whole batch.
        spawn_ranks(command_train_phase1, args, ranks, cuda=device.type == "cuda")
        return
    lead = rank == 0                              # validates, logs and saves
    if world > 1:
        args.gpus = "1"                           # each rank sees only its own GPU
        config["data"]["num_workers"] = -(-int(config["data"]["num_workers"]) // world)
    execution = _configure_execution(config, args, device)
    seed_everything(config["phase1"]["initialization_seed"])
    normalizer = build_normalizer(manifest["meta"])
    model = build_phase1_model(config, normalizer)
    validation_dataset = _dataset(config, manifest, "valid", fixed_realization=True)
    _fixed_validation_bank(validation_dataset, config["monitor"].get("validation_bank_size", 64))
    validation_loader = _loader(config, validation_dataset, config["phase1"]["batch_size"], train=False)
    if config["model"].get("encoder_norm_calibration", False) and not args.resume:
        # Fresh runs only, on the whole fixed validation bank -- the same on every rank, and what
        # the latent gate measures. Eight samples misjudge the IMU, whose scale varies by window.
        # A resumed run loads its calibrated gains from the checkpoint.
        bank = list(validation_loader)
        measured = model.calibrate_centre_norms(
            *(torch.cat([b[key] for b in bank]) for key in ("image_noisy", "imu_noisy_phys",
                                                            "image_clean", "imu_clean_phys")))
        if lead:
            print("centre norm calibration | RMS before, first -> last layer:"
                  f" image {measured['image'][0]:.2e} -> {measured['image'][-1]:.2e}"
                  f" | IMU {measured['imu'][0]:.2e} -> {measured['imu'][-1]:.2e}")
    if config["model"].get("vit_input_standardize") is not None and not args.resume:
        # Fresh runs only, on the fixed validation bank, the same on every rank; a resumed run
        # loads its gains from the checkpoint.
        bank = list(validation_loader)
        measured = model.calibrate_vit_inputs(
            *(torch.cat([b[key] for b in bank]) for key in ("image_noisy", "imu_noisy_phys",
                                                            "image_clean", "imu_clean_phys")))
        if lead:
            image_rms, imu_rms = measured["image"], measured["imu"]
            print("vit input standardization | coefficient RMS per channel:"
                  f" image {min(image_rms):.3f}-{max(image_rms):.3f} | IMU {min(imu_rms):.3f}-{max(imu_rms):.3f}")
    trainer = _phase1_trainer(model, config, device, manifest["meta"]["manifest_hash"])
    resume_payload = None
    if args.resume:
        resume_payload = load_checkpoint(args.resume, device)
        require_phase1_checkpoint(resume_payload)
        if resume_payload["metadata"].get("manifest_hash") != manifest["meta"]["manifest_hash"]:
            raise ValueError("Phase-1 resume manifest differs from training data")
        if resume_payload["metadata"].get("normalizer_hash") != state_dict_hash(normalizer):
            raise ValueError("Phase-1 resume normalization differs from training data")
        if resume_payload["metadata"].get("configuration_hash") != configuration_hash(config, "phase1"):
            raise ValueError("Phase-1 resume config changes training semantics")
        model.load_state_dict(resume_payload["model"], strict=True)
        trainer.optimizer.load_state_dict(resume_payload["optimizer"])
        trainer.successful_updates = int(resume_payload["successful_updates"])
        if resume_payload["metadata"].get("data_microbatches_consumed") != trainer.successful_updates:
            raise ValueError("Phase-1 resume checkpoint has inconsistent data progress")
        trainer.initialization_hash = resume_payload["metadata"]["initialization_hash"]
        restore_rng_state(resume_payload["rng"])

    train_dataset = _train_dataset(config, manifest, "phase1")
    batches = _training_batch_stream(
        config,
        train_dataset,
        config["phase1"]["batch_size"],
        start_microbatch=trainer.successful_updates,
        namespace="phase1",
        rank=rank,
        world=world,
    )
    output = Path(args.output or config["runtime"]["output_dir"]) / "phase1"
    # Chi rank 0: mot loi o day phai toi duoc rank dang cho share_rank0_rng ben duoi.
    with rank0_section():
        if lead:
            _prepare_run(output, args.resume, trainer.successful_updates)
            _write_resolved(config, output)
            write_json(output / "execution.json", execution)
            write_json(output / "validation_bank.json", [sample.sample_id for sample in validation_dataset.samples])
        log = _Jsonl(output / "train.jsonl" if lead else None)
        if resume_payload is None:
            reference = _validate_latent(
                model, validation_loader, device, len(validation_loader), trainer.evaluation_model
            ) if lead else {}
            warning_checks = 0
            log.write({"event": "initialization_reference", **reference})
        else:
            saved_metrics = resume_payload.get("latent_metrics", {})
            if "reference" not in saved_metrics:
                raise ValueError("Resume checkpoint lacks the initialization latent reference")
            reference = saved_metrics["reference"]
            warning_checks = int(saved_metrics.get("consecutive_warning_checks", 0))
    share_rank0_rng()                             # the reference drew on rank 0 only
    maximum = config["phase1"]["max_successful_updates"]
    checkpoint_every = config["runtime"]["checkpoint_every_updates"]
    started, first_update, waited = time.perf_counter(), trainer.successful_updates, 0.0
    recent = _RecentMean()
    while trainer.successful_updates < maximum:
        batch, wait = _next_timed(batches)
        waited += wait
        metrics = trainer.step(batch)
        metrics["data_wait_seconds"] = wait
        log.write(metrics)
        if metrics.get("skipped"):
            raise FloatingPointError(f"Phase-1 update skipped: {metrics}")
        recent.add(metrics, ("loss", "jepa"))
        update = trainer.successful_updates
        if update % config["runtime"]["log_every_updates"] == 0 or update == 1:
            anchor = f" recon={metrics['reconstruction']:.6f}" if "reconstruction" in metrics else ""
            print(
                f"phase1 update={_progress(update, maximum)} loss={metrics['loss']:.6f}"
                f" jepa={metrics['jepa']:.6f}{anchor} {recent.text(('loss', 'jepa'))}"
                f" {_pace(started, first_update, update, waited)}"
            )
        if update % checkpoint_every == 0 or update == maximum:
            if not lead:
                # Rank 0 validates, gates and saves; take its RNG and its restart decision.
                share_rank0_rng()
                _restart_if_memory_high(config, _memory_mib(), update, maximum)
                continue
            # Chi rank 0: gate FAIL hay loi validate/luu thi cac rank kia cung dung, khong cho 60 phut.
            with rank0_section():
                validation = _validate_latent(
                    model, validation_loader, device, len(validation_loader), trainer.evaluation_model
                )
                passed, gate_reasons = _latent_gate(
                    reference, validation, config["monitor"],
                    absolute_scale=config["model"].get("encoder_norm", "group") == "centre")
                warning_checks = 0 if passed else warning_checks + 1
                gate_status = "PASS" if passed else (
                    "FAIL" if warning_checks >= config["monitor"]["consecutive_warning_checks"] else "WARN"
                )
                memory = _memory_mib()
                log.write(
                    {
                        "successful_updates": update,
                        "latent_gate_status": gate_status,
                        "latent_gate_reasons": gate_reasons,
                        **memory,
                        **validation,
                    }
                )
                print(
                    f"  gate update={_progress(update, maximum)} {gate_status}"
                    f" | RSS {memory['rss_mib']:.0f} MiB"
                    f" + worker {memory['children_rss_mib']:.0f} MiB"
                )
                atomic_torch_save(
                    trainer.checkpoint_payload(
                        serializable_config(config),
                        latent_gate_status=gate_status,
                        latent_metrics={
                            "reference": reference,
                            "current": validation,
                            "gate_reasons": gate_reasons,
                            "consecutive_warning_checks": warning_checks,
                        },
                    ),
                    output / "last.pt",
                )
                if gate_status == "FAIL":
                    plot_training(output)
                    raise RuntimeError(
                        "Latent diversity/scale gate failed on consecutive checks; inspect phase1/train.jsonl"
                    )
            share_rank0_rng()
            _restart_if_memory_high(config, memory, update, maximum)
    if lead:
        plot_training(output)
        print(f"Saved phase-1 checkpoint and training_curves.png: {output}")


def _load_phase1_for_phase2(
    config: dict[str, Any], manifest: dict[str, Any], checkpoint: str, device: torch.device
) -> tuple[LatentPretrainingModel, dict[str, Any]]:
    payload = load_checkpoint(checkpoint, device)
    require_phase1_checkpoint(payload)
    if payload["metadata"].get("configuration_hash") != configuration_hash(config, "phase1"):
        raise ValueError("Current config does not match the phase-1 training contract")
    if payload["metadata"].get("latent_gate_status") != "PASS":
        raise ValueError(
            "Phase-1 checkpoint has not passed latent diversity/scale gates; phase 2 is blocked"
        )
    if payload["successful_updates"] != config["phase1"]["max_successful_updates"]:
        raise ValueError("Complete the configured phase-1 update budget before phase 2")
    if payload["metadata"].get("manifest_hash") != manifest["meta"]["manifest_hash"]:
        raise ValueError("Phase-1 checkpoint and current manifest hashes differ")
    model = build_phase1_model(config, build_normalizer(manifest["meta"])).to(device)
    expected_normalizer_hash = state_dict_hash(model.normalizer)
    model.load_state_dict(payload["model"], strict=True)
    if state_dict_hash(model.backbone) != payload["metadata"]["backbone_hash"]:
        raise ValueError("Loaded backbone hash differs from phase-1 checkpoint metadata")
    if state_dict_hash(model.normalizer) != expected_normalizer_hash:
        raise ValueError("Phase-1 normalizer does not match current train statistics")
    return model, payload


def command_train_phase2(args: argparse.Namespace) -> None:
    config = load_config(_config_path(args.config))
    init_checkpoint = getattr(args, "decoder_init_checkpoint", None)
    decoder_init_source = str(Path(init_checkpoint).resolve()) if init_checkpoint else None
    if args.resume and init_checkpoint:
        raise ValueError("Use --resume alone; decoder initialization is already in the checkpoint")
    if "full_guard" in config["phase2"] and not (args.resume or init_checkpoint):
        raise ValueError("phase2.full_guard requires --decoder-init-checkpoint on a new run")
    manifest, _ = _manifest(config, args.manifest)
    checkpoint = args.backbone_checkpoint or config["phase2"].get("backbone_checkpoint")
    if not checkpoint:
        raise ValueError("Pass --backbone-checkpoint from a completed v3 phase 1")
    device = resolve_device(args.device or config["runtime"]["device"])
    rank, world = rank_and_world()
    ranks = _ddp_world(config, args, device)
    if ranks > 1 and world == 1:
        # runtime.parallel ddp: this process only launches one process per GPU.
        spawn_ranks(command_train_phase2, args, ranks, cuda=device.type == "cuda")
        return
    lead = rank == 0                              # validates, logs and saves
    if world > 1:
        args.gpus = "1"                           # each rank sees only its own GPU
        # The 4 vCPUs are shared, and each rank loads only its share of a batch.
        config["data"]["num_workers"] = -(-int(config["data"]["num_workers"]) // world)
    execution = _configure_execution(config, args, device)
    phase1_model, parent_payload = _load_phase1_for_phase2(config, manifest, checkpoint, device)
    seed_everything(config["phase2"]["decoder_initialization_seed"])
    system = RestorationSystem(
        phase2_backbone(config, phase1_model), phase1_model.normalizer, build_decoders(config),
        # phase2.decoder_predictor_input: the trained phase-1 predictor stays, frozen.
        *phase2_latent_modules(config, phase1_model),
    )
    del phase1_model, parent_payload
    if init_checkpoint:
        # Only decoder weights are needed; do not allocate the old optimizer on GPU.
        initialized = load_checkpoint(init_checkpoint, "cpu")
        metadata = initialized.get("metadata", {})
        old_config = initialized.get("config", {})
        if metadata.get("phase") != "latent_decoder_train" or metadata.get("pipeline_version") != 3:
            raise ValueError("Decoder initialization must be a phase-2 checkpoint")
        if metadata.get("configuration_hash") != configuration_hash(old_config, "phase2"):
            raise ValueError("Decoder initialization config hash mismatch")
        if metadata.get("manifest_hash") != manifest["meta"]["manifest_hash"]:
            raise ValueError("Decoder initialization manifest differs")
        if configuration_hash(old_config, "phase1") != configuration_hash(config, "phase1"):
            raise ValueError("Decoder initialization uses a different phase-1 contract")
        if metadata.get("frozen_backbone_hash") != state_dict_hash(system.backbone):
            raise ValueError("Decoder initialization uses a different frozen backbone")
        if metadata.get("frozen_normalizer_hash") != state_dict_hash(system.normalizer):
            raise ValueError("Decoder initialization uses a different IMU normalizer")
        if metadata.get("backbone_finetuned"):
            # Its decoders learned on the fine-tuned backbone, which a new run does not load.
            raise ValueError("Decoder initialization comes from a run whose backbone was fine-tuned (LP-FT); "
                             "its decoders do not fit the phase-1 backbone")
        prefix = "decoders."
        decoder_state = {
            key[len(prefix):]: value for key, value in initialized["system"].items()
            if key.startswith(prefix)
        }
        system.decoders.load_state_dict(decoder_state, strict=True)
        if metadata.get("decoder_current_hash") != state_dict_hash(system.decoders):
            raise ValueError("Decoder initialization weights do not match checkpoint metadata")
        del initialized
    trainer = Phase2Trainer(
        system, config, device, str(Path(checkpoint).resolve()), manifest["meta"]["manifest_hash"]
    )
    best_validation = math.inf
    best_blur_score = math.inf
    best_guarded_score = math.inf
    guard_reference = None
    if args.resume:
        payload = load_checkpoint(args.resume, device)
        if payload.get("metadata", {}).get("phase") != "latent_decoder_train":
            raise ValueError("Resume checkpoint is not a phase-2 checkpoint")
        if payload["metadata"].get("configuration_hash") != configuration_hash(config, "phase2"):
            raise ValueError("Phase-2 resume config changes training semantics")
        if payload["metadata"].get("manifest_hash") != manifest["meta"]["manifest_hash"]:
            raise ValueError("Phase-2 resume manifest differs from training data")
        if payload["metadata"].get("frozen_backbone_hash") != trainer.frozen_backbone_hash:
            raise ValueError("Phase-2 resume checkpoint has a different phase-1 parent")
        # Before the optimizer state: past backbone_finetune_after_updates it has two groups.
        if payload["metadata"].get("latent_predictor_hash") != trainer.latent_predictor_hash:
            raise ValueError("Phase-2 resume checkpoint has a different phase-1 predictor")
        trainer.successful_updates = int(payload["successful_updates"])
        trainer.prepare_backbone_finetune()
        system.load_state_dict(payload["system"], strict=True)
        trainer.assert_backbone_frozen()             # also the predictor it just loaded
        trainer.decoder_initialization_hash = payload["metadata"]["decoder_initialization_hash"]
        trainer.optimizer.load_state_dict(payload["optimizer"])
        if trainer.amp and payload.get("scaler"):
            trainer.scaler.load_state_dict(payload["scaler"])
        expected_microbatches = trainer.successful_updates * config["phase2"]["gradient_accumulation"]
        if payload["metadata"].get("data_microbatches_consumed") != expected_microbatches:
            raise ValueError("Phase-2 resume checkpoint has inconsistent data progress")
        restore_rng_state(payload["rng"])
        best_validation = float(payload.get("best_joint_validation_score", math.inf))
        best_blur_score = float(payload.get("best_blur_validation_score", math.inf))
        best_guarded_score = float(payload.get("best_guarded_blur_score", math.inf))
        guard_reference = payload.get("guard_reference")
        decoder_init_source = payload.get("decoder_init_checkpoint")
        if "full_guard" in config["phase2"] and guard_reference is None:
            raise ValueError("Guarded phase-2 resume checkpoint lacks its initialization reference")

    train_dataset = _train_dataset(config, manifest, "phase2", scenarios=config["phase2"].get("train_scenarios"))
    validation_dataset = _dataset(config, manifest, "valid", fixed_realization=True)
    _fixed_validation_bank(validation_dataset, config["runtime"]["validation_batches"] * config["phase2"]["batch_size"])
    validation_loader = _loader(config, validation_dataset, config["phase2"]["batch_size"], train=False)
    blur_dataset = blur_loader = None
    if "blur_validation_samples" in config["phase2"]:
        blur_dataset = _dataset(config, manifest, "valid", fixed_realization=True,
                                image_mode="blur_only", imu_mode="clean")
        _fixed_validation_bank(blur_dataset, config["phase2"]["blur_validation_samples"])
        blur_loader = _loader(config, blur_dataset, config["phase2"]["batch_size"], train=False)
    batches = _training_batch_stream(
        config,
        train_dataset,
        config["phase2"]["batch_size"],
        start_microbatch=trainer.successful_updates * config["phase2"]["gradient_accumulation"],
        namespace="phase2",
        rank=rank,
        world=world,
    )
    output = Path(args.output or config["runtime"]["output_dir"]) / "phase2"
    # Chi rank 0: mot loi o day phai toi duoc rank dang cho share_rank0_rng ben duoi.
    with rank0_section():
        if lead:
            _prepare_run(output, args.resume, trainer.successful_updates)
        if lead and "full_guard" in config["phase2"] and guard_reference is None:
            if blur_loader is None:
                raise ValueError("Guarded phase 2 needs blur validation")
            guard_reference = {
                "full": _evaluate_with_overlap(
                    system, validation_loader, validation_dataset, device,
                    config["runtime"]["validation_batches"],
                    config["phase2"]["smooth_l1_beta"],
                    forward_model=trainer.evaluation_model,
                ),
                "blur": _validate_active_blur(system, blur_loader, device,
                                                forward_model=trainer.evaluation_model),
            }
            print(
                "  guarded init reference"
                f" | full PSNR {guard_reference['full']['image_psnr_db']:.2f}"
                f" | SSIM {guard_reference['full']['image_ssim']:.3f}"
                f" | accel {guard_reference['full']['accel_rmse']:.3f}"
                f" | gyro {guard_reference['full']['gyro_rmse']:.3f}"
                f" | blur MAE {guard_reference['blur']['image_mae_restored']:.5f}"
                f" | edge {guard_reference['blur']['strong_edge_gradient_mae_restored']:.5f}"
            )
        if lead:
            _write_resolved(config, output)
            write_json(output / "execution.json", execution)
            write_json(output / "validation_bank.json", [sample.sample_id for sample in validation_dataset.samples])
            if blur_dataset is not None:
                write_json(output / "blur_validation_bank.json", [sample.sample_id for sample in blur_dataset.samples])
            if guard_reference is not None:
                write_json(output / "guard_reference.json", guard_reference)
        log = _Jsonl(output / "train.jsonl" if lead else None)
        maximum = config["phase2"]["max_successful_updates"]
        checkpoint_every = config["runtime"]["checkpoint_every_updates"]
        if lead and args.resume and math.isfinite(best_validation) and not (output / "best_joint_validation.pt").exists():
            prior_best = Path(args.resume).parent / "best_joint_validation.pt"
            if not prior_best.exists():
                raise ValueError("Resume needs best_joint_validation.pt beside last.pt; copy the complete phase2 folder")
            import shutil
            shutil.copy2(prior_best, output / "best_joint_validation.pt")
        if lead and args.resume and math.isfinite(best_blur_score) and not (output / "best_blur_validation.pt").exists():
            prior_best_blur = Path(args.resume).parent / "best_blur_validation.pt"
            if not prior_best_blur.exists():
                raise ValueError("Resume needs best_blur_validation.pt beside last.pt")
            import shutil
            shutil.copy2(prior_best_blur, output / "best_blur_validation.pt")
        if lead and args.resume and math.isfinite(best_guarded_score) and not (output / "best_guarded_validation.pt").exists():
            prior_best_guarded = Path(args.resume).parent / "best_guarded_validation.pt"
            if not prior_best_guarded.exists():
                raise ValueError("Resume needs best_guarded_validation.pt beside last.pt")
            import shutil
            shutil.copy2(prior_best_guarded, output / "best_guarded_validation.pt")
    share_rank0_rng()                             # the guard reference drew on rank 0 only
    started, first_update, waited = time.perf_counter(), trainer.successful_updates, 0.0
    recent = _RecentMean()
    while trainer.successful_updates < maximum:
        timed = [_next_timed(batches) for _ in range(config["phase2"]["gradient_accumulation"])]
        wait = sum(seconds for _, seconds in timed)
        waited += wait
        metrics = trainer.step([batch for batch, _ in timed])
        metrics["data_wait_seconds"] = wait
        log.write(metrics)
        if metrics.get("skipped"):
            raise FloatingPointError(f"Phase-2 update skipped: {metrics}")
        recent.add(metrics, ("loss", "image_l1"))
        update = trainer.successful_updates
        if update % config["runtime"]["log_every_updates"] == 0 or update == 1:
            print(
                f"phase2 update={_progress(update, maximum)} loss={metrics['loss']:.6f}"
                f" image_l1={metrics['image_l1']:.6f} {recent.text(('loss', 'image_l1'))}"
                f" {_pace(started, first_update, update, waited)}"
            )
        if update % checkpoint_every == 0 or update == maximum:
            if not lead:
                # Rank 0 validates and saves; take its RNG and its restart decision.
                share_rank0_rng()
                _restart_if_memory_high(config, _memory_mib(), update, maximum)
                continue
            # Chi rank 0: loi validate/luu thi cac rank kia cung dung, khong cho 60 phut.
            with rank0_section():
                trainer.assert_backbone_frozen()
                evaluation = _evaluate_with_overlap(
                    system,
                    validation_loader,
                    validation_dataset,
                    device,
                    config["runtime"]["validation_batches"],
                    config["phase2"]["smooth_l1_beta"],
                    forward_model=trainer.evaluation_model,
                )
                validation = {f"validation_{key}": value for key, value in evaluation.items()}
                blur_validation = {}
                blur_score = math.inf
                if blur_loader is not None:
                    blur = _validate_active_blur(system, blur_loader, device,
                                                  forward_model=trainer.evaluation_model)
                    blur_validation = {f"blur_validation_{key}": value for key, value in blur.items()}
                    blur_score = max(
                        float(blur["image_mae_restored"]) / max(float(blur["image_mae_input"]), 1e-12),
                        float(blur["strong_edge_gradient_mae_restored"]) /
                        max(float(blur["strong_edge_gradient_mae_input"]), 1e-12),
                    )
                guard_pass = False
                guard_failures: list[str] = []
                if guard_reference is not None:
                    guard_pass, guard_failures = _phase2_full_blur_guard(
                        evaluation, blur, guard_reference, config["phase2"]["full_guard"]
                    )
                memory = _memory_mib()
                record = {"successful_updates": update, **memory, **validation, **blur_validation}
                if blur_loader is not None:
                    record["blur_validation_worst_ratio"] = blur_score
                if guard_reference is not None:
                    record["guard_pass"] = guard_pass
                    record["guard_failures"] = guard_failures
                log.write(record)
                # Validation la thu duy nhat tra loi "model co hoat dong khong"; no
                # chay 48 lan trong mot run nen phai nhin thay duoc, khong chi nam
                # trong train.jsonl ma kernel dang bi chan khong doc duoc.
                beats = (
                    validation["validation_image_psnr_db"] > validation["validation_baseline_image_psnr_db"]
                    and validation["validation_image_ssim"] > validation["validation_baseline_image_ssim"]
                    and validation["validation_accel_rmse"] < validation["validation_baseline_accel_rmse"]
                )
                print(
                    f"  validation update={_progress(update, maximum)}"
                    f" | PSNR {validation['validation_image_psnr_db']:.2f}"
                    f" vs {validation['validation_baseline_image_psnr_db']:.2f}"
                    f" | SSIM {validation['validation_image_ssim']:.3f}"
                    f" vs {validation['validation_baseline_image_ssim']:.3f}"
                    f" | accel {validation['validation_accel_rmse']:.3f}"
                    f" vs {validation['validation_baseline_accel_rmse']:.3f}"
                    f" | {'VUOT baseline' if beats else 'chua vuot'}"
                    f" | RSS {memory['rss_mib']:.0f}+{memory['children_rss_mib']:.0f} MiB"
                )
                kinds = kind_summary(validation, "validation_")
                if kinds:
                    print(f"  theo loai anh: {kinds}")
                if blur_validation:
                    print(
                        f"  blur actual={blur_validation['blur_validation_active_frames']}"
                        f" | MAE {blur_validation['blur_validation_image_mae_input']:.5f}"
                        f" -> {blur_validation['blur_validation_image_mae_restored']:.5f}"
                        f" | edge {blur_validation['blur_validation_strong_edge_gradient_mae_input']:.5f}"
                        f" -> {blur_validation['blur_validation_strong_edge_gradient_mae_restored']:.5f}"
                        f" | worst ratio {blur_score:.3f}"
                    )
                if guard_reference is not None:
                    print(f"  full+blur guard: {'PASS' if guard_pass else 'FAIL'}"
                          f" | {', '.join(guard_failures) if guard_failures else 'all checks passed'}")
                payload = trainer.checkpoint_payload(serializable_config(config))
                improved = validation["validation_joint_validation_score"] < best_validation
                if improved:
                    best_validation = float(validation["validation_joint_validation_score"])
                improved_blur = blur_score < best_blur_score
                if improved_blur:
                    best_blur_score = blur_score
                improved_guarded = guard_pass and blur_score < best_guarded_score
                if improved_guarded:
                    best_guarded_score = blur_score
                payload["best_joint_validation_score"] = best_validation
                if blur_loader is not None:
                    payload["best_blur_validation_score"] = best_blur_score
                    payload["blur_validation_metrics"] = blur_validation
                if guard_reference is not None:
                    payload["guard_reference"] = guard_reference
                    payload["best_guarded_blur_score"] = best_guarded_score
                    payload["guard_pass"] = guard_pass
                    payload["guard_failures"] = guard_failures
                if decoder_init_source is not None:
                    payload["decoder_init_checkpoint"] = decoder_init_source
                payload["validation_metrics"] = validation
                if improved:
                    atomic_torch_save(payload, output / "best_joint_validation.pt")
                if improved_blur:
                    atomic_torch_save(payload, output / "best_blur_validation.pt")
                if improved_guarded:
                    atomic_torch_save(payload, output / "best_guarded_validation.pt")
                atomic_torch_save(payload, output / "last.pt")
            share_rank0_rng()
            _restart_if_memory_high(config, memory, update, maximum)
    if lead:
        plot_training(output)
        print(f"Saved phase-2 checkpoints and training_curves.png: {output}")


def _system_from_phase2(checkpoint: str, device: torch.device) -> tuple[RestorationSystem, dict[str, Any]]:
    payload = load_checkpoint(checkpoint, device)
    metadata = payload.get("metadata", {})
    if metadata.get("pipeline_version") != 3 or metadata.get("phase") != "latent_decoder_train":
        raise ValueError("Checkpoint is not a QWT-JEPA v3 phase-2 checkpoint")
    config = payload["config"]
    if metadata.get("configuration_hash") != configuration_hash(config, "phase2"):
        raise ValueError("Phase-2 checkpoint config hash mismatch")
    backbone = build_backbone(config)
    system = RestorationSystem(backbone, ImuNormalizer(), build_decoders(config),
                               *phase2_latent_modules(config)).to(device)
    system.load_state_dict(payload["system"], strict=True)
    system.freeze_backbone()
    # After LP-FT the backbone is the fine-tuned one; frozen_backbone_hash names its parent.
    expected_backbone = (metadata["backbone_current_hash"] if metadata.get("backbone_finetuned")
                         else metadata["frozen_backbone_hash"])
    if state_dict_hash(system.backbone) != expected_backbone:
        raise ValueError("Phase-2 frozen backbone hash mismatch")
    if state_dict_hash(system.normalizer) != metadata["frozen_normalizer_hash"]:
        raise ValueError("Phase-2 frozen normalizer hash mismatch")
    if latent_predictor_hash(system) != metadata.get("latent_predictor_hash"):
        raise ValueError("Phase-2 latent predictor hash mismatch")
    if state_dict_hash(system.decoders) != metadata["decoder_current_hash"]:
        raise ValueError("Phase-2 decoder hash mismatch")
    system.eval()
    return system, config


def command_evaluate(args: argparse.Namespace) -> None:
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("--max-batches must be positive; omit it for the full split")
    if args.every < 1:
        raise ValueError("--every must be at least 1 (1 = every frame)")
    if args.panels < 0:
        raise ValueError("--panels cannot be negative")
    if args.scenarios is not None:
        if not args.protocol:
            raise ValueError("--scenarios picks among the --protocol scenarios; add --protocol")
        picked = [name.strip() for name in args.scenarios.split(",") if name.strip()]
        unknown = sorted(set(picked) - set(PROTOCOL_SCENARIOS))
        if not picked or unknown:
            raise ValueError(f"--scenarios: unknown {unknown}; choose from {list(PROTOCOL_SCENARIOS)}")
    device = resolve_device(args.device or "auto")
    system, config = _system_from_phase2(args.checkpoint, device)
    execution = _configure_execution(config, args, device, use_saved_setting=False)
    forward = RestorationForward(system)
    # --amp: fp16 autocast as phase 2 trained with it; outputs and metrics stay fp32.
    forward.amp = bool(args.amp) and device.type == "cuda"
    runner, _ = parallel_forward(forward, device, config["runtime"]["gpu_count"])
    manifest, _ = _manifest(config, args.manifest)
    if args.protocol:
        # In the protocol's order, whatever order --scenarios names them in.
        scenarios = {name: modes for name, modes in PROTOCOL_SCENARIOS.items()
                     if args.scenarios is None or name in picked}
    else:
        scenarios = {"requested": (args.image_mode, args.imu_mode)}
    results = {}
    output = Path(args.output or Path(args.checkpoint).parent / f"evaluation_{args.split}")
    if (output / "metrics.json").exists():
        raise ValueError(f"Evaluation output already exists: {output}; choose a new --output")
    write_json(output / "evaluation_config.json", {
        "checkpoint": str(Path(args.checkpoint).resolve()), "split": args.split,
        "manifest_hash": manifest["meta"]["manifest_hash"], "config": serializable_config(config),
        "scenarios": scenarios, "max_batches": args.max_batches, "every": args.every, "amp": forward.amp,
        "validation_realization": config["data"]["validation_realization"],
        "evaluation_clean_probability": 0.0,
        "execution": execution,
    })
    for number, (name, (image_mode, imu_mode)) in enumerate(scenarios.items(), start=1):
        dataset = _dataset(
            config,
            manifest,
            args.split,
            fixed_realization=True,
            image_mode=image_mode,
            imu_mode=imu_mode,
            full_frame=bool(getattr(args, "full_frame", False)),
        )
        if args.every > 1:
            _every_nth_frame(dataset, args.every)
        loader = _loader(config, dataset, config["phase2"]["batch_size"], train=False)
        print(f"[{number}/{len(scenarios)}] {name}: ảnh {image_mode}, IMU {imu_mode} · {len(dataset)} mẫu"
              + (f" (1/{args.every} frame mỗi trajectory)" if args.every > 1 else ""), flush=True)
        results[name] = _evaluate_with_overlap(
            system,
            loader,
            dataset,
            device,
            args.max_batches or len(loader),
            config["phase2"]["smooth_l1_beta"],
            output=output / name, panels=args.panels,
            label=f"{config.get('run_kind', 'main').upper()} | {args.split} | {name}",
            forward_model=runner,
            progress=True,
        )
        if kind_summary(results[name]):
            print(f"  {name} theo loai anh: {kind_summary(results[name])}", flush=True)
    scope = f"max-batches={args.max_batches}" if args.max_batches else (
        f"1/{args.every} frame" if args.every > 1 else "full split")
    evaluation_summary(output, results, label=f"{config.get('run_kind', 'main').upper()} | {args.split} | {scope}")
    print(json.dumps(results, indent=2))
    print(f"Saved metrics, panels, IMU arrays and comparison.png to {output}")


def _read_imu_csv(path: str, length: int) -> tuple[np.ndarray, np.ndarray | None]:
    first = Path(path).read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
    headers = {"ax,ay,az,gx,gy,gz", "timestamp,ax,ay,az,gx,gy,gz"}
    values = np.loadtxt(path, delimiter=",", ndmin=2, skiprows=int(first in headers))
    if not np.isfinite(values).all():
        raise ValueError("IMU CSV contains NaN/Inf")
    if values.shape == (length, 6):
        return values.astype(np.float32), None
    if values.shape == (length, 7):
        return values[:, 1:].astype(np.float32), values[:, 0].astype(np.float64)
    raise ValueError(f"Expected IMU CSV [{length},6] or [{length},7], got {values.shape}")


def command_infer(args: argparse.Namespace) -> None:
    device = resolve_device(args.device or "auto")
    system, config = _system_from_phase2(args.checkpoint, device)
    # data.source_size: the model trained on crops of frames this size and runs on whole ones.
    image = load_rgb(args.image, tuple(config["data"].get("source_size") or config["data"]["image_size"]))
    imu, timestamps = _read_imu_csv(args.imu, config["data"]["imu_window"])
    if timestamps is None:
        timestamps = np.arange(len(imu), dtype=np.float64) * args.imu_dt
    image_time = args.image_time if args.image_time is not None else 0.5 * (timestamps[0] + timestamps[-1])
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    imu_tensor = torch.from_numpy(imu.T).unsqueeze(0).float().to(device)
    with torch.no_grad():
        restored = system(
            image_tensor,
            imu_tensor,
            torch.tensor([image_time], dtype=torch.float64, device=device),
            torch.from_numpy(timestamps).unsqueeze(0).to(device),
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    image_out = restored.image[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.uint8(image_out * 255.0), mode="RGB").save(output / "image_restored.png")
    imu_out = restored.imu_physical[0].transpose(0, 1).cpu().numpy()
    np.savetxt(
        output / "imu_restored.csv",
        np.column_stack((timestamps, imu_out)),
        delimiter=",",
        header="timestamp,ax,ay,az,gx,gy,gz",
        comments="",
    )
    (output / "metadata.json").write_text(
        json.dumps({"checkpoint": str(Path(args.checkpoint).resolve()), "image_time": image_time}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved restored image and IMU to {output}")


def command_preview_corruption(args: argparse.Namespace) -> None:
    config = load_config(_config_path(args.config))
    image_corruptor, _ = build_corruptors(config)
    clean = load_rgb(args.image, tuple(config["data"]["image_size"]))
    noisy, parameters = image_corruptor(
        clean,
        split="preview",
        realization=args.realization,
        trajectory="preview",
        timestamp=0.0,
        frame_index=0,
        mode=args.mode,
    )
    panel = np.concatenate((clean, noisy, np.abs(clean - noisy)), axis=1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(panel, 0, 1) * 255.0), mode="RGB").save(output)
    output.with_suffix(".json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    print(f"Saved clean | corrupted | absolute error panel to {output}")


def _synthetic_batch(config: dict[str, Any]) -> dict[str, Any]:
    batch_size = config["phase1"]["batch_size"]
    height, width = config["data"]["image_size"]
    length = config["data"]["imu_window"]
    yy, xx = np.mgrid[0:height, 0:width]
    image_corruptor, imu_corruptor = build_corruptors(config)
    clean_images, noisy_images, clean_imus, noisy_imus = [], [], [], []
    noise_free_images, degradations, stop_maps = [], [], []
    timestamps = np.arange(length, dtype=np.float64) * 0.01
    for sample in range(batch_size):
        clean = np.stack(
            (
                (xx + sample) / max(1, width + batch_size - 1),
                (yy + 2 * sample) / max(1, height + 2 * batch_size - 1),
                0.5 + 0.25 * np.sin((xx + yy + sample) / 4.0),
            ),
            axis=-1,
        ).clip(0, 1).astype(np.float32)
        time = timestamps
        imu_clean = np.stack(
            [np.sin(time * (axis + 1) + sample * 0.2) for axis in range(6)], axis=-1
        ).astype(np.float32)
        noisy_image, noise_free, image_parameters = image_corruptor.render_with_sensor_reference(
            clean,
            split="smoke",
            realization=0,
            trajectory=f"synthetic/{sample}",
            timestamp=float(time.mean()),
            frame_index=sample,
            # Same contract as the real dataset: the clean gyro drives the blur.
            gyro=imu_clean[:, 3:6].astype(np.float64),
            imu_times=time,
        )
        noise_free_images.append(torch.from_numpy(noise_free.transpose(2, 0, 1)))
        degradations.append(torch.from_numpy(degradation_vector(image_parameters)))
        stop_maps.append(torch.from_numpy(brightness_stops(image_parameters, height, width))[None])
        noisy_imu, _ = imu_corruptor.window(
            imu_clean,
            time,
            0,
            length,
            split="smoke",
            realization=0,
            trajectory=f"synthetic/{sample}",
        )
        clean_images.append(torch.from_numpy(clean.transpose(2, 0, 1)))
        noisy_images.append(torch.from_numpy(noisy_image.transpose(2, 0, 1)))
        clean_imus.append(torch.from_numpy(imu_clean.T))
        noisy_imus.append(torch.from_numpy(noisy_imu.T))
    return {
        "image_clean": torch.stack(clean_images).float(),
        "image_noisy": torch.stack(noisy_images).float(),
        "imu_clean_phys": torch.stack(clean_imus).float(),
        "imu_noisy_phys": torch.stack(noisy_imus).float(),
        "image_time": torch.full((batch_size,), float(timestamps.mean())),
        "imu_times": torch.from_numpy(timestamps).float().repeat(batch_size, 1),
        "image_noise_free": torch.stack(noise_free_images).float(),
        "image_degradation": torch.stack(degradations).float(),
        "image_stops": torch.stack(stop_maps).float(),
        "sample_id": [f"synthetic-{index}" for index in range(batch_size)],
    }


def command_smoke(args: argparse.Namespace) -> None:
    config = load_config(_config_path(args.config or "configs/smoke.yaml"))
    device = resolve_device(args.device or config["runtime"]["device"])
    _configure_execution(config, args, device)
    seed_everything(config["phase1"]["initialization_seed"])
    batch = _synthetic_batch(config)
    model = build_phase1_model(config, ImuNormalizer())
    qwt_input = batch["image_clean"][:1].to(device)
    transform = model.backbone.image_transform.to(device)
    coeff, layout = transform.analysis(qwt_input)
    # Luminance QWT: synthesis gives Y back, so compare with the Y it analysed.
    qwt_error = float((transform.synthesis(coeff, layout) - transform.prepare(qwt_input)).abs().max())
    trainer1 = _phase1_trainer(model, config, device, "synthetic")
    phase1_metrics = trainer1.step(batch)
    seed_everything(config["phase2"]["decoder_initialization_seed"])
    system = RestorationSystem(model.backbone, model.normalizer, build_decoders(config),
                               *phase2_latent_modules(config, model))
    trainer2 = Phase2Trainer(system, config, device, "synthetic-phase1", "synthetic")
    phase2_batch = {key: value[:1] if isinstance(value, torch.Tensor) else value[:1] for key, value in batch.items()}
    phase2_metrics = trainer2.step([phase2_batch])
    trainer2.assert_backbone_frozen()
    result = {
        "qwt_roundtrip_max_abs_error": qwt_error,
        "phase1": phase1_metrics,
        "phase2": phase2_metrics,
        "phase1_has_decoder_attribute": hasattr(model, "decoder"),
        "backbone_frozen_after_phase2": True,
    }
    output = Path(args.output or config["runtime"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("build-manifest", help="Audit and pair a TartanAir dataset")
    manifest.add_argument("--config")
    manifest.add_argument("--data-root")
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(function=command_build_manifest)

    phase1 = subparsers.add_parser("train-phase1", help="Train latent JEPA without a decoder")
    phase1.add_argument("--config")
    phase1.add_argument("--manifest")
    phase1.add_argument("--output")
    phase1.add_argument("--device")
    phase1.add_argument("--gpus", choices=("auto", "1", "2"), help="Override runtime.gpu_count; batch_size remains global")
    phase1.add_argument("--resume")
    phase1.set_defaults(function=command_train_phase1)

    phase2 = subparsers.add_parser("train-phase2", help="Freeze backbone and train latent-only decoders")
    phase2.add_argument("--config")
    phase2.add_argument("--manifest")
    phase2.add_argument("--backbone-checkpoint")
    phase2.add_argument("--decoder-init-checkpoint", help="Start a new phase-2 run from existing decoder weights")
    phase2.add_argument("--output")
    phase2.add_argument("--device")
    phase2.add_argument("--gpus", choices=("auto", "1", "2"), help="Override runtime.gpu_count; batch_size remains global")
    phase2.add_argument("--resume")
    phase2.set_defaults(function=command_train_phase2)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a completed phase-2 checkpoint")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--manifest")
    evaluate.add_argument("--split", choices=("valid", "test"), default="test")
    evaluate.add_argument("--max-batches", type=int,
                          help="Only the first batches -- the first trajectories in manifest order; --every spreads")
    evaluate.add_argument("--every", type=int, default=1,
                          help="Every N-th frame of each trajectory (all trajectories kept; 8 = ~8x faster)")
    evaluate.add_argument("--device")
    evaluate.add_argument("--gpus", choices=("auto", "1", "2"), default="auto")
    evaluate.add_argument("--output", help="Directory for metrics, image/IMU panels and merged arrays")
    evaluate.add_argument("--panels", type=int, default=6, help="Image panels and IMU trajectory plots per scenario")
    evaluate.add_argument("--protocol", action="store_true", help="Evaluate all clean/noise/blur groups")
    evaluate.add_argument("--scenarios",
                          help="With --protocol: only these comma-separated scenarios (e.g. one half per GPU)")
    evaluate.add_argument("--amp", action="store_true",
                          help="fp16 autocast on CUDA, as a phase 2 trained with precision amp_fp16; metrics stay fp32")
    evaluate.add_argument("--full-frame", action="store_true",
                          help="With data.source_size: score whole source frames (e.g. 640x640), not training crops")
    evaluate.add_argument(
        "--image-mode",
        choices=("full", "clean", "low_light_only", "blur_only", "sensor_noise_only"),
        default="full",
    )
    evaluate.add_argument(
        "--imu-mode",
        choices=("full", "clean", "white_noise_only", "bias_only", "bandwidth_only"),
        default="full",
    )
    evaluate.set_defaults(function=command_evaluate)

    infer = subparsers.add_parser("infer", help="Restore one image and one IMU window")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--image", required=True)
    infer.add_argument("--imu", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--image-time", type=float)
    infer.add_argument("--imu-dt", type=float, default=0.01)
    infer.add_argument("--device")
    infer.set_defaults(function=command_infer)

    preview = subparsers.add_parser("preview-corruption", help="Preview low-light camera degradation")
    preview.add_argument("--config")
    preview.add_argument("--image", required=True)
    preview.add_argument("--output", required=True)
    preview.add_argument("--mode", choices=("full", "clean", "low_light_only", "blur_only", "sensor_noise_only"), default="full")
    preview.add_argument("--realization", type=int, default=0)
    preview.set_defaults(function=command_preview_corruption)

    smoke = subparsers.add_parser("smoke", help="Run one synthetic update in each phase")
    smoke.add_argument("--config")
    smoke.add_argument("--output")
    smoke.add_argument("--device")
    smoke.add_argument("--gpus", choices=("auto", "1", "2"))
    smoke.set_defaults(function=command_smoke)
    plots = subparsers.add_parser("plot-training", help="Generate PNG curves and CSV summaries from JSONL logs")
    plots.add_argument("--run-dir", required=True)
    plots.set_defaults(function=lambda args: print("\n".join(str(path) for path in plot_training(args.run_dir))))
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.function(args)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main(sys.argv[1:])
