"""Time the real training data loader and optimizer step without saving a checkpoint.

Use --config to benchmark a fresh phase-1 run before any checkpoint exists, or
--checkpoint to benchmark phase 2. Weights are updated only in memory. Run this
in a separate process while actual training is stopped. Compare the same
--workers and --updates with --gpus 1 and 2.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qjepa.cli import _dataset, _memory_mib, _phase1_trainer, _system_from_phase2, _training_batch_stream
from qjepa.config import build_normalizer, build_phase1_model, load_config, seed_everything
from qjepa.data import read_manifest
from qjepa.training.phase2 import Phase2Trainer


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median_s": float(np.median(values)),
        "mean_s": float(np.mean(values)),
        "p90_s": float(np.percentile(values, 90)),
    }


def probe(checkpoint: Path | None, manifest_path: Path, *, config_path: Path | None = None,
          gpus: int, workers: int,
          updates: int, warmup: int, start_update: int = 0,
          output: Path | None = None) -> dict:
    if gpus not in (1, 2) or workers < 0 or updates < 1 or warmup < 0 or start_update < 0:
        raise ValueError("Use --gpus 1/2, --workers >= 0, --updates >= 1, --warmup >= 0, --start-update >= 0")
    if not torch.cuda.is_available():
        raise RuntimeError("This training performance probe requires CUDA")
    if (checkpoint is None) == (config_path is None):
        raise ValueError("Pass exactly one of --config (phase 1) or --checkpoint (phase 2)")
    device = torch.device("cuda")
    manifest = read_manifest(manifest_path)
    if checkpoint is None:
        phase = "phase1"
        config = load_config(config_path)
        seed_everything(config[phase]["initialization_seed"])
    else:
        phase = "phase2"
        system, config = _system_from_phase2(str(checkpoint), device)
    config["data"] = dict(config["data"], num_workers=workers)
    config["runtime"] = dict(config["runtime"], gpu_count=gpus)
    if phase == "phase1":
        model = build_phase1_model(config, build_normalizer(manifest["meta"]))
        trainer = _phase1_trainer(model, config, device, manifest["meta"]["manifest_hash"])
    else:
        trainer = Phase2Trainer(
            system, config, device,
            parent_checkpoint="performance_probe_only",
            manifest_hash=manifest["meta"]["manifest_hash"],
        )
    trainer.successful_updates = start_update
    dataset = _dataset(
        config, manifest, "train", fixed_realization=False,
        scenarios=config["phase2"].get("train_scenarios") if phase == "phase2" else None,
    )
    accumulation = config[phase]["gradient_accumulation"]
    stream = _training_batch_stream(
        config, dataset, config[phase]["batch_size"],
        start_microbatch=start_update * accumulation,
        namespace=phase,
    )
    waits: list[float] = []
    steps: list[float] = []
    for index in range(warmup + updates):
        start = time.perf_counter()
        batches = [next(stream) for _ in range(accumulation)]
        loaded = time.perf_counter()
        result = trainer.step(batches[0] if phase == "phase1" else batches)
        torch.cuda.synchronize()
        finished = time.perf_counter()
        if result.get("skipped"):
            raise RuntimeError(f"Optimizer step skipped during probe: {result}")
        if index >= warmup:
            waits.append(loaded - start)
            steps.append(finished - loaded)
    elapsed = sum(waits) + sum(steps)
    report = {
        "phase": phase,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "config": str(config_path) if config_path is not None else None,
        "gpus": gpus,
        "workers": workers,
        "pin_memory": bool(config["data"]["pin_memory"]),
        "batch_size_per_microbatch": config[phase]["batch_size"],
        "gradient_accumulation": accumulation,
        "start_update": start_update,
        "warmup_updates": warmup,
        "measured_updates": updates,
        "data_wait": _summary(waits),
        "optimizer_step": _summary(steps),
        "wall_seconds_per_update": elapsed / updates,
        "data_wait_fraction": sum(waits) / elapsed,
        "memory_mib": _memory_mib(),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Existing phase-2 checkpoint")
    source.add_argument("--config", type=Path, help="Resolved config; benchmark fresh phase 1")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gpus", required=True, type=int, choices=(1, 2))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--start-update", type=int, default=0,
                        help="Simulate a later phase schedule, such as phase-1 sensitivity after update 500")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    probe(args.checkpoint, args.manifest, config_path=args.config,
          gpus=args.gpus, workers=args.workers,
          updates=args.updates, warmup=args.warmup, start_update=args.start_update,
          output=args.output)


if __name__ == "__main__":
    main()
