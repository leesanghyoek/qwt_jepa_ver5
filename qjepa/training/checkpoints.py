"""Versioned checkpoint helpers with strict phase provenance."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def configuration_hash(config: dict[str, Any], phase: str) -> str:
    """Hash training semantics while allowing paths/worker counts to move."""
    data = {
        key: value
        for key, value in config["data"].items()
        if key not in {"root", "manifest_dir", "num_workers", "pin_memory", "halo_root", "flare_pairs_root"}
    }
    contract: dict[str, Any] = {
        "pipeline_version": config["pipeline_version"],
        "run_kind": config.get("run_kind", "main"),
        "validation_recipe": "fixed_bank_full_corruption_v2",
        "data": data,
        "model": config["model"],
        "corruption": config["corruption"],
    }
    if phase == "phase1":
        contract.update(
            {
                "phase1": config["phase1"],
                "encoder_sensitivity": config["encoder_sensitivity"],
                "monitor": config["monitor"],
            }
        )
    elif phase == "phase2":
        phase2 = {key: value for key, value in config["phase2"].items() if key != "backbone_checkpoint"}
        contract["phase2"] = phase2
        contract["validation_batches"] = config["runtime"]["validation_batches"]
    else:
        raise ValueError(f"Unknown configuration phase {phase!r}")
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def state_dict_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        digest.update(buffer.getvalue())
    return digest.hexdigest()


def target_backbone_hash(model: nn.Module) -> str:
    """The backbone phase 2 freezes with phase2.backbone_weights: target -- the phase-1
    model's backbone carrying its EMA teacher's encoder weights (the fusion, if any, stays)."""
    backbone = copy.deepcopy(model.backbone)
    model.teachers.load_into(backbone)
    return state_dict_hash(backbone)


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        # Resume may move between one and two GPUs. Additional GPUs keep the
        # initialization seed; nonexistent device states must not be restored.
        for index, value in enumerate(state["cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(value.cpu(), device=index)


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=False)


def require_phase1_checkpoint(payload: dict[str, Any]) -> None:
    metadata = payload.get("metadata", {})
    for key, expected in (("pipeline_version", 3), ("phase", "latent_pretrain")):
        if metadata.get(key) != expected:
            raise ValueError(f"Invalid phase-1 checkpoint metadata {key}={metadata.get(key)!r}")
    # Decoder phase 1 chi la mo neo cho latent va bi vut sau do; phase 2 van dung
    # decoder moi tinh. Hai truong nay phai nhat quan de metadata khong noi doi.
    reconstruction = metadata.get("trained_with_reconstruction")
    calls = metadata.get("phase1_decoder_forward_calls")
    if reconstruction is not True and reconstruction is not False:
        raise ValueError(f"Invalid phase-1 checkpoint metadata trained_with_reconstruction={reconstruction!r}")
    if not isinstance(calls, int) or calls < 0:
        raise ValueError(f"Invalid phase-1 checkpoint metadata phase1_decoder_forward_calls={calls!r}")
    if reconstruction != (calls > 0):
        raise ValueError(
            f"Phase-1 metadata disagrees: trained_with_reconstruction={reconstruction}"
            f" but decoder ran {calls} times"
        )
