"""Phase 2 on two GPUs as two processes (DDP), with the loss on the whole batch.

On Kaggle's torch 2.10 + cuDNN 9.10 (T4 x2), DataParallel's replica threads race
in fp32 ("CUDA error: misaligned address"); one process per GPU shares nothing.
Each rank loads its share of every microbatch and runs the model on it. The
outputs and the batch are gathered, so every rank computes the loss of the whole
microbatch -- the loss one GPU computes. Only the rank's own share keeps its
graph, so the loss is multiplied by the world size before backward: DDP averages
the gradients, and the average of world x (gradient through each share) is the
one-GPU gradient.

Rank 0 validates, logs and saves; the others wait for it at the next collective.
"""

from __future__ import annotations

import os
import socket
import sys
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def rank_and_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def gather_shares(tensor: torch.Tensor, keep_graph: bool) -> torch.Tensor:
    """Every rank's share concatenated along dim 0 in rank order, i.e. in batch order.

    With ``keep_graph`` the local share is the tensor itself, so gradients reach
    this rank's model; the other shares are values only.
    """
    rank, world = rank_and_world()
    if world == 1:
        return tensor
    shares = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(shares, tensor.detach().contiguous())
    if keep_graph:
        shares[rank] = tensor
    return torch.cat(shares, dim=0)


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")


def any_rank(flag: bool) -> bool:
    """True on every rank when it is true on any rank."""
    if rank_and_world()[1] == 1:
        return flag
    value = torch.tensor([int(flag)], device=_device())
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def share_rank0_rng() -> None:
    """Give every rank rank 0's CPU RNG states.

    Only rank 0 validates, and a validation DataLoader draws its worker seed from
    the global generator; without this the ranks would crop VGG's window in
    different places from the first validation on.
    """
    if rank_and_world()[1] == 1:
        return
    import random

    import numpy as np

    states = [(random.getstate(), np.random.get_state(), torch.random.get_rng_state())]
    dist.broadcast_object_list(states, src=0, device=_device())
    python, numpy_state, torch_state = states[0]
    random.setstate(python)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_state.cpu())


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _entry(rank: int, world: int, port: int, cuda: bool, target: Callable[[Any], None], args: Any) -> None:
    if cuda:
        # One visible GPU per process: "cuda" is this rank's GPU everywhere, and
        # nothing (RNG state, stray tensors) creates a context on the other one.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        ids = visible.split(",") if visible else [str(index) for index in range(world)]
        os.environ["CUDA_VISIBLE_DEVICES"] = ids[rank].strip()
    if rank:
        sys.stdout = open(os.devnull, "w")          # one log: rank 0's; errors still reach stderr
    dist.init_process_group("nccl" if cuda else "gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=world)
    try:
        target(args)
    finally:
        dist.destroy_process_group()


def spawn(target: Callable[[Any], None], args: Any, world: int, cuda: bool) -> None:
    """Run ``target(args)`` in ``world`` processes; a failed rank's exit code becomes ours.

    Exit 75 (checkpoint saved, restart to free memory) and a SIGKILL reach the
    notebook as they would from a single process.
    """
    try:
        mp.spawn(_entry, args=(world, _free_port(), cuda, target, args), nprocs=world, join=True)
    except mp.ProcessExitedException as failed:
        if failed.exit_code < 0:
            os.kill(os.getpid(), -failed.exit_code)
        raise SystemExit(failed.exit_code) from None
