"""Deterministic trajectory-diverse batches for phase-1 statistics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class TrajectoryDiverseBatchSampler(Sampler[list[int]]):
    """Sample reproducible full batches containing multiple trajectories.

    Minority trajectories may be cycled when the dataset is strongly imbalanced.
    Same-position variance from adjacent frames of one motion is less useful than
    modest trajectory oversampling for the anti-collapse objective.
    """

    def __init__(
        self,
        trajectory_keys: list[str],
        batch_size: int,
        minimum_trajectories: int,
        seed: int,
    ) -> None:
        if batch_size < 1 or len(trajectory_keys) < batch_size:
            raise ValueError("Need at least one full batch")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, key in enumerate(trajectory_keys):
            grouped[key].append(index)
        if minimum_trajectories < 1 or minimum_trajectories > batch_size:
            raise ValueError("minimum_trajectories must lie in [1,batch_size]")
        if len(grouped) < minimum_trajectories:
            raise ValueError(
                f"Dataset has {len(grouped)} trajectories but batches require {minimum_trajectories}"
            )
        self.groups = {key: tuple(indices) for key, indices in sorted(grouped.items())}
        self.keys = tuple(self.groups)
        self.batch_size = batch_size
        self.minimum_trajectories = minimum_trajectories
        self.seed = int(seed)
        self.num_batches = len(trajectory_keys) // batch_size

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed)
        queues: dict[str, list[int]] = {}
        pointers: dict[str, int] = {}

        def refill(key: str) -> None:
            source = self.groups[key]
            permutation = torch.randperm(len(source), generator=generator).tolist()
            queues[key] = [source[index] for index in permutation]
            pointers[key] = 0

        def draw(key: str, forbidden: set[int]) -> int | None:
            for _ in range(2):
                if key not in queues or pointers[key] >= len(queues[key]):
                    refill(key)
                while pointers[key] < len(queues[key]):
                    value = queues[key][pointers[key]]
                    pointers[key] += 1
                    if value not in forbidden:
                        return value
            return None

        for _ in range(self.num_batches):
            order = torch.randperm(len(self.keys), generator=generator).tolist()
            mandatory = [self.keys[index] for index in order[: self.minimum_trajectories]]
            chosen: list[int] = []
            used: set[int] = set()
            for key in mandatory:
                value = draw(key, used)
                if value is None:
                    raise RuntimeError("Could not draw a unique mandatory trajectory sample")
                chosen.append(value)
                used.add(value)
            while len(chosen) < self.batch_size:
                candidate_order = torch.randperm(len(self.keys), generator=generator).tolist()
                value = None
                for key_index in candidate_order:
                    value = draw(self.keys[key_index], used)
                    if value is not None:
                        break
                if value is None:
                    raise RuntimeError("Could not construct a full batch without duplicate samples")
                chosen.append(value)
                used.add(value)
            yield chosen


class FlarePairSlots(Sampler[list[int]]):
    """Add ``per_batch`` FlareImage slots to every batch of ``inner`` (data.flare_pairs_*), spread over the batch.

    Slot j of the b-th batch of the epoch is index ``first_index + b * per_batch + j``: past the TartanAir samples,
    so PairedCameraImuDataset reads it as its (b * per_batch + j)-th FlareImage draw of the realization. The index
    follows from the batch number alone, so skipping batches on resume (_SkipBatches, outside) still yields the
    batches the run would have seen. The slots sit at evenly spaced positions, last one at the end, so each DDP
    rank's contiguous share (_RankShare) loads its part of them.
    """

    def __init__(self, inner, first_index: int, per_batch: int) -> None:
        if per_batch < 1:
            raise ValueError("per_batch must be at least 1")
        self.inner, self.first_index, self.per_batch = inner, int(first_index), int(per_batch)

    def __iter__(self) -> Iterator[list[int]]:
        for number, batch in enumerate(self.inner):
            size = len(batch) + self.per_batch
            slots = {round((j + 1) * size / self.per_batch) - 1 for j in range(self.per_batch)}
            draws = iter(range(self.first_index + number * self.per_batch,
                               self.first_index + (number + 1) * self.per_batch))
            samples = iter(batch)
            yield [next(draws) if position in slots else next(samples) for position in range(size)]

    def __len__(self) -> int:
        return len(self.inner)

