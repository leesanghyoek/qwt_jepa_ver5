"""Training speed on 2 x T4: what the log says about it, and the IMU caches.

Pins: the progress line reports the seconds per update spent waiting for data next
to the seconds per update; both phases log data_wait_seconds per update; a dataset
sizes both IMU caches to its trajectories (capped), and a corruptor that caches
every trajectory builds each once yet returns the very windows a small cache does.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from qjepa.cli import _next_timed, _pace
from qjepa.corruptions import ImuCorruptionConfig, TrajectoryImuCorruptor
from qjepa.data.dataset import MAX_CACHED_TRAJECTORIES, PairedCameraImuDataset


def test_the_progress_line_reports_the_wait_for_data():
    started = time.perf_counter() - 3.0
    assert _pace(started, 10, 12).endswith(" s/update")                  # unchanged without a wait
    line = _pace(started, 10, 12, waited=1.0)
    assert 1.4 < float(line.split()[0]) < 1.7 and line.endswith("(cho du lieu 0.50)")


def test_next_timed_returns_the_batch_and_the_time_blocked_on_it():
    def slow():
        time.sleep(0.05)
        yield "batch"
    batch, seconds = _next_timed(slow())
    assert batch == "batch" and 0.04 < seconds < 1.0


def test_a_dataset_caches_every_trajectory_up_to_the_cap():
    samples = [SimpleNamespace(trajectory_key=f"env/Data_easy/P{index % 40:03d}") for index in range(400)]
    dataset = PairedCameraImuDataset(samples)
    assert dataset.cache.size == 40 and dataset.imu_corruptor.cache_size == 40
    many = [SimpleNamespace(trajectory_key=f"t{index}") for index in range(MAX_CACHED_TRAJECTORIES + 5)]
    assert PairedCameraImuDataset(many).cache.size == MAX_CACHED_TRAJECTORIES


def test_a_full_cache_builds_each_trajectory_once_and_returns_the_same_windows():
    rng = np.random.default_rng(0)
    trajectories = {f"t{index}": rng.normal(size=(600, 6)) for index in range(20)}
    times = np.arange(600) * 0.01
    small, full = (TrajectoryImuCorruptor(ImuCorruptionConfig(), 73128, cache_size=size) for size in (8, 64))
    builds = {}
    for corruptor in (small, full):
        original = corruptor._build
        def counted(*args, _original=original, _corruptor=corruptor, **kwargs):
            builds[id(_corruptor)] = builds.get(id(_corruptor), 0) + 1
            return _original(*args, **kwargs)
        corruptor._build = counted
    for _ in range(2):
        for key, imu in trajectories.items():
            context = dict(split="train", realization=0, trajectory=key, mode="full")
            a, _ = small.window(imu, times, 100, 228, **context)
            b, _ = full.window(imu, times, 100, 228, **context)
            assert np.array_equal(a, b)
    assert builds[id(full)] == 20 and builds[id(small)] == 40
