"""tools/training_report.py: the train log of each phase, summarised along its updates.

The user (p27): the report should carry the phase-1 and phase-2 train logs -- the values the runs trained
at, update by update, in short. The first/lowest/last table hides a loss that falls, climbs back and
falls again (what p26's phase 1 did). Pins: each phase prints ten contiguous update ranges covering the
whole run, every cell the median of its range; a mid-run climb shows; skipped updates are left out and
counted; the schedule (learning rate, weight decay, teacher momentum) sits next to the loss; a run
shorter than ten updates gets one row per update.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import median

import pytest

REPO = Path(__file__).resolve().parents[1]


def _write(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _phase1(updates: int):
    for update in range(1, updates + 1):
        t = update / updates
        loss = 0.3 * math.exp(-6 * t) + 0.05 + (0.1 if 0.4 < t <= 0.6 else 0.0)   # climbs back mid-run
        yield {"skipped": False, "loss": loss, "jepa": loss, "jepa_image": 1.1 * loss, "jepa_imu": 0.9 * loss,
               "gradient_norm": 2.0, "learning_rate": 1e-3, "weight_decay": 0.04 + 0.36 * t,
               "teacher_momentum": 0.996 + 0.004 * t, "successful_updates": update}
        if update == 7:
            yield {"skipped": True, "reason": "non_finite_loss", "loss": float("nan"), "successful_updates": update}


def _phase2(updates: int):
    for update in range(1, updates + 1):
        yield {"skipped": False, "loss": 2.0 - update / updates, "image_l1": 0.5, "gradient_norm": 3.0,
               "learning_rate": 2e-4, "successful_updates": update}


def _report(run: Path) -> str:
    result = subprocess.run([sys.executable, "tools/training_report.py", "--run", str(run)], cwd=REPO,
                            capture_output=True, text=True, check=True)
    return result.stdout


def _table(text: str, title: str) -> list[list[str]]:
    """The rows of the first block of a trajectory table: [update range, cells...]."""
    lines = text[text.index(title):].splitlines()
    header = next(index for index, line in enumerate(lines) if line.strip().startswith("update "))
    rows = []
    for line in lines[header + 1:]:
        if not line.strip() or not line.strip()[0].isdigit():
            break
        rows.append(line.split())
    return rows


def test_both_phases_print_ten_contiguous_ranges_of_medians(tmp_path):
    records = list(_phase1(1000))
    _write(tmp_path / "phase1/train.jsonl", records)
    _write(tmp_path / "phase2/train.jsonl", _phase2(500))
    text = _report(tmp_path)
    rows = _table(text, "Log train phase 1 — diễn biến theo update")
    assert len(rows) == 10
    ranges = [tuple(int(v) for v in row[0].split("–")) for row in rows]
    assert ranges[0][0] == 1 and ranges[-1][1] == 1000
    assert all(previous[1] + 1 == current[0] for previous, current in zip(ranges, ranges[1:]))
    done = [r for r in records if not r["skipped"]]
    for (first, last), row in zip(ranges, rows):
        expected = median(r["loss"] for r in done if first <= r["successful_updates"] <= last)
        assert float(row[1]) == pytest.approx(expected, abs=5e-7)      # six decimals, as printed
    losses = [float(row[1]) for row in rows]
    assert losses[4] > losses[3] and losses[6] < losses[5]          # the mid-run climb shows
    assert "1 update bị bỏ (không tính)" in text
    header = text[text.index("Log train phase 1"):].split("Log train phase 2")[0]
    assert "weight_decay" in header and "teacher_momentum" in header and "learning_rate" in header
    phase2 = _table(text, "Log train phase 2 — diễn biến theo update")
    assert len(phase2) == 10 and phase2[-1][0] == "451–500"
    assert "Khoá khác" not in text                                   # weight_decay is a known key now


def test_a_run_shorter_than_ten_updates_gets_one_row_per_update(tmp_path):
    _write(tmp_path / "phase1/train.jsonl", [r for r in _phase1(4) if not r["skipped"]])
    rows = _table(_report(tmp_path), "Log train phase 1 — diễn biến theo update")
    assert [row[0] for row in rows] == ["1–1", "2–2", "3–3", "4–4"]
