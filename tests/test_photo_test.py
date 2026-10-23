"""tools/photo_test.py: any photo with any IMU window through a phase-2 checkpoint, without a dataset.

Pins: every --imu form gives the window it names (zeros, a constant, a CSV with or without timestamps, gravity plus
noise reproducible per photo) and a wrong one is refused; one value serves every photo or one per photo; both sizes
run (the 256 crop and the whole frame cut to multiples of 16); each photo gets its panels, the restored image and an
IMU CSV holding the window given and the one restored. A folder stands for every photo in it (not its subfolders),
the checkpoint is found next to the photos, and the results land in <folder>/ket_qua_test.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from test_flare_pairs import _train_tiny
from tools.photo_test import collect_images, imu_window, main


def test_every_imu_form_gives_the_window_it_names(tmp_path):
    photo = Path("a.jpg")
    zeros, times, _ = imu_window("khong", photo, 8, 0.01, (0.5, 0.1))
    assert not zeros.any() and times[1] == pytest.approx(0.01)
    still, _, _ = imu_window("0,0,9.81,0,0,0", photo, 8, 0.01, (0.5, 0.1))
    assert (still == np.float32([0, 0, 9.81, 0, 0, 0])).all()
    noisy, _, _ = imu_window("ngau_nhien", photo, 8, 0.01, (0.5, 0.1))
    again, _, _ = imu_window("ngau_nhien", photo, 8, 0.01, (0.5, 0.1))
    assert (noisy == again).all() and abs(noisy[:, 2].mean() - 9.81) < 1 and noisy[:, 3:].std() > 0
    rows = np.column_stack((np.arange(8) * 0.005, np.arange(48).reshape(8, 6)))
    np.savetxt(tmp_path / "t.csv", rows, delimiter=",", header="timestamp,ax,ay,az,gx,gy,gz", comments="")
    window, stamps, _ = imu_window(str(tmp_path / "t.csv"), photo, 8, 0.01, (0.5, 0.1))
    assert (window == rows[:, 1:]).all() and stamps[1] == pytest.approx(0.005)
    with pytest.raises(ValueError, match="6 so"):
        imu_window("1,2,3", photo, 8, 0.01, (0.5, 0.1))


def test_photos_with_their_imu_windows_through_a_checkpoint(tmp_path):
    torch.set_num_threads(1)
    last, _ = _train_tiny(tmp_path, "flare", True)
    photos = []
    for name, size in (("xe.jpg", (90, 70)), ("b.png", (40, 52))):
        rng = np.random.default_rng(len(name))
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(tmp_path / name)
        photos.append(str(tmp_path / name))
    output = tmp_path / "out"
    main([*photos, "--checkpoint", str(last), "--output", str(output), "--device", "cpu", "--khong-hien",
          "--imu", "0,0,9.81,0,0,0", "ngau_nhien"])
    for stem in ("xe", "b"):
        for size in ("256", "full"):
            assert (output / f"{stem}_{size}_so_sanh.png").stat().st_size > 1000
        table = np.loadtxt(output / f"{stem}_imu.csv", delimiter=",", skiprows=1)
        assert table.shape[1] == 13 and np.isfinite(table).all()
    full = Image.open(output / "xe_full_ra.png")
    assert full.size == (80, 64)                                     # 90 x 70 cut to multiples of 16
    assert (np.loadtxt(output / "xe_imu.csv", delimiter=",", skiprows=1)[:, 1:7] == np.float32(
        [0, 0, 9.81, 0, 0, 0])).all()
    with pytest.raises(SystemExit):
        main([*photos, "--checkpoint", str(last), "--output", str(output), "--imu", "khong", "khong", "khong", "--khong-hien"])


def test_a_folder_is_every_photo_in_it_with_the_checkpoint_beside_them(tmp_path):
    torch.set_num_threads(1)
    last, _ = _train_tiny(tmp_path, "flare", True)
    folder = tmp_path / "Downloads"
    (folder / "con").mkdir(parents=True)
    for name in ("b.png", "a.jpg", "con/c.png"):
        Image.fromarray(np.full((40, 48, 3), 120, np.uint8)).save(folder / name)
    (folder / "ghi_chu.txt").write_text("khong phai anh")
    shutil.copy(last, folder / "best_joint_validation.pt")
    assert [path.name for path in collect_images([str(folder)])] == ["a.jpg", "b.png"]
    main([str(folder), "--device", "cpu", "--khong-hien", "--size", "full"])
    results = folder / "ket_qua_test"
    assert sorted(path.name for path in results.glob("*_so_sanh.png")) == ["a_full_so_sanh.png", "b_full_so_sanh.png"]
    (folder / "trong").mkdir()
    with pytest.raises(FileNotFoundError, match="khong co anh"):
        collect_images([str(folder / "trong")])
