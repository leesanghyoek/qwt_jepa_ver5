"""Kaggle refuses to commit a notebook whose source is 1 MB or more ("The kernel source must be less than 1 megabytes
in size", 30/10/2026: the p54 notebook was 1.22 MB, 0.98 MB of it the architecture PNG of Cell 0). Pin: the training
notebook stays under that limit with room for the next run's intro (the PNG goes in with a 128-colour palette)."""

from pathlib import Path

KAGGLE_SOURCE_LIMIT = 1_000_000


def test_the_training_notebook_fits_kaggles_source_limit():
    size = Path("qwt-jaco-jepa-ijepa.ipynb").stat().st_size
    assert size < 0.8 * KAGGLE_SOURCE_LIMIT, f"{size} bytes: Kaggle commits only < {KAGGLE_SOURCE_LIMIT}"
