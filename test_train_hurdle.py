"""Tests for Phase 3 batch-by-year Hurdle GBDT, Focal Loss, and 20 km LOBBO split."""

from datetime import date
import io
from pathlib import Path
import re

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from clean_firms import GRID_N_COLS, ProgressTracker
from train_hurdle import (
    LOBBO_BUFFER_CELLS,
    build_lobbo_row_bounds,
    focal_loss_objective,
    train_hurdle_in_batches,
)


def test_focal_loss_and_lobbo_20km_buffer() -> None:
    rows = np.arange(0, 100, dtype=np.int32)
    val_min, val_max, buf_low, buf_high = build_lobbo_row_bounds(rows, n_blocks=4, val_block=1)
    assert val_min == 25
    assert val_max == 49
    assert buf_low == 25 - LOBBO_BUFFER_CELLS
    assert buf_high == 49 + LOBBO_BUFFER_CELLS

    X = np.zeros((4, 2), dtype=np.float32)
    y = np.array([1, 0, 1, 0], dtype=np.int32)
    ds = lgb.Dataset(X, label=y)
    ds.construct()
    grad, hess = focal_loss_objective(gamma=2.0, alpha=0.75)(np.zeros(4), ds)
    assert np.all(np.isfinite(grad))
    assert np.all(hess > 0.0)
    assert grad[0] < 0.0 and grad[1] > 0.0


def test_batch_training_on_yearly_parquets(tmp_path: Path) -> None:
    daily_dir = tmp_path / "daily"
    models_dir = tmp_path / "models"
    rng = np.random.default_rng(7)

    for year in (2022, 2023):
        part_dir = daily_dir / f"year={year}"
        part_dir.mkdir(parents=True)
        rows = []
        for grid_row in range(0, 60, 3):
            cell_id = grid_row * GRID_N_COLS + 10
            for day in range(1, 21):
                is_fire = 1 if (day % 7 == 0 and grid_row % 2 == 0) else 0
                phi_m = float(15.0 + rng.random() * 10.0) if is_fire else 0.0
                phi_v = float(phi_m * 2.5) if is_fire else 0.0
                rows.append(
                    {
                        "cell_id": cell_id,
                        "fecha": date(year, 1, day),
                        "n_modis": 1 if is_fire else 0,
                        "phi_modis": phi_m,
                        "phi_max_modis": phi_m,
                        "scan_bar_modis": 1.1 if is_fire else None,
                        "track_bar_modis": 1.0 if is_fire else None,
                        "y_occ_modis": is_fire,
                        "n_modis_3d": 1 if is_fire else 0,
                        "n_modis_7d": 1 if is_fire else 0,
                        "n_modis_30d": 1 if is_fire else 0,
                        "phi_modis_3d": phi_m,
                        "phi_modis_7d": phi_m,
                        "phi_modis_30d": phi_m,
                        "n_viirs": 3 if is_fire else 0,
                        "phi_viirs": phi_v,
                        "phi_max_viirs": phi_v * 0.6,
                        "scan_bar_viirs": 0.4 if is_fire else None,
                        "track_bar_viirs": 0.4 if is_fire else None,
                        "y_occ_viirs": is_fire,
                        "n_viirs_3d": 3 if is_fire else 0,
                        "n_viirs_7d": 3 if is_fire else 0,
                        "n_viirs_30d": 3 if is_fire else 0,
                        "phi_viirs_3d": phi_v,
                        "phi_viirs_7d": phi_v,
                        "phi_viirs_30d": phi_v,
                    }
                )
        pq.write_table(pa.Table.from_pylist(rows), part_dir / "data.parquet")

    buf = io.StringIO()
    stats = train_hurdle_in_batches(
        input_path=daily_dir,
        models_dir=models_dir,
        start_year=2022,
        end_year=2023,
        neg_sample_rate=0.5,
        n_lobbo_blocks=3,
        val_block=0,
        trees_per_batch=10,
        progress_factory=lambda total: ProgressTracker(total, stream=buf, is_tty=False),
    )

    assert stats.batches_processed == 2
    assert stats.occ_rows_used > 0
    assert stats.int_rows_used > 0
    assert (models_dir / "hurdle_occ_focal.txt").exists()
    assert (models_dir / "hurdle_int_tweedie.txt").exists()
    pcts = [float(m) for m in re.findall(r"(\d+\.\d+)%", buf.getvalue())]
    assert pcts[-1] == 100.0
