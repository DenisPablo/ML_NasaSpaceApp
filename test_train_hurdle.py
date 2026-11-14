"""Tests for Phase 3 sparse active-cell Tweedie GBDTs (NOAA-20 count + FRP) and dual validation."""

from datetime import date
import io
from pathlib import Path
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from clean_firms import ProgressTracker
from train_hurdle import (
    LOBBO_BUFFER_CELLS,
    build_lobbo_row_bounds,
    train_models_with_dual_validation,
)


def test_lobbo_20km_buffer_bounds() -> None:
    rows = np.arange(0, 100, dtype=np.int32)
    val_min, val_max, buf_low, buf_high = build_lobbo_row_bounds(rows, n_blocks=4, val_block=1)
    assert val_min == 25
    assert val_max == 49
    assert buf_low == 25 - LOBBO_BUFFER_CELLS
    assert buf_high == 49 + LOBBO_BUFFER_CELLS


def test_dual_validation_training_on_sparse_parquets(tmp_path: Path) -> None:
    daily_dir = tmp_path / "daily"
    models_dir = tmp_path / "models"
    rng = np.random.default_rng(7)

    for year in (2021, 2022, 2023, 2024):
        part_dir = daily_dir / f"year={year}"
        part_dir.mkdir(parents=True)
        rows = []
        for grid_row in range(0, 60, 2):
            cell_id = grid_row * 600 + 10
            for day in (5, 15, 25):
                n_m = int(rng.integers(1, 5))
                n_aqua = max(1, n_m - 1)
                n_terra = n_m - n_aqua
                phi_m = float(n_m * 12.0 + rng.random() * 5.0)
                rows.append(
                    {
                        "cell_id": cell_id,
                        "grid_row": grid_row,
                        "grid_col": 10,
                        "lat_centro": -33.3,
                        "lon_centro": -66.3,
                        "fecha": date(year, 8, day),
                        "anio": year,
                        "mes": 8,
                        "n_modis": n_m,
                        "n_modis_aqua": n_aqua,
                        "n_modis_terra": n_terra,
                        "phi_modis": phi_m,
                        "phi_modis_aqua": phi_m * 0.7,
                        "phi_modis_terra": phi_m * 0.3,
                        "phi_max_modis": phi_m * 0.8,
                        "scan_bar_modis": 1.2,
                        "track_bar_modis": 1.1,
                        "n_modis_3d": n_m,
                        "n_modis_7d": n_m,
                        "phi_modis_7d": phi_m,
                        "n_viirs_n20": int(round(n_m * 5.5)),
                        "phi_viirs_n20": float(phi_m * 2.1),
                    }
                )
        pq.write_table(pa.Table.from_pylist(rows), part_dir / "data.parquet")

    buf = io.StringIO()
    stats = train_models_with_dual_validation(
        input_path=daily_dir,
        models_dir=models_dir,
        start_date=date(2021, 1, 1),
        end_date=date(2024, 12, 31),
        temporal_split_year=2022,
        n_lobbo_blocks=3,
        val_block=0,
        n_estimators=20,
        progress_factory=lambda total: ProgressTracker(total, stream=buf, is_tty=False),
    )

    assert stats.train_rows == 360
    assert stats.temporal_val_rows == 180
    assert stats.lobbo_val_rows > 0
    assert (models_dir / "n20_count_tweedie.txt").exists()
    assert (models_dir / "n20_frp_tweedie.txt").exists()
    pcts = [float(m) for m in re.findall(r"(\d+\.\d+)%", buf.getvalue())]
    assert pcts[-1] == 100.0
