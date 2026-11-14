"""Tests for Phase 2 sparse active-cell daily aggregation (Aqua/Terra + NOAA-20)."""

from datetime import date, datetime
import io
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq

from aggregate_daily import run_aggregation_pipeline
from clean_firms import OUTPUT_SCHEMA, ProgressTracker


def _make_clean_row(
    cell_id: int,
    fecha: date,
    frp: float,
    scan: float,
    acq_time: str,
    sensor_family: str = "VIIRS",
    satellite: str | None = None,
) -> dict:
    sat = satellite if satellite is not None else ("Aqua" if sensor_family == "MODIS" else "N20")
    return {
        "cell_id": cell_id,
        "col": cell_id % 600,
        "row": cell_id // 600,
        "fecha_local": fecha,
        "timestamp_utc3": datetime(fecha.year, fecha.month, fecha.day, 12, 0),
        "latitude": -33.3,
        "longitude": -66.3,
        "x_aea": 1900000.0,
        "y_aea": 400000.0,
        "brightness": 320.0,
        "bright_t31": 295.0,
        "frp": frp,
        "scan": scan,
        "track": 0.4,
        "confidence": "n",
        "satellite": sat,
        "instrument": sensor_family,
        "sensor_family": sensor_family,
        "version": "2",
        "daynight": "D",
        "type": 0,
        "source_quality": "SP",
        "acq_date": fecha.isoformat(),
        "acq_time": acq_time,
    }


def test_sparse_daily_aggregation_aqua_terra_and_null_pre_n20(tmp_path: Path) -> None:
    clean_parquet = tmp_path / "clean.parquet"
    daily_parquet = tmp_path / "daily.parquet"

    rows = [
        # Pre-N20 era (2010): MODIS Aqua + Terra, VIIRS must be NULL
        _make_clean_row(100, date(2010, 8, 1), frp=12.0, scan=1.5, acq_time="1510", sensor_family="MODIS", satellite="Aqua"),
        _make_clean_row(100, date(2010, 8, 1), frp=8.0, scan=1.1, acq_time="1310", sensor_family="MODIS", satellite="Terra"),
        # N20 era (2024): N20 + N21 on same day -> only N20 counted by default
        _make_clean_row(100, date(2024, 1, 1), frp=15.0, scan=0.4, acq_time="1500", sensor_family="VIIRS", satellite="N20"),
        _make_clean_row(100, date(2024, 1, 1), frp=25.0, scan=0.4, acq_time="1600", sensor_family="VIIRS", satellite="N21"),
        _make_clean_row(100, date(2024, 1, 3), frp=10.0, scan=1.2, acq_time="1500", sensor_family="MODIS", satellite="Aqua"),
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA), clean_parquet)

    buf = io.StringIO()
    stats = run_aggregation_pipeline(
        input_parquet=clean_parquet,
        output_parquet=daily_parquet,
        progress_factory=lambda total: ProgressTracker(total, stream=buf, is_tty=False),
    )

    # Sparse output: only the 3 active (cell_id, fecha) pairs are emitted
    assert stats.total_rows == 3
    assert stats.modis_active_rows == 2
    assert stats.n20_active_rows == 1

    out = sorted(pq.read_table(daily_parquet).to_pylist(), key=lambda r: r["fecha"])
    assert out[0]["fecha"] == date(2010, 8, 1)
    assert out[0]["n_modis"] == 2
    assert out[0]["n_modis_aqua"] == 1
    assert out[0]["n_modis_terra"] == 1
    assert out[0]["phi_modis"] == 20.0
    assert out[0]["n_viirs_n20"] is None
    assert out[0]["phi_viirs_n20"] is None

    assert out[1]["fecha"] == date(2024, 1, 1)
    assert out[1]["n_viirs_n20"] == 1
    assert out[1]["phi_viirs_n20"] == 15.0

    assert out[2]["fecha"] == date(2024, 1, 3)
    assert out[2]["n_viirs_n20"] == 0
    assert out[2]["phi_viirs_n20"] == 0.0
    assert out[2]["n_modis_3d"] == 1

    pcts = [float(m) for m in re.findall(r"(\d+\.\d+)%", buf.getvalue())]
    assert pcts[-1] == 100.0


def test_yearly_partitioned_preserves_cross_year_7d_window(tmp_path: Path) -> None:
    clean_parquet = tmp_path / "clean.parquet"
    out_dir = tmp_path / "daily_by_year"

    rows = [
        _make_clean_row(100, date(2023, 12, 30), frp=15.0, scan=1.2, acq_time="1500", sensor_family="MODIS", satellite="Aqua"),
        _make_clean_row(100, date(2024, 1, 2), frp=5.0, scan=1.2, acq_time="1500", sensor_family="MODIS", satellite="Aqua"),
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA), clean_parquet)

    run_aggregation_pipeline(
        input_parquet=clean_parquet,
        output_parquet=out_dir,
        progress_factory=lambda total: ProgressTracker(total, stream=io.StringIO(), is_tty=False),
    )

    assert (out_dir / "year=2023" / "data.parquet").exists()
    assert (out_dir / "year=2024" / "data.parquet").exists()

    y2024 = pq.read_table(out_dir / "year=2024" / "data.parquet").to_pylist()
    assert len(y2024) == 1
    assert y2024[0]["fecha"] == date(2024, 1, 2)
    assert y2024[0]["n_modis_3d"] == 1
    assert y2024[0]["n_modis_7d"] == 2
    assert y2024[0]["phi_modis_7d"] == 20.0
