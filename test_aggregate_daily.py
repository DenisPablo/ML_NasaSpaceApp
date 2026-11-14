"""Tests for Phase 2 daily aggregation, zero imputation, and rolling windows."""

from datetime import date, datetime
import io
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq

from aggregate_daily import run_aggregation_pipeline
from clean_firms import OUTPUT_SCHEMA, ProgressTracker


def _make_clean_row(cell_id: int, fecha: date, frp: float, scan: float, acq_time: str) -> dict:
    return {
        "cell_id": cell_id,
        "col": cell_id,
        "row": 0,
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
        "satellite": "SNPP",
        "instrument": "SNPP",
        "sensor_family": "VIIRS",
        "version": "2",
        "daynight": "D",
        "type": 0,
        "source_quality": "SP",
        "acq_date": fecha.isoformat(),
        "acq_time": acq_time,
    }


def test_daily_aggregation_zero_imputation_and_rolling_windows(tmp_path: Path) -> None:
    clean_parquet = tmp_path / "clean.parquet"
    daily_parquet = tmp_path / "daily.parquet"

    rows = [
        _make_clean_row(100, date(2024, 1, 1), frp=10.0, scan=0.4, acq_time="1500"),
        _make_clean_row(100, date(2024, 1, 1), frp=25.0, scan=0.6, acq_time="1700"),
        _make_clean_row(200, date(2024, 1, 4), frp=5.0, scan=0.5, acq_time="1300"),
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA), clean_parquet)

    buf = io.StringIO()
    stats = run_aggregation_pipeline(
        input_parquet=clean_parquet,
        output_parquet=daily_parquet,
        progress_factory=lambda total: ProgressTracker(total, stream=buf, is_tty=False),
    )

    assert stats.active_cells == 2
    assert stats.total_days == 4
    assert stats.total_rows == 8
    assert stats.positive_rows == 2
    assert stats.zero_rate == 0.75

    tbl = pq.read_table(daily_parquet)
    assert tbl.schema.field("phi_viirs").type == pa.float32()
    assert tbl.schema.field("phi_viirs_7d").type == pa.float32()
    assert tbl.schema.field("n_viirs").type == pa.int32()

    cell_100 = [r for r in tbl.to_pylist() if r["cell_id"] == 100]
    assert cell_100[0]["n_viirs"] == 2
    assert cell_100[0]["phi_viirs"] == 35.0
    assert cell_100[0]["phi_max_viirs"] == 25.0
    assert cell_100[0]["y_occ"] == 1

    assert cell_100[1]["n_viirs"] == 0
    assert cell_100[1]["phi_viirs_3d"] == 35.0
    assert cell_100[2]["phi_viirs_3d"] == 35.0

    assert cell_100[3]["phi_viirs_3d"] == 0.0
    assert cell_100[3]["phi_viirs_7d"] == 35.0
    assert cell_100[3]["phi_viirs_30d"] == 35.0

    pcts = [float(m) for m in re.findall(r"(\d+\.\d+)%", buf.getvalue())]
    assert pcts[-1] == 100.0
