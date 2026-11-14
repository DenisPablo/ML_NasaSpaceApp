"""Tests for Phase 4 NOAA-20 Tweedie serving inference and 2018-04-01 splice."""

from datetime import date
import io
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq

from aggregate_daily import run_aggregation_pipeline
from clean_firms import OUTPUT_SCHEMA, ProgressTracker
from predict_harmonized import run_harmonized_inference
from test_aggregate_daily import _make_clean_row
from train_hurdle import MODEL_VERSION, train_models_with_dual_validation


def test_serving_backfill_and_n20_splice(tmp_path: Path) -> None:
    clean_parquet = tmp_path / "clean.parquet"
    daily_dir = tmp_path / "daily_by_year"
    models_dir = tmp_path / "models"
    serving_dir = tmp_path / "serving_by_year"

    rows = [
        _make_clean_row(1205, date(2008, 8, 10), frp=45.0, scan=1.2, acq_time="1500", sensor_family="MODIS", satellite="Aqua"),
        _make_clean_row(1805, date(2018, 3, 15), frp=20.0, scan=1.1, acq_time="1500", sensor_family="MODIS", satellite="Terra"),
        _make_clean_row(1805, date(2018, 5, 10), frp=25.0, scan=1.1, acq_time="1500", sensor_family="MODIS", satellite="Aqua"),
        _make_clean_row(1805, date(2018, 5, 10), frp=60.0, scan=0.4, acq_time="1510", sensor_family="VIIRS", satellite="N20"),
    ]
    for yr in (2021, 2022, 2023):
        for row_idx in range(0, 12):
            cid = row_idx * 600 + 5
            rows.append(_make_clean_row(cid, date(yr, 7, 1), frp=30.0, scan=1.1, acq_time="1500", sensor_family="MODIS", satellite="Aqua"))
            rows.append(_make_clean_row(cid, date(yr, 7, 1), frp=65.0, scan=0.4, acq_time="1510", sensor_family="VIIRS", satellite="N20"))

    pq.write_table(pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA), clean_parquet)
    run_aggregation_pipeline(input_parquet=clean_parquet, output_parquet=daily_dir)

    train_models_with_dual_validation(
        input_path=daily_dir,
        models_dir=models_dir,
        start_date=date(2018, 4, 1),
        end_date=date(2023, 12, 31),
        temporal_split_year=2022,
        n_estimators=20,
    )

    buf = io.StringIO()
    stats = run_harmonized_inference(
        input_path=daily_dir,
        models_dir=models_dir,
        output_path=serving_dir,
        start_year=2008,
        end_year=2023,
        batch_size=50,
        progress_factory=lambda total: ProgressTracker(total, stream=buf, is_tty=False),
    )

    assert stats.files_processed == 16
    assert stats.estimated_rows == 2
    assert stats.observed_rows == 37

    y2008 = pq.read_table(serving_dir / "year=2008" / "data.parquet").to_pylist()
    assert len(y2008) == 1
    assert y2008[0]["is_estimate"] is True
    assert y2008[0]["n_viirs_n20"] is None
    assert y2008[0]["phi_viirs_n20"] is None
    assert y2008[0]["n_scaled_baseline"] == 6
    assert y2008[0]["n_harmonized"] >= 0.0
    assert y2008[0]["model_version"] == MODEL_VERSION
    assert y2008[0]["f_nube"] is None

    y2018 = sorted(pq.read_table(serving_dir / "year=2018" / "data.parquet").to_pylist(), key=lambda r: r["fecha"])
    assert y2018[0]["fecha"] == date(2018, 3, 15)
    assert y2018[0]["is_estimate"] is True
    assert y2018[0]["n_viirs_n20"] is None
    assert y2018[1]["fecha"] == date(2018, 5, 10)
    assert y2018[1]["is_estimate"] is False
    assert y2018[1]["n_viirs_n20"] == 1
    assert y2018[1]["n_harmonized"] == 1.0
    assert y2018[1]["phi_harmonized"] == 60.0

    pcts = [float(m) for m in re.findall(r"(\d+\.\d+)%", buf.getvalue())]
    assert pcts[-1] == 100.0
