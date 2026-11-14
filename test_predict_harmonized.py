"""Tests for Phase 4 streamed Hurdle inference and 2000-2026 harmonized backfilling."""

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
from train_hurdle import train_hurdle_in_batches


def test_harmonized_backfill_and_observed_splice(tmp_path: Path) -> None:
    clean_parquet = tmp_path / "clean.parquet"
    daily_dir = tmp_path / "daily_by_year"
    models_dir = tmp_path / "models"
    final_dir = tmp_path / "harmonized_by_year"

    rows = []
    # Pre-VIIRS era (2008) with MODIS detections
    rows.append(_make_clean_row(1205, date(2008, 8, 10), frp=45.0, scan=1.2, acq_time="1500", sensor_family="MODIS"))
    rows.append(_make_clean_row(1805, date(2008, 8, 11), frp=0.5, scan=1.0, acq_time="1500", sensor_family="MODIS"))

    # Overlap era (2018-2019) for LOBBO training
    for yr in (2018, 2019):
        for row_idx in range(0, 12):
            cid = row_idx * 600 + 5
            rows.append(_make_clean_row(cid, date(yr, 7, 1), frp=30.0, scan=1.1, acq_time="1500", sensor_family="MODIS"))
            rows.append(_make_clean_row(cid, date(yr, 7, 1), frp=60.0, scan=0.4, acq_time="1510", sensor_family="VIIRS", satellite="SNPP"))

    pq.write_table(pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA), clean_parquet)
    run_aggregation_pipeline(input_parquet=clean_parquet, output_parquet=daily_dir)

    train_hurdle_in_batches(
        input_path=daily_dir,
        models_dir=models_dir,
        start_year=2018,
        end_year=2019,
        neg_sample_rate=0.5,
        trees_per_batch=10,
    )

    buf = io.StringIO()
    stats = run_harmonized_inference(
        input_path=daily_dir,
        models_dir=models_dir,
        output_path=final_dir,
        start_year=2008,
        end_year=2019,
        batch_size=50,
        progress_factory=lambda total: ProgressTracker(total, stream=buf, is_tty=False),
    )

    assert stats.files_processed == 12
    assert stats.total_rows > 0
    assert stats.backfilled_rows > 0
    assert stats.observed_rows > 0

    y2008 = pq.read_table(final_dir / "year=2008" / "data.parquet")
    assert y2008.schema.field("phi_harmonized").type == pa.float32()
    assert y2008.schema.field("is_backfilled").type == pa.bool_()
    for r in y2008.to_pylist():
        assert r["is_backfilled"] is True
        assert r["phi_harmonized"] == r["y_hat_viirs_eq"]

    y2012 = pq.read_table(final_dir / "year=2012" / "data.parquet")
    for r in y2012.to_pylist():
        expected_bf = r["fecha"] < date(2012, 1, 20)
        assert r["is_backfilled"] is expected_bf

    y2018 = pq.read_table(final_dir / "year=2018" / "data.parquet")
    for r in y2018.to_pylist():
        assert r["is_backfilled"] is False
        assert r["phi_harmonized"] == r["phi_viirs_obs"]

    pcts = [float(m) for m in re.findall(r"(\d+\.\d+)%", buf.getvalue())]
    assert pcts[-1] == 100.0
