"""Tests for clean_firms.py ETL cleaning rules and progress percentages."""

from datetime import date
import io
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq

from clean_firms import ProgressTracker, run_cleaning_pipeline


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    lines = [",".join(header)] + [",".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_cleaning_rules_utc_rollover_filters_dedup_and_progress(tmp_path: Path) -> None:
    input_dir = tmp_path / "firms"
    input_dir.mkdir()
    output_parquet = tmp_path / "out" / "clean.parquet"

    archive_header = [
        "latitude",
        "longitude",
        "brightness",
        "scan",
        "track",
        "acq_date",
        "acq_time",
        "satellite",
        "instrument",
        "confidence",
        "version",
        "bright_t31",
        "frp",
        "daynight",
        "type",
    ]
    nrt_header = archive_header[:-1]  # NRT omits 'type'

    # San Luis coordinates: (-33.30, -66.33)
    # Out-of-San-Luis coordinates (Buenos Aires): (-34.60, -58.38)
    _write_csv(
        input_dir / "fire_archive_SV-C2_test.csv",
        archive_header,
        [
            # Valid San Luis detection at 01:30 UTC -> rolls back to 2024-05-09 UTC-3
            ["-33.30", "-66.33", "320.5", "0.4", "0.4", "2024-05-10", "0130", "SNPP", "SNPP", "n", "2", "295.0", "5.5", "N", "0"],
            # Valid San Luis detection at 03:00 UTC -> stays 2024-05-10 UTC-3
            ["-33.31", "-66.34", "330.0", "0.4", "0.4", "2024-05-10", "0300", "SNPP", "SNPP", "h", "2", "296.0", "8.2", "N", "0"],
            # Dropped: low confidence 'l'
            ["-33.32", "-66.35", "310.0", "0.4", "0.4", "2024-05-10", "0400", "SNPP", "SNPP", "l", "2", "294.0", "2.1", "N", "0"],
            # Dropped: industrial/static type == 2
            ["-33.33", "-66.36", "340.0", "0.4", "0.4", "2024-05-10", "0400", "SNPP", "SNPP", "h", "2", "294.0", "12.0", "N", "2"],
            # Dropped: outside San Luis bbox
            ["-34.60", "-58.38", "330.0", "0.4", "0.4", "2024-05-10", "0400", "SNPP", "SNPP", "h", "2", "294.0", "9.0", "N", "0"],
        ],
    )

    _write_csv(
        input_dir / "fire_nrt_SV-C2_test.csv",
        nrt_header,
        [
            # Duplicate of first SP row -> must be deduplicated in favor of SP
            ["-33.30", "-66.33", "320.5", "0.4", "0.4", "2024-05-10", "0130", "SNPP", "SNPP", "n", "2.0NRT", "295.0", "5.5", "N"],
            # Unique NRT San Luis detection -> kept with type == -1
            ["-33.40", "-66.20", "318.0", "0.4", "0.4", "2026-08-05", "455", "SNPP", "SNPP", "n", "2.0NRT", "293.0", "3.4", "N"],
        ],
    )

    progress_buf = io.StringIO()
    stats = run_cleaning_pipeline(
        input_dir=input_dir,
        output_parquet=output_parquet,
        region="san_luis",
        progress_factory=lambda total: ProgressTracker(total, stream=progress_buf, is_tty=False),
    )

    assert stats.files_processed == 2
    assert stats.rows_read == 7
    assert stats.rows_kept == 4  # 2 from SP + 2 from NRT before dedup
    assert stats.rows_deduped == 3

    table = pq.read_table(output_parquet)
    assert table.num_rows == 3
    assert table.schema.field("frp").type == pa.float32()
    assert table.schema.field("x_aea").type == pa.float32()
    assert table.schema.field("cell_id").type == pa.int32()

    rows = table.to_pylist()
    # Check UTC -> UTC-3 rollover
    assert rows[0]["fecha_local"] == date(2024, 5, 9)
    assert rows[0]["source_quality"] == "SP"
    assert rows[0]["type"] == 0

    assert rows[1]["fecha_local"] == date(2024, 5, 10)

    assert rows[2]["fecha_local"] == date(2026, 8, 5)
    assert rows[2]["source_quality"] == "NRT"
    assert rows[2]["type"] == -1
    assert rows[2]["acq_time"] == "0455"

    # Verify progress percentages were emitted and reached 100.0%
    pcts = [float(m) for m in re.findall(r"(\d+\.\d+)%", progress_buf.getvalue())]
    assert pcts, "Expected percentage output in progress stream"
    assert pcts == sorted(pcts)
    assert pcts[-1] == 100.0


def test_recursive_multisensor_modis_and_viirs_n20(tmp_path: Path) -> None:
    root = tmp_path / "Data-Argentina"
    modis_dir = root / "DL_FIRE_M-C61_test"
    viirs_dir = root / "DL_FIRE_J1V-C2_test"
    modis_dir.mkdir(parents=True)
    viirs_dir.mkdir(parents=True)
    output_parquet = tmp_path / "out" / "multisensor.parquet"

    header = [
        "latitude", "longitude", "brightness", "scan", "track",
        "acq_date", "acq_time", "satellite", "instrument",
        "confidence", "version", "bright_t31", "frp", "daynight", "type",
    ]
    _write_csv(
        modis_dir / "fire_archive_M-C61_test.csv",
        header,
        [
            # Kept: MODIS numeric confidence 55 >= 30
            ["-33.30", "-66.33", "310.0", "1.2", "1.1", "2005-08-10", "1400", "Terra", "MODIS", "55", "6.03", "290.0", "18.5", "D", "0"],
            # Dropped: MODIS numeric confidence 20 < 30
            ["-33.31", "-66.34", "305.0", "1.2", "1.1", "2005-08-10", "1400", "Terra", "MODIS", "20", "6.03", "290.0", "12.0", "D", "0"],
        ],
    )
    _write_csv(
        viirs_dir / "fire_archive_J1V-C2_test.csv",
        header,
        [
            # Kept: NOAA-20 VIIRS
            ["-33.35", "-66.30", "325.0", "0.4", "0.4", "2020-09-01", "1600", "N20", "VIIRS", "h", "2", "292.0", "4.2", "D", "0"],
        ],
    )

    stats = run_cleaning_pipeline(
        input_dir=root,
        output_parquet=output_parquet,
        region="san_luis",
        progress_factory=lambda total: ProgressTracker(total, stream=io.StringIO(), is_tty=False),
    )

    assert stats.files_processed == 2
    assert stats.rows_deduped == 2
    rows = pq.read_table(output_parquet).to_pylist()
    assert [r["sensor_family"] for r in rows] == ["MODIS", "VIIRS"]
    assert [r["satellite"] for r in rows] == ["Terra", "N20"]


def test_nrt_filters_recurrent_type2_static_sources(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    output_parquet = tmp_path / "out" / "clean.parquet"

    sp_header = [
        "latitude", "longitude", "brightness", "scan", "track",
        "acq_date", "acq_time", "satellite", "instrument",
        "confidence", "version", "bright_t31", "frp", "daynight", "type",
    ]
    nrt_header = sp_header[:-1]

    _write_csv(
        input_dir / "fire_archive_J1V-C2_test.csv",
        sp_header,
        [
            # Recurrent type == 2 industrial flare at (-33.30, -66.33)
            ["-33.301", "-66.331", "330.0", "0.4", "0.4", "2024-01-01", "1500", "N20", "VIIRS", "h", "2", "295.0", "8.0", "D", "2"],
            ["-33.302", "-66.332", "330.0", "0.4", "0.4", "2024-01-02", "1500", "N20", "VIIRS", "h", "2", "295.0", "8.0", "D", "2"],
        ],
    )
    _write_csv(
        input_dir / "fire_nrt_J1V-C2_test.csv",
        nrt_header,
        [
            # Dropped in NRT because (-33.30, -66.33) is a recurrent type == 2 flare
            ["-33.303", "-66.333", "330.0", "0.4", "0.4", "2026-09-01", "1500", "N20", "VIIRS", "h", "2.0NRT", "295.0", "8.0", "D"],
            # Kept in NRT because (-33.45, -66.10) is not in blacklist
            ["-33.450", "-66.100", "330.0", "0.4", "0.4", "2026-09-01", "1500", "N20", "VIIRS", "h", "2.0NRT", "295.0", "12.0", "D"],
        ],
    )

    stats = run_cleaning_pipeline(
        input_dir=input_dir,
        output_parquet=output_parquet,
        region="san_luis",
        progress_factory=lambda total: ProgressTracker(total, stream=io.StringIO(), is_tty=False),
    )

    assert stats.rows_deduped == 1
    rows = pq.read_table(output_parquet).to_pylist()
    assert abs(rows[0]["latitude"] - (-33.45)) < 1e-3
