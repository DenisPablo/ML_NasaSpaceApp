"""Phase 2 ETL: Sparse active-cell daily aggregation (MODIS Aqua/Terra + VIIRS NOAA-20) in DuckDB."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import time
from typing import Callable

import duckdb
import numpy as np
import pyarrow as pa
from pyproj import Transformer

from clean_firms import (
    ARGENTINA_ALBERS_PROJ,
    GRID_CELL_SIZE_M,
    GRID_N_COLS,
    GRID_X_MIN,
    GRID_Y_MIN,
    ProgressTracker,
)

VIIRS_N20_START_DATE = date(2018, 4, 1)


@dataclass
class AggregationStats:
    """Summary metrics for the sparse daily aggregated grid."""

    total_rows: int = 0
    active_cells: int = 0
    total_days: int = 0
    modis_active_rows: int = 0
    n20_active_rows: int = 0
    min_date: date | None = None
    max_date: date | None = None
    elapsed_seconds: float = 0.0


def build_cell_coords_table(con: duckdb.DuckDBPyConnection, cells_source_posix: str) -> int:
    """Populate TEMP TABLE cell_coords with deterministic Albers centroids in WGS84."""
    cell_ids = (
        con.sql(f"SELECT DISTINCT cell_id::INT32 AS cell_id FROM read_parquet('{cells_source_posix}')")
        .to_arrow_table()
        .column(0)
        .to_numpy(zero_copy_only=False)
        .astype(np.int32)
    )
    if cell_ids.size == 0:
        return 0

    grid_row = (cell_ids // GRID_N_COLS).astype(np.int32)
    grid_col = (cell_ids % GRID_N_COLS).astype(np.int32)
    x_center = GRID_X_MIN + (grid_col.astype(np.float64) + 0.5) * GRID_CELL_SIZE_M
    y_center = GRID_Y_MIN + (grid_row.astype(np.float64) + 0.5) * GRID_CELL_SIZE_M

    inv_tf = Transformer.from_crs(ARGENTINA_ALBERS_PROJ, "EPSG:4326", always_xy=True)
    lon_c, lat_c = inv_tf.transform(x_center, y_center)

    coords_tbl = pa.table(
        {
            "cell_id": pa.array(cell_ids, type=pa.int32()),
            "grid_row": pa.array(grid_row, type=pa.int32()),
            "grid_col": pa.array(grid_col, type=pa.int32()),
            "lat_centro": pa.array(np.asarray(lat_c, dtype=np.float32), type=pa.float32()),
            "lon_centro": pa.array(np.asarray(lon_c, dtype=np.float32), type=pa.float32()),
        }
    )
    con.register("_coords_arrow", coords_tbl)
    con.execute("CREATE TEMP TABLE cell_coords AS SELECT * FROM _coords_arrow")
    con.unregister("_coords_arrow")
    return int(cell_ids.size)


def build_daily_aggregation_sql(
    input_posix: str,
    start_date: date,
    end_date: date,
    emit_start_date: date | None = None,
    emit_end_date: date | None = None,
    viirs_satellite: str | None = "N20",
    viirs_start_date: date = VIIRS_N20_START_DATE,
) -> str:
    """Build DuckDB SQL for sparse active-cell daily aggregation and causal 3d/7d windows."""
    out_start = emit_start_date if emit_start_date is not None else start_date
    out_end = emit_end_date if emit_end_date is not None else end_date

    if viirs_satellite and viirs_satellite.upper() != "ALL":
        safe_sat = viirs_satellite.replace("'", "''")
        viirs_filter = f"sensor_family = 'VIIRS' AND satellite = '{safe_sat}'"
    else:
        viirs_filter = "sensor_family = 'VIIRS'"

    return f"""
    WITH firms_diario AS (
        SELECT
            cell_id::INT32 AS cell_id,
            fecha_local::DATE AS fecha,
            COUNT(*) FILTER (WHERE sensor_family = 'MODIS')::INT32 AS n_modis,
            COUNT(*) FILTER (WHERE sensor_family = 'MODIS' AND satellite = 'Aqua')::INT32 AS n_modis_aqua,
            COUNT(*) FILTER (WHERE sensor_family = 'MODIS' AND satellite = 'Terra')::INT32 AS n_modis_terra,
            COALESCE(SUM(frp) FILTER (WHERE sensor_family = 'MODIS'), 0.0)::FLOAT AS phi_modis,
            COALESCE(SUM(frp) FILTER (WHERE sensor_family = 'MODIS' AND satellite = 'Aqua'), 0.0)::FLOAT AS phi_modis_aqua,
            COALESCE(SUM(frp) FILTER (WHERE sensor_family = 'MODIS' AND satellite = 'Terra'), 0.0)::FLOAT AS phi_modis_terra,
            COALESCE(MAX(frp) FILTER (WHERE sensor_family = 'MODIS'), 0.0)::FLOAT AS phi_max_modis,
            AVG(scan) FILTER (WHERE sensor_family = 'MODIS')::FLOAT AS scan_bar_modis,
            AVG(track) FILTER (WHERE sensor_family = 'MODIS')::FLOAT AS track_bar_modis,
            COUNT(*) FILTER (WHERE {viirs_filter})::INT32 AS n_viirs_raw,
            COALESCE(SUM(frp) FILTER (WHERE {viirs_filter}), 0.0)::FLOAT AS phi_viirs_raw
        FROM read_parquet('{input_posix}')
        WHERE fecha_local BETWEEN DATE '{start_date.isoformat()}' AND DATE '{end_date.isoformat()}'
        GROUP BY cell_id, fecha_local
    ),
    lags AS (
        SELECT UNNEST(GENERATE_SERIES(0, 6))::INT32 AS d
    ),
    ventanas_activas AS (
        SELECT
            f.cell_id,
            (f.fecha + l.d * INTERVAL '1 day')::DATE AS fecha,
            SUM(f.n_modis) FILTER (WHERE l.d <= 2)::INT32 AS n_modis_3d,
            SUM(f.n_modis)::INT32 AS n_modis_7d,
            SUM(f.phi_modis)::FLOAT AS phi_modis_7d
        FROM firms_diario f
        CROSS JOIN lags l
        WHERE f.n_modis > 0
          AND (f.fecha + l.d * INTERVAL '1 day')::DATE BETWEEN DATE '{out_start.isoformat()}' AND DATE '{out_end.isoformat()}'
        GROUP BY f.cell_id, (f.fecha + l.d * INTERVAL '1 day')::DATE
    )
    SELECT
        f.cell_id,
        c.grid_row,
        c.grid_col,
        c.lat_centro,
        c.lon_centro,
        f.fecha,
        YEAR(f.fecha)::INT16 AS anio,
        MONTH(f.fecha)::INT16 AS mes,
        f.n_modis,
        f.n_modis_aqua,
        f.n_modis_terra,
        f.phi_modis,
        f.phi_modis_aqua,
        f.phi_modis_terra,
        f.phi_max_modis,
        f.scan_bar_modis,
        f.track_bar_modis,
        COALESCE(v.n_modis_3d, 0)::INT32 AS n_modis_3d,
        COALESCE(v.n_modis_7d, 0)::INT32 AS n_modis_7d,
        COALESCE(v.phi_modis_7d, 0.0)::FLOAT AS phi_modis_7d,
        CASE WHEN f.fecha < DATE '{viirs_start_date.isoformat()}' THEN NULL::INT32 ELSE f.n_viirs_raw END AS n_viirs_n20,
        CASE WHEN f.fecha < DATE '{viirs_start_date.isoformat()}' THEN NULL::FLOAT ELSE f.phi_viirs_raw END AS phi_viirs_n20
    FROM firms_diario f
    JOIN cell_coords c ON f.cell_id = c.cell_id
    LEFT JOIN ventanas_activas v ON f.cell_id = v.cell_id AND f.fecha = v.fecha
    WHERE f.fecha BETWEEN DATE '{out_start.isoformat()}' AND DATE '{out_end.isoformat()}'
      AND (f.n_modis > 0 OR (f.fecha >= DATE '{viirs_start_date.isoformat()}' AND f.n_viirs_raw > 0))
    """


def run_aggregation_pipeline(
    input_parquet: Path,
    output_parquet: Path,
    mask_parquet: Path | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    viirs_satellite: str | None = "N20",
    viirs_start_date: date = VIIRS_N20_START_DATE,
    progress_factory: Callable[[int], ProgressTracker] | None = None,
) -> AggregationStats:
    """Execute Phase 2 DuckDB sparse daily aggregation (single file or year-partitioned dir)."""
    if not input_parquet.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_parquet}")

    is_partitioned = output_parquet.suffix.lower() != ".parquet"
    if is_partitioned:
        output_parquet.mkdir(parents=True, exist_ok=True)
    else:
        output_parquet.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    try:
        con.execute("SET memory_limit = '4GB'; SET preserve_insertion_order = false;")
        input_posix = input_parquet.as_posix()
        cells_source = mask_parquet.as_posix() if mask_parquet is not None else input_posix
        active_cells = build_cell_coords_table(con, cells_source)
        if active_cells == 0:
            raise ValueError(f"No se encontraron celdas activas en {cells_source}")

        min_d, max_d, input_rows = con.execute(
            f"SELECT MIN(fecha_local)::DATE, MAX(fecha_local)::DATE, COUNT(*) FROM read_parquet('{input_posix}')"
        ).fetchone()
        if input_rows == 0 or min_d is None or max_d is None:
            raise ValueError(f"El archivo limpio {input_parquet} no contiene filas.")

        eff_start = start_date if start_date is not None else min_d
        eff_end = end_date if end_date is not None else max_d
        total_days = (eff_end - eff_start).days + 1
        if total_days <= 0:
            raise ValueError(f"Rango de fechas invalido: {eff_start} .. {eff_end}")

        years = list(range(eff_start.year, eff_end.year + 1)) if is_partitioned else [0]
        tracker = (
            progress_factory(len(years) + 1)
            if progress_factory is not None
            else ProgressTracker(total_units=len(years) + 1)
        )

        total_rows = 0
        modis_active_rows = 0
        n20_active_rows = 0

        if is_partitioned:
            for year in years:
                y_start = max(eff_start, date(year, 1, 1))
                y_end = min(eff_end, date(year, 12, 31))
                lookback_start = max(eff_start, y_start - timedelta(days=6))
                part_file = output_parquet / f"year={year}" / "data.parquet"
                part_file.parent.mkdir(parents=True, exist_ok=True)
                query = build_daily_aggregation_sql(
                    input_posix=input_posix,
                    start_date=lookback_start,
                    end_date=y_end,
                    emit_start_date=y_start,
                    emit_end_date=y_end,
                    viirs_satellite=viirs_satellite,
                    viirs_start_date=viirs_start_date,
                )
                part_posix = part_file.as_posix()
                con.execute(f"COPY ({query}) TO '{part_posix}' (FORMAT PARQUET, COMPRESSION ZSTD)")
                y_tot, y_m, y_v = con.execute(
                    f"""
                    SELECT
                        COUNT(*)::BIGINT,
                        COUNT(*) FILTER (WHERE n_modis > 0)::BIGINT,
                        COUNT(*) FILTER (WHERE COALESCE(n_viirs_n20, 0) > 0)::BIGINT
                    FROM read_parquet('{part_posix}')
                    """
                ).fetchone()
                total_rows += int(y_tot or 0)
                modis_active_rows += int(y_m or 0)
                n20_active_rows += int(y_v or 0)
                tracker.update(
                    1,
                    label=f"Anio {year} exportado",
                    rows_read=int(input_rows),
                    rows_kept=total_rows,
                    force=True,
                )
        else:
            query = build_daily_aggregation_sql(
                input_posix=input_posix,
                start_date=eff_start,
                end_date=eff_end,
                viirs_satellite=viirs_satellite,
                viirs_start_date=viirs_start_date,
            )
            out_posix = output_parquet.as_posix()
            con.execute(f"COPY ({query}) TO '{out_posix}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            tracker.update(1, label="Parquet ZSTD escrito", rows_read=int(input_rows), rows_kept=0, force=True)
            r_tot, r_m, r_v = con.execute(
                f"""
                SELECT
                    COUNT(*)::BIGINT,
                    COUNT(*) FILTER (WHERE n_modis > 0)::BIGINT,
                    COUNT(*) FILTER (WHERE COALESCE(n_viirs_n20, 0) > 0)::BIGINT
                FROM read_parquet('{out_posix}')
                """
            ).fetchone()
            total_rows = int(r_tot or 0)
            modis_active_rows = int(r_m or 0)
            n20_active_rows = int(r_v or 0)

        tracker.update(1, label="Fase 2 completada", rows_read=int(input_rows), rows_kept=total_rows, force=True)
        return AggregationStats(
            total_rows=total_rows,
            active_cells=active_cells,
            total_days=total_days,
            modis_active_rows=modis_active_rows,
            n20_active_rows=n20_active_rows,
            min_date=eff_start,
            max_date=eff_end,
            elapsed_seconds=time.perf_counter() - tracker.start_time,
        )
    finally:
        con.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for Phase 2 sparse daily aggregation."""
    parser = argparse.ArgumentParser(
        description="Fase 2 ETL: Matriz diaria dispersa (MODIS Aqua/Terra + VIIRS NOAA-20) en DuckDB."
    )
    parser.add_argument("--input", type=Path, default=Path("data/clean/firms_argentina_clean.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/firms_argentina_daily"))
    parser.add_argument("--mask-parquet", type=Path, default=None)
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument(
        "--viirs-satellite",
        type=str,
        default="N20",
        help="Satelite VIIRS de referencia unica (N20 por defecto).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    stats = run_aggregation_pipeline(
        input_parquet=args.input,
        output_parquet=args.output,
        mask_parquet=args.mask_parquet,
        start_date=args.start_date,
        end_date=args.end_date,
        viirs_satellite=args.viirs_satellite,
    )
    print(
        f"\nFase 2 dispersa completada en {stats.elapsed_seconds:.2f}s | "
        f"Celdas activas: {stats.active_cells:,} | Filas activas: {stats.total_rows:,} | "
        f"MODIS>0: {stats.modis_active_rows:,} | N20>0: {stats.n20_active_rows:,} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
