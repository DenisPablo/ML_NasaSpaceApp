"""Phase 2 ETL: Daily aggregation, explicit zero imputation, and rolling windows in DuckDB."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import time
from typing import Callable

import duckdb

from clean_firms import ProgressTracker


@dataclass
class AggregationStats:
    """Summary metrics for the daily aggregated grid."""

    total_rows: int = 0
    active_cells: int = 0
    total_days: int = 0
    positive_rows: int = 0
    zero_rate: float = 0.0
    min_date: date | None = None
    max_date: date | None = None
    elapsed_seconds: float = 0.0


def build_daily_aggregation_sql(
    input_posix: str,
    start_date: date,
    end_date: date,
    mask_posix: str | None = None,
    emit_start_date: date | None = None,
    emit_end_date: date | None = None,
    use_temp_cells: bool = False,
) -> str:
    """Build DuckDB SQL for cartesian zero imputation and 3d/7d/30d rolling windows."""
    cells_source = mask_posix if mask_posix is not None else input_posix
    out_start = emit_start_date if emit_start_date is not None else start_date
    out_end = emit_end_date if emit_end_date is not None else end_date
    cells_cte = (
        ""
        if use_temp_cells
        else f"""celdas_activas AS (
        SELECT DISTINCT cell_id::INT32 AS cell_id
        FROM read_parquet('{cells_source}')
    ),
    """
    )
    return f"""
    WITH {cells_cte}calendario AS (
        SELECT UNNEST(
            GENERATE_SERIES(DATE '{start_date.isoformat()}', DATE '{end_date.isoformat()}', INTERVAL 1 DAY)
        )::DATE AS fecha
    ),
    grilla_base AS (
        SELECT c.cell_id, cal.fecha
        FROM celdas_activas c
        CROSS JOIN calendario cal
    ),
    firms_diario AS (
        SELECT
            cell_id::INT32 AS cell_id,
            fecha_local::DATE AS fecha,
            COUNT(*) FILTER (WHERE sensor_family = 'MODIS')::INT32 AS n_modis,
            COALESCE(SUM(frp) FILTER (WHERE sensor_family = 'MODIS'), 0.0)::FLOAT AS phi_modis,
            COALESCE(MAX(frp) FILTER (WHERE sensor_family = 'MODIS'), 0.0)::FLOAT AS phi_max_modis,
            AVG(scan) FILTER (WHERE sensor_family = 'MODIS')::FLOAT AS scan_bar_modis,
            AVG(track) FILTER (WHERE sensor_family = 'MODIS')::FLOAT AS track_bar_modis,
            COUNT(*) FILTER (WHERE sensor_family = 'VIIRS')::INT32 AS n_viirs,
            COALESCE(SUM(frp) FILTER (WHERE sensor_family = 'VIIRS'), 0.0)::FLOAT AS phi_viirs,
            COALESCE(MAX(frp) FILTER (WHERE sensor_family = 'VIIRS'), 0.0)::FLOAT AS phi_max_viirs,
            AVG(scan) FILTER (WHERE sensor_family = 'VIIRS')::FLOAT AS scan_bar_viirs,
            AVG(track) FILTER (WHERE sensor_family = 'VIIRS')::FLOAT AS track_bar_viirs
        FROM read_parquet('{input_posix}')
        WHERE fecha_local BETWEEN DATE '{start_date.isoformat()}' AND DATE '{end_date.isoformat()}'
        GROUP BY cell_id, fecha_local
    ),
    imputado AS (
        SELECT
            g.cell_id,
            g.fecha,
            COALESCE(f.n_modis, 0)::INT32 AS n_modis,
            COALESCE(f.phi_modis, 0.0)::FLOAT AS phi_modis,
            COALESCE(f.phi_max_modis, 0.0)::FLOAT AS phi_max_modis,
            f.scan_bar_modis::FLOAT AS scan_bar_modis,
            f.track_bar_modis::FLOAT AS track_bar_modis,
            CASE WHEN COALESCE(f.n_modis, 0) > 0 THEN 1::TINYINT ELSE 0::TINYINT END AS y_occ_modis,
            COALESCE(f.n_viirs, 0)::INT32 AS n_viirs,
            COALESCE(f.phi_viirs, 0.0)::FLOAT AS phi_viirs,
            COALESCE(f.phi_max_viirs, 0.0)::FLOAT AS phi_max_viirs,
            f.scan_bar_viirs::FLOAT AS scan_bar_viirs,
            f.track_bar_viirs::FLOAT AS track_bar_viirs,
            CASE WHEN COALESCE(f.n_viirs, 0) > 0 THEN 1::TINYINT ELSE 0::TINYINT END AS y_occ_viirs
        FROM grilla_base g
        LEFT JOIN firms_diario f
            ON g.cell_id = f.cell_id AND g.fecha = f.fecha
    ),
    ventanas AS (
        SELECT
            cell_id,
            fecha,
            n_modis,
            phi_modis,
            phi_max_modis,
            scan_bar_modis,
            track_bar_modis,
            y_occ_modis,
            SUM(n_modis) OVER w3::INT32 AS n_modis_3d,
            SUM(n_modis) OVER w7::INT32 AS n_modis_7d,
            SUM(n_modis) OVER w30::INT32 AS n_modis_30d,
            SUM(phi_modis) OVER w3::FLOAT AS phi_modis_3d,
            SUM(phi_modis) OVER w7::FLOAT AS phi_modis_7d,
            SUM(phi_modis) OVER w30::FLOAT AS phi_modis_30d,
            n_viirs,
            phi_viirs,
            phi_max_viirs,
            scan_bar_viirs,
            track_bar_viirs,
            y_occ_viirs,
            SUM(n_viirs) OVER w3::INT32 AS n_viirs_3d,
            SUM(n_viirs) OVER w7::INT32 AS n_viirs_7d,
            SUM(n_viirs) OVER w30::INT32 AS n_viirs_30d,
            SUM(phi_viirs) OVER w3::FLOAT AS phi_viirs_3d,
            SUM(phi_viirs) OVER w7::FLOAT AS phi_viirs_7d,
            SUM(phi_viirs) OVER w30::FLOAT AS phi_viirs_30d
        FROM imputado
        WINDOW
            w3 AS (PARTITION BY cell_id ORDER BY fecha ROWS BETWEEN 2 PRECEDING AND CURRENT ROW),
            w7 AS (PARTITION BY cell_id ORDER BY fecha ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
            w30 AS (PARTITION BY cell_id ORDER BY fecha ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
    )
    SELECT *
    FROM ventanas
    WHERE fecha BETWEEN DATE '{out_start.isoformat()}' AND DATE '{out_end.isoformat()}'
    ORDER BY cell_id, fecha
    """


def run_aggregation_pipeline(
    input_parquet: Path,
    output_parquet: Path,
    mask_parquet: Path | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    progress_factory: Callable[[int], ProgressTracker] | None = None,
) -> AggregationStats:
    """Execute Phase 2 DuckDB daily aggregation (single file or year-partitioned dir)."""
    if not input_parquet.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_parquet}")

    is_partitioned = output_parquet.suffix.lower() != ".parquet"
    if is_partitioned:
        output_parquet.mkdir(parents=True, exist_ok=True)
    else:
        output_parquet.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    try:
        con.execute("SET memory_limit = '5GB'; SET preserve_insertion_order = false;")
        input_posix = input_parquet.as_posix()
        mask_posix = mask_parquet.as_posix() if mask_parquet is not None else None
        cells_source = mask_posix if mask_posix is not None else input_posix
        con.execute(
            f"CREATE TEMP TABLE celdas_activas AS SELECT DISTINCT cell_id::INT32 AS cell_id FROM read_parquet('{cells_source}')"
        )

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
        total_units = len(years) + 1
        tracker = (
            progress_factory(total_units)
            if progress_factory is not None
            else ProgressTracker(total_units=total_units)
        )

        total_rows = 0
        positive_rows = 0
        active_cells = 0

        if is_partitioned:
            for year in years:
                y_start = max(eff_start, date(year, 1, 1))
                y_end = min(eff_end, date(year, 12, 31))
                lookback_start = max(eff_start, y_start - timedelta(days=29))
                part_file = output_parquet / f"year={year}" / "data.parquet"
                part_file.parent.mkdir(parents=True, exist_ok=True)
                query = build_daily_aggregation_sql(
                    input_posix=input_posix,
                    start_date=lookback_start,
                    end_date=y_end,
                    mask_posix=mask_posix,
                    emit_start_date=y_start,
                    emit_end_date=y_end,
                    use_temp_cells=True,
                )
                part_posix = part_file.as_posix()
                con.execute(f"COPY ({query}) TO '{part_posix}' (FORMAT PARQUET, COMPRESSION ZSTD)")
                y_rows, y_pos, y_cells = con.execute(
                    f"SELECT COUNT(*)::BIGINT, SUM(y_occ_viirs)::BIGINT, COUNT(DISTINCT cell_id)::BIGINT FROM read_parquet('{part_posix}')"
                ).fetchone()
                total_rows += int(y_rows or 0)
                positive_rows += int(y_pos or 0)
                active_cells = max(active_cells, int(y_cells or 0))
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
                mask_posix=mask_posix,
                use_temp_cells=True,
            )
            out_posix = output_parquet.as_posix()
            con.execute(f"COPY ({query}) TO '{out_posix}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            tracker.update(
                1,
                label="Parquet ZSTD escrito",
                rows_read=int(input_rows),
                rows_kept=0,
                force=True,
            )
            r_tot, r_cells, r_pos = con.execute(
                f"SELECT COUNT(*)::BIGINT, COUNT(DISTINCT cell_id)::BIGINT, SUM(y_occ_viirs)::BIGINT FROM read_parquet('{out_posix}')"
            ).fetchone()
            total_rows = int(r_tot or 0)
            active_cells = int(r_cells or 0)
            positive_rows = int(r_pos or 0)

        zero_rate = ((total_rows - positive_rows) / total_rows) if total_rows > 0 else 0.0
        tracker.update(
            1,
            label="Fase 2 completada",
            rows_read=int(input_rows),
            rows_kept=total_rows,
            force=True,
        )

        return AggregationStats(
            total_rows=total_rows,
            active_cells=active_cells,
            total_days=total_days,
            positive_rows=positive_rows,
            zero_rate=zero_rate,
            min_date=eff_start,
            max_date=eff_end,
            elapsed_seconds=time.perf_counter() - tracker.start_time,
        )
    finally:
        con.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for Phase 2 daily aggregation."""
    parser = argparse.ArgumentParser(
        description="Fase 2 ETL: Agregacion diaria, imputacion de ceros y ventanas moviles en DuckDB."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/clean/firms_argentina_clean.parquet"),
        help="Parquet limpio generado en la Fase 1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/firms_argentina_daily"),
        help="Archivo .parquet o directorio particionado por anio (year=YYYY).",
    )
    parser.add_argument(
        "--mask-parquet",
        type=Path,
        default=None,
        help="Parquet opcional con columna cell_id (ej. dim_celdas_san_luis.parquet).",
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=None,
        help="Fecha inicial YYYY-MM-DD (opcional).",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=None,
        help="Fecha final YYYY-MM-DD (opcional).",
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
    )
    print(
        f"\nFase 2 completada en {stats.elapsed_seconds:.2f}s | "
        f"Celdas: {stats.active_cells:,} | Dias: {stats.total_days:,} | "
        f"Filas totales: {stats.total_rows:,} | "
        f"Positivas: {stats.positive_rows:,} ({(1.0 - stats.zero_rate) * 100:.2f}%) | "
        f"Ceros: {stats.zero_rate * 100:.2f}% -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
