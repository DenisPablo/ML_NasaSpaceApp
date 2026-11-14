"""Phase 2 ETL: Daily aggregation, explicit zero imputation, and rolling windows in DuckDB."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
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
) -> str:
    """Build DuckDB SQL for cartesian zero imputation and 3d/7d/30d rolling windows."""
    cells_source = mask_posix if mask_posix is not None else input_posix
    return f"""
    WITH celdas_activas AS (
        SELECT DISTINCT cell_id::INT32 AS cell_id
        FROM read_parquet('{cells_source}')
    ),
    calendario AS (
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
            COUNT(*)::INT32 AS n_viirs,
            SUM(frp)::FLOAT AS phi_viirs,
            MAX(frp)::FLOAT AS phi_max_viirs,
            AVG(scan)::FLOAT AS scan_bar,
            AVG(track)::FLOAT AS track_bar
        FROM read_parquet('{input_posix}')
        WHERE fecha_local BETWEEN DATE '{start_date.isoformat()}' AND DATE '{end_date.isoformat()}'
        GROUP BY cell_id, fecha_local
    ),
    imputado AS (
        SELECT
            g.cell_id,
            g.fecha,
            COALESCE(f.n_viirs, 0)::INT32 AS n_viirs,
            COALESCE(f.phi_viirs, 0.0)::FLOAT AS phi_viirs,
            COALESCE(f.phi_max_viirs, 0.0)::FLOAT AS phi_max_viirs,
            f.scan_bar::FLOAT AS scan_bar,
            f.track_bar::FLOAT AS track_bar,
            CASE WHEN COALESCE(f.n_viirs, 0) > 0 THEN 1::TINYINT ELSE 0::TINYINT END AS y_occ
        FROM grilla_base g
        LEFT JOIN firms_diario f
            ON g.cell_id = f.cell_id AND g.fecha = f.fecha
    )
    SELECT
        cell_id,
        fecha,
        n_viirs,
        phi_viirs,
        phi_max_viirs,
        scan_bar,
        track_bar,
        y_occ,
        SUM(n_viirs) OVER (
            PARTITION BY cell_id ORDER BY fecha ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        )::INT32 AS n_viirs_3d,
        SUM(n_viirs) OVER (
            PARTITION BY cell_id ORDER BY fecha ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        )::INT32 AS n_viirs_7d,
        SUM(n_viirs) OVER (
            PARTITION BY cell_id ORDER BY fecha ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        )::INT32 AS n_viirs_30d,
        SUM(phi_viirs) OVER (
            PARTITION BY cell_id ORDER BY fecha ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        )::FLOAT AS phi_viirs_3d,
        SUM(phi_viirs) OVER (
            PARTITION BY cell_id ORDER BY fecha ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        )::FLOAT AS phi_viirs_7d,
        SUM(phi_viirs) OVER (
            PARTITION BY cell_id ORDER BY fecha ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        )::FLOAT AS phi_viirs_30d
    FROM imputado
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
    """Execute Phase 2 DuckDB daily aggregation with progress percentages."""
    if not input_parquet.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_parquet}")

    tracker = (
        progress_factory(100)
        if progress_factory is not None
        else ProgressTracker(total_units=100)
    )
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    try:
        input_posix = input_parquet.as_posix()
        mask_posix = mask_parquet.as_posix() if mask_parquet is not None else None

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

        tracker.update(
            15,
            label="Leyendo metadata y grilla",
            rows_read=int(input_rows),
            rows_kept=0,
            force=True,
        )

        query = build_daily_aggregation_sql(
            input_posix=input_posix,
            start_date=eff_start,
            end_date=eff_end,
            mask_posix=mask_posix,
        )
        tracker.update(
            25,
            label="Calculando grilla cartesiana + ventanas 3d/7d/30d",
            rows_read=int(input_rows),
            rows_kept=0,
            force=True,
        )

        out_posix = output_parquet.as_posix()
        con.execute(f"COPY ({query}) TO '{out_posix}' (FORMAT PARQUET, COMPRESSION ZSTD)")

        tracker.update(
            45,
            label="Escribiendo Parquet ZSTD",
            rows_read=int(input_rows),
            rows_kept=0,
            force=True,
        )

        total_rows, active_cells, positive_rows = con.execute(
            f"""
            SELECT
                COUNT(*)::BIGINT,
                COUNT(DISTINCT cell_id)::BIGINT,
                SUM(y_occ)::BIGINT
            FROM read_parquet('{out_posix}')
            """
        ).fetchone()

        total_rows = int(total_rows or 0)
        active_cells = int(active_cells or 0)
        positive_rows = int(positive_rows or 0)
        zero_rate = ((total_rows - positive_rows) / total_rows) if total_rows > 0 else 0.0

        tracker.update(
            15,
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
        default=Path("data/clean/viirs_snpp_san_luis_clean.parquet"),
        help="Parquet limpio generado en la Fase 1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/viirs_snpp_san_luis_daily.parquet"),
        help="Parquet diario de salida con ceros imputados y ventanas moviles.",
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
