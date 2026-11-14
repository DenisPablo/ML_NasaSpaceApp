"""Phase 3 ML: Sparse active-cell Tweedie GBDTs (NOAA-20 count p=1.45 & FRP p=1.62) + Dual Validation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import time
from typing import Callable

import duckdb
import lightgbm as lgb
import numpy as np

from clean_firms import ProgressTracker

FEATURE_NAMES = [
    "n_modis",
    "n_modis_aqua",
    "n_modis_terra",
    "phi_modis",
    "phi_max_modis",
    "scan_bar_modis",
    "track_bar_modis",
    "n_modis_3d",
    "n_modis_7d",
    "phi_modis_7d",
    "grid_row",
    "grid_col",
    "doy_sin",
    "doy_cos",
]

CALIBRATION_START = date(2018, 4, 1)
CALIBRATION_END = date(2025, 12, 31)
TEMPORAL_SPLIT_END = date(2022, 12, 31)
BASELINE_SCALE_FACTOR = 5.5
LOBBO_BUFFER_CELLS = 4  # 4 x 5 km = 20 km spatial exclusion buffer
MODEL_VERSION = "tweedie-n20-v2"


@dataclass
class TrainingStats:
    """Summary metrics for dual-validated NOAA-20 Tweedie models and 5.5x baseline."""

    train_rows: int = 0
    temporal_val_rows: int = 0
    temporal_r2_count: float = 0.0
    temporal_r2_baseline: float = 0.0
    temporal_r2_frp: float = 0.0
    lobbo_val_rows: int = 0
    lobbo_r2_count: float = 0.0
    lobbo_mbe_count: float = 0.0
    lobbo_r2_frp: float = 0.0
    elapsed_seconds: float = 0.0


def build_lobbo_row_bounds(
    rows: np.ndarray,
    n_blocks: int = 4,
    val_block: int = 0,
) -> tuple[int, int, int, int]:
    """Compute (val_row_min, val_row_max, buf_low, buf_high) for 20 km LOBBO split."""
    if rows.size == 0:
        raise ValueError("No hay filas para calcular bloques LOBBO.")
    r_min = int(np.min(rows))
    r_max = int(np.max(rows))
    edges = np.linspace(r_min, r_max + 1, n_blocks + 1, dtype=np.int32)
    val_min = int(edges[val_block])
    val_max = int(edges[val_block + 1] - 1)
    return val_min, val_max, val_min - LOBBO_BUFFER_CELLS, val_max + LOBBO_BUFFER_CELLS


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0


def discover_parquet_batches(input_path: Path) -> list[Path]:
    """Return sorted yearly Parquet files or single Parquet file."""
    if input_path.is_file():
        return [input_path]
    files = sorted(input_path.glob("year=*/data.parquet"))
    if not files:
        files = sorted(input_path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos Parquet en {input_path}")
    return files


def _load_calibration_arrays(
    con: duckdb.DuckDBPyConnection,
    year_files: list[Path],
    start_date: date,
    end_date: date,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all active calibration rows (n_modis > 0, n_viirs_n20 IS NOT NULL) without subsampling."""
    files_sql = ", ".join(f"'{p.as_posix()}'" for p in year_files)
    sql = f"""
    SELECT
        n_modis::FLOAT AS n_modis,
        n_modis_aqua::FLOAT AS n_modis_aqua,
        n_modis_terra::FLOAT AS n_modis_terra,
        phi_modis::FLOAT AS phi_modis,
        phi_max_modis::FLOAT AS phi_max_modis,
        COALESCE(scan_bar_modis, -1.0)::FLOAT AS scan_bar_modis,
        COALESCE(track_bar_modis, -1.0)::FLOAT AS track_bar_modis,
        n_modis_3d::FLOAT AS n_modis_3d,
        n_modis_7d::FLOAT AS n_modis_7d,
        phi_modis_7d::FLOAT AS phi_modis_7d,
        grid_row::FLOAT AS grid_row,
        grid_col::FLOAT AS grid_col,
        SIN(2.0 * PI() * DAYOFYEAR(fecha) / 365.25)::FLOAT AS doy_sin,
        COS(2.0 * PI() * DAYOFYEAR(fecha) / 365.25)::FLOAT AS doy_cos,
        n_viirs_n20::FLOAT AS y_count,
        phi_viirs_n20::FLOAT AS y_frp,
        YEAR(fecha)::INT32 AS anio
    FROM read_parquet([{files_sql}])
    WHERE fecha BETWEEN DATE '{start_date.isoformat()}' AND DATE '{end_date.isoformat()}'
      AND n_modis > 0
      AND n_viirs_n20 IS NOT NULL
    """
    tbl = con.sql(sql).to_arrow_table()
    if tbl.num_rows == 0:
        raise RuntimeError("No se encontraron filas activas en la ventana de calibracion.")

    X = np.column_stack(
        [tbl[col].to_numpy(zero_copy_only=False).astype(np.float32) for col in FEATURE_NAMES]
    )
    y_count = tbl["y_count"].to_numpy(zero_copy_only=False).astype(np.float32)
    y_frp = tbl["y_frp"].to_numpy(zero_copy_only=False).astype(np.float32)
    years = tbl["anio"].to_numpy(zero_copy_only=False).astype(np.int32)
    return X, y_count, y_frp, years


def _fit_tweedie_pair(
    X_train: np.ndarray,
    y_count_train: np.ndarray,
    y_frp_train: np.ndarray,
    n_estimators: int = 150,
) -> tuple[lgb.LGBMRegressor, lgb.LGBMRegressor]:
    """Fit regularized LightGBM Tweedie pair: Model A (count p=1.45) and Model B (FRP p=1.62)."""
    common_params = {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 50,
        "n_estimators": n_estimators,
        "force_col_wise": True,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 42,
    }
    count_model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.45,
        **common_params,
    )
    frp_model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.62,
        **common_params,
    )
    count_model.fit(X_train, y_count_train)
    frp_model.fit(X_train, y_frp_train)
    return count_model, frp_model


def train_models_with_dual_validation(
    input_path: Path,
    models_dir: Path,
    start_date: date = CALIBRATION_START,
    end_date: date = CALIBRATION_END,
    temporal_split_year: int = TEMPORAL_SPLIT_END.year,
    n_lobbo_blocks: int = 4,
    val_block: int = 0,
    n_estimators: int = 150,
    progress_factory: Callable[[int], ProgressTracker] | None = None,
) -> TrainingStats:
    """Run Temporal + Spatial LOBBO validation (no subsampling) and fit final NOAA-20 models."""
    parquet_files = discover_parquet_batches(input_path)
    year_files = [
        p for p in parquet_files
        if not p.parent.name.startswith("year=")
        or start_date.year <= int(p.parent.name.split("=", 1)[1]) <= end_date.year
    ]
    if not year_files:
        raise ValueError(f"No hay archivos Parquet entre {start_date} y {end_date} en {input_path}")

    tracker = (
        progress_factory(4)
        if progress_factory is not None
        else ProgressTracker(total_units=4)
    )
    models_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    try:
        con.execute("SET memory_limit = '4GB'; SET preserve_insertion_order = false;")
        X, y_count, y_frp, years = _load_calibration_arrays(con, year_files, start_date, end_date)
    finally:
        con.close()

    n_rows = int(X.shape[0])
    tracker.update(1, label="Calibracion cargada (sin submuestreo)", rows_read=n_rows, rows_kept=n_rows, force=True)

    # 1. Temporal validation (2018-04..2022 -> 2023..2025)
    t_train = years <= temporal_split_year
    t_val = years > temporal_split_year
    temp_val_rows = int(np.sum(t_val))
    temp_r2_count, temp_r2_base, temp_r2_frp = 0.0, 0.0, 0.0
    if np.any(t_train) and np.any(t_val):
        m_cnt_t, m_frp_t = _fit_tweedie_pair(X[t_train], y_count[t_train], y_frp[t_train], n_estimators=n_estimators)
        pred_cnt_t = np.maximum(m_cnt_t.predict(X[t_val]), 0.0)
        pred_frp_t = np.maximum(m_frp_t.predict(X[t_val]), 0.0)
        base_cnt_t = np.round(X[t_val, 0] * BASELINE_SCALE_FACTOR)
        temp_r2_count = _r2_score(y_count[t_val], pred_cnt_t)
        temp_r2_base = _r2_score(y_count[t_val], base_cnt_t)
        temp_r2_frp = _r2_score(y_frp[t_val], pred_frp_t)
    tracker.update(1, label="Validacion Temporal", rows_read=n_rows, rows_kept=temp_val_rows, force=True)

    # 2. Spatial LOBBO validation (20 km exclusion buffer)
    grid_rows = X[:, 10].astype(np.int32)
    val_min, val_max, buf_low, buf_high = build_lobbo_row_bounds(
        grid_rows, n_blocks=n_lobbo_blocks, val_block=val_block
    )
    s_train = (grid_rows < buf_low) | (grid_rows > buf_high)
    s_val = (grid_rows >= val_min) & (grid_rows <= val_max)
    lobbo_val_rows = int(np.sum(s_val))
    lobbo_r2_count, lobbo_mbe_count, lobbo_r2_frp = 0.0, 0.0, 0.0
    if np.any(s_train) and np.any(s_val):
        m_cnt_s, m_frp_s = _fit_tweedie_pair(X[s_train], y_count[s_train], y_frp[s_train], n_estimators=n_estimators)
        pred_cnt_s = np.maximum(m_cnt_s.predict(X[s_val]), 0.0)
        pred_frp_s = np.maximum(m_frp_s.predict(X[s_val]), 0.0)
        lobbo_r2_count = _r2_score(y_count[s_val], pred_cnt_s)
        lobbo_mbe_count = float(np.mean(pred_cnt_s - y_count[s_val]))
        lobbo_r2_frp = _r2_score(y_frp[s_val], pred_frp_s)
    tracker.update(1, label="Validacion Espacial LOBBO (20 km)", rows_read=n_rows, rows_kept=lobbo_val_rows, force=True)

    # 3. Final production models on full 2018-04..2025 calibration set
    final_count_model, final_frp_model = _fit_tweedie_pair(X, y_count, y_frp, n_estimators=n_estimators)
    final_count_model.booster_.save_model(str(models_dir / "n20_count_tweedie.txt"))
    final_frp_model.booster_.save_model(str(models_dir / "n20_frp_tweedie.txt"))
    tracker.update(1, label="Modelos finales guardados", rows_read=n_rows, rows_kept=n_rows, force=True)

    return TrainingStats(
        train_rows=n_rows,
        temporal_val_rows=temp_val_rows,
        temporal_r2_count=temp_r2_count,
        temporal_r2_baseline=temp_r2_base,
        temporal_r2_frp=temp_r2_frp,
        lobbo_val_rows=lobbo_val_rows,
        lobbo_r2_count=lobbo_r2_count,
        lobbo_mbe_count=lobbo_mbe_count,
        lobbo_r2_frp=lobbo_r2_frp,
        elapsed_seconds=time.perf_counter() - tracker.start_time,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for Phase 3 NOAA-20 Tweedie training."""
    parser = argparse.ArgumentParser(
        description="Fase 3: Baseline 5.5x + LightGBM Tweedie (Conteos p=1.45 & FRP p=1.62) sobre celdas activas."
    )
    parser.add_argument("--input", type=Path, default=Path("data/processed/firms_argentina_daily"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--start-date", type=date.fromisoformat, default=CALIBRATION_START)
    parser.add_argument("--end-date", type=date.fromisoformat, default=CALIBRATION_END)
    parser.add_argument("--temporal-split-year", type=int, default=TEMPORAL_SPLIT_END.year)
    parser.add_argument("--n-lobbo-blocks", type=int, default=4)
    parser.add_argument("--val-block", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=150)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    stats = train_models_with_dual_validation(
        input_path=args.input,
        models_dir=args.models_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        temporal_split_year=args.temporal_split_year,
        n_lobbo_blocks=args.n_lobbo_blocks,
        val_block=args.val_block,
        n_estimators=args.n_estimators,
    )
    print(
        f"\nFase 3 completada en {stats.elapsed_seconds:.2f}s | Filas activas: {stats.train_rows:,} | "
        f"Temporal R2 Count={stats.temporal_r2_count:.4f} (Base 5.5x={stats.temporal_r2_baseline:.4f}), "
        f"FRP={stats.temporal_r2_frp:.4f} | "
        f"LOBBO R2 Count={stats.lobbo_r2_count:.4f} (MBE={stats.lobbo_mbe_count:.4f}), FRP={stats.lobbo_r2_frp:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
