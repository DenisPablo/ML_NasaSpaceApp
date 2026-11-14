"""Phase 3 ML: Batch-by-year Hurdle GBDT (Focal Loss + Tweedie) with 20 km LOBBO split."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

import duckdb
import lightgbm as lgb
import numpy as np

from clean_firms import GRID_N_COLS, ProgressTracker

FEATURE_NAMES = [
    "n_modis",
    "phi_modis",
    "phi_max_modis",
    "scan_bar_modis",
    "track_bar_modis",
    "n_modis_3d",
    "n_modis_7d",
    "n_modis_30d",
    "phi_modis_3d",
    "phi_modis_7d",
    "phi_modis_30d",
    "grid_row",
    "grid_col",
    "doy_sin",
    "doy_cos",
]

LOBBO_BUFFER_CELLS = 4  # 4 x 5 km = 20 km spatial exclusion buffer


@dataclass
class TrainingStats:
    """Summary metrics for batch-by-year Hurdle GBDT training."""

    batches_processed: int = 0
    occ_rows_used: int = 0
    int_rows_used: int = 0
    val_rows: int = 0
    val_r2: float = 0.0
    val_mbe: float = 0.0
    val_zero_rate: float = 0.0
    elapsed_seconds: float = 0.0


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    z_clip = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


def focal_loss_objective(
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> Callable[[np.ndarray, lgb.Dataset], tuple[np.ndarray, np.ndarray]]:
    """Return custom Binary Focal Loss (grad, hess) objective for LightGBM."""
    cached: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _fobj(preds: np.ndarray, train_data: lgb.Dataset) -> tuple[np.ndarray, np.ndarray]:
        did = id(train_data)
        entry = cached.get(did)
        if entry is None:
            y = train_data.get_label()
            pos = y == 1
            alpha_t = np.where(pos, alpha, 1.0 - alpha).astype(np.float32)
            sign_t = np.where(pos, 1.0, -1.0).astype(np.float32)
            entry = (pos, alpha_t, sign_t)
            cached[did] = entry
        pos, alpha_t, sign_t = entry

        z = np.clip(preds.astype(np.float32, copy=False), -30.0, 30.0)
        p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-7, 1.0 - 1e-7)
        pt = np.where(pos, p, 1.0 - p)
        om_pt = 1.0 - pt
        om_g = (om_pt * om_pt) if gamma == 2.0 else (om_pt ** gamma)

        grad = alpha_t * sign_t * om_g * (gamma * pt * np.log(pt) - om_pt)
        # Positive Gauss-Newton / curvature approximation for stable tree splits
        hess = np.maximum(alpha_t * om_g * pt * om_pt, 1e-6)
        return grad.astype(np.float64), hess.astype(np.float64)

    return _fobj


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
    buf_low = val_min - LOBBO_BUFFER_CELLS
    buf_high = val_max + LOBBO_BUFFER_CELLS
    return val_min, val_max, buf_low, buf_high


def _feature_select_sql(source_glob: str, where_clause: str) -> str:
    return f"""
    SELECT
        n_modis::FLOAT AS n_modis,
        phi_modis::FLOAT AS phi_modis,
        phi_max_modis::FLOAT AS phi_max_modis,
        COALESCE(scan_bar_modis, -1.0)::FLOAT AS scan_bar_modis,
        COALESCE(track_bar_modis, -1.0)::FLOAT AS track_bar_modis,
        n_modis_3d::FLOAT AS n_modis_3d,
        n_modis_7d::FLOAT AS n_modis_7d,
        n_modis_30d::FLOAT AS n_modis_30d,
        phi_modis_3d::FLOAT AS phi_modis_3d,
        phi_modis_7d::FLOAT AS phi_modis_7d,
        phi_modis_30d::FLOAT AS phi_modis_30d,
        (cell_id // {GRID_N_COLS})::FLOAT AS grid_row,
        (cell_id % {GRID_N_COLS})::FLOAT AS grid_col,
        SIN(2.0 * PI() * DAYOFYEAR(fecha) / 365.25)::FLOAT AS doy_sin,
        COS(2.0 * PI() * DAYOFYEAR(fecha) / 365.25)::FLOAT AS doy_cos,
        y_occ_viirs::INT32 AS y_occ,
        phi_viirs::FLOAT AS phi_viirs
    FROM read_parquet('{source_glob}')
    WHERE {where_clause}
    """


def _fetch_float32_batch(
    con: duckdb.DuckDBPyConnection,
    sql: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute feature SQL and return (X_float32, y_occ, y_frp)."""
    tbl = con.sql(sql).to_arrow_table()
    if tbl.num_rows == 0:
        return (
            np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )
    X = np.column_stack(
        [tbl[col].to_numpy(zero_copy_only=False).astype(np.float32) for col in FEATURE_NAMES]
    )
    y_occ = tbl["y_occ"].to_numpy(zero_copy_only=False).astype(np.int32)
    y_frp = tbl["phi_viirs"].to_numpy(zero_copy_only=False).astype(np.float32)
    return X, y_occ, y_frp


def predict_hurdle(
    clf_booster: lgb.Booster,
    reg_model: lgb.LGBMRegressor,
    X: np.ndarray,
    occ_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict combined VIIRS-equivalent FRP = pi_hat * mu_hat."""
    if X.shape[0] == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    raw_scores = clf_booster.predict(X, raw_score=True)
    pi_hat = sigmoid(raw_scores).astype(np.float32)
    y_hat = np.zeros(X.shape[0], dtype=np.float32)
    active_mask = pi_hat >= occ_threshold
    if np.any(active_mask):
        mu_active = np.maximum(reg_model.predict(X[active_mask]), 0.0).astype(np.float32)
        y_hat[active_mask] = (pi_hat[active_mask] * mu_active).astype(np.float32)
    return y_hat, pi_hat


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


def train_hurdle_in_batches(
    input_path: Path,
    models_dir: Path,
    start_year: int = 2012,
    end_year: int = 2024,
    neg_sample_rate: float = 0.02,
    n_lobbo_blocks: int = 4,
    val_block: int = 0,
    trees_per_batch: int = 40,
    progress_factory: Callable[[int], ProgressTracker] | None = None,
) -> TrainingStats:
    """Train Hurdle GBDT (Stage 1a Focal Loss in year batches + Stage 1b Tweedie)."""
    parquet_files = discover_parquet_batches(input_path)
    year_files = [
        p for p in parquet_files
        if not p.parent.name.startswith("year=")
        or start_year <= int(p.parent.name.split("=", 1)[1]) <= end_year
    ]
    if not year_files:
        raise ValueError(f"No hay archivos Parquet entre {start_year} y {end_year} en {input_path}")

    total_units = len(year_files) + 2
    tracker = (
        progress_factory(total_units)
        if progress_factory is not None
        else ProgressTracker(total_units=total_units)
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        con.execute("SET memory_limit = '5GB'; SET preserve_insertion_order = false;")
        rows_arr = (
            con.sql(
                f"SELECT DISTINCT (cell_id // {GRID_N_COLS})::INT32 FROM read_parquet('{year_files[0].as_posix()}')"
            )
            .to_arrow_table()
            .column(0)
            .to_numpy(zero_copy_only=False)
        )
        val_min, val_max, buf_low, buf_high = build_lobbo_row_bounds(
            rows_arr, n_blocks=n_lobbo_blocks, val_block=val_block
        )

        train_spatial = f"((cell_id // {GRID_N_COLS}) < {buf_low} OR (cell_id // {GRID_N_COLS}) > {buf_high})"
        val_spatial = f"(cell_id // {GRID_N_COLS}) BETWEEN {val_min} AND {val_max}"
        date_window = f"YEAR(fecha) BETWEEN {start_year} AND {end_year}"
        neg_thresh = max(1, int(round(neg_sample_rate * 1000)))

        occ_X_parts: list[np.ndarray] = []
        occ_y_parts: list[np.ndarray] = []
        pos_X_parts: list[np.ndarray] = []
        pos_y_parts: list[np.ndarray] = []
        val_X_parts: list[np.ndarray] = []
        val_y_occ_parts: list[np.ndarray] = []
        val_y_frp_parts: list[np.ndarray] = []
        occ_rows_used = 0

        # Extract sampled training, positive intensity, and LOBBO validation in one Parquet scan per year
        for p_file in year_files:
            p_posix = p_file.as_posix()
            where_combined = (
                f"{date_window} AND ({train_spatial} OR {val_spatial}) AND "
                f"(y_occ_viirs = 1 OR (ABS(HASH(cell_id, fecha)) % 1000) < {neg_thresh})"
            )
            X_all, y_occ_all, y_frp_all = _fetch_float32_batch(
                con, _feature_select_sql(p_posix, where_combined)
            )
            if X_all.shape[0] > 0:
                grid_row = X_all[:, 11]
                train_mask = (grid_row < buf_low) | (grid_row > buf_high)
                val_mask = ~train_mask
                pos_mask = train_mask & (y_occ_all == 1) & (y_frp_all > 0.0)

                if np.any(train_mask):
                    X_b = X_all[train_mask]
                    occ_X_parts.append(X_b)
                    occ_y_parts.append(y_occ_all[train_mask])
                    occ_rows_used += int(X_b.shape[0])

                if np.any(pos_mask):
                    pos_X_parts.append(X_all[pos_mask])
                    pos_y_parts.append(y_frp_all[pos_mask])

                if np.any(val_mask):
                    val_X_parts.append(X_all[val_mask])
                    val_y_occ_parts.append(y_occ_all[val_mask])
                    val_y_frp_parts.append(y_frp_all[val_mask])

            tracker.update(
                1,
                label=f"Extrayendo batch ({p_file.parent.name})",
                rows_read=occ_rows_used,
                rows_kept=occ_rows_used,
                force=True,
            )

        if not occ_X_parts or not pos_X_parts:
            raise RuntimeError("No se encontraron filas de entrenamiento en la ventana indicada.")

        X_occ = np.vstack(occ_X_parts)
        y_occ = np.concatenate(occ_y_parts)

        X_pos = np.vstack(pos_X_parts)
        y_pos = np.concatenate(pos_y_parts)

        fobj = focal_loss_objective(gamma=2.0, alpha=0.75)
        lgb_params = {
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_bin": 255,
            "force_col_wise": True,
            "n_jobs": -1,
            "verbose": -1,
            "seed": 42,
        }

        # Stage 1a: Train Focal Loss classifier once across all combined years (no chronological drift)
        dtrain = lgb.Dataset(X_occ, label=y_occ, free_raw_data=True)
        clf_booster = lgb.train(
            {**lgb_params, "objective": fobj},
            dtrain,
            num_boost_round=trees_per_batch * len(year_files),
        )

        # Stage 1b: Tweedie regressor strictly on positive detections (Y > 0)
        reg_model = lgb.LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.62,
            n_estimators=200,
            **lgb_params,
        )
        reg_model.fit(X_pos, y_pos)
        int_rows_used = int(X_pos.shape[0])

        tracker.update(
            1,
            label="Etapa 1a Focal + Etapa 1b Tweedie p=1.62",
            rows_read=occ_rows_used + int_rows_used,
            rows_kept=int_rows_used,
            force=True,
        )

        # Evaluate on held-out LOBBO validation block
        if val_X_parts:
            X_val = np.vstack(val_X_parts)
            y_val_occ = np.concatenate(val_y_occ_parts)
            y_val_frp = np.concatenate(val_y_frp_parts)
            y_hat, _ = predict_hurdle(clf_booster, reg_model, X_val)
            ss_res = float(np.sum((y_val_frp - y_hat) ** 2))
            ss_tot = float(np.sum((y_val_frp - np.mean(y_val_frp)) ** 2))
            val_r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0
            val_mbe = float(np.mean(y_hat - y_val_frp))
            neg_mask = y_val_occ == 0
            val_zero_rate = float(np.mean(y_hat[neg_mask] == 0.0)) if np.any(neg_mask) else 1.0
            val_rows = int(X_val.shape[0])
        else:
            val_rows, val_r2, val_mbe, val_zero_rate = 0, 0.0, 0.0, 1.0

        clf_booster.save_model(str(models_dir / "hurdle_occ_focal.txt"))
        reg_model.booster_.save_model(str(models_dir / "hurdle_int_tweedie.txt"))

        tracker.update(
            1,
            label="Modelos guardados",
            rows_read=occ_rows_used + int_rows_used,
            rows_kept=val_rows,
            force=True,
        )

        return TrainingStats(
            batches_processed=len(year_files),
            occ_rows_used=occ_rows_used,
            int_rows_used=int_rows_used,
            val_rows=val_rows,
            val_r2=val_r2,
            val_mbe=val_mbe,
            val_zero_rate=val_zero_rate,
            elapsed_seconds=time.perf_counter() - tracker.start_time,
        )
    finally:
        con.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for Phase 3 batch training."""
    parser = argparse.ArgumentParser(description="Fase 3: Entrenamiento Hurdle GBDT por batches anuales.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/firms_argentina_daily"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--neg-sample-rate", type=float, default=0.02)
    parser.add_argument("--n-lobbo-blocks", type=int, default=4)
    parser.add_argument("--val-block", type=int, default=0)
    parser.add_argument("--trees-per-batch", type=int, default=40)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    stats = train_hurdle_in_batches(
        input_path=args.input,
        models_dir=args.models_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        neg_sample_rate=args.neg_sample_rate,
        n_lobbo_blocks=args.n_lobbo_blocks,
        val_block=args.val_block,
        trees_per_batch=args.trees_per_batch,
    )
    print(
        f"\nFase 3 completada en {stats.elapsed_seconds:.2f}s | "
        f"Batches: {stats.batches_processed} | Occ rows: {stats.occ_rows_used:,} | "
        f"Tweedie rows: {stats.int_rows_used:,} | LOBBO R2: {stats.val_r2:.4f} | "
        f"MBE: {stats.val_mbe:.4f} | Zero-rate: {stats.val_zero_rate * 100:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
