"""Phase 4 Inference: Streamed Hurdle GBDT backfilling (2000-2011) and harmonized series (2000-2026)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import time
from typing import Callable

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from clean_firms import GRID_N_COLS, ProgressTracker
from train_hurdle import discover_parquet_batches, sigmoid

VIIRS_SNPP_START_DATE = date(2012, 1, 20)

REQUIRED_INPUT_COLS = [
    "cell_id", "fecha", "n_modis", "phi_modis", "phi_max_modis",
    "scan_bar_modis", "track_bar_modis", "n_modis_3d", "n_modis_7d",
    "n_modis_30d", "phi_modis_3d", "phi_modis_7d", "phi_modis_30d",
    "n_viirs", "phi_viirs",
]

HARMONIZED_SCHEMA = pa.schema([
    ("cell_id", pa.int32()), ("fecha", pa.date32()),
    ("n_modis", pa.int32()), ("phi_modis", pa.float32()),
    ("n_viirs_obs", pa.int32()), ("phi_viirs_obs", pa.float32()),
    ("pi_hat", pa.float32()), ("mu_hat", pa.float32()),
    ("y_hat_viirs_eq", pa.float32()), ("phi_harmonized", pa.float32()),
    ("is_backfilled", pa.bool_()),
])


@dataclass
class InferenceStats:
    """Summary metrics for Phase 4 harmonized backfilling."""

    files_processed: int = 0
    total_rows: int = 0
    backfilled_rows: int = 0
    observed_rows: int = 0
    positive_harmonized_rows: int = 0
    elapsed_seconds: float = 0.0


def build_feature_matrix_from_batch(batch: pa.RecordBatch) -> np.ndarray:
    """Build float32 feature matrix matching train_hurdle.FEATURE_NAMES from an Arrow batch."""
    cell_id = batch.column("cell_id").to_numpy(zero_copy_only=False).astype(np.int64)
    grid_row = (cell_id // GRID_N_COLS).astype(np.float32)
    grid_col = (cell_id % GRID_N_COLS).astype(np.float32)

    doy = pc.day_of_year(batch.column("fecha")).to_numpy(zero_copy_only=False).astype(np.float64)
    angle = (2.0 * np.pi * doy) / 365.25
    doy_sin = np.sin(angle).astype(np.float32)
    doy_cos = np.cos(angle).astype(np.float32)

    scan_bar = pc.fill_null(batch.column("scan_bar_modis"), -1.0).to_numpy(zero_copy_only=False).astype(np.float32)
    track_bar = pc.fill_null(batch.column("track_bar_modis"), -1.0).to_numpy(zero_copy_only=False).astype(np.float32)

    def _f32(name: str) -> np.ndarray:
        return batch.column(name).to_numpy(zero_copy_only=False).astype(np.float32)

    return np.column_stack([
        _f32("n_modis"), _f32("phi_modis"), _f32("phi_max_modis"),
        scan_bar, track_bar, _f32("n_modis_3d"), _f32("n_modis_7d"),
        _f32("n_modis_30d"), _f32("phi_modis_3d"), _f32("phi_modis_7d"),
        _f32("phi_modis_30d"), grid_row, grid_col, doy_sin, doy_cos,
    ])


def _infer_and_build_table(
    batch: pa.RecordBatch,
    clf_booster: lgb.Booster,
    reg_booster: lgb.Booster,
    occ_threshold: float,
    viirs_start_date: date,
    audit_all_years: bool = False,
) -> tuple[pa.Table, int, int, int]:
    """Run Hurdle inference on one Arrow batch, skipping observed/zero-MODIS rows when possible."""
    n_rows = batch.num_rows
    fecha_arr = batch.column("fecha")
    is_backfilled_arr = pc.less(fecha_arr, pa.scalar(viirs_start_date, type=pa.date32()))
    is_backfilled_np = is_backfilled_arr.to_numpy(zero_copy_only=False).astype(bool)

    if not audit_all_years and not np.any(is_backfilled_np):
        zeros_f32 = pa.repeat(pa.scalar(0.0, type=pa.float32()), n_rows)
        phi_obs_arr = pc.cast(batch.column("phi_viirs"), pa.float32())
        n_pos = int(pc.sum(pc.greater(phi_obs_arr, 0.0)).as_py() or 0)
        table = pa.table(
            {
                "cell_id": pc.cast(batch.column("cell_id"), pa.int32()),
                "fecha": pc.cast(fecha_arr, pa.date32()),
                "n_modis": pc.cast(batch.column("n_modis"), pa.int32()),
                "phi_modis": pc.cast(batch.column("phi_modis"), pa.float32()),
                "n_viirs_obs": pc.cast(batch.column("n_viirs"), pa.int32()),
                "phi_viirs_obs": phi_obs_arr,
                "pi_hat": zeros_f32,
                "mu_hat": zeros_f32,
                "y_hat_viirs_eq": zeros_f32,
                "phi_harmonized": phi_obs_arr,
                "is_backfilled": pc.cast(is_backfilled_arr, pa.bool_()),
            },
            schema=HARMONIZED_SCHEMA,
        )
        return table, 0, n_rows, n_pos

    pi_hat = np.zeros(n_rows, dtype=np.float32)
    mu_hat = np.zeros(n_rows, dtype=np.float32)
    y_hat = np.zeros(n_rows, dtype=np.float32)

    if audit_all_years:
        eval_mask = np.ones(n_rows, dtype=bool)
    else:
        modis_30d_np = batch.column("n_modis_30d").to_numpy(zero_copy_only=False) > 0
        eval_mask = is_backfilled_np & modis_30d_np

    if np.any(eval_mask):
        eval_batch = batch if np.all(eval_mask) else batch.filter(pa.array(eval_mask))
        X_eval = build_feature_matrix_from_batch(eval_batch)
        raw_scores = clf_booster.predict(X_eval, raw_score=True)
        pi_sub = sigmoid(raw_scores).astype(np.float32)

        mu_sub = np.zeros(X_eval.shape[0], dtype=np.float32)
        y_sub = np.zeros(X_eval.shape[0], dtype=np.float32)
        active_mask = pi_sub >= occ_threshold
        if np.any(active_mask):
            mu_active = np.maximum(reg_booster.predict(X_eval[active_mask]), 0.0).astype(np.float32)
            mu_sub[active_mask] = mu_active
            y_sub[active_mask] = (pi_sub[active_mask] * mu_active).astype(np.float32)

        pi_hat[eval_mask] = pi_sub
        mu_hat[eval_mask] = mu_sub
        y_hat[eval_mask] = y_sub

    phi_viirs_obs = batch.column("phi_viirs").to_numpy(zero_copy_only=False).astype(np.float32)
    phi_harmonized = np.where(is_backfilled_np, y_hat, phi_viirs_obs).astype(np.float32)

    n_backfill = int(np.sum(is_backfilled_np))
    n_obs = int(n_rows - n_backfill)
    n_pos = int(np.sum(phi_harmonized > 0.0))

    table = pa.table(
        {
            "cell_id": pc.cast(batch.column("cell_id"), pa.int32()),
            "fecha": pc.cast(fecha_arr, pa.date32()),
            "n_modis": pc.cast(batch.column("n_modis"), pa.int32()),
            "phi_modis": pc.cast(batch.column("phi_modis"), pa.float32()),
            "n_viirs_obs": pc.cast(batch.column("n_viirs"), pa.int32()),
            "phi_viirs_obs": pa.array(phi_viirs_obs, type=pa.float32()),
            "pi_hat": pa.array(pi_hat, type=pa.float32()),
            "mu_hat": pa.array(mu_hat, type=pa.float32()),
            "y_hat_viirs_eq": pa.array(y_hat, type=pa.float32()),
            "phi_harmonized": pa.array(phi_harmonized, type=pa.float32()),
            "is_backfilled": pa.array(is_backfilled_np, type=pa.bool_()),
        },
        schema=HARMONIZED_SCHEMA,
    )
    return table, n_backfill, n_obs, n_pos


def run_harmonized_inference(
    input_path: Path,
    models_dir: Path,
    output_path: Path,
    start_year: int = 2000,
    end_year: int = 2026,
    occ_threshold: float = 0.5,
    viirs_start_date: date = VIIRS_SNPP_START_DATE,
    batch_size: int = 500_000,
    audit_all_years: bool = False,
    progress_factory: Callable[[int], ProgressTracker] | None = None,
) -> InferenceStats:
    """Stream daily Parquet partitions through trained Hurdle models to build the harmonized series."""
    occ_model_path = models_dir / "hurdle_occ_focal.txt"
    int_model_path = models_dir / "hurdle_int_tweedie.txt"
    if not occ_model_path.exists() or not int_model_path.exists():
        raise FileNotFoundError(
            f"Faltan modelos entrenados en {models_dir} (hurdle_occ_focal.txt / hurdle_int_tweedie.txt)."
        )

    parquet_files = discover_parquet_batches(input_path)
    year_files = [
        p for p in parquet_files
        if not p.parent.name.startswith("year=")
        or start_year <= int(p.parent.name.split("=", 1)[1]) <= end_year
    ]
    if not year_files:
        raise ValueError(f"No hay archivos Parquet entre {start_year} y {end_year} en {input_path}")

    clf_booster = lgb.Booster(model_file=str(occ_model_path))
    reg_booster = lgb.Booster(model_file=str(int_model_path))

    is_partitioned = output_path.suffix.lower() != ".parquet"
    if is_partitioned:
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    tracker = (
        progress_factory(len(year_files))
        if progress_factory is not None
        else ProgressTracker(total_units=len(year_files))
    )

    total_rows = 0
    backfilled_rows = 0
    observed_rows = 0
    positive_rows = 0

    single_writer: pq.ParquetWriter | None = None
    try:
        if not is_partitioned:
            single_writer = pq.ParquetWriter(output_path, schema=HARMONIZED_SCHEMA, compression="zstd")

        for p_file in year_files:
            pf = pq.ParquetFile(p_file)
            part_writer = single_writer
            if is_partitioned:
                part_dir = output_path / (p_file.parent.name if p_file.parent.name.startswith("year=") else "data")
                part_dir.mkdir(parents=True, exist_ok=True)
                part_writer = pq.ParquetWriter(part_dir / "data.parquet", schema=HARMONIZED_SCHEMA, compression="zstd")

            try:
                for batch in pf.iter_batches(batch_size=batch_size, columns=REQUIRED_INPUT_COLS):
                    out_tbl, n_bf, n_ob, n_pos = _infer_and_build_table(
                        batch=batch,
                        clf_booster=clf_booster,
                        reg_booster=reg_booster,
                        occ_threshold=occ_threshold,
                        viirs_start_date=viirs_start_date,
                        audit_all_years=audit_all_years,
                    )
                    assert part_writer is not None
                    part_writer.write_table(out_tbl)
                    total_rows += out_tbl.num_rows
                    backfilled_rows += n_bf
                    observed_rows += n_ob
                    positive_rows += n_pos
            finally:
                if is_partitioned and part_writer is not None:
                    part_writer.close()

            tracker.update(
                1,
                label=f"Inferencia ({p_file.parent.name})",
                rows_read=total_rows,
                rows_kept=positive_rows,
                force=True,
            )
    finally:
        if single_writer is not None:
            single_writer.close()

    return InferenceStats(
        files_processed=len(year_files),
        total_rows=total_rows,
        backfilled_rows=backfilled_rows,
        observed_rows=observed_rows,
        positive_harmonized_rows=positive_rows,
        elapsed_seconds=time.perf_counter() - tracker.start_time,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for Phase 4 harmonized inference."""
    parser = argparse.ArgumentParser(
        description="Fase 4: Inferencia Hurdle GBDT en streaming y serie armonizada 2000-2026."
    )
    parser.add_argument("--input", type=Path, default=Path("data/processed/firms_argentina_daily"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("data/final/firms_argentina_harmonized"))
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--occ-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument(
        "--audit-all-years",
        action="store_true",
        help="Ejecutar inferencia ML tambien en 2012-2026 para auditar y_hat vs phi_viirs_obs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    stats = run_harmonized_inference(
        input_path=args.input,
        models_dir=args.models_dir,
        output_path=args.output,
        start_year=args.start_year,
        end_year=args.end_year,
        occ_threshold=args.occ_threshold,
        batch_size=args.batch_size,
        audit_all_years=args.audit_all_years,
    )
    print(
        f"\nFase 4 completada en {stats.elapsed_seconds:.2f}s | "
        f"Archivos: {stats.files_processed} | Filas totales: {stats.total_rows:,} | "
        f"Backfilled (<2012-01-20): {stats.backfilled_rows:,} | "
        f"Observadas SNPP: {stats.observed_rows:,} | "
        f"Positivas armonizadas: {stats.positive_harmonized_rows:,} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
