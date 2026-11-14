"""Phase 4 Inference: Streamed NOAA-20 Tweedie backfilling (<2018-04) + Serving Parquet (2000-2026)."""

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

from aggregate_daily import VIIRS_N20_START_DATE
from clean_firms import ProgressTracker
from train_hurdle import BASELINE_SCALE_FACTOR, MODEL_VERSION, discover_parquet_batches

REQUIRED_INPUT_COLS = [
    "cell_id", "grid_row", "grid_col", "lat_centro", "lon_centro",
    "fecha", "anio", "mes", "n_modis", "n_modis_aqua", "n_modis_terra",
    "phi_modis", "phi_max_modis", "scan_bar_modis", "track_bar_modis",
    "n_modis_3d", "n_modis_7d", "phi_modis_7d", "n_viirs_n20", "phi_viirs_n20",
]

SERVING_SCHEMA = pa.schema([
    ("cell_id", pa.int32()), ("grid_row", pa.int32()), ("grid_col", pa.int32()),
    ("lat_centro", pa.float32()), ("lon_centro", pa.float32()),
    ("fecha", pa.date32()), ("anio", pa.int16()), ("mes", pa.int16()),
    ("n_modis", pa.int32()), ("n_modis_aqua", pa.int32()), ("n_modis_terra", pa.int32()),
    ("n_viirs_n20", pa.int32()), ("n_scaled_baseline", pa.int32()), ("n_harmonized", pa.float32()),
    ("phi_modis", pa.float32()), ("phi_viirs_n20", pa.float32()), ("phi_harmonized", pa.float32()),
    ("is_estimate", pa.bool_()), ("model_version", pa.string()), ("f_nube", pa.float32()),
])


@dataclass
class InferenceStats:
    """Summary metrics for Phase 4 serving dataset generation."""

    files_processed: int = 0
    total_rows: int = 0
    estimated_rows: int = 0
    observed_rows: int = 0
    elapsed_seconds: float = 0.0


def build_feature_matrix_from_batch(batch: pa.RecordBatch) -> np.ndarray:
    """Build float32 feature matrix matching train_hurdle.FEATURE_NAMES from an Arrow batch."""
    doy = pc.day_of_year(batch.column("fecha")).to_numpy(zero_copy_only=False).astype(np.float64)
    angle = (2.0 * np.pi * doy) / 365.25
    doy_sin = np.sin(angle).astype(np.float32)
    doy_cos = np.cos(angle).astype(np.float32)

    scan_bar = pc.fill_null(batch.column("scan_bar_modis"), -1.0).to_numpy(zero_copy_only=False).astype(np.float32)
    track_bar = pc.fill_null(batch.column("track_bar_modis"), -1.0).to_numpy(zero_copy_only=False).astype(np.float32)

    def _f32(name: str) -> np.ndarray:
        return batch.column(name).to_numpy(zero_copy_only=False).astype(np.float32)

    return np.column_stack([
        _f32("n_modis"), _f32("n_modis_aqua"), _f32("n_modis_terra"),
        _f32("phi_modis"), _f32("phi_max_modis"), scan_bar, track_bar,
        _f32("n_modis_3d"), _f32("n_modis_7d"), _f32("phi_modis_7d"),
        _f32("grid_row"), _f32("grid_col"), doy_sin, doy_cos,
    ])


def _infer_and_build_serving_table(
    batch: pa.RecordBatch,
    count_booster: lgb.Booster,
    frp_booster: lgb.Booster,
    viirs_start_date: date,
    model_version: str = MODEL_VERSION,
) -> tuple[pa.Table, int, int]:
    """Build serving Arrow table: ML estimates when < viirs_start_date & n_modis > 0, else observed N20."""
    n_rows = batch.num_rows
    fecha_arr = batch.column("fecha")
    is_est_arr = pc.less(fecha_arr, pa.scalar(viirs_start_date, type=pa.date32()))
    is_est_np = is_est_arr.to_numpy(zero_copy_only=False).astype(bool)

    n_modis_np = batch.column("n_modis").to_numpy(zero_copy_only=False).astype(np.int32)
    n_scaled_baseline = np.rint(n_modis_np.astype(np.float64) * BASELINE_SCALE_FACTOR).astype(np.int32)

    n_viirs_arr = pc.cast(batch.column("n_viirs_n20"), pa.int32())
    phi_viirs_arr = pc.cast(batch.column("phi_viirs_n20"), pa.float32())
    n_viirs_filled = pc.fill_null(n_viirs_arr, 0).to_numpy(zero_copy_only=False).astype(np.float32)
    phi_viirs_filled = pc.fill_null(phi_viirs_arr, 0.0).to_numpy(zero_copy_only=False).astype(np.float32)

    n_harmonized = n_viirs_filled.copy()
    phi_harmonized = phi_viirs_filled.copy()

    # Pre-2018-04 rows with n_modis == 0 stay 0.0; only predict where is_estimate & n_modis > 0
    eval_mask = is_est_np & (n_modis_np > 0)
    if np.any(is_est_np):
        n_harmonized[is_est_np] = 0.0
        phi_harmonized[is_est_np] = 0.0
    if np.any(eval_mask):
        eval_batch = batch if np.all(eval_mask) else batch.filter(pa.array(eval_mask))
        X_eval = build_feature_matrix_from_batch(eval_batch)
        n_harmonized[eval_mask] = np.maximum(count_booster.predict(X_eval), 0.0).astype(np.float32)
        phi_harmonized[eval_mask] = np.maximum(frp_booster.predict(X_eval), 0.0).astype(np.float32)

    n_est = int(np.sum(is_est_np))
    n_obs = int(n_rows - n_est)

    table = pa.table(
        {
            "cell_id": pc.cast(batch.column("cell_id"), pa.int32()),
            "grid_row": pc.cast(batch.column("grid_row"), pa.int32()),
            "grid_col": pc.cast(batch.column("grid_col"), pa.int32()),
            "lat_centro": pc.cast(batch.column("lat_centro"), pa.float32()),
            "lon_centro": pc.cast(batch.column("lon_centro"), pa.float32()),
            "fecha": pc.cast(fecha_arr, pa.date32()),
            "anio": pc.cast(batch.column("anio"), pa.int16()),
            "mes": pc.cast(batch.column("mes"), pa.int16()),
            "n_modis": pa.array(n_modis_np, type=pa.int32()),
            "n_modis_aqua": pc.cast(batch.column("n_modis_aqua"), pa.int32()),
            "n_modis_terra": pc.cast(batch.column("n_modis_terra"), pa.int32()),
            "n_viirs_n20": n_viirs_arr,
            "n_scaled_baseline": pa.array(n_scaled_baseline, type=pa.int32()),
            "n_harmonized": pa.array(n_harmonized, type=pa.float32()),
            "phi_modis": pc.cast(batch.column("phi_modis"), pa.float32()),
            "phi_viirs_n20": phi_viirs_arr,
            "phi_harmonized": pa.array(phi_harmonized, type=pa.float32()),
            "is_estimate": pa.array(is_est_np, type=pa.bool_()),
            "model_version": pa.repeat(pa.scalar(model_version, type=pa.string()), n_rows),
            "f_nube": pa.nulls(n_rows, type=pa.float32()),
        },
        schema=SERVING_SCHEMA,
    )
    return table, n_est, n_obs


def run_harmonized_inference(
    input_path: Path,
    models_dir: Path,
    output_path: Path,
    start_year: int = 2000,
    end_year: int = 2026,
    viirs_start_date: date = VIIRS_N20_START_DATE,
    batch_size: int = 500_000,
    progress_factory: Callable[[int], ProgressTracker] | None = None,
) -> InferenceStats:
    """Stream sparse daily Parquet partitions through trained NOAA-20 Tweedie models."""
    count_model_path = models_dir / "n20_count_tweedie.txt"
    frp_model_path = models_dir / "n20_frp_tweedie.txt"
    if not count_model_path.exists() or not frp_model_path.exists():
        raise FileNotFoundError(
            f"Faltan modelos entrenados en {models_dir} (n20_count_tweedie.txt / n20_frp_tweedie.txt)."
        )

    parquet_files = discover_parquet_batches(input_path)
    year_files = [
        p for p in parquet_files
        if not p.parent.name.startswith("year=")
        or start_year <= int(p.parent.name.split("=", 1)[1]) <= end_year
    ]
    if not year_files:
        raise ValueError(f"No hay archivos Parquet entre {start_year} y {end_year} en {input_path}")

    count_booster = lgb.Booster(model_file=str(count_model_path))
    frp_booster = lgb.Booster(model_file=str(frp_model_path))

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

    total_rows, estimated_rows, observed_rows = 0, 0, 0
    single_writer: pq.ParquetWriter | None = None
    try:
        if not is_partitioned:
            single_writer = pq.ParquetWriter(output_path, schema=SERVING_SCHEMA, compression="zstd")

        for p_file in year_files:
            pf = pq.ParquetFile(p_file)
            part_writer = single_writer
            if is_partitioned:
                part_dir = output_path / (p_file.parent.name if p_file.parent.name.startswith("year=") else "data")
                part_dir.mkdir(parents=True, exist_ok=True)
                part_writer = pq.ParquetWriter(part_dir / "data.parquet", schema=SERVING_SCHEMA, compression="zstd")

            try:
                for batch in pf.iter_batches(batch_size=batch_size, columns=REQUIRED_INPUT_COLS):
                    out_tbl, n_est, n_obs = _infer_and_build_serving_table(
                        batch=batch,
                        count_booster=count_booster,
                        frp_booster=frp_booster,
                        viirs_start_date=viirs_start_date,
                    )
                    assert part_writer is not None
                    part_writer.write_table(out_tbl)
                    total_rows += out_tbl.num_rows
                    estimated_rows += n_est
                    observed_rows += n_obs
            finally:
                if is_partitioned and part_writer is not None:
                    part_writer.close()

            tracker.update(
                1,
                label=f"Serving ({p_file.parent.name})",
                rows_read=total_rows,
                rows_kept=estimated_rows,
                force=True,
            )
    finally:
        if single_writer is not None:
            single_writer.close()

    return InferenceStats(
        files_processed=len(year_files),
        total_rows=total_rows,
        estimated_rows=estimated_rows,
        observed_rows=observed_rows,
        elapsed_seconds=time.perf_counter() - tracker.start_time,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for Phase 4 serving dataset generation."""
    parser = argparse.ArgumentParser(
        description="Fase 4: Inferencia NOAA-20 Tweedie y generacion de Parquet Serving (2000-2026)."
    )
    parser.add_argument("--input", type=Path, default=Path("data/processed/firms_argentina_daily"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("data/serving/firms_argentina_harmonized"))
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=500_000)
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
        batch_size=args.batch_size,
    )
    print(
        f"\nFase 4 completada en {stats.elapsed_seconds:.2f}s | "
        f"Archivos: {stats.files_processed} | Filas serving: {stats.total_rows:,} | "
        f"Estimadas (<2018-04): {stats.estimated_rows:,} | Observadas N20: {stats.observed_rows:,} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
