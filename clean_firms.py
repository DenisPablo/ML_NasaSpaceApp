"""ETL cleaning pipeline for NASA FIRMS MODIS/VIIRS CSVs with live progress percentages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import time
from typing import Callable

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
import pyarrow.parquet as pq
from pyproj import Transformer

# Custom Argentina Albers Equal Area Conic (GRS 1980 / POSGAR 2007 datum)
# NOTE: Do NOT use EPSG:5347 (which is POSGAR 2007 / Argentina zone 5 Transverse Mercator).
ARGENTINA_ALBERS_PROJ = (
    "+proj=aea +lat_0=-37 +lon_0=-60 +lat_1=-26 +lat_2=-48 "
    "+x_0=2500000 +y_0=0 +ellps=GRS80 +units=m +no_defs"
)

GRID_X_MIN = 1_400_000.0
GRID_Y_MIN = -2_100_000.0
GRID_CELL_SIZE_M = 5_000.0
GRID_N_COLS = 460
GRID_N_ROWS = 780

# Provisional San Luis bounding box (EPSG:4326) with ~5 km margin when IGN mask is absent
SAN_LUIS_BBOX = {
    "lat_min": -36.05,
    "lat_max": -31.80,
    "lon_min": -67.45,
    "lon_max": -64.80,
}

OUTPUT_SCHEMA = pa.schema(
    [
        ("cell_id", pa.int32()),
        ("col", pa.int32()),
        ("row", pa.int32()),
        ("fecha_local", pa.date32()),
        ("timestamp_utc3", pa.timestamp("us")),
        ("latitude", pa.float32()),
        ("longitude", pa.float32()),
        ("x_aea", pa.float32()),
        ("y_aea", pa.float32()),
        ("brightness", pa.float32()),
        ("bright_t31", pa.float32()),
        ("frp", pa.float32()),
        ("scan", pa.float32()),
        ("track", pa.float32()),
        ("confidence", pa.string()),
        ("satellite", pa.string()),
        ("instrument", pa.string()),
        ("version", pa.string()),
        ("daynight", pa.string()),
        ("type", pa.int8()),
        ("source_quality", pa.string()),
        ("acq_date", pa.string()),
        ("acq_time", pa.string()),
    ]
)


@dataclass
class CleaningStats:
    """Summary counters collected across all processed FIRMS files."""

    files_processed: int = 0
    rows_read: int = 0
    rows_kept: int = 0
    rows_deduped: int = 0
    elapsed_seconds: float = 0.0


class ProgressTracker:
    """Tracks byte-weighted or row-weighted progress percentages across files."""

    def __init__(
        self,
        total_units: int,
        stream=None,
        bar_width: int = 24,
        is_tty: bool | None = None,
    ) -> None:
        self.total_units = max(total_units, 1)
        self.completed_units = 0
        self.start_time = time.perf_counter()
        self.stream = stream if stream is not None else sys.stdout
        self.bar_width = bar_width
        self.is_tty = is_tty if is_tty is not None else getattr(self.stream, "isatty", lambda: False)()
        self._last_printed_pct = -1.0

    def format_line(self, pct: float, label: str, rows_read: int, rows_kept: int) -> str:
        """Render a single progress bar line with percentage, counts, and ETA."""
        clamped = min(max(pct, 0.0), 100.0)
        filled = int(round((clamped / 100.0) * self.bar_width))
        bar = "#" * filled + "-" * (self.bar_width - filled)
        elapsed = time.perf_counter() - self.start_time
        if clamped > 0.5 and clamped < 100.0:
            eta_s = elapsed * (100.0 - clamped) / clamped
            eta_str = f"ETA {eta_s:4.1f}s"
        elif clamped >= 100.0:
            eta_str = f"Total {elapsed:4.1f}s"
        else:
            eta_str = "ETA  --.-s"

        return (
            f"[{bar}] {clamped:5.1f}% | {label} | "
            f"Leidas: {rows_read:,} | Limpias: {rows_kept:,} | {eta_str}"
        )

    def update(self, delta_units: int, label: str, rows_read: int, rows_kept: int, force: bool = False) -> None:
        """Advance progress by delta_units and emit output when appropriate."""
        self.completed_units = min(self.completed_units + max(delta_units, 0), self.total_units)
        pct = (self.completed_units / self.total_units) * 100.0

        if not force and not self.is_tty and (pct - self._last_printed_pct) < 5.0 and pct < 100.0:
            return

        line = self.format_line(pct, label, rows_read, rows_kept)
        if self.is_tty:
            end = "\n" if pct >= 100.0 or force else "\r"
            self.stream.write(f"\r{line}{end}")
        else:
            self.stream.write(f"{line}\n")
        self.stream.flush()
        self._last_printed_pct = pct


def infer_source_quality(filename: str) -> str:
    """Classify FIRMS file as 'NRT' or 'SP' based on filename."""
    lower = filename.lower()
    return "NRT" if "nrt" in lower else "SP"


def build_confidence_mask(conf_col: pa.Array) -> pa.Array:
    """Keep nominal/high confidence detections ('n', 'h' for VIIRS; >=30 for MODIS)."""
    if pa.types.is_integer(conf_col.type) or pa.types.is_floating(conf_col.type):
        return pc.fill_null(pc.greater_equal(pc.cast(conf_col, pa.float64()), 30.0), False)

    conf_str = pc.utf8_lower(pc.utf8_trim_whitespace(pc.cast(conf_col, pa.string())))
    is_viirs_valid = pc.fill_null(pc.is_in(conf_str, value_set=pa.array(["n", "h"])), False)
    numeric_only = pc.if_else(
        pc.utf8_is_digit(conf_str),
        conf_str,
        pa.scalar(None, type=pa.string()),
    )
    is_modis_valid = pc.fill_null(pc.greater_equal(pc.cast(numeric_only, pa.float64()), 30.0), False)
    return pc.or_kleene(is_viirs_valid, is_modis_valid)


def convert_utc_to_local_utc3(acq_date: pa.Array, acq_time: pa.Array) -> tuple[pa.Array, pa.Array]:
    """Combine acq_date + acq_time (UTC) and shift -3h to Argentina local time."""
    date_str = pc.cast(acq_date, pa.string())
    time_raw = pc.utf8_trim_whitespace(pc.cast(acq_time, pa.string()))
    time_padded = pc.utf8_lpad(time_raw, width=4, padding="0")
    dt_str = pc.binary_join_element_wise(date_str, time_padded, " ")
    ts_utc = pc.strptime(dt_str, format="%Y-%m-%d %H%M", unit="us")
    ts_utc3 = pc.subtract(ts_utc, pa.scalar(3 * 3600 * 1_000_000, type=pa.duration("us")))
    fecha_local = pc.cast(ts_utc3, pa.date32())
    return fecha_local, ts_utc3


def project_and_hash_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    transformer: Transformer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project EPSG:4326 lon/lat to Argentina Albers and compute O(1) 5x5 km cell_id."""
    x_aea, y_aea = transformer.transform(lon, lat)
    x_aea = np.asarray(x_aea, dtype=np.float64)
    y_aea = np.asarray(y_aea, dtype=np.float64)

    col = np.floor((x_aea - GRID_X_MIN) / GRID_CELL_SIZE_M).astype(np.int32)
    row = np.floor((y_aea - GRID_Y_MIN) / GRID_CELL_SIZE_M).astype(np.int32)
    in_bounds = (
        np.isfinite(x_aea)
        & np.isfinite(y_aea)
        & (col >= 0)
        & (col < GRID_N_COLS)
        & (row >= 0)
        & (row < GRID_N_ROWS)
    )
    cell_id = (row * GRID_N_COLS + col).astype(np.int32)
    return (
        x_aea.astype(np.float32),
        y_aea.astype(np.float32),
        col,
        row,
        cell_id,
        in_bounds,
    )


def clean_batch(
    batch: pa.RecordBatch,
    source_quality: str,
    transformer: Transformer,
    region: str = "san_luis",
    valid_cell_ids: set[int] | None = None,
) -> pa.Table | None:
    """Apply FIRMS quality filters, UTC->UTC-3 conversion, projection, and float32 casting."""
    if batch.num_rows == 0:
        return None

    table = pa.Table.from_batches([batch])
    col_names = set(table.column_names)

    # 1. Basic numeric & spatial validity + optional San Luis bbox pre-filter
    lat = pc.cast(table["latitude"], pa.float64())
    lon = pc.cast(table["longitude"], pa.float64())
    frp = pc.cast(table["frp"], pa.float64())

    mask = pc.and_kleene(
        pc.and_kleene(pc.is_finite(lat), pc.is_finite(lon)),
        pc.and_kleene(pc.is_finite(frp), pc.greater(frp, 0.0)),
    )

    if region == "san_luis" and valid_cell_ids is None:
        bbox_mask = pc.and_kleene(
            pc.and_kleene(
                pc.greater_equal(lat, SAN_LUIS_BBOX["lat_min"]),
                pc.less_equal(lat, SAN_LUIS_BBOX["lat_max"]),
            ),
            pc.and_kleene(
                pc.greater_equal(lon, SAN_LUIS_BBOX["lon_min"]),
                pc.less_equal(lon, SAN_LUIS_BBOX["lon_max"]),
            ),
        )
        mask = pc.and_kleene(mask, bbox_mask)

    # 2. Type == 0 filter (vegetation fire) when column exists (SP archive)
    if "type" in col_names:
        type_int = pc.cast(table["type"], pa.int8(), safe=False)
        type_mask = pc.fill_null(pc.equal(type_int, 0), False)
        mask = pc.and_kleene(mask, type_mask)

    # 3. Confidence filter (exclude 'l' / < 30%)
    mask = pc.and_kleene(mask, build_confidence_mask(table["confidence"]))

    filtered = table.filter(mask)
    if filtered.num_rows == 0:
        return None

    # 4. UTC -> UTC-3 conversion
    acq_date_str = pc.cast(filtered["acq_date"], pa.string())
    acq_time_str = pc.utf8_lpad(
        pc.utf8_trim_whitespace(pc.cast(filtered["acq_time"], pa.string())),
        width=4,
        padding="0",
    )
    fecha_local, ts_utc3 = convert_utc_to_local_utc3(acq_date_str, acq_time_str)

    # 5. Reprojection & O(1) spatial hashing
    lon_np = filtered["longitude"].to_numpy(zero_copy_only=False).astype(np.float64)
    lat_np = filtered["latitude"].to_numpy(zero_copy_only=False).astype(np.float64)
    x_aea, y_aea, col_np, row_np, cell_id_np, in_bounds = project_and_hash_grid(
        lon_np, lat_np, transformer
    )

    if valid_cell_ids is not None:
        in_bounds = in_bounds & np.isin(cell_id_np, list(valid_cell_ids))

    if not np.any(in_bounds):
        return None

    # 6. Assemble normalized float32/int32 table
    type_arr = (
        pc.cast(filtered["type"], pa.int8())
        if "type" in col_names
        else pa.array(np.full(filtered.num_rows, -1, dtype=np.int8), type=pa.int8())
    )
    quality_arr = pa.array([source_quality] * filtered.num_rows, type=pa.string())

    bright_col = "brightness" if "brightness" in col_names else "bright_ti4"
    t31_col = "bright_t31" if "bright_t31" in col_names else "bright_ti5"

    out_table = pa.table(
        {
            "cell_id": pa.array(cell_id_np, type=pa.int32()),
            "col": pa.array(col_np, type=pa.int32()),
            "row": pa.array(row_np, type=pa.int32()),
            "fecha_local": fecha_local,
            "timestamp_utc3": ts_utc3,
            "latitude": pc.cast(filtered["latitude"], pa.float32()),
            "longitude": pc.cast(filtered["longitude"], pa.float32()),
            "x_aea": pa.array(x_aea, type=pa.float32()),
            "y_aea": pa.array(y_aea, type=pa.float32()),
            "brightness": pc.cast(filtered[bright_col], pa.float32()),
            "bright_t31": pc.cast(filtered[t31_col], pa.float32()),
            "frp": pc.cast(filtered["frp"], pa.float32()),
            "scan": pc.cast(filtered["scan"], pa.float32()),
            "track": pc.cast(filtered["track"], pa.float32()),
            "confidence": pc.utf8_lower(pc.cast(filtered["confidence"], pa.string())),
            "satellite": pc.cast(filtered["satellite"], pa.string()),
            "instrument": pc.cast(filtered["instrument"], pa.string()),
            "version": pc.cast(filtered["version"], pa.string()),
            "daynight": pc.cast(filtered["daynight"], pa.string()),
            "type": type_arr,
            "source_quality": quality_arr,
            "acq_date": acq_date_str,
            "acq_time": acq_time_str,
        },
        schema=OUTPUT_SCHEMA,
    )

    if not np.all(in_bounds):
        out_table = out_table.filter(pa.array(in_bounds))

    return out_table if out_table.num_rows > 0 else None


def load_valid_cell_ids(mask_parquet: Path | None) -> set[int] | None:
    """Load active cell_id set from dim_celdas_san_luis.parquet if provided."""
    if mask_parquet is None:
        return None
    tbl = pq.read_table(mask_parquet, columns=["cell_id"])
    return set(tbl["cell_id"].to_pylist())


def deduplicate_to_parquet(staged_parquet: Path, output_parquet: Path) -> int:
    """Deduplicate staged records preferring SP over NRT and write final Parquet."""
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        staged_posix = staged_parquet.as_posix()
        out_posix = output_parquet.as_posix()
        con.execute(
            f"""
            COPY (
                SELECT * EXCLUDE (rn)
                FROM (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY latitude, longitude, acq_date, acq_time, satellite
                            ORDER BY CASE WHEN source_quality = 'SP' THEN 0 ELSE 1 END
                        ) AS rn
                    FROM read_parquet('{staged_posix}')
                )
                WHERE rn = 1
                ORDER BY fecha_local, cell_id
            ) TO '{out_posix}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        final_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_posix}')").fetchone()[0]
        return int(final_count)
    finally:
        con.close()


def discover_csv_files(input_dir: Path) -> list[Path]:
    """Return sorted FIRMS CSV files with archive (SP) processed before NRT."""
    files = sorted(input_dir.glob("*.csv"))
    return sorted(files, key=lambda p: (0 if infer_source_quality(p.name) == "SP" else 1, p.name))


def run_cleaning_pipeline(
    input_dir: Path,
    output_parquet: Path,
    region: str = "san_luis",
    mask_parquet: Path | None = None,
    block_size_bytes: int = 8 * 1024 * 1024,
    progress_factory: Callable[[int], ProgressTracker] | None = None,
) -> CleaningStats:
    """Stream-clean FIRMS CSV files with live progress percentages and export Parquet."""
    csv_files = discover_csv_files(input_dir)
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {input_dir}")

    # Reserve 5% of progress weight for final DuckDB deduplication & ZSTD write
    total_file_bytes = sum(p.stat().st_size for p in csv_files)
    dedup_weight = max(int(total_file_bytes * 0.05), 1)
    total_units = total_file_bytes + dedup_weight

    tracker = (
        progress_factory(total_units)
        if progress_factory is not None
        else ProgressTracker(total_units=total_units)
    )
    transformer = Transformer.from_crs("EPSG:4326", ARGENTINA_ALBERS_PROJ, always_xy=True)
    valid_cell_ids = load_valid_cell_ids(mask_parquet)

    stats = CleaningStats()
    read_opts = pv.ReadOptions(block_size=block_size_bytes)
    convert_opts = pv.ConvertOptions(
        column_types={"acq_date": pa.string(), "acq_time": pa.string(), "version": pa.string()}
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        staged_path = Path(tmp_dir) / "staged.parquet"
        writer = pq.ParquetWriter(staged_path, OUTPUT_SCHEMA, compression="snappy")

        try:
            for csv_path in csv_files:
                file_size = csv_path.stat().st_size
                source_quality = infer_source_quality(csv_path.name)
                stats.files_processed += 1

                # Estimate bytes per batch proportionally using batch rows vs average row width (~85 bytes)
                # and reconcile exact file_size at EOF so progress stays monotonic and accurate.
                bytes_accounted = 0
                reader = pv.open_csv(csv_path, read_options=read_opts, convert_options=convert_opts)
                for batch in reader:
                    stats.rows_read += batch.num_rows
                    cleaned = clean_batch(
                        batch,
                        source_quality=source_quality,
                        transformer=transformer,
                        region=region,
                        valid_cell_ids=valid_cell_ids,
                    )
                    if cleaned is not None:
                        writer.write_table(cleaned)
                        stats.rows_kept += cleaned.num_rows

                    est_delta = min(batch.num_rows * 82, max(file_size - bytes_accounted, 0))
                    bytes_accounted += est_delta
                    tracker.update(
                        est_delta,
                        label=f"{csv_path.name} ({source_quality})",
                        rows_read=stats.rows_read,
                        rows_kept=stats.rows_kept,
                    )

                remaining = max(file_size - bytes_accounted, 0)
                if remaining > 0:
                    tracker.update(
                        remaining,
                        label=f"{csv_path.name} ({source_quality})",
                        rows_read=stats.rows_read,
                        rows_kept=stats.rows_kept,
                    )
        finally:
            writer.close()

        if stats.rows_kept == 0:
            output_parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist([], schema=OUTPUT_SCHEMA), output_parquet)
            stats.rows_deduped = 0
        else:
            stats.rows_deduped = deduplicate_to_parquet(staged_path, output_parquet)

        tracker.update(
            dedup_weight,
            label="Deduplicacion + Parquet ZSTD",
            rows_read=stats.rows_read,
            rows_kept=stats.rows_deduped,
            force=True,
        )

    stats.elapsed_seconds = time.perf_counter() - tracker.start_time
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for FIRMS cleaning script."""
    parser = argparse.ArgumentParser(
        description="Limpieza y normalizacion espacial/temporal de CSVs NASA FIRMS (VIIRS/MODIS)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("DL_FIRE_SV-C2_847884"),
        help="Carpeta con archivos CSV de FIRMS.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/clean/viirs_snpp_san_luis_clean.parquet"),
        help="Ruta del archivo Parquet de salida.",
    )
    parser.add_argument(
        "--region",
        choices=["san_luis", "argentina"],
        default="san_luis",
        help="Alcance espacial (por defecto: san_luis).",
    )
    parser.add_argument(
        "--mask-parquet",
        type=Path,
        default=None,
        help="Parquet opcional con columna cell_id (ej. dim_celdas_san_luis.parquet).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    stats = run_cleaning_pipeline(
        input_dir=args.input_dir,
        output_parquet=args.output,
        region=args.region,
        mask_parquet=args.mask_parquet,
    )
    print(
        f"\nLimpieza completada en {stats.elapsed_seconds:.2f}s | "
        f"Archivos: {stats.files_processed} | "
        f"Leidas: {stats.rows_read:,} | "
        f"Limpias (pre-dedup): {stats.rows_kept:,} | "
        f"Finales (Parquet): {stats.rows_deduped:,} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
