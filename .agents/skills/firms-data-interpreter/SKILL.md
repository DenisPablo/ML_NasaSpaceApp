---
name: firms-data-interpreter
description: "Trigger: interpret data, explain columns, analyze parquet, FIRMS metrics, FRP, cell_id, VIIRS MODIS. Interpret raw CSV, clean Parquet, daily grid, and Hurdle ML fire data."
license: Apache-2.0
metadata:
  author: "DenisPablo"
  version: "1.0"
---

## Activation Contract

Load this skill when the user asks to interpret, inspect, or explain raw FIRMS CSVs, clean Parquet detections, daily `5 x 5 km` grid tables, rolling features, `cell_id` coordinates, or Hurdle model (`pi_hat`, `mu_hat`, `y_hat`) outputs.

## Hard Rules

- Never guess or invent row counts, distributions, dates, or schema fields; query the actual file with DuckDB or PyArrow via `.venv\Scripts\python.exe` first.
- Always distinguish **pre-VIIRS unobserved dates (`< 2012-01-20`)** from true zero-fire observations (`y_occ_viirs = 0` during `2012+`).
- Interpret `phi_*` and `frp` strictly in **MW (megawatts)** and `cell_id` strictly as a **`5 x 5 km` (`25 km^2`)** Argentina Albers grid cell, never as a point or `EPSG:5347`.
- Remember `scan_bar_*` and `track_bar_*` are `NULL` on zero-fire days (`n == 0`); ML queries must impute `-1.0` before blaming data corruption.
- Distinguish `SP` (Standard Processing, `type == 0`) from `NRT` (Near Real-Time, `type == -1`) when explaining detection quality.

## Decision Gates

| User Request | Data Layer to Inspect | Action |
| :--- | :--- | :--- |
| Raw detection anomalies, `type`, `confidence` | `Data-Argentina/**/*.csv` | Read CSV header/sample; check `source_quality` and sensor rules |
| Projected points, deduplication, `UTC-3` | `data/clean/*.parquet` | Query schema, date bounds, `sensor_family`, `source_quality` |
| Daily zero-inflation, `3d`/`7d`/`30d` persistence | `data/processed/**/data.parquet` | Query `(cell_id, fecha)` stats, positive rate, rolling sums |
| Model predictions or LOBBO metrics | `models/` or `train_hurdle.py` output | Separate `pi_hat` (occurrence), `mu_hat` (intensity), `y_hat` |

## Execution Steps

1. Read `references/data-dictionary.md` for exact column definitions, units, and sensor eras.
2. Identify which layer (`raw`, `clean`, `daily`, or `model`) matches the user's question.
3. Run a bounded read-only DuckDB/PyArrow query using `.venv\Scripts\python.exe` to inspect actual rows, counts, min/max, or null rates for the target slice.
4. Translate physical and statistical meaning clearly:
   - Map `cell_id` to `(row, col)` via `row = cell_id // 600`, `col = cell_id % 600` (and approximate Albers `x_aea`, `y_aea` if needed).
   - Explain `phi_modis` vs `phi_viirs` scale differences (~375 m VIIRS vs ~1 km MODIS pixel sensitivity).
   - Flag any sensor-era boundary (`2012-01-20` SNPP start, `2018-04` N20, `2023-03` N21) affecting counts.

## Output Contract

Return:
- **Layer & Scope**: Exact file/partition inspected and query filter used.
- **Observed Values**: Verified counts, ranges, or sample rows from tool output.
- **Physical & ML Interpretation**: Plain explanation of what the numbers mean for fire behavior, sensor physics, and Hurdle modeling.
- **Caveats**: Any null-imputation, era boundary, or `SP`/`NRT` distinction that applies.

## References

- `references/data-dictionary.md` — Verified schemas, units, Albers grid math, and sensor eras.
- `BACKEND_CONTRACTS.md` — Backend consumption contracts for `cell_id`, Parquet columns, and metrics.
- `DOCUMENTACION_PIPELINE.md` — End-to-end phase outputs and validation rules.
