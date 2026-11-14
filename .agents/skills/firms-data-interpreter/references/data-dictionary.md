# FIRMS MODIS x VIIRS Data Dictionary

## 1. Spatial Grid & Time Reference

- **Projection**: Argentina Albers Equal Area Conic (`+proj=aea +lat_0=-37 +lon_0=-60 +lat_1=-26 +lat_2=-48 +x_0=2500000 +y_0=0 +ellps=GRS80 +units=m +no_defs`).
- **Grid cell size**: `5000 m` (`5 x 5 km`, `25 km^2` per cell).
- **Grid origin**: `X0 = 1,000,000 m`, `Y0 = -2,000,000 m`, `GRID_N_COLS = 600`.
- **Cell coordinates**: `col = floor((x_aea - X0) / 5000)`, `row = floor((y_aea - Y0) / 5000)`, `cell_id = row * 600 + col`.
- **Timezone**: Fixed `UTC-3` (`timestamp_utc3`, `fecha_local`, `fecha`). Never apply historical DST (`UTC-2`).

## 2. Sensor Coverage Eras

| Sensor Family | Satellites | Start Date | Role |
| :--- | :--- | :--- | :--- |
| `MODIS` | `Terra`, `Aqua` | `2000-11` / `2002-07` | Input features (`X`) across `2000-2026` |
| `VIIRS` | `SNPP`, `N20`, `N21` | `2012-01` / `2018-04` / `2023-03` | Target (`Y`) in `2012-2024`; `0` before `2012-01-20` is unobserved, not zero fire |

## 3. Clean Layer (`data/clean/*.parquet`)

| Column | Type / Unit | Meaning |
| :--- | :--- | :--- |
| `cell_id`, `col`, `row` | `int32` | Deterministic 5x5 km grid cell index and row/col |
| `fecha_local`, `timestamp_utc3` | `date32`, `timestamp[us]` | Local date and timestamp in fixed `UTC-3` |
| `latitude`, `longitude` | `float32` (deg) | Original WGS84 coordinates |
| `x_aea`, `y_aea` | `float32` (m) | Projected Argentina Albers coordinates |
| `brightness`, `bright_t31` | `float32` (K) | Brightness temperature (channel I4/21 and I5/31) |
| `frp` | `float32` (MW) | Fire Radiative Power per detection |
| `scan`, `track` | `float32` (km) | Ground pixel footprint dimensions |
| `confidence` | `string` | Raw confidence (`n`/`h` for VIIRS, `>= 30` for MODIS) |
| `satellite`, `instrument`, `sensor_family` | `string` | Platform, raw instrument, normalized `MODIS` or `VIIRS` |
| `source_quality`, `type` | `string`, `int32` | `SP` (`type == 0`, vegetation fire) or `NRT` (`type == -1`) |

## 4. Daily Grid Layer (`data/processed/firms_argentina_daily/year=YYYY/data.parquet`)

| Column Pattern | Unit | Meaning |
| :--- | :--- | :--- |
| `n_modis`, `n_viirs` | count | Active detections in `(cell_id, fecha)` |
| `phi_modis`, `phi_viirs` | MW | Sum of `frp` in `(cell_id, fecha)` |
| `phi_max_modis`, `phi_max_viirs` | MW | Max `frp` in `(cell_id, fecha)` |
| `scan_bar_*`, `track_bar_*` | km | Mean pixel `scan` / `track` (`NULL` when `n == 0`) |
| `y_occ_modis`, `y_occ_viirs` | `0` or `1` | Binary fire occurrence (`n > 0`) |
| `n_*_3d`, `n_*_7d`, `n_*_30d` | count | Causal rolling sum of detections (`[t-k+1, t]`) |
| `phi_*_3d`, `phi_*_7d`, `phi_*_30d` | MW | Causal rolling sum of `phi` (`[t-k+1, t]`) |

## 5. Hurdle Model Outputs

- `pi_hat`: Calibrated probability `P(y_occ_viirs = 1 | X)` from Stage 1a (Focal Loss).
- `mu_hat`: Conditional expected `phi_viirs` (MW) given `Y > 0` from Stage 1b (Tweedie `p=1.62`).
- `y_hat`: Final harmonized VIIRS-equivalent FRP (`pi_hat * mu_hat` when `pi_hat >= threshold`, else `0.0`).
