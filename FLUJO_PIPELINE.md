# Flujo del Pipeline: Datos Originales → Limpieza → Matriz Dispersa → ML → Serving

Guía operativa del flujo de datos para **[NASA Space Apps 2026 — Harmonization of MODIS and VIIRS Hot Spots](https://www.spaceappschallenge.org/2026/challenges/modis-and-viirs-hot-spots/)** a **escala Nacional (Argentina Continental)**.

> **Nota:** este documento describe las fases 2–4 experimentales (referencia NOAA-20 y factor `5,5×`). El producto publicado para la API usa S-NPP (`2012-01-20 → 2026-11-02`) y NOAA-20 desde `2026-11-03`, con factores jerárquicos provincia × mes: ver [CLASIFICACION.md](CLASIFICACION.md).

---

## Diagrama de Fases

```text
[Data-Argentina/*.csv (6 archivos, 4.104.670 filas)]
   │  MODIS (M-C61) + VIIRS NOAA-20 (J1V-C2) + VIIRS NOAA-21 (J2V-C2)
   ▼
1. clean_firms.py  ──► data/clean/firms_argentina_clean.parquet
   │  type=0 (y lista negra de antorchas en NRT), confianza, UTC-3 fijo,
   │  Albers Argentina (+proj=aea) 5x5 km (cell_id), dedup SP > NRT
   ▼
2. aggregate_daily.py ──► data/processed/firms_argentina_daily/year=YYYY/data.parquet
   │  Matriz dispersa de celdas activas (n_modis > 0 o n_viirs_n20 > 0)
   │  Desglose Aqua/Terra + Target único NOAA-20 (NULL antes de 2018-04-01)
   ▼
3. train_hurdle.py ──► models/ (Tweedie Conteos p=1.45 + Tweedie FRP p=1.62)
   │  Calibración 2018-04-01 a 2025-12-31 + Baseline 5,5x
   │  Validación doble sin submuestreo: Temporal (2023-2025) + Espacial LOBBO (20 km)
   ▼
4. predict_harmonized.py ──► data/serving/firms_argentina_harmonized/year=YYYY/data.parquet
   │  Backfill < 2018-04-01 (is_estimate=true) + Observado N20 >= 2018-04-01 (is_estimate=false)
   ▼
[Backend FastAPI + DuckDB (VPS DonWeb 4 GB RAM / 2 vCPU)]
   └── /api/v1/calendar (Matriz Año × Mes + mediana histórica) + /hotspots.csv + GeoJSON
```

---

## Salida y Rol de Cada Fase

### 1. Datos Originales (`Data-Argentina/`)
- **Contenido:** 6 archivos CSV (`fire_archive` + `fire_nrt` de `M-C61`, `J1V-C2` y `J2V-C2`, **4.104.670 filas**).
- **Nota sobre `SV-C2`:** No se incluye Suomi-NPP (`2012–2017`), por lo que la serie VIIRS real arranca el **`2018-04-01`** con **NOAA-20 (`N20`)**. Se descarta sumar `NOAA-21` al target para evitar el salto artificial de `×1,92` en FRP y `+37%` en celda-días desde marzo de 2023.

### 2. Limpieza (`clean_firms.py`)
- **Entrada:** CSVs crudos en streaming.
- **Transformaciones:**
  - Filtro de confianza (`'n'/'h'` en VIIRS; `>= 30` en MODIS) y `type == 0` en `SP`.
  - **Lista negra estática en `NRT`:** Como los `NRT` (agosto–noviembre 2026) omiten `type`, se filtran coordenadas redondeadas a `0.01°` con historial recurrente de `type == 2` (antorchas industriales/refinerías) en `SP`.
  - Conversión **`UTC → UTC-3` fijo** (mantiene constante la hora solar local de satélites heliosíncronos).
  - Proyección a **Argentina Albers (`+proj=aea`)** y grilla `5 × 5 km` (`25 km²`). Ejemplo en San Luis (`lat = -33.3014`, `lon = -66.3321`): `cell_id = 229643` (`col = 163`, `row = 499`).
  - Deduplicación priorizando `SP > NRT`.

### 3. Agregación Diaria Dispersa (`aggregate_daily.py`)
- **Cambio clave de arquitectura:** En vez de expandir ceros sobre `111.248` celdas × `9.500` días (~1.000M de filas), conserva únicamente **celdas-día activas** (`n_modis > 0` o `n_viirs_n20 > 0`), reduciendo la serie nacional de 26 años a **~150.000–300.000 filas activas**.
- **Desglose MODIS:** Separa `n_modis_aqua` ($r = 0,967$ con N20), `n_modis_terra` ($r = 0,918$), `phi_modis_aqua`, `phi_modis_terra` y totales `n_modis`, `phi_modis`, `phi_max_modis`, `scan_bar_modis` (km), `track_bar_modis` (km), más ventanas `3d`/`7d`.
- **Target VIIRS único:** `n_viirs_n20` y `phi_viirs_n20` (solo `satellite == 'N20'`). Antes de `2018-04-01` valen **`NULL`** (no `0`).

### 4. Entrenamiento ML (`train_hurdle.py`)
- **Ventana:** `2018-04-01` a `2025-12-31` sobre filas con `n_modis > 0` (si `n_modis == 0` antes de `2018-04-01`, la salida es `0` sin inferencia).
- **Baseline + 2 Modelos Tweedie (`num_leaves=31`, `min_child_samples=50`, `n_estimators=150`):**
  1. **Baseline estadístico:** `n_scaled_baseline = ROUND(n_modis * 5.5)` (o factor mensual $k_m$).
  2. **Modelo A (Conteos — producto oficial NASA):** LightGBM Tweedie `p=1.45` → `n_harmonized_ml`.
  3. **Modelo B (FRP MW — capa energética):** LightGBM Tweedie `p=1.62` → `phi_harmonized_ml`.
- **Validación doble sin submuestreo:** Temporal (`2018-04..2022-12` → `2023..2025`) y Espacial LOBBO (buffer `20 km` / 4 celdas), con foco en **San Luis** y **Delta del Paraná**.

### 5. Inferencia y *Serving* (`predict_harmonized.py`)
- **Salida:** `data/serving/firms_argentina_harmonized/year=YYYY/data.parquet`.
- **Empalme:**
  - `< 2018-04-01` (`is_estimate = true`): `n_harmonized = n_harmonized_ml`, `phi_harmonized = phi_harmonized_ml`, `n_viirs_n20 = NULL`, `phi_viirs_n20 = NULL`.
  - `>= 2018-04-01` (`is_estimate = false`): `n_harmonized = n_viirs_n20`, `phi_harmonized = phi_viirs_n20`.
- **Consumo en Backend:** DuckDB sobre VPS DonWeb (`4 GB RAM / 2 vCPU`) responde consultas espaciales y calendario en `< 20 ms`.
