# Armonización Satelital de Focos de Calor MODIS × VIIRS (`2000–2026`) — Escala Nacional (Argentina)

Pipeline de ingeniería de datos y Machine Learning para el desafío oficial **[NASA Space Apps 2026 — Harmonization of MODIS and VIIRS Hot Spots](https://www.spaceappschallenge.org/2026/challenges/modis-and-viirs-hot-spots/)**.

Reconstruye una serie diaria nacional homogénea (`2000–2026`) de **Conteo de Focos Armonizados (`n_harmonized`)** y **Fire Radiative Power (`phi_harmonized`, MW)** sobre **Argentina Continental completa** en una grilla de área igual de `5 × 5 km` (`25 km²`), calibrando MODIS (`Terra`/`Aqua`) contra el sensor de referencia único **VIIRS NOAA-20 (`N20`)**.

---

## Arquitectura Unificada (Auditoría v2)

| Componente | Especificación |
| :--- | :--- |
| **Alcance geográfico** | **Argentina Continental completa** (`111.248` celdas totales), con presets rápidos para **San Luis**, **Delta del Paraná** y **Gran Chaco / Córdoba / Corrientes**. |
| **Estructura de datos** | **Matriz diaria dispersa de celdas activas** (`n_modis > 0` o `n_viirs > 0`), reduciendo ~1.000M de filas densas a **~150.000–300.000 filas activas** entrenables en segundos en CPU local y servibles en `< 20 ms` en VPS (`4 GB RAM / 2 vCPU`). |
| **Proyección espacial** | **Argentina Albers Equal Area Conic (`+proj=aea`)** custom (`+lat_0=-37 +lon_0=-60 +lat_1=-26 +lat_2=-48 +x_0=2500000 +y_0=0 +ellps=GRS80 +units=m +no_defs`). **Nunca usar `EPSG:5347`.** |
| **Huso horario** | **`UTC-3` fijo** para toda la serie `2000–2026` (preserva la hora solar local de órbitas heliosíncronas). |
| **Sensor de referencia** | **Producto publicado (clasificación, [CLASIFICACION.md](CLASIFICACION.md)):** VIIRS **S-NPP** (`2012-01-20 → 2026-11-02`) y **NOAA-20** desde `2026-11-03`, con factores jerárquicos provincia × mes (nacional ≈ `3,33`). Nunca se suma `N21`. *La variante NOAA-20 + `5,5×` corresponde a las fases 2–4 experimentales.* |
| **Ventana de calibración** | **`2018-04-01` a `2025-12-31`** (sin `SV-C2` en `2012–2017`; los valores observados VIIRS `< 2018-04-01` se almacenan como `NULL`). |
| **Predictores (`X`)** | Desglose `Aqua` vs. `Terra` (`n_modis_aqua`, `n_modis_terra`, `phi_modis_aqua`, `phi_modis_terra`) + `phi_modis`, `phi_max_modis`, `scan_bar_modis`, `track_bar_modis` (km), ventanas `3d`/`7d` y estacionalidad. |
| **Modelos** | **Baseline estadístico (`5,5×` / razón mediana mensual)** + **2 regresores LightGBM Tweedie** sobre celdas activas (`p=1.45` para conteos `n_harmonized_ml`, `p=1.62` para FRP `phi_harmonized_ml`). |
| **Producto oficial NASA** | **Calendario de Actividad de Quema (Matriz Año × Mes, `2000–2026`)** con comparación contra **mediana histórica multianual** y descarga CSV. |

---

## Orden de Ejecución

```powershell
# 0. Instalar dependencias en .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 1. Fase 1: Limpieza, filtro de antorchas NRT, UTC-3 y proyección Albers
.\.venv\Scripts\python.exe clean_firms.py --input-dir Data-Argentina --region argentina

# 1b. Clasificación (Propuesta B) — CAMINO OFICIAL para generar los datos de la API
#    (FireHistoryInterfaceAPI). Ver CLASIFICACION.md y EJECUCION_CLASIFICACION.md.
.\.venv\Scripts\python.exe run_classification.py --clean data/clean/firms_argentina_clean.parquet --provinces-geojson data/ref/provincias.geojson --raw-dir Data-Argentina --out data/serving_api

# --- Fases 2–4: EXPERIMENTALES / OPCIONALES (LightGBM). No generan los datos de la API. ---

# 2. Fase 2: Matriz diaria dispersa (Aqua/Terra + target NOAA-20) y ventanas 3d/7d
.\.venv\Scripts\python.exe aggregate_daily.py

# 3. Fase 3: Baseline 5,5x + LightGBM Tweedie (Conteos p=1.45 y FRP p=1.62) + Validación doble
.\.venv\Scripts\python.exe train_hurdle.py

# 4. Fase 4: Backfilling (2000-2018-03) + empalme observado NOAA-20 -> Serving Parquet
.\.venv\Scripts\python.exe predict_harmonized.py

# 5. Suite de tests unitarios
.\.venv\Scripts\python.exe -m pytest -v
```

---

## Documentación del Repositorio

- [DOCUMENTACION_PIPELINE.md](DOCUMENTACION_PIPELINE.md): Especificación técnica completa, decisiones de la Auditoría v2, variables físicas y protocolo de validación doble.
- [FLUJO_PIPELINE.md](FLUJO_PIPELINE.md): Guía paso a paso del flujo `Datos originales → Limpieza → Agregación dispersa → ML → Serving`.
- [EJECUCION_CLASIFICACION.md](EJECUCION_CLASIFICACION.md): Cómo ejecutar la clasificación, qué archivos devuelve y cuáles se entregan al backend.
- [CLASIFICACION.md](CLASIFICACION.md): Etapa de clasificación y armonización **Propuesta B** (`run_classification.py`) que publica los archivos del contrato de FireHistoryInterfaceAPI (factores provincia × mes, rangos del 80 %, puntos industriales, satélites operativos).
- [BACKEND_CONTRACTS.md](BACKEND_CONTRACTS.md): ⚠️ **Reemplazado** por el contrato de FireHistoryInterfaceAPI (`docs/02`); describe la salida experimental de las fases 2–4. Esquema exacto de la tabla de *serving* y contratos FastAPI (`/calendar`, `/hotspots.csv`, `/grid/geojson`, `/cells/{cell_id}/series`).
