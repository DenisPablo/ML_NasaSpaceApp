# Contratos de Datos y Backend (Serving Parquet + FastAPI + DuckDB)

Contratos técnicos de almacenamiento y API REST/GeoJSON para servir la serie armonizada nacional (`2000–2026`) en un VPS (**DonWeb: 4 GB RAM / 2 vCPU**) mediante **DuckDB + FastAPI**, cumpliendo los requisitos de **[NASA Space Apps 2026 — Harmonization of MODIS and VIIRS Hot Spots](https://www.spaceappschallenge.org/2026/challenges/modis-and-viirs-hot-spots/)**.

---

## 1. Esquema de Serving (`data/serving/firms_argentina_harmonized/year=YYYY/data.parquet`)

Tabla **dispersa de celdas activas** (`n_modis > 0` o `n_viirs_n20 > 0`, ~150.000–300.000 filas totales en 26 años), proyectada en **Argentina Albers (`+proj=aea`, nunca `EPSG:5347`)** con celdas `5 × 5 km`. Ejemplo de referencia en San Luis (`lat = -33.3014`, `lon = -66.3321`): `cell_id = 229643`, `grid_col = 163`, `grid_row = 499`.

| Columna | Tipo | Nullable | Descripción |
| :--- | :--- | :---: | :--- |
| `cell_id` | `INT32` | No | Identificador único de celda `5 × 5 km`. |
| `grid_row`, `grid_col` | `INT32` | No | Fila y columna en la grilla Albers nacional. |
| `lat_centro`, `lon_centro` | `FLOAT32` | No | Centroide WGS84 de la celda (para filtrado `bbox` y mapas). |
| `fecha` | `DATE32` | No | Fecha operativa local (`UTC-3` fijo). |
| `anio`, `mes` | `INT16` | No | Año (`2000–2026`) y mes (`1–12`) precalculados para agregación de calendario. |
| `n_modis` | `INT32` | No | Conteo total MODIS (`Aqua + Terra`). |
| `n_modis_aqua`, `n_modis_terra` | `INT32` | No | Conteo desglosado por satélite MODIS. |
| `n_viirs_n20` | `INT32` | Sí | Conteo observado **VIIRS NOAA-20** (`NULL` antes de `2018-04-01`). |
| `n_scaled_baseline` | `INT32` | No | Baseline estadístico `ROUND(n_modis * 5.5)`. |
| `n_harmonized` | `INT32` / `FLOAT32` | No | Conteo oficial armonizado (`n_harmonized_ml` si `is_estimate`, sino `n_viirs_n20`). |
| `phi_modis` | `FLOAT32` | No | FRP acumulado MODIS (MW). |
| `phi_viirs_n20` | `FLOAT32` | Sí | FRP observado **VIIRS NOAA-20** en MW (`NULL` antes de `2018-04-01`). |
| `phi_harmonized` | `FLOAT32` | No | FRP armonizado en MW (`phi_harmonized_ml` si `is_estimate`, sino `phi_viirs_n20`). |
| `is_estimate` | `BOOL` | No | `true` si `fecha < 2018-04-01` (estimado ML); `false` si es observación real `N20`. |
| `model_version` | `VARCHAR` | No | Versión del artefacto ML (ej. `tweedie-n20-v2`). |
| `f_nube` | `FLOAT32` | Sí | Fracción de cobertura nubosa (`NULL` si no disponible). |

> **Regla de serialización de nulos:** Si `f_nube IS NULL`, el campo `cloud_occluded` en cualquier respuesta JSON de la API **debe serializarse como `null`** (nunca como `false`).

---

## 2. Presets Regionales Soportados

Todos los endpoints aceptan `bbox=min_lon,min_lat,max_lon,max_lat` libre sobre Argentina Continental o un parámetro `region` predefinido:
- `argentina` (nacional completo)
- `san_luis` (Provincia de San Luis — interfaz urbano-forestal serrana)
- `delta_parana` (Delta del Paraná — estrés hídrico y pajonales)
- `gran_chaco` / `cordoba` / `corrientes`

---

## 3. Endpoints FastAPI Obligatorios

### 3.1 `GET /api/v1/calendar` (Producto Principal NASA)
Devuelve la **Matriz Año × Mes (`2000–2026`)** de actividad de quema para el área consultada.

- **Query params:** `bbox` o `region`, `start_year` (default `2000`), `end_year` (default `2026`), `metric` (`"count" | "frp"`, default `"count"`).
- **Respuesta JSON (por celda año-mes):**
```json
{
  "region": "san_luis",
  "metric": "count",
  "cells": [
    {
      "anio": 2008,
      "mes": 8,
      "n_modis_raw": 412,
      "n_viirs_obs": null,
      "n_scaled_baseline": 2266,
      "n_harmonized": 2340,
      "frp_harmonized": 118450.5,
      "is_estimate": true,
      "historical_monthly_median": 1480.0,
      "anomaly_pct": 58.11
    }
  ]
}
```

### 3.2 `GET /api/v1/hotspots.csv`
Descarga directa en **CSV** (`Content-Type: text/csv`) de la serie y focos armonizados filtrados.

- **Query params:** `bbox` o `region`, `start_date` (`YYYY-MM-DD`), `end_date` (`YYYY-MM-DD`).
- **Columnas CSV:** `fecha,cell_id,lat_centro,lon_centro,n_modis,n_viirs_n20,n_scaled_baseline,n_harmonized,phi_modis,phi_viirs_n20,phi_harmonized,is_estimate,model_version`.

### 3.3 `GET /api/v1/grid/geojson`
Devuelve una `FeatureCollection` GeoJSON de las celdas activas agregadas en un rango de fechas, lista para renderizar mapas conmutando entre conteo y energía.

- **Query params:** `bbox` o `region`, `start_date`, `end_date`, `metric` (`"count" | "frp"`).
- **Properties por Feature:** `cell_id`, `grid_row`, `grid_col`, `n_modis`, `n_viirs_n20`, `n_scaled_baseline`, `n_harmonized`, `phi_modis`, `phi_viirs_n20`, `phi_harmonized`, `is_estimate`, `model_version`, `cloud_occluded` (`null` si `f_nube IS NULL`).

### 3.4 `GET /api/v1/cells/{cell_id}/series`
Devuelve la serie temporal diaria completa (`2000–2026`) de una celda puntual.

- **Path param:** `cell_id` (ej. `229643`).
- **Query params:** `start_date`, `end_date`.
- **Campos por registro:** `fecha`, `n_modis`, `n_modis_aqua`, `n_modis_terra`, `n_viirs_n20`, `n_scaled_baseline`, `n_harmonized`, `phi_modis`, `phi_viirs_n20`, `phi_harmonized`, `is_estimate`, `model_version`, `cloud_occluded`.
