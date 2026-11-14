# Contratos de Datos y API para Backend (DuckDB + FastAPI)

Especificación exacta del esquema de **Parquet de serving** y los contratos REST/GeoJSON de **FastAPI** para el visor en **React + MapLibre GL**, ejecutando consultas DuckDB *in-process* directo sobre disco dentro del límite de **4 GB RAM / 2 vCPU** (DonWeb Cloud).

---

## 1. Estructura exacta del Parquet de Serving

Ubicación esperada en disco (particionado por año para lectura rápida con *partition pruning* en DuckDB):

```text
data/serving/firms_argentina_harmonized/
  ├── year=2000/data.parquet
  ├── ...
  └── year=2026/data.parquet
```

### Esquema de columnas por `(cell_id, fecha)`

| Columna | Tipo Parquet | Nullable | Descripción |
| :--- | :--- | :--- | :--- |
| `cell_id` | `INT32` | No | ID único de celda `5 × 5 km` (`row * 460 + col` en Argentina Albers `+proj=aea`) |
| `grid_row` | `INT32` | No | Fila en grilla (`0..779`) |
| `grid_col` | `INT32` | No | Columna en grilla (`0..459`) |
| `lat_centro` | `FLOAT` (`float32`) | No | Latitud del centroide de la celda (`EPSG:4326`) para GeoJSON/MapLibre |
| `lon_centro` | `FLOAT` (`float32`) | No | Longitud del centroide de la celda (`EPSG:4326`) para GeoJSON/MapLibre |
| `fecha` | `DATE` | No | Fecha operativa local (`UTC-3`, `YYYY-MM-DD`) |
| `n_modis` | `INT32` | No | Focos diarios crudos MODIS (`0` si no hubo detección) |
| `phi_modis` | `FLOAT` (`float32`) | No | FRP diario acumulado crudo MODIS en MW |
| `pi_hat` | `FLOAT` (`float32`) | No | Probabilidad de ocurrencia predicha por Etapa 1a (`[0.0, 1.0]`) |
| `mu_hat` | `FLOAT` (`float32`) | No | Intensidad condicional predicha por Etapa 1b en MW |
| `phi_harmonized` | `FLOAT` (`float32`) | No | **FRP diario armonizado VIIRS-equivalente** ($\hat{\pi} \cdot \hat{\mu}$) en MW |
| `n_viirs` | `INT32` | No | Focos diarios reales VIIRS (`0` antes de 2012-01-20) |
| `phi_viirs` | `FLOAT` (`float32`) | No | FRP diario real VIIRS en MW (`0.0` antes de 2012-01-20) |
| `f_nube` | `FLOAT` (`float32`) | Sí | Fracción de nubosidad `[0.0, 1.0]` (`NULL` si no se acopló QA de nubes aún) |

---

## 2. Endpoints FastAPI (Contratos REST / GeoJSON)

### `GET /api/v1/health`
Verifica disponibilidad de la API y acceso de DuckDB a los Parquet de serving.

```json
{
  "status": "ok",
  "duckdb_ready": true,
  "min_date": "2000-11-01",
  "max_date": "2026-11-13",
  "active_cells": 65420
}
```

### `GET /api/v1/grid/geojson`
Devuelve las celdas agregadas para un rango de fechas y *bounding box* del viewport de MapLibre GL (conmutador **MODIS Crudo vs. Armonizado vs. VIIRS**).

**Query params:**
- `start_date` (`date`, requerido, `YYYY-MM-DD`)
- `end_date` (`date`, requerido, `YYYY-MM-DD`)
- `layer` (`enum`: `"harmonized"` | `"modis"` | `"viirs"`, default `"harmonized"`)
- `bbox` (`string`, opcional: `"west,south,east,north"` en `EPSG:4326` para filtrar al viewport)
- `min_frp` (`float`, default `0.0`, filtra celdas con FRP `> min_frp` para no enviar ceros al mapa)

**Response (`200 OK`, `application/geo+json`):**
```json
{
  "type": "FeatureCollection",
  "metadata": {
    "start_date": "2024-08-01",
    "end_date": "2024-08-31",
    "layer": "harmonized",
    "feature_count": 124
  },
  "features": [
    {
      "type": "Feature",
      "id": 211610,
      "geometry": {
        "type": "Point",
        "coordinates": [-66.3321, -33.3014]
      },
      "properties": {
        "cell_id": 211610,
        "frp_modis_crudo": 142.5,
        "frp_armonizado": 389.2,
        "frp_viirs_real": 401.8,
        "n_modis": 4,
        "n_viirs": 19,
        "cloud_occluded": false
      }
    }
  ]
}
```
> `cloud_occluded`: `true` cuando `AVG(f_nube) > 0.80` en el rango consultado (`false` si `f_nube` es `NULL`).

### `GET /api/v1/cells/{cell_id}/series`
Serie temporal diaria (o mensual) para graficar la continuidad `2000–2026` al hacer clic en una celda.

**Query params:**
- `start_date` (`date`, default `"2000-11-01"`)
- `end_date` (`date`, default `"2026-11-13"`)
- `granularity` (`enum`: `"day"` | `"month"` | `"year"`, default `"month"`)

**Response (`200 OK`, `application/json`):**
```json
{
  "cell_id": 211610,
  "granularity": "month",
  "points": [
    {
      "period": "2011-09-01",
      "frp_modis_crudo": 85.4,
      "frp_armonizado": 241.0,
      "frp_viirs_real": 0.0
    },
    {
      "period": "2012-09-01",
      "frp_modis_crudo": 92.1,
      "frp_armonizado": 258.7,
      "frp_viirs_real": 264.3
    }
  ]
}
```

### `GET /api/v1/series/summary`
Serie agregada nacional (o por `bbox`) para auditar la continuidad multidecadal `2000–2026` sin escalón artificial en 2012.

**Query params:** `start_date`, `end_date`, `granularity` (`"month"` | `"year"`), `bbox` (opcional).

---

## 3. Reglas de implementación en FastAPI + DuckDB (4 GB RAM)

1. **Conexión DuckDB *stateless* / *read-only*:** abrir `duckdb.connect(":memory:", config={"memory_limit": "1536MB", "threads": 2})` por consulta o pool liviano, sin mantener tablas materializadas en RAM.
2. **Partition pruning:** filtrar siempre por `fecha BETWEEN ? AND ?` y rutas `year=YYYY/data.parquet` correspondientes al rango pedido.
3. **Nunca enviar ceros masivos al frontend:** en `/grid/geojson`, aplicar `HAVING frp > :min_frp` (por defecto `0.0`) para devolver solo celdas con actividad térmica en el rango.
