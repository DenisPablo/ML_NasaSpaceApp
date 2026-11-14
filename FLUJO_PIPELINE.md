# Flujo Completo del Pipeline: Datos Originales → Limpieza → Resultado ETL → ML → Output Final

Guía de referencia rápida sobre cómo se transforma cada dato desde los CSV crudos de NASA FIRMS hasta la serie diaria armonizada **2000–2026** lista para el backend y el mapa web.

```text
[0. Datos Originales]   Data-Argentina/**/*.csv (~7,08 M de filas)
         │              1 fila = 1 píxel detectado por 1 satélite en UTC
         ▼
[1. Limpieza]           clean_firms.py ──► data/clean/firms_argentina_clean.parquet
         │              1 fila = 1 foco válido en vegetación, en UTC-3 y asignado a celda 5×5 km
         ▼
[2. Resultado ETL]      aggregate_daily.py ──► data/processed/firms_argentina_daily/year=YYYY/data.parquet
         │              1 fila = 1 celda × 1 día calendario (con ceros explícitos + ventanas 3d/7d/30d)
         ▼
[3. Entrenamiento ML]   train_hurdle.py ──► models/hurdle_occ_focal.txt + models/hurdle_int_tweedie.txt
         │              Aprende en 2012–2024 cómo traducir detecciones MODIS a escala VIIRS
         ▼
[4. Output Final]       Inferencia ──► data/serving/firms_argentina_harmonized/year=YYYY/data.parquet
                        1 fila = 1 celda × 1 día con 3 curvas: MODIS crudo, Armonizado y VIIRS real
```

---

## Etapa 0: Datos Originales (`Data-Argentina/**/*.csv`)

- **Qué contiene:** 8 archivos CSV de NASA FIRMS divididos en 4 carpetas por constelación (`fire_archive_*.csv` científico `SP` con columna `type`, y `fire_nrt_*.csv` tiempo casi real `NRT` sin `type`).
- **Satélites incluidos:** **MODIS** (`Terra` y `Aqua`, 1 km, desde Nov 2000) y **VIIRS** (`SNPP`, `N20`, `N21`, 375 m, desde Ene 2012).
- **Qué representa 1 fila:** un único píxel con fuego detectado en ese instante. No existen filas para días sin fuego.

```csv
latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight,type
-33.3014,-66.3321,325.4,1.8,1.3,2020-08-15,0130,Terra,MODIS,75,6.03,291.2,24.5,N,0
```

- **Por qué no se usa directo:** fechas en `UTC` (`0130 UTC` = `22:30` del día anterior en Argentina), incluye detecciones industriales (`type=2`) y baja confianza (`'l'` / `<30%`), coordenadas en grados (`EPSG:4326`) y esquemas distintos entre `SP`/`NRT` y `MODIS`/`VIIRS`.

---

## Etapa 1: Limpieza (`clean_firms.py`)

### Transformaciones aplicadas
1. **Lectura en *streaming*:** procesa los 8 CSV por bloques de 8 MB con progreso porcentual (`0% → 100%`).
2. **Filtros de calidad:** descarta `frp <= 0`, conserva `type == 0` en archivos `SP` (asigna `type = -1` en `NRT`) y descarta confianza baja (`'l'` en VIIRS o `< 30` en MODIS).
3. **Huso horario (`UTC → UTC-3`):** resta 3 horas antes de derivar `fecha_local`.
4. **Proyección y *Spatial Hashing* $O(1)$:** reproyecta a **Argentina Albers (`+proj=aea`)** y asigna `col`, `row` y `cell_id = row * 460 + col` (celdas de `5.000 m × 5.000 m`).
5. **Normalización y deduplicación:** genera `sensor_family` (`"MODIS"` / `"VIIRS"`), castea métricas a `float32`/`int32` y deduplica en DuckDB priorizando `SP` sobre `NRT`.

### Output de Fase 1
- **Archivo:** `data/clean/firms_argentina_clean.parquet` (Parquet comprimido `ZSTD`).
- **Qué representa 1 fila:** un foco válido individual, en hora local `UTC-3` y asignado a su celda de 5×5 km.

| `cell_id` | `col` | `row` | `fecha_local` | `timestamp_utc3` | `x_aea` | `y_aea` | `frp` | `scan` | `sensor_family` | `satellite` | `source_quality` |
| ---: | ---: | ---: | :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| `211610` | `10` | `460` | `2020-08-14` | `2020-08-14 22:30:00` | `1452130.0` | `201450.0` | `24.5` | `1.8` | `"MODIS"` | `"Terra"` | `"SP"` |

---

## Etapa 2: Resultado ETL (`aggregate_daily.py`)

### Transformaciones aplicadas
1. **Agregación diaria por sensor:** agrupa en DuckDB por `(cell_id, fecha_local)` separando features **MODIS** (`n_modis`, `phi_modis`, `phi_max_modis`, `scan_bar_modis`, `track_bar_modis`) y target **VIIRS** (`n_viirs`, `phi_viirs`, `phi_max_viirs`, `y_occ_viirs`).
2. **Imputación explícita de ceros:** producto cartesiano `celdas × días calendario`, rellenando con `0` los días sin fuego (>99,9% de la matriz).
3. **Ventanas móviles retrospectivas:** acumulados a **3, 7 y 30 días** (`_3d`, `_7d`, `_30d`) por celda para modelar memoria térmica.
4. **Particionado anual con *lookback*:** escribe `year=YYYY/data.parquet` leyendo 29 días previos del año anterior para preservar continuidad en enero.

### Output de Fase 2
- **Directorio:** `data/processed/firms_argentina_daily/year=YYYY/data.parquet`
- **Qué representa 1 fila:** el estado diario completo de **1 celda × 1 día calendario**, alineando entrada MODIS y referencia VIIRS.

| `cell_id` | `fecha` | `n_modis` | `phi_modis` | `scan_bar_modis` | `phi_modis_7d` | `y_occ_viirs` | `n_viirs` | `phi_viirs` |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `211610` | `2020-08-13` | `0` | `0.0` | `NULL` | `0.0` | `0` | `0` | `0.0` |
| `211610` | `2020-08-14` | `2` | `45.0` | `2.1` | `45.0` | `1` | `11` | `138.4` |

---

## Etapa 3: Entrenamiento ML — Hurdle GBDT (`train_hurdle.py`)

Calibra sobre la ventana de solapamiento **2012–2024** usando 15 features (`MODIS` diario + ventanas `3d/7d/30d` + `grid_row`/`grid_col` + estacionalidad cíclica `doy_sin`/`doy_cos`) en `float32`:

1. **Etapa 1a — Clasificador de Ocurrencia (`Focal Loss`, $\gamma=2.0, \alpha=0.75$):** estima $\hat{\pi}_{g,t} = \Pr(\text{VIIRS} > 0 \mid X)$. Lee de a un año (*batch*) con DuckDB tomando 100% de positivos y 2% estratificado de ceros con `sample_weight`, actualizando árboles incrementalmente (`init_model`) para operar dentro de 15–16 GB de RAM.
2. **Etapa 1b — Regresor de Intensidad (`Tweedie`, $p=1.62$):** entrena únicamente sobre filas positivas (`phi_viirs > 0`) para predecir la intensidad condicional $\hat{\mu}_{g,t}$.
3. **Validación espacial LOBBO:** retiene un bloque latitudinal completo con un **buffer de exclusión de 20 km (4 celdas)** para evitar fuga espacial y audita $R^2$, MBE y tasa de ceros.

### Output de Fase 3
- **Modelos entrenados:** `models/hurdle_occ_focal.txt` y `models/hurdle_int_tweedie.txt`.
- **Métricas en consola:** $R^2$ LOBBO, MBE y *zero-rate* sobre el bloque retenido.

---

## Etapa 4: Output Final / Serving (`data/serving/firms_argentina_harmonized/`)

Al aplicar los dos modelos entrenados sobre toda la serie `2000–2026`, se obtiene el FRP diario equivalente a VIIRS mediante:

$$\hat{y}_{\text{VIIRS-equiv}} = \hat{\pi}_{g,t} \cdot \hat{\mu}_{g,t}$$

- **Ubicación:** `data/serving/firms_argentina_harmonized/year=YYYY/data.parquet` (esquema detallado en [BACKEND_CONTRACTS.md](BACKEND_CONTRACTS.md)).
- **Qué representa 1 fila:** una celda y fecha con las tres series listas para consultar desde **DuckDB + FastAPI** hacia **React + MapLibre GL**.

| `cell_id` | `lat_centro` | `lon_centro` | `fecha` | `phi_modis` (Crudo) | `phi_harmonized` (ML) | `phi_viirs` (Real) | Uso en la serie |
| ---: | ---: | ---: | :--- | ---: | ---: | ---: | :--- |
| `211610` | `-33.3014` | `-66.3321` | `2006-09-10` | `52.0` | `148.6` | `0.0` | **Backfilling (2000–2011):** reconstruye energía VIIRS antes de su lanzamiento. |
| `211610` | `-33.3014` | `-66.3321` | `2020-08-14` | `45.0` | `134.2` | `138.4` | **Solapamiento (2012–2026):** permite auditar continuidad sin salto en 2012. |
