# Documentación Completa del Pipeline de Armonización MODIS × VIIRS (Argentina)

Documento unificado de arquitectura, flujo de datos, suficiencia muestral, entrenamiento **Hurdle GBDT** y contratos REST/GeoJSON para reconstruir la serie diaria homogénea de **Fire Radiative Power (FRP)** en Megavatios (MW) entre **MODIS** (`2000–presente`) y **VIIRS** (`2012–presente`) sobre una grilla de `5 × 5 km` en **Argentina Albers (`+proj=aea`)**.

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

## 1. Ruta Rápida de Ejecución y Entorno

Diseñado para entrenar en local dentro de **15–16 GB de RAM** y servir en producción dentro de **4 GB RAM / 2 vCPU** (DonWeb Cloud):

```powershell
# 0. Crear entorno virtual e instalar dependencias
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 1. Fase 1: Limpieza multi-sensor + proyección + spatial hashing (con progreso %)
.\.venv\Scripts\python.exe clean_firms.py

# 2. Fase 2: Agregación diaria por año + ceros explícitos + ventanas 3d/7d/30d
.\.venv\Scripts\python.exe aggregate_daily.py

# 3. Fase 3: Entrenamiento Hurdle GBDT por batches anuales + validación LOBBO (20 km)
.\.venv\Scripts\python.exe train_hurdle.py

# Tests unitarios completos
.\.venv\Scripts\python.exe -m pytest -v
```

### Notas técnicas críticas
- **CRS (`+proj=aea`):** usar la definición custom **Argentina Albers Cónica de Área Equivalente** sobre GRS 1980 / POSGAR 2007 (`+proj=aea +lat_0=-37 +lon_0=-60 +lat_1=-26 +lat_2=-48 +x_0=2500000 +y_0=0 +ellps=GRS80 +units=m +no_defs`). **No utilizar `EPSG:5347`**, ya que corresponde a *POSGAR 2007 / Argentina 5* (Transversa de Mercator / Gauss-Krüger), incompatible con la grilla.
- **Presupuesto de RAM:** no materializar la matriz nacional densa completa (~528M–1.000M de filas = ~38 GB en `float32`) en memoria. Fase 2 particiona por año en DuckDB y Fase 3 entrena Etapa 1a incrementalmente (`init_model`) año por año.

---

## 2. Flujo Detallado por Etapas y Outputs

### Etapa 0: Datos Originales (`Data-Argentina/**/*.csv`)
- **Contenido:** 8 archivos CSV de NASA FIRMS divididos en 4 carpetas por constelación: `fire_archive_*.csv` (**SP** científico con columna `type`) y `fire_nrt_*.csv` (**NRT** tiempo casi real sin columna `type`).
- **Satélites:** **MODIS** (`Terra` desde Nov 2000, `Aqua` desde Jul 2002; píxel 1 km) y **VIIRS** (`SNPP` desde Ene 2012, `N20` desde Abr 2018, `N21` desde Mar 2023; píxel 375 m).
- **Qué representa 1 fila:** un único píxel con fuego detectado en ese instante (no hay filas para días sin fuego).

```csv
latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight,type
-33.3014,-66.3321,325.4,1.8,1.3,2020-08-15,0130,Terra,MODIS,75,6.03,291.2,24.5,N,0
```

- **Por qué requiere limpieza:** timestamps en `UTC` (`0130 UTC` = `22:30` del día anterior en Argentina), incluye detecciones industriales (`type=2`) y baja confianza (`'l'` / `<30%`), coordenadas en grados (`EPSG:4326`) con deformación longitudinal >36% entre el norte y el sur, y esquemas distintos entre `SP`/`NRT` y `MODIS`/`VIIRS`.

---

### Etapa 1: Limpieza y Normalización Espacial (`clean_firms.py`)
1. **Lectura en *streaming*:** procesa los 8 CSV por bloques de 8 MB con barra de progreso porcentual (`0.0% → 100.0%`) y ETA.
2. **Filtros de calidad:** descarta `frp <= 0` y valores no finitos; conserva `type == 0` (vegetación) en archivos `SP` (asigna `type = -1` en `NRT`); descarta confianza baja (`'l'` en VIIRS o `< 30` en MODIS).
3. **Conversión horaria (`UTC → UTC-3`):** resta 3 horas antes de derivar `fecha_local` para asignar pasadas nocturnas (`00:00–02:59 UTC`) al día operativo argentino anterior.
4. **Proyección y *Spatial Hashing* $O(1)$:** reproyecta a Argentina Albers (`+proj=aea`) y calcula `col = floor((X - 1_400_000) / 5_000)`, `row = floor((Y + 2_100_000) / 5_000)` y `cell_id = row * 460 + col`.
5. **Unificación y deduplicación:** normaliza `sensor_family` (`"MODIS"` / `"VIIRS"`), castea métricas a `float32`/`int32` y deduplica en DuckDB priorizando `SP` sobre `NRT`.

**Output de Fase 1:** `data/clean/firms_argentina_clean.parquet` (1 fila = 1 foco válido en `UTC-3` asignado a su celda `5 × 5 km`).

| `cell_id` | `col` | `row` | `fecha_local` | `timestamp_utc3` | `x_aea` | `y_aea` | `frp` | `scan` | `sensor_family` | `satellite` | `source_quality` |
| ---: | ---: | ---: | :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| `211610` | `10` | `460` | `2020-08-14` | `2020-08-14 22:30:00` | `1452130.0` | `201450.0` | `24.5` | `1.8` | `"MODIS"` | `"Terra"` | `"SP"` |

---

### Etapa 2: Resultado ETL — Matriz Diaria e Imputación de Ceros (`aggregate_daily.py`)
1. **Agregación diaria por sensor:** agrupa en DuckDB por `(cell_id, fecha_local)` separando features **MODIS** (`n_modis`, `phi_modis`, `phi_max_modis`, `scan_bar_modis`, `track_bar_modis`) y target **VIIRS** (`n_viirs`, `phi_viirs`, `phi_max_viirs`, `y_occ_viirs`).
2. **Producto cartesiano (`celdas × calendario`):** imputa explícitamente `0` en los días sin fuego (>99,9% de la matriz) para evitar alucinaciones de falsos positivos.
3. **Ventanas móviles retrospectivas (`3d`, `7d`, `30d`):** sumas acumuladas por celda para aportar memoria térmica ante oclusión temporal o pasadas desfavorables.
4. **Particionado anual con *lookback*:** escribe `year=YYYY/data.parquet` leyendo los últimos 29 días del año previo para no cortar las ventanas de enero.

**Output de Fase 2:** `data/processed/firms_argentina_daily/year=YYYY/data.parquet` (1 fila = 1 celda × 1 día calendario).

| `cell_id` | `fecha` | `n_modis` | `phi_modis` | `scan_bar_modis` | `phi_modis_7d` | `y_occ_viirs` | `n_viirs` | `phi_viirs` |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `211610` | `2020-08-13` | `0` | `0.0` | `NULL` | `0.0` | `0` | `0` | `0.0` |
| `211610` | `2020-08-14` | `2` | `45.0` | `2.1` | `45.0` | `1` | `11` | `138.4` |

---

### Etapa 3: Entrenamiento ML — Hurdle GBDT por *Batches* (`train_hurdle.py`)
Resuelve el salto instrumental de 2012 calibrando en la ventana de solapamiento **2012–2024** con 15 features `float32` (`n_modis`, `phi_modis`, `phi_max_modis`, `scan_bar_modis`, `track_bar_modis`, ventanas `3d/7d/30d`, `grid_row`, `grid_col`, `doy_sin`, `doy_cos`):

1. **Etapa 1a — Clasificador de Ocurrencia (`Binary Focal Loss`, $\gamma=2.0, \alpha=0.75$):** estima $\hat{\pi}_{g,t} = \Pr(\text{VIIRS} > 0 \mid X)$. Lee año por año con DuckDB conservando el 100% de los positivos y un muestreo estratificado de ceros (`--neg-sample-rate 0.02`) compensado con `sample_weight`, actualizando árboles vía `init_model`.
2. **Etapa 1b — Regresor de Intensidad (`Tweedie`, $p=1.62$):** entrena únicamente sobre registros positivos (`y_occ_viirs = 1 AND phi_viirs > 0`) para predecir la intensidad condicional $\hat{\mu}_{g,t} > 0$.
3. **Validación espacial LOBBO (buffer 20 km):** retiene un bloque latitudinal completo con una franja de exclusión de 4 celdas (20 km) alrededor para suprimir autocorrelación espacial y reporta $R^2$, MBE y *zero-rate*.

**Output de Fase 3:** `models/hurdle_occ_focal.txt` y `models/hurdle_int_tweedie.txt`.

---

### Etapa 4: Output Final / Serving (`data/serving/firms_argentina_harmonized/`)
Aplica ambos modelos sobre toda la serie `2000–2026` mediante $\hat{y}_{\text{VIIRS-equiv}} = \hat{\pi}_{g,t} \cdot \hat{\mu}_{g,t}$. No es un modelo predictivo climático 24h–72h (diferido a fase posterior), sino la reconstrucción histórica homogénea **MODIS Crudo vs. Armonizado vs. VIIRS Real**.

| `cell_id` | `lat_centro` | `lon_centro` | `fecha` | `phi_modis` | `phi_harmonized` | `phi_viirs` | Rol en la serie |
| ---: | ---: | ---: | :--- | ---: | ---: | ---: | :--- |
| `211610` | `-33.3014` | `-66.3321` | `2006-09-10` | `52.0` | `148.6` | `0.0` | **Backfilling (`2000–2011`):** reconstruye energía VIIRS antes de su lanzamiento. |
| `211610` | `-33.3014` | `-66.3321` | `2020-08-14` | `45.0` | `134.2` | `138.4` | **Solapamiento (`2012–2026`):** audita continuidad sin escalón en 2012. |

---

## 3. Suficiencia de los Datos y Brechas Conocidas

- **Por qué son suficientes para el MVP:** cubren toda la constelación disponible en Sudamérica (MODIS `Terra/Aqua` + VIIRS `SNPP/N20/N21`), incluyen >14 años de solapamiento simultáneo (`2012–2026`) y aportan `frp`, `scan` y `track` para corregir el *efecto Bow-Tie* junto con ventanas `3d/7d/30d`.
- **Brechas conocidas (mejoras opcionales según métrica LOBBO):**
  1. **Sin `f_nube`:** los CSV FIRMS no incluyen fracción de nubosidad (proviene de QA raster `MOD09GA`/`VNP09GA`); se deja `f_nube = NULL` hasta acoplarlo.
  2. **Sin capa explícita de ecorregiones:** hoy `grid_row`/`grid_col` actúan como *proxy* geográfico continuo; si el $R^2$ nacional en LOBBO queda debajo de `0,85`, conviene sumar ecorregiones APN y `brightness`/`bright_t31`.
  3. **Sin máscara oficial IGN (`dim_celdas_argentina.parquet`):** las celdas activas se derivan de las celdas con al menos una detección en 26 años.

---

## 4. Contratos de Datos y API para Backend (DuckDB + FastAPI)

### Esquema del Parquet de Serving (`data/serving/firms_argentina_harmonized/year=YYYY/data.parquet`)

| Columna | Tipo Parquet | Nullable | Descripción |
| :--- | :--- | :--- | :--- |
| `cell_id`, `grid_row`, `grid_col` | `INT32` | No | Identificador y coordenadas de grilla `5 × 5 km` (`N_COLS = 460`) |
| `lat_centro`, `lon_centro` | `FLOAT` (`float32`) | No | Centroide de la celda en `EPSG:4326` para GeoJSON/MapLibre |
| `fecha` | `DATE` | No | Fecha operativa local (`UTC-3`, `YYYY-MM-DD`) |
| `n_modis`, `phi_modis` | `INT32`, `FLOAT` | No | Conteo y FRP crudo diario MODIS (MW) |
| `pi_hat`, `mu_hat`, `phi_harmonized` | `FLOAT` | No | Probabilidad ($\hat{\pi}$), intensidad ($\hat{\mu}$) y FRP armonizado ($\hat{\pi} \cdot \hat{\mu}$) |
| `n_viirs`, `phi_viirs` | `INT32`, `FLOAT` | No | Conteo y FRP real diario VIIRS (`0` antes de 2012-01-20) |
| `f_nube` | `FLOAT` | Sí | Fracción de nubosidad `[0.0, 1.0]` (`NULL` si no está acoplada) |

### Endpoints FastAPI

- **`GET /api/v1/health`** → estado de DuckDB, rango de fechas (`min_date`, `max_date`) y `active_cells`.
- **`GET /api/v1/grid/geojson`** (`start_date`, `end_date`, `layer`: `"harmonized" | "modis" | "viirs"`, `bbox` opcional `"west,south,east,north"`, `min_frp` default `0.0`) → `FeatureCollection` GeoJSON con propiedades `cell_id`, `frp_modis_crudo`, `frp_armonizado`, `frp_viirs_real`, `n_modis`, `n_viirs` y `cloud_occluded` (`AVG(f_nube) > 0.80`).
- **`GET /api/v1/cells/{cell_id}/series`** (`start_date`, `end_date`, `granularity`: `"day" | "month" | "year"`) → arreglo `points` con `period`, `frp_modis_crudo`, `frp_armonizado`, `frp_viirs_real`.
- **`GET /api/v1/series/summary`** (`start_date`, `end_date`, `granularity`: `"month" | "year"`, `bbox` opcional) → serie agregada para auditar continuidad `2000–2026`.

### Reglas operativas en servidor (4 GB RAM)
1. Abrir DuckDB *in-process* con `memory_limit = '1536MB'` y `threads = 2` sin tablas persistidas en RAM.
2. Aplicar *partition pruning* por `year=YYYY/data.parquet` y filtro `fecha BETWEEN ? AND ?`.
3. En `/grid/geojson`, filtrar `HAVING frp > :min_frp` para nunca serializar ceros masivos hacia MapLibre GL. *(Para exportación a CSV liviano, aplicar el mismo filtro de actividad o agregación mensual/anual).*
