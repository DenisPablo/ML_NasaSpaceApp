# Pipeline de Armonización MODIS × VIIRS (Argentina)

Pipeline ETL y de entrenamiento **Hurdle GBDT** para reconstruir una serie temporal homogénea de **Fire Radiative Power (FRP)** diario en Megavatios (MW) entre **MODIS** (`2000–presente`) y **VIIRS** (`2012–presente`) sobre una grilla de `5 × 5 km` en proyección **Argentina Albers (`+proj=aea`)**, optimizado para correr dentro de **15–16 GB de RAM**.

---

## Ruta rápida de ejecución

```powershell
# 0. Activar entorno e instalar dependencias (si aún no está creado)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 1. Fase 1: Limpieza multi-sensor + proyección + spatial hashing (con progreso %)
.\.venv\Scripts\python.exe clean_firms.py

# 2. Fase 2: Agregación diaria por año + ceros explícitos + ventanas 3d/7d/30d
.\.venv\Scripts\python.exe aggregate_daily.py

# 3. Fase 3: Entrenamiento Hurdle GBDT por batches anuales + validación LOBBO (20 km)
.\.venv\Scripts\python.exe train_hurdle.py

# Tests unitarios
.\.venv\Scripts\python.exe -m pytest -v
```

---

## Arquitectura del Pipeline

| Fase | Script | Entrada | Salida | Qué resuelve |
| :--- | :--- | :--- | :--- | :--- |
| **Fase 1** | `clean_firms.py` | `Data-Argentina/**/*.csv` | `data/clean/firms_argentina_clean.parquet` | Filtra `type == 0` (en `SP`), `confidence` (`n`/`h` o `>= 30`), `frp > 0`, convierte `UTC → UTC-3`, proyecta a Argentina Albers (`+proj=aea`), asigna `cell_id` en $O(1)$ y deduplica (`SP > NRT`). |
| **Fase 2** | `aggregate_daily.py` | `data/clean/firms_argentina_clean.parquet` | `data/processed/firms_argentina_daily/year=YYYY/data.parquet` | Separa features **MODIS** y target **VIIRS**, imputa ceros explícitos (`celdas × días`) y calcula ventanas móviles `3d`/`7d`/`30d` por año con *lookback* de 29 días. |
| **Fase 3** | `train_hurdle.py` | `data/processed/firms_argentina_daily` | `models/hurdle_occ_focal.txt`, `models/hurdle_int_tweedie.txt` | Entrena Etapa 1a (**Binary Focal Loss**, $\gamma=2.0, \alpha=0.75$) por *batches* anuales con submuestreo estratificado de ceros y Etapa 1b (**Tweedie** `p=1.62`) sobre `Y > 0`, validando con **LOBBO** (buffer de 20 km). |

---

## Esquema de salida esperado tras la armonización (Inferencia)

Al aplicar los modelos entrenados sobre la serie `2000–2026`, la tabla resultante de armonización (Parquet particionado por año, o CSV filtrado a celdas con actividad) contiene una fila por `(cell_id, fecha)`:

| Columna | Tipo | Rol en Armonización |
| :--- | :--- | :--- |
| `cell_id` | `int32` | Identificador de celda (`5 × 5 km`, `N_COLS = 460`) |
| `fecha` | `date` | Día operativo local (`UTC-3`, `YYYY-MM-DD`) |
| `n_modis`, `phi_modis` | `int32`, `float32` | Conteo y FRP crudo de MODIS (MW) |
| `pi_hat`, `mu_hat` | `float32`, `float32` | Probabilidad de ocurrencia e intensidad condicional predichas |
| `phi_harmonized` | `float32` | **FRP armonizado VIIRS-equivalente** ($\hat{\pi} \cdot \hat{\mu}$) |
| `n_viirs`, `phi_viirs` | `int32`, `float32` | Conteo y FRP real de VIIRS (`2012–2026`, `0` antes de 2012) |

---

## Notas técnicas clave

- **CRS (`+proj=aea`):** usar la definición custom Argentina Albers (`+proj=aea +lat_0=-37 +lon_0=-60 +lat_1=-26 +lat_2=-48 +x_0=2500000 +y_0=0 +ellps=GRS80 +units=m +no_defs`). No usar `EPSG:5347` (corresponde a Gauss-Krüger Faja 5).
- **Memoria (15–16 GB RAM):** nunca cargar toda la grilla nacional densa en un único tensor; Fase 2 particiona por año en DuckDB y Fase 3 entrena Etapa 1a incrementalmente (`init_model`) por *batches* anuales con `float32`.
