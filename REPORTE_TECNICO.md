# Reporte Técnico: Pipeline de Armonización Satelital MODIS × VIIRS (`2000–2026`)

El pipeline completo de ingeniería de datos y Machine Learning para armonizar **26 años de detecciones de fuego en Argentina (`2000–2026`)** está implementado, optimizado para ejecutarse dentro de **15.2 GB de RAM**, verificado con **8 tests automatizados (`8 passed in 5.43s`)** y publicado en GitHub (`main` en `c9db274` previo a este reporte).

---

## 1. Resumen Ejecutivo: Qué hace cada fase

| Fase | Script principal | Entrada | Salida | Qué resuelve |
| :--- | :--- | :--- | :--- | :--- |
| **Fase 1: Limpieza** | [clean_firms.py](clean_firms.py) | `Data-Argentina/**/*.csv` (MODIS + VIIRS) | `data/clean/firms_argentina_clean.parquet` (`152.68 MB`) | Filtra ruido (`type=0`, confianza nominal/alta), convierte a `UTC-3` fijo, proyecta a grilla Albers `5 × 5 km` (`cell_id`) y deduplica (`SP > NRT`). |
| **Fase 2: Grilla Diaria** | [aggregate_daily.py](aggregate_daily.py) | `firms_argentina_clean.parquet` | `data/processed/firms_argentina_daily/year=YYYY/data.parquet` | Empareja `(cell_id, fecha)`, imputa ceros reales, separa predictores MODIS (`X`) y target VIIRS `SNPP` puro (`Y`), y calcula ventanas móviles `3d`/`7d`/`30d`. |
| **Fase 3: Entrenamiento ML** | [train_hurdle.py](train_hurdle.py) | Parquets `2012–2024` (`339.7M` filas) | `models/hurdle_occ_focal.txt` + `models/hurdle_int_tweedie.txt` | Entrena modelo **Hurdle GBDT** en dos etapas: **Focal Loss** ($\gamma=2.0, \alpha=0.75$) para ocurrencia y **Tweedie** ($p=1.62$) para intensidad FRP (MW), con validación espacial **LOBBO (buffer 20 km)**. |
| **Fase 4: Inferencia / *Backfill*** | [predict_harmonized.py](predict_harmonized.py) | Parquets `2000–2026` + `models/` | `data/final/firms_argentina_harmonized/year=YYYY/data.parquet` | Reconstruye el FRP equivalente VIIRS antes del `2012-01-20` (`is_backfilled=True`) y empalma con `phi_viirs_obs` desde el `2012-01-20` en adelante. |

---

## 2. Decisiones Técnicas y Optimizaciones Aplicadas

| Problema detectado | Decisión / Solución implementada | Impacto verificado |
| :--- | :--- | :--- |
| **CRS erróneo (`EPSG:5347`) en especificación inicial** | Se reemplazó por **Argentina Albers Equal Area Conic (`+proj=aea`)** de área igual (`25 km²` exactos por celda `5 × 5 km`). | Evita distorsión de área entre el norte y la Patagonia en el cálculo de densidad de energía (MW). |
| **Cambio horario DST `UTC-2` (`2007–2009`)** | Se mantuvo **`UTC-3` fijo** en toda la serie `2000–2026` (`clean_firms.py`). | Respeta la física de órbitas heliosíncronas (hora solar local) y evita partir la grilla entre provincias. |
| **Salto de densidad VIIRS en `2018` (`N20`) y `2023` (`N21`)** | Se agregó `--viirs-satellite SNPP` (por defecto) en `aggregate_daily.py`. | El target `phi_viirs` y `y_occ_viirs` mantiene un único sensor de referencia (`SNPP`) homogéneo entre `2012` y `2026`. |
| **Cuello de botella en ventanas móviles `3d`/`7d`/`30d`** | En `aggregate_daily.py`, las ventanas se calculan de forma **dispersa** solo sobre detecciones activas (`GENERATE_SERIES(0, 29)`) antes de cruzar con la grilla de ceros. | Evita ordenar cientos de millones de filas en DuckDB y mantiene el límite en `5 GB` de RAM. |
| **Sesgo temporal (*chronological drift*) al entrenar por año** | En `train_hurdle.py`, se extrae 100% positivos + 2% ceros año por año en 1 sola lectura por archivo, pero el GBDT entrena viendo **todos los años `2012–2024` juntos**. | Los árboles ven toda la variabilidad climática de los 13 años sin sobreajustar al último año leído, reduciendo de 39 a 13 los escaneos Parquet. |
| **Velocidad de `focal_loss_objective` en CPU** | Se simplificó algebraicamente el gradiente cancelando la división por `p_t`, cacheando etiquetas por dataset y operando en `float32`. | Equivalencia numérica verificada (`max diff = 3.33e-16` en gradiente y `1.38e-17` en hessiano) y ~2.7× menos tiempo por iteración en benchmark de `1M` filas (`1.375s` → `0.514s`). |
| **Inferencia sobre ~680M de filas (`2000–2026`)** | En `predict_harmonized.py`: (1) ruta *zero-copy* en PyArrow para años observados (`>= 2012-01-20`), (2) filtro `n_modis_30d > 0` en *backfill*, y (3) Tweedie solo donde `pi_hat >= 0.5`. | En validación `year=2024` (`8,015` filas), filtrar por `n_modis_30d > 0` mantuvo/mejoró $R^2$ (`0.580462` → `0.580472`) y *zero-rate* (`0.9991` → `0.999242`), evitando correr árboles en el ~95% de celdas vacías. |

---

## 3. Suite de Tests Ejecutados (`pytest`)

Comando ejecutado:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

**Resultado observado:** `8 passed in 5.43s`

| Archivo de test | Test unitario / de integración | Qué verifica | Estado |
| :--- | :--- | :--- | :---: |
| [test_clean_firms.py](test_clean_firms.py) | `test_cleaning_rules_utc_rollover_filters_dedup_and_progress` | Cambio de día `UTC → UTC-3`, filtro `type == 0`, confianza, deduplicación `SP > NRT` y progreso al `100.0%`. | PASSED |
| [test_clean_firms.py](test_clean_firms.py) | `test_recursive_multisensor_modis_and_viirs_n20` | Lectura recursiva y normalización simultánea de `MODIS` (`conf >= 30`) y `VIIRS` (`'n'/'h'`). | PASSED |
| [test_aggregate_daily.py](test_aggregate_daily.py) | `test_daily_aggregation_zero_imputation_and_rolling_windows` | Producto cartesiano de ceros, separación `*_modis` / `*_viirs` y sumas móviles causales `3d`, `7d`, `30d`. | PASSED |
| [test_aggregate_daily.py](test_aggregate_daily.py) | `test_yearly_partitioned_output_preserves_cross_year_rolling_windows` | Escritura particionada `year=YYYY/data.parquet` preservando los 29 días de diciembre previo. | PASSED |
| [test_aggregate_daily.py](test_aggregate_daily.py) | `test_viirs_satellite_filter_isolates_snpp_target` | Aislamiento del satélite `SNPP` por defecto frente a `N20`, y modo `--viirs-satellite ALL`. | PASSED |
| [test_train_hurdle.py](test_train_hurdle.py) | `test_focal_loss_and_lobbo_20km_buffer` | Signo y estabilidad finita de `grad`/`hess` en Focal Loss y buffer espacial LOBBO de `20 km`. | PASSED |
| [test_train_hurdle.py](test_train_hurdle.py) | `test_batch_training_on_yearly_parquets` | Entrenamiento sobre Parquets anuales y guardado de `hurdle_occ_focal.txt` y `hurdle_int_tweedie.txt`. | PASSED |
| [test_predict_harmonized.py](test_predict_harmonized.py) | `test_harmonized_backfill_and_observed_splice` | *Backfill* pre-VIIRS (`2008`), corte en `2012-01-20`, empalme con `phi_viirs_obs` (`2018`) y esquema Parquet final. | PASSED |

---

## 4. Documentación, Skill e Historial de Ramas

- [README.md](README.md): Guía rápida de instalación, arquitectura y ejecución.
- [DOCUMENTACION_PIPELINE.md](DOCUMENTACION_PIPELINE.md) y [FLUJO_PIPELINE.md](FLUJO_PIPELINE.md): Flujo detallado `Datos originales → Limpieza → Agregación → ML → Serie Final`.
- [BACKEND_CONTRACTS.md](BACKEND_CONTRACTS.md): Contratos de esquema Parquet, fórmulas de `cell_id` y endpoints REST/GeoJSON.
- [SKILL.md](.agents/skills/firms-data-interpreter/SKILL.md) y [data-dictionary.md](.agents/skills/firms-data-interpreter/references/data-dictionary.md): Skill `firms-data-interpreter` para consultar e interpretar las 4 capas de datos con DuckDB/PyArrow.

### Próximo paso de ejecución

```powershell
.\.venv\Scripts\python.exe predict_harmonized.py
```
