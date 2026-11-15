# Veredicto de la armonización MODIS → VIIRS: cálculos y réplica

Esta carpeta tiene todo lo necesario para repetir la evaluación científica de la Propuesta B: factores provincia × mes, rangos del 80 %, día operativo, reglas de confianza y anomalía.

- **Informe:** [veredicto-armonizacion.md](veredicto-armonizacion.md). Veredicto: **sólido con reservas**, con 20 hallazgos, severidad y responsable.
- **Scripts:** [`scripts/`](scripts), del `m0` al `m9`.
- **Salidas de la corrida publicada:** [`resultados/`](resultados), un `.txt` por script.
- **Datos:** `serving_api/20261114T174945/` de este repo. Es idéntico a `data/current/` de [FireHistoryInterfaceAPI](https://github.com/smlopezsm/FireHistoryInterfaceAPI).
- **Código de calibración:** `classification/` de este repo (igual a `dcf5b18`). Los scripts lo importan y no lo copian.

## Cómo replicar

Desde la raíz del repo (Python 3.12; con `uv`):

```bash
uv venv --python 3.12 .venv-analisis
uv pip install --python .venv-analisis/bin/python -r analisis/veredicto-armonizacion/requirements.txt
bash analisis/veredicto-armonizacion/run_all.sh .venv-analisis/bin/python
```

- Tarda ~30 s y sobrescribe `resultados/`. Después, `git diff analisis/veredicto-armonizacion/resultados` tiene que quedar vacío.
- Para otra versión de datos: `FH_DATA=/ruta/al/paquete bash analisis/veredicto-armonizacion/run_all.sh …`.
- Las simulaciones de `m5_conf.py` usan una semilla fija (`42`), así que también se repiten exactas.

**Primer control:** `resultados/m1_repro.txt` tiene que mostrar `max|diff| 0.0` para `modis` y `terra`, y `cov in-sample 83.1088…`. Con eso se confirma que los scripts parten de los mismos factores, rangos y cobertura que publicó el pipeline.

## De dónde sale cada cifra

| Hallazgo | Cifra clave | Script | Línea a buscar en `resultados/` |
|---|---|---|---|
| Reproducción | Factores con diferencia 0,0; cobertura 83,1 % | `m1_repro.py` | `max\|diff\|`, `cov in-sample` |
| H10 Estabilidad | V/M 2012–25: 3,02–3,83, CV 6,0 %, Spearman p = 0,77 | `m2_factor.py` | `CV anual`, `Spearman trend` |
| H11 Sobredispersión | CV interanual 0,164 contra 0,035 Poisson (×4,9) | `m2_factor.py` | `prov-mes con >=5 años` |
| H12 Shrinkage | Error LOYO 14,84 % (K = 200) contra 18,04 % (solo país) | `m3_cv.py` | `K por LOYO` |
| H3 Rangos | LOYO 81,0 %; hacia adelante 77,1 % (64,0 % en > 500) | `m3_cv.py` | `Cobertura LOYO`, `Cobertura hacia adelante` |
| H4 Rangos por provincia | De 66,5 % (Salta) a 89,2 % (Santa Fe) | `m3_cv.py` | `por provincia` |
| H5 Rango nacional | 98,8 % de cobertura sumando límites | `m3_cv.py` | `rango NACIONAL` |
| H13 LOPO | \|sesgo\| mediano 7,7 %; Neuquén −29,7 % | `m3_cv.py` | tabla `sesgo_LOPO_%` |
| H7 Días parciales | 31 días y 7.054 focos (S-NPP < 0,5 × NOAA-20); razón 1,024 | `m4b_gaps.py` | `focos faltantes`, `razón global` |
| H8 Calibración | 450 días excluidos (8,9 %), sobre todo en abril–julio | `m4_oper.py` | `excluidos de calibración` |
| H9 Ceros silenciosos | 201 días; ≈ 3.817 MODIS ≈ 12.700 VIIRS-eq | `m4_oper.py` | `Pre-VIIRS`, `MODIS faltante` |
| H14 frac ≤ 0,1 | Error nacional mediano 1,3 %, p90 4,1 % | `m5_conf.py` | `Simulación nacional` |
| H15 Umbral 21 | Error suave con M, sin quiebre en 21 | `m5_conf.py` | `Error relativo LOYO por M` |
| H1 Anomalía | −2,65 %/año; 2003–11 contra 2012–25; 68 % de coincidencia | `m6_anom.py`, `m7_conf.py` | `Theil-Sen`, `coinciden`, `MODIS crudo` |
| H16 Terra → Terra + Aqua | Razón 0,981 en 2002–2011 | `m7_conf.py` | `vía solo Terra` |
| H17 Deriva orbital | Terra 11,3 → 10,3 h; Aqua 15,0 → 16,1 h (2025) | `m8_orbit.py` | tabla de horas |
| H19 Composición | Factor implícito 3,26 contra 3,33 | `m8_orbit.py` | `factor nacional implícito` |
| H6 Solo Terra | Cobertura 73,3 % con los rangos publicados | `m9_extra.py` | `Terra sola LOYO` |

## Scripts

| Script | Qué hace | Genera |
|---|---|---|
| `common.py` | Lee los Parquet con DuckDB, importa `classification/` y aplica los rangos | — |
| `m0_inspect.py` | Esquemas, factores, rangos y días caídos por satélite | — |
| `m1_repro.py` | Reajusta los factores y rangos con el código del pipeline y los compara con lo publicado | `mm.parquet`, `mt.parquet` |
| `m2_factor.py` | Tendencia anual, sobredispersión (GLM Poisson) y tendencia por provincia | — |
| `m3_cv.py` | LOYO para varios K, cobertura por tramo, provincia y año, prueba hacia adelante y LOPO | `loyo200.parquet` |
| `m4_oper.py` | Días "operativos" sospechosos, efecto en los factores, días excluidos y ceros silenciosos | — |
| `m4b_gaps.py` | S-NPP contra NOAA-20 día por día (coberturas parciales) | — |
| `m5_conf.py` | Simulación de días estimados (semilla 42) y error por cantidad de focos MODIS | — |
| `m6_anom.py` | Réplica nacional de la lógica de la API, eras y categorías de anomalía | `nat.parquet` |
| `m7_conf.py` | Categorías por era, MODIS crudo, Aqua/Terra y consistencia del cambio de fuente | — |
| `m8_orbit.py` | Hora de paso mediana, FRP, composición y distancia temporal | — |
| `m9_extra.py` | Rangos para solo Terra y error por clase de confianza | — |

## Límites

Los detalles están en el §6 del informe. Los más importantes:
- No se levantó la API: su lógica se replicó a escala nacional.
- Los umbrales para detectar días parciales los fijamos nosotros (ver `m4_oper.py` y `m4b_gaps.py`).
- `m6_anom.py` mide H2 con la lógica de anomalía anterior. En `main` de la API ese punto ya está corregido.
