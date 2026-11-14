# Etapa de clasificación y armonización — Propuesta B

Esta etapa toma el Parquet limpio de la **Fase 1** (`clean_firms.py --region argentina`) y produce
**exactamente** los archivos que lee la API ([FireHistoryInterfaceAPI · `docs/02-contratos-de-datos.md`](https://github.com/smlopezsm/FireHistoryInterfaceAPI/blob/docs/documentacion-inicial/docs/02-contratos-de-datos.md)).

Es **independiente** de las fases 2–4 (agregación diaria, LightGBM, serving v2): no las modifica ni las reemplaza.

## Dónde se hace cada cosa

| Capa | Qué hace | Dónde |
|---|---|---|
| **Pipeline (este repo)** | Limpieza, detección de puntos industriales, conteos celda-día por satélite, días con satélite caído, **calibración** de factores y rangos, validación del contrato | `clean_firms.py` + `run_classification.py` |
| Backend (FastAPI + DuckDB) | Sumar conteos por zona y período, aplicar `factor × MODIS`, rangos, `estado` / `confianza` / anomalía | FireHistoryInterfaceAPI |
| Frontend (React) | Solo mostrar | FireHistoryInterface |

> Guía paso a paso para el equipo: [EJECUCION_CLASIFICACION.md](EJECUCION_CLASIFICACION.md).

## Cómo se ejecuta

```bash
# 1. Fase 1 para todo el país (si no está hecha)
python clean_firms.py --region argentina

# 2. Límites provinciales oficiales (IGN vía Georef), una sola vez
curl -L -o data/ref/provincias.geojson https://apis.datos.gob.ar/georef/api/v2.0/provincias.geojson

# 3. Clasificación
python run_classification.py \
  --clean data/clean/firms_argentina_clean.parquet \
  --provinces-geojson data/ref/provincias.geojson \
  --raw-dir data/raw \
  --out data/serving_api
```

`--raw-dir` es opcional: se usa solo para la regla "marcado industrial por FIRMS" (`type = 2`),
porque la Fase 1 ya descarta esas filas. Sin él, esa regla queda apagada y se avisa en el log.

## Pasos

| # | Paso | Módulo | Resultado |
|---|---|---|---|
| 1 | Normalizar el esquema de la Fase 1 al vocabulario del contrato | `counts.prepare_detections` | `fecha`, `satelite` (`Terra/Aqua/SNPP/N20/N21`), `confianza`… |
| 1b | Recortar cada satélite a su misión (`MISSION_START` … `MISSION_END`) y cortar en `fecha_max` | `counts.trim_to_missions`, `counts.default_fecha_max` | `recorte_misiones` en `validacion.json` |
| 2 | Máscara nacional de celdas 5 × 5 km con fracción dentro de cada provincia | `geo.build_cell_mask` | `dim_celdas` |
| 3 | Provincias simplificadas para el mapa | `geo.provinces_table` | `dim_provincias` |
| 4 | Puntos de calor fijos industriales (3 reglas) | `static_sources.detect_static_sources` | `puntos_industriales` |
| 5 | Quitar focos en esos puntos | `counts.exclude_static` | — |
| 6 | Días × satélite operativo (apagón ≠ cero fuego) | `counts.build_satelite_operativo` | `satelite_operativo` |
| 7 | Conteos dispersos celda × día por satélite (INT16) | `counts.build_conteos` | `conteos_celda_dia/anio=…` |
| 8 | Focos individuales con `foco_id` estable | `counts.build_focos` | `focos/anio=…` |
| 9 | Factores jerárquicos provincia × mes (`modis` y `terra`) | `calibrate.fit_hierarchical` | `factores_provincia_mes` y `factores_region_mes` (niveles región y país del mismo ajuste) |
| 10 | Rangos del 80 % (dejando un año afuera) | `calibrate.calibrate_ranges` | `rangos` |
| 11 | Pruebas hacia atrás / adelante y cobertura | `calibrate.evaluate` | `validacion.json` |
| 12 | Validar contrato y publicar de forma atómica (`current` → versión) | `publish` | `metadata.json` |

## Reglas científicas (resumen)

| Tema | Regla |
|---|---|
| Referencia VIIRS | S-NPP (2012-01-20 → 2026-11-02); NOAA-20 desde 2026-11-03. **Nunca** se suma NOAA-21. |
| Entrada MODIS | Terra + Aqua (`tipo = modis`); solo Terra antes del 2002-07-04 (`tipo = terra`, calibrado 2012–2019) |
| Factor | `f_p,m = (V_p,m + K·f_región,m) / (M_p,m + K)`, con `K = 200` y cascada región → país-mes → país |
| Provincias con pocos datos | CABA, Santa Cruz y Tierra del Fuego usan el factor de su región |
| Bordes de la serie | Se descartan los focos con fecha local fuera de la misión de su satélite (p. ej. Terra 2000-11-01 UTC que cae el 2000-10-31 local). La serie termina en el último día completo, `min(fecha máxima de los datos, ayer en UTC−3)`; un `--fecha-max` solo puede adelantar ese corte, nunca pasarlo |
| Días válidos | Solo días en que todos los satélites involucrados tienen al menos un foco en el país |
| Punto persistente | ≥ 30 días, ≥ 3 años y ≥ 6 meses con focos en la misma celda de ~1 km |
| Punto industrial | nocturno ≥ 60 % y ≥ 10 meses · `type = 2` ≥ 30 % · nocturno ≥ 80 % y ≥ 5 años |
| Tramos de rangos | 0 · 1–5 · 6–20 · 21–100 · 101–500 · 501+ focos MODIS en el mes |

Todos los parámetros están en `classification/config.py`.

## Resultado con los datos reales de Argentina (corrida de prueba)

Corrida sobre las 8 descargas FIRMS de Argentina (MODIS C6.1, S-NPP, NOAA-20, NOAA-21; SP + NRT),
simulando la salida de la Fase 1 con pandas:

| Métrica | Valor | Referencia en la propuesta |
|---|---:|---:|
| Celdas en la máscara | 112.427 | — |
| Puntos industriales / focos excluidos | 489 / 81.087 | 488 / 80.681 |
| Factor nacional MODIS → S-NPP | 3,33 | 3,32 |
| Factor nacional Terra → S-NPP | 9,08 | 9,18 |
| Error mensual por provincia, prueba hacia atrás (cal. 2015–22, prueba 2012–14) | 15,5 % (sesgo 0,0 %) | 15,1 % |
| Ídem, prueba hacia adelante (cal. 2012–19, prueba 2020–25) | 16,5 % (sesgo −4,9 %) | 18,3 % |
| Error usando MODIS crudo | ≈ 70 % | ≈ 70 % |
| Cobertura de los rangos del 80 % (en la ventana de calibración) | 83,1 % | 77,9 % (fuera de muestra) |

Las diferencias menores se deben a que esta corrida usa la ventana completa 2012–2025 y la exclusión de
días con satélite caído. Los factores `terra` llegan a ~26 en julio (Terra sola ve pocos focos en invierno),
por eso su validación admite hasta 40 (en `modis` el límite sigue siendo 20).

## Pruebas

```bash
python -m pytest test_classification.py -v
```

Las pruebas usan datos sintéticos y una proyección falsa: no necesitan `pyproj` ni `pyarrow`.
La escritura Parquet (`publish.write_all`) y `make_albers_transforms` requieren las dependencias de
`requirements.txt`.
