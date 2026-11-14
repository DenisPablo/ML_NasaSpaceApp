# Cómo ejecutar la clasificación y entregar los datos al backend

Guía práctica para generar los archivos de la **Propuesta B** en una computadora del equipo y pasarlos a la API ([FireHistoryInterfaceAPI](https://github.com/smlopezsm/FireHistoryInterfaceAPI)).
El detalle técnico de cada paso está en [CLASIFICACION.md](CLASIFICACION.md).

---

## 1. Requisitos

| Requisito | Valor |
|---|---|
| Python | 3.11 o superior |
| RAM libre | 4–6 GB (≈ 7 M de detecciones en memoria) |
| Disco libre | ~3 GB (CSV crudos + Parquet limpio + salida) |
| Tiempo estimado | Fase 1: unos minutos · Clasificación: ~2–5 min |
| Datos de entrada | Las 8 descargas FIRMS de Argentina (`DL_FIRE_M-C61_*`, `DL_FIRE_SV-C2_*`, `DL_FIRE_J1V-C2_*`, `DL_FIRE_J2V-C2_*`, cada una con `fire_archive_*.csv` y `fire_nrt_*.csv`) |

## 2. Preparación (una sola vez)

```bash
git clone https://github.com/DenisPablo/ml_nasaspaceapp && cd ml_nasaspaceapp
git checkout feat/clasificacion-propuesta-b        # hasta que se fusione en main

python3 -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt

# Datos crudos: copiar las 8 carpetas DL_FIRE_* dentro de Data-Argentina/
mkdir -p Data-Argentina

# Límites provinciales oficiales (IGN vía API Georef)
mkdir -p data/ref
curl -L -o data/ref/provincias.geojson https://apis.datos.gob.ar/georef/api/v2.0/provincias.geojson

# Verificar la instalación
python -m pytest test_classification.py -v           # 17 pruebas en verde
```

## 3. Ejecución

| Paso | Comando | Genera |
|---|---|---|
| 1. Limpieza (Fase 1) | `python clean_firms.py --input-dir Data-Argentina --region argentina` | `data/clean/firms_argentina_clean.parquet` |
| 2. Clasificación | ver abajo | `data/serving_api/<versión>/` + enlace `data/serving_api/current` |

```bash
python run_classification.py \
  --clean data/clean/firms_argentina_clean.parquet \
  --provinces-geojson data/ref/provincias.geojson \
  --raw-dir Data-Argentina \
  --out data/serving_api
```

| Opción | Obligatoria | Para qué |
|---|---|---|
| `--clean` | Sí | Salida de la Fase 1 |
| `--provinces-geojson` | Sí | Límites provinciales del IGN |
| `--raw-dir` | Recomendada | CSV crudos: activa la regla "marcado industrial por FIRMS" (`type = 2`) |
| `--out` | No (`data/serving_api`) | Carpeta de salida |
| `--fecha-max` | No | Cortar en una fecha (`YYYY-MM-DD`) |
| `--k` | No (`200`) | Colchón de los factores; no cambiar sin recalibrar |
| `--factores-version` | No (`AAAA.MM-b1`) | Etiqueta de la tabla de factores |

### Qué tiene que mostrar el log

```
[1/7] Leyendo Parquet limpio
[2/7] Provincias y máscara nacional de celdas
[3/7] Fuentes fijas industriales        → ~489 puntos · ~81.000 focos excluidos
[4/7] Conteos celda-día, focos y satélites operativos
[5/7] Calibración de factores (Propuesta B)
[6/7] Validación                        → JSON con las métricas
[7/7] Publicando en data/serving_api
Listo: data/serving_api/<versión>
```

Si el contrato no se cumple, el programa termina con **código 2**, lista los errores y **no publica nada**.

### Valores de control (corrida de referencia con los datos de Argentina)

| Métrica (`validacion.json`) | Esperado | Si da muy distinto |
|---|---:|---|
| `factor_nacional.modis` | ≈ 3,3 | Revisar que la Fase 1 corrió con `--region argentina` |
| `factor_nacional.terra` | ≈ 9,1 | Ídem |
| `hacia_atras.error_pct` | ≈ 15 % | Revisar filtros de confianza de la Fase 1 |
| `hacia_adelante.error_pct` | ≈ 16–18 % | Ídem |
| `hacia_atras.error_modis_crudo_pct` | ≈ 70 % | — (es la línea base sin armonizar) |
| `cobertura_rangos_pct` | 78–83 % | — |
| `puntos_industriales.n_puntos` | ≈ 489 | Sin `--raw-dir` da algunos menos |
| `filas.dim_celdas` | ≈ 112.000 | Revisar el GeoJSON de provincias |
| `filas.conteos_celda_dia` | ≈ 1,9 M | — |
| `filas.focos` | ≈ 6,2 M | — |

---

## 4. Qué devuelve y con qué formato

Todo queda en `data/serving_api/<versión>/` (por ejemplo `20261114T170000`), y `data/serving_api/current` apunta a la última versión.

```
data/serving_api/
├── current -> 20261114T170000
└── 20261114T170000/
    ├── focos/
    │   ├── anio=2000/part-0.parquet
    │   ├── …
    │   └── anio=2026/part-0.parquet
    ├── conteos_celda_dia/
    │   ├── anio=2000/part-0.parquet
    │   └── …
    ├── dim_celdas.parquet
    ├── dim_provincias.parquet
    ├── factores_provincia_mes.parquet
    ├── factores_region_mes.parquet
    ├── rangos.parquet
    ├── satelite_operativo.parquet
    ├── puntos_industriales.parquet
    ├── metadata.json
    └── validacion.json
```

| Archivo | Formato | Filas aprox. | Para qué lo usa el backend |
|---|---|---:|---|
| `focos/anio=YYYY/part-0.parquet` | Parquet ZSTD, particionado Hive (la columna `anio` va en el nombre de la carpeta) | 6,2 M | Mapa de puntos y descarga CSV |
| `conteos_celda_dia/anio=YYYY/part-0.parquet` | Ídem; disperso (sin ceros), conteos `INT16` | 1,9 M | **Base del calendario**: suma por zona/período |
| `dim_celdas.parquet` | Parquet ZSTD | 112 k | Máscara nacional, celda → provincia/región, geometría |
| `dim_provincias.parquet` | Parquet ZSTD | 24 | Selector de provincia y polígonos del mapa |
| `factores_provincia_mes.parquet` | Parquet ZSTD | 576 (24 × 12 × 2 tipos) | `estimado = factor × focos MODIS` |
| `factores_region_mes.parquet` | Parquet ZSTD | 144 (6 × 12 × 2 tipos: 5 regiones + `Argentina`) | Respaldo si falta una combinación provincia × mes |
| `rangos.parquet` | Parquet ZSTD | 6 | Rango del 80 % por tramo de focos MODIS |
| `satelite_operativo.parquet` | Parquet ZSTD | ~40 k | Distinguir "sin datos" de "cero focos" |
| `puntos_industriales.parquet` | Parquet ZSTD | ~489 | Capa informativa del mapa |
| `metadata.json` | JSON UTF-8 | — | Versiones, fechas, satélite de referencia |
| `validacion.json` | JSON UTF-8 | — | Métricas de la corrida (auditoría; la API no lo necesita) |

Columnas y tipos exactos de cada archivo: [FireHistoryInterfaceAPI · docs/02-contratos-de-datos.md](https://github.com/smlopezsm/FireHistoryInterfaceAPI/blob/main/docs/02-contratos-de-datos.md).

---

## 5. Qué hay que pasar al backend

| Se entrega | No se entrega |
|---|---|
| **La carpeta de versión completa** `data/serving_api/<versión>/` (los 9 Parquet/carpetas + `metadata.json`). `validacion.json` puede ir: sirve como registro y la API lo ignora. | CSV crudos (`Data-Argentina/`) |
| | Parquet limpio (`data/clean/`) |
| | `provincias.geojson` (ya está dentro de `dim_provincias`) |
| | Salidas de las fases 2–4 (`data/daily`, `models/`, serving v2) |

Reglas:
- **Nunca se entregan archivos sueltos:** la versión va completa, porque los factores, los conteos y `metadata.json` tienen que corresponder a la misma corrida.
- No renombrar archivos ni carpetas, ni mover la columna `anio` adentro de los Parquet.
- Si cambia el formato de alguna tabla, se sube `schema_version` en `metadata.json` y se avisa al backend.

### Cómo se carga en el servidor (VPS)

```bash
VERSION=$(readlink data/serving_api/current)

# 1. Subir la versión nueva a releases/
rsync -av --progress data/serving_api/$VERSION/ deploy@<vps>:/var/data/releases/$VERSION/

# 2. Activarla con un cambio atómico del enlace
ssh deploy@<vps> "ln -sfn /var/data/releases/$VERSION /var/data/current"

# 3. Avisar a la API para que recargue (o reiniciar el contenedor desde Dokploy)
curl -X POST https://<api>/admin/reload -H "Authorization: Bearer $ADMIN_TOKEN"
```

Volver a la versión anterior: `ln -sfn /var/data/releases/<versión-anterior> /var/data/current` y repetir el paso 3.

### En desarrollo local (sin VPS)

Montar la carpeta en el contenedor de la API como solo lectura:

```bash
docker run -p 8000:8000 \
  -v "$(pwd)/data/serving_api:/var/data:ro" \
  -e DATA_DIR=/var/data/current \
  firehistory-api
```

---

## 6. Problemas frecuentes

| Síntoma | Causa probable | Solución |
|---|---|---|
| `ModuleNotFoundError: pyarrow / pyproj / pandas` | Dependencias sin instalar | `pip install -r requirements.txt` con el venv activado |
| `FileNotFoundError ... firms_argentina_clean.parquet` | Falta la Fase 1 | Correr el paso 1 |
| `Sin focos MODIS en la ventana de calibración` | La Fase 1 se corrió solo para San Luis o sin MODIS | Repetir con `--region argentina` y las 8 carpetas |
| Código 2: `factor fuera de (…)` | Datos incompletos para algún período | Revisar que estén los archivos `archive` (SP) de los 4 productos |
| Código 2: `cell_id fuera de dim_celdas` | GeoJSON de provincias distinto o recortado | Volver a bajarlo de la API Georef |
| Muy pocos puntos industriales | Falta `--raw-dir` | Agregarlo apuntando a `Data-Argentina` |
| Memoria insuficiente | Menos de 4 GB libres | Cerrar programas o usar `--fecha-max` para probar con un período menor |
