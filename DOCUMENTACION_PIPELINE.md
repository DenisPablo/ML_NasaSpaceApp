# Documentación Técnica del Pipeline MODIS × VIIRS (`2000–2026`) — Escala Argentina

Documento maestro de arquitectura y ciencia de datos para **[NASA Space Apps 2026 — Harmonization of MODIS and VIIRS Hot Spots](https://www.spaceappschallenge.org/2026/challenges/modis-and-viirs-hot-spots/)**, alineado con la **Auditoría v2** sobre los datos reales de NASA FIRMS para **Argentina Continental**.

---

## 1. Alcance Geográfico y Arquitectura Dispersa

| Dimensión | Decisión de Arquitectura |
| :--- | :--- |
| **Dominio principal** | **Argentina Continental completa** (`4.104.670` detecciones crudas en 6 CSVs de `Data-Argentina/`: `fire_archive` y `fire_nrt` de `M-C61`, `J1V-C2`, `J2V-C2`). |
| **Grilla espacial** | **Argentina Albers Equal Area Conic (`+proj=aea`)** en celdas de `5 × 5 km` (`25 km²` exactos). **Prohibido usar `EPSG:5347`** (Transversa de Mercator zona 5, distorsiona áreas en el oeste/sur). |
| **Definición PROJ** | `+proj=aea +lat_0=-37 +lon_0=-60 +lat_1=-26 +lat_2=-48 +x_0=2500000 +y_0=0 +ellps=GRS80 +units=m +no_defs` |
| **Ejemplo de referencia** | San Luis (`lat = -33.3014`, `lon = -66.3321`) → `cell_id = 229643` (`col = 163`, `row = 499`). |
| **Representación dispersa** | En lugar de materializar ceros sobre `111.248` celdas × `9.500` días (~1.000M de filas), se almacena la **matriz diaria dispersa de celdas activas** (`n_modis > 0` o `n_viirs_n20 > 0`), resultando en **~150.000–300.000 filas activas** para los 26 años. |
| **Presets regionales (UI/API)** | Cualquier `bbox` o `lat/lon` del país, con accesos rápidos a **Provincia de San Luis** (interfaz urbano-forestal serrana), **Delta del Paraná** (estrés hídrico y pajonales) y **Gran Chaco / Córdoba / Corrientes**. |

---

## 2. Reglas de Sensores, Calibración y Calidad (Auditoría v2)

### 2.1 Sensor de Referencia Único: VIIRS NOAA-20 (`N20`)
- **Por qué no usar `SV-C2` ni sumar `N21`:** En los 6 CSVs disponibles no está `DL_FIRE_SV-C2` (Suomi-NPP, que además finalizó su entrega operativa el 2 de noviembre de 2026), por lo que no hay datos VIIRS en `2012–2017`. A su vez, sumar `NOAA-21` (`N21`) desde marzo de 2023 introduce un **salto artificial de `×1,92` en el FRP y `+37%` en celda-días**.
- **Regla estricta:**
  - Target único de conteo y energía: **`satellite == 'N20'`** (`n_viirs_n20`, `phi_viirs_n20`).
  - Ventana de entrenamiento/calibración: **`2018-04-01` a `2025-12-31`**.
  - Valores observados VIIRS `< 2018-04-01`: almacenados como **`NULL`** (nunca `0`), distinguiendo ausencia de sensor de ausencia de fuego.

### 2.2 Desglose MODIS (`Aqua` vs. `Terra`) y Período Inicial
- **Correlación con NOAA-20:** **MODIS Aqua** (pasada vespertina ~13:30 hs, coincidente con N20) alcanza una correlación log mensual de **$r = 0,967$** con NOAA-20, frente a **$r = 0,918$** de **MODIS Terra** (pasada matutina ~10:30 hs y deriva orbital creciente en 2025–2026).
- `aggregate_daily.py` desglosa explícitamente `n_modis_aqua`, `n_modis_terra`, `phi_modis_aqua` y `phi_modis_terra` junto a los agregados totales.
- **Bandera de confianza `2000-11` a `2002-06`:** Período **solo-Terra** previo al lanzamiento de Aqua; las estimaciones en ese tramo llevan menor confianza o ajuste específico.

### 2.3 Limpieza Temporal, Física de Píxel y Antorchas en `NRT`
- **Huso horario (`UTC-3` fijo):** Se aplica `UTC - 3h` uniforme en `2000–2026` porque los satélites heliosíncronos siguen la hora solar local, no los decretos civiles. *(Nota metodológica: el horario de verano `UTC-2` de 2008–2009 afectó solo ~409 focos nocturnos en San Luis y no debe aplicarse).*
- **`scan_bar_modis` y `track_bar_modis`:** Representan las **dimensiones físicas medias del píxel en kilómetros (km)** (desde `1.0 km` en nadir hasta `4.8 km` en el borde, donde el área supera `9,8 km²` por efecto *Bow-Tie*), actuando como *proxy* directo del ángulo cenital de barrido.
- **Filtro de antorchas en `NRT`:** Dado que los archivos `NRT` (agosto–noviembre 2026) no incluyen `type`, `clean_firms.py` construye una **lista negra** de celdas/coordenadas (redondeadas a `0.01°`) con historial recurrente de `type == 2` (fuentes térmicas estáticas/refinerías) en `SP` y las excluye también de `NRT`.

---

## 3. Doble Rol de Conteos y FRP + Modelado

### 3.1 Alineación con el Desafío Oficial de NASA
1. **Producto Principal (Obligatorio NASA):** **Calendario de Actividad de Quema (Matriz Año × Mes, `2000–2026`)** basado en **Conteos Armonizados (`n_harmonized`)**, comparando cada mes contra su **mediana histórica multianual** y con descarga directa en **CSV**.
2. **Doble rol del FRP (`phi_*`, MW):**
   - **Por dentro (Feature física):** Un `1 foco` de MODIS puede ser una quema agrícola de `8 MW` o un incendio forestal de `200 MW`. Usar `phi_modis`, `phi_max_modis`, `scan_bar_modis` y `track_bar_modis` permite al modelo estimar cuántos focos de 375 m detectaría VIIRS dentro de ese píxel.
   - **Por fuera (Capa analítica UI):** Conmutador en calendario y mapa entre **`[ Conteo de Focos Armonizados | Intensidad Energética FRP (MW) ]`**.

### 3.2 Baseline Estadístico y Modelos LightGBM Tweedie
- **Regla de no-alucinación:** Antes de `2018-04-01`, si `n_modis == 0` entonces `n_harmonized = 0` y `phi_harmonized = 0.0`.
- **Baseline Estadístico (`n_scaled_baseline`):** Factor global **`5,5×`** (la mediana nacional `NOAA-20 / Aqua` es `5,19×`, y en zonas serranas como San Luis oscila entre `5,15×` y `7,23×`, promediando `~5,5×`) o razón mediana mensual $k_m = \text{MEDIAN}(N_{\text{N20}} / N_{\text{MODIS}})$ en `2018–2022`.
- **Modelos GBDT sobre celdas-día activas (`n_modis > 0`, `2018-04` a `2025`):**
  - Hiperparámetros comunes: `num_leaves=31`, `min_child_samples=50`, `n_estimators=150`.
  - Features (14): `[n_modis, n_modis_aqua, n_modis_terra, phi_modis, phi_max_modis, scan_bar_modis, track_bar_modis, n_modis_3d, n_modis_7d, phi_modis_7d, grid_row, grid_col, doy_sin, doy_cos]`.
  - **Modelo A (Conteo `n_harmonized_ml`):** `objective="tweedie"`, `tweedie_variance_power=1.45`, target = `n_viirs_n20`.
  - **Modelo B (FRP `phi_harmonized_ml`):** `objective="tweedie"`, `tweedie_variance_power=1.62`, target = `phi_viirs_n20`.

### 3.3 Protocolo de Validación Doble (Sin Submuestreo)
1. **Validación Temporal:** Entrenar en `2018-04` a `2022-12` y evaluar fuera de muestra en `2023-01` a `2025-12` (conjunto completo, sin submuestreo), comprobando mejora o paridad frente al baseline `5,5×`.
2. **Validación Espacial (LOBBO 20 km):** Bloques latitudinales/ecorregionales con franja de exclusión de `4` celdas (`20 km`), incluyendo evaluación específica en **San Luis** y **Delta del Paraná**.
