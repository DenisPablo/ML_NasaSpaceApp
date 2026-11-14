# Contexto Técnico para Agente de Programación: Armonización Satelital MODIS × VIIRS (NASA Space Apps 2026)

Este documento consolida las especificaciones técnicas, matemáticas, geoespaciales y de arquitectura de software para el proyecto de armonización de focos de calor **MODIS** y **VIIRS**. Sirve como fuente de verdad operativa para implementar el pipeline ETL, el motor de entrenamiento **Hurdle GBDT**, la validación espacial **LOBBO**, el *serving* analítico con **DuckDB + FastAPI**, la interfaz en **React** y los benchmarks de rendimiento en **CPU**.

---

## 1. Resumen Ejecutivo, Alcance Estricto del MVP y Restricción de 16 Horas

- **Decisión estratégica de alcance (San Luis Only para el MVP):** El MVP se acota **estrictamente a la Provincia de San Luis (~76.748 km², ~3.070 celdas activas de 5 × 5 km)**, descartando el procesamiento a escala nacional (111.248 celdas) durante esta etapa.
  - **Justificación operativa (Ventana de 16 horas):** Esta reducción de dominio (~36× menos celdas) evita la saturación de memoria RAM y los cuellos de botella de cómputo al entrenar en hardware local (**CPU Ryzen 7**), bajando la matriz de entrenamiento 2012–2024 de ~528,2 millones de filas a **~14,57 millones de filas**. Esto permite completar el pipeline ETL, el entrenamiento Hurdle GBDT, la validación y el *serving* en cuestión de horas, encajando dentro de la ventana disponible de **16 horas de desarrollo** con asistencia de IA.
  - **Separación Entorno Local vs. Servidor:** El entrenamiento pesado del modelo se realiza **en local (CPU Ryzen 7)** con tensores `float32`. La ingesta diaria por lotes, la inferencia ligera de árboles y el *serving* web corren en producción sobre un **Cloud Server de DonWeb (4 GB RAM, 2 núcleos)**.
  - **Arquitectura preservada:** Se mantienen intactos la proyección **Argentina Albers Cónica de Área Equivalente (CRS custom sobre GRS 1980)**, la integración **DuckDB + Parquet**, el modelo **Hurdle GBDT (Focal Loss + Tweedie)** y los protocolos de validación **LOBBO**. La máscara nacional (`dim_celdas_argentina.parquet`) queda solo como referencia de escalabilidad futura.
- **Objetivo central:** Resolver la discontinuidad instrumental generada por el recambio de hardware satelital de NASA (**MODIS** 1 km, 2000–presente → **VIIRS** 375 m, 2012–presente) para reconstruir una serie temporal homogénea de **24 años (2000–2024)** sin quiebre estructural en 2012.
- **Indicador operativo único:** **Fire Radiative Power (FRP)** diario acumulado en Megavatios (MW) por celda de 5 × 5 km. Se descartan los conteos simples de focos y los agregados mensuales rígidos.
- **Fuera de alcance del MVP (Fase 2 / Post-MVP):**
  - Expansión territorial a las 111.248 celdas de Argentina Continental.
  - Módulo predictivo de riesgo en tiempo real (24h–72h) con variables meteorológicas dinámicas (ERA5-Land, SMN/Open-Meteo FWI) y de combustible (NDVI/NDWI, SRTM/OSM). *(Nota: en Fase 2, la inferencia diaria aplica el modelo ya entrenado sobre datos nuevos sin reentrenar; no se congela un dataset estático porque perdería capacidad de alerta dinámica, por lo que todo el bloque predictivo se difiere post-MVP).*

### Documentación de Referencia del Proyecto
- [Plan de Implementación: Sistema de Detección y Modelado de Incendios](https://docs.google.com/document/d/1PuuX2eQcFcjZ7wEsyE2kx_KHBv2WmF96l68pPNVx1HM/edit?usp=drivesdk)
- [Propuesta de Proyecto MVP: Monitoreo Térmico y Armonización Satelital (San Luis)](https://docs.google.com/document/d/1qsUazsXUI9JPp0OYWBDgdLCwQ2h5Bzb0TlnVs__1fPc/edit?usp=drivesdk)
- [Diccionario Técnico de Variables Satelitales: MODIS y VIIRS (NASA FIRMS)](https://docs.google.com/document/d/1QY3YhC4Cq72D_6wQusrylWm-Xnelv66x5U9DBlGfKz4/edit?usp=drivesdk)
- [Armonización Operativa de Datos Satelitales de Fuego MODIS y VIIRS (RS-ARG-FUEGO-2026-V1)](https://drive.google.com/file/d/1EinvncjYU5wJAnTY5o4MeQnQHk3es09q/view?usp=drivesdk)

---

## 2. Fundamentos Físicos del FRP y Discrepancias entre Sensores

### 2.1 Comparativa de Hardware (NASA FIRMS)

| Parámetro Técnico | MODIS (Terra / Aqua) | VIIRS Banda I (S-NPP / NOAA-20 / NOAA-21) | Impacto en el Pipeline |
| :--- | :--- | :--- | :--- |
| **Producto FIRMS** | `MOD14` / `MYD14` / `MCD14DL` (C6.1) | `VNP14IMG` / `VJ114IMG` (375 m) | Preferir archivo científico estándar (`SP`) sobre `NRT` para entrenamiento. |
| **Resolución en Nadir** | 1.000 m × 1.000 m (1,0 km²) | 375 m × 375 m (0,14 km²) | VIIRS detecta entre **3,5× y 7,1×** más píxeles activos que MODIS. |
| **Resolución en Borde** | ~4.800 m × 2.000 m (~9,8 km² a 55°) | ~800 m × 800 m (≤ 0,65 km²) | MODIS sufre severo **efecto Bow-Tie** ($\propto \sec^3\theta$); VIIRS agrega 3:1, 2:1, 1:1 a bordo. |
| **Área mínima con llama** | 100–750 m² | 15–35 m² | MODIS omite quemas agrícolas y frentes subpíxel de baja intensidad. |
| **Umbral mínimo FRP** | ~10–20 MW / píxel | ~0,75–3,0 MW / píxel | Requiere modelado no lineal dependiente de bioma y ángulo de visión. |
| **Serie disponible** | Feb 2000 (Terra) / May 2002 (Aqua) | Ene 2012 (S-NPP) / Ene 2018 (NOAA-20) | Ventana de solapamiento para calibración: **2012–2024**. |

### 2.2 Física Radiativa y Ecuación de Wooster (FRP)
- **Ventanas espectrales (Ley de Planck):** Los algoritmos contextuales explotan el contraste entre el **Infrarrojo Medio (MIR, ~3,7–4,0 µm)** (`brightness` en MODIS; `bright_ti4` en VIIRS), donde las llamas (600–1400 K) generan un pico exponencial, y el **Infrarrojo Térmico (TIR, ~10,5–12,5 µm)** (`bright_t31` / `bright_ti5`), que mide la temperatura de fondo (~300 K) y detecta oclusión por nubes.
- **Aproximación subpíxel de Dozier:** $L_\lambda = p_{\text{fuego}} B(\lambda, T_{\text{fuego}}) + (1 - p_{\text{fuego}}) B(\lambda, T_{\text{fondo}})$.
- **Potencia Radiativa del Fuego (FRP en MW):** Derivada mediante la relación empírica de Wooster a partir del exceso de radiancia MIR:
  $$\text{FRP} = \frac{A_{\text{píxel}} \cdot \sigma}{a} \left(L_{\text{MIR}} - \bar{L}_{\text{MIR,fondo}}\right)$$
  donde $A_{\text{píxel}}$ es el área de la huella en superficie, $\sigma = 5,6704 \times 10^{-8}\text{ W m}^{-2}\text{ K}^{-4}$ es la constante de Stefan-Boltzmann y $a$ es un coeficiente espectral del sensor.
- **Por qué FRP resuelve la armonización:** Por conservación de energía radiativa, la suma del FRP de múltiples píxeles de 375 m de VIIRS dentro de una celda de 5 × 5 km converge hacia la energía total emitida, permitiendo calibrar la subestimación de MODIS mediante covariables geométricas ($\bar{\theta}$, `scan`, `track`) y atmosféricas ($f_{\text{nube}}$).

### 2.3 Reglas Críticas de Limpieza e Ingesta FIRMS
1. **Conversión Horaria Obligatoria (`UTC` → `UTC-3`):** Argentina opera en `UTC-3` sin horario de verano. Las pasadas nocturnas entre las 21:00 y 23:59 hora local aparecen en FIRMS con fecha del día siguiente (`00:00`–`02:59 UTC`). Convertir `acq_date` + `acq_time` a `America/Argentina/Buenos_Aires` (o `UTC-3` fijo) **antes** de agrupar por día operativo (`00:00` a `23:59`).
2. **Filtro de Tipo de Fuente (`type`):** En archivos científicos (`SP`), conservar exclusivamente `type == 0` (fuego presunto en vegetación), descartando volcanes (`1`), antorchas/fuentes industriales estáticas (`2`) y detecciones *offshore* (`3`).
3. **Filtro de Confianza (`confidence`):** Descartar detecciones de confianza baja (`< 30%` en MODIS; `'l'` en VIIRS) para suprimir falsos positivos por reflejo solar (*sun glint*) en suelos desnudos.

---

## 3. Infraestructura Espacial: CRS POSGAR 2007 y Grilla de 5 × 5 km

### 3.1 Sistema de Referencia Cartográfico (Argentina Albers Custom — `+proj=aea`)
Trabajar en grados decimales (`EPSG:4326`) introduce una deformación relativa >36% entre el norte y el sur de Argentina (1° de longitud ≈ 103,2 km en Jujuy vs. 65,6 km en Tierra del Fuego). Por ello, toda geometría se proyecta a una **Proyección Albers Cónica de Área Equivalente para Argentina (CRS custom sobre elipsoide GRS 1980 / datum POSGAR 2007)**:
- **Elipsoide:** GRS 1980 ($a = 6.378.137\text{ m}$, $1/f \approx 298,257$)
- **Latitud de Origen ($\phi_0$):** $-37^\circ 00' 00''$ S | **Meridiano Central ($\lambda_0$):** $-60^\circ 00' 00''$ O
- **Paralelos Estándar:** $\phi_1 = -26^\circ 00' 00''$ S, $\phi_2 = -48^\circ 00' 00''$ S
- **Falso Este / Norte:** $2.500.000,0\text{ m}$ / $0,0\text{ m}$
- **Error de escala superficial:** Acotado estrictamente dentro de $\pm 0,05\%$.
- **Definición PROJ para `pyproj` / DuckDB / Shapely:**
  ```text
  +proj=aea +lat_0=-37 +lon_0=-60 +lat_1=-26 +lat_2=-48 +x_0=2500000 +y_0=0 +ellps=GRS80 +units=m +no_defs
  ```
  > **Advertencia técnica:** No existe un código EPSG oficial para Albers sobre POSGAR 2007. No utilizar `EPSG:5347`, ya que corresponde a *POSGAR 2007 / Argentina 5* (Transversa de Mercator / Gauss-Krüger), incompatible con esta grilla.

### 3.2 Geometría de Grilla y *Spatial Hashing* en $O(1)$

> **Nota de diseño (Grilla Regular 5×5 km vs. Grilla Hexagonal):** Aunque en discusiones exploratorias se evaluó el uso de grillas hexagonales (p. ej. H3), la especificación operativa adoptada en el *Plan de Implementación* utiliza una **grilla regular cuadrada de 5.000 m × 5.000 m (25 km² / 2.500 ha)** sobre el CRS **Argentina Albers (`+proj=aea`)**. Esta decisión permite asignar cada foco satelital a su celda mediante división entera en tiempo $O(1)$ sin intersecciones geométricas costosas (*point-in-polygon*), acelerando el pipeline dentro del sprint de 16 horas.

- **Rectángulo contenedor de referencia (en metros, Argentina Albers `+proj=aea`):**
  - $X \in [1.400.000,\ 3.700.000]$ m Este ($N_{\text{cols}} = 460$)
  - $Y \in [-2.100.000,\ 1.800.000]$ m Norte ($N_{\text{rows}} = 780$)
- **Fórmulas de *Spatial Hashing* $O(1)$:**
  ```python
  col = np.floor((X - 1_400_000.0) / 5_000.0).astype(np.int32)
  row = np.floor((Y - (-2_100_000.0)) / 5_000.0).astype(np.int32)
  cell_id = row * N_COLS + col
  ```
- **Máscara Territorial Activa del MVP:**
  - `dim_celdas_san_luis.parquet`: **~3.070 celdas activas** (recorte oficial del IGN para la Provincia de San Luis). *(Nota: `dim_celdas_argentina.parquet` con 111.248 celdas queda reservada para Fase 2).*

---

## 4. Pipeline ETL, Volumetría e Integración con DuckDB

### 4.1 Frecuencia de Ejecución: Normalización Histórica Única vs. Procesamiento Diario
- **Normalización histórica (ejecución única en local):** Todo el pipeline de normalización para el Modelo 1 de armonización (reproyección a Argentina Albers `+proj=aea`, conversión `UTC` → `UTC-3`, imputación explícita de ceros, cálculo de ventanas móviles en DuckDB y conversión de columnas numéricas a **`float32`** para proteger la memoria) se ejecuta **una sola vez** sobre el archivo histórico (2000–2024) para construir y persistir la serie base limpia en Parquet.
- **Operación continua (ingesta diaria incremental en servidor):** Una vez consolidada esa línea base histórica, no se vuelve a normalizar el historial completo; únicamente se procesan por lotes los **datos diarios nuevos** directamente sobre Parquet con DuckDB para mantener el consumo de RAM mínimo dentro del límite de **4 GB** del servidor.

### 4.2 Imputación Explícita de Ceros y Comparativa de Volumetría
Para entrenar un modelo capaz de distinguir cuándo **no** hay fuego y evitar alucinaciones, la tabla de entrenamiento realiza el producto cartesiano completo entre celdas activas y días calendario:

$$\text{Total de filas} = \text{Celdas activas} \times \text{Días de la ventana}$$

| Escenario | Celdas Activas | Ventana 2012–2024 (~4.749 días) | Backfilling 2000–2024 (~9.070 días) | Estado en el MVP |
| :--- | :--- | :--- | :--- | :--- |
| **Piloto San Luis (MVP)** | **3.070** | **~14,57 millones de filas** | **~27,84 millones de filas** | **Activo (Objetivo 16h)** |
| Región Cuyo (Benchmark escala) | ~12.000–14.000 | ~57–66,5 millones de filas | ~109–127 millones de filas | Solo benchmark sintético |
| Nacional (Argentina) | 111.248 | ~528,2 millones de filas | ~1.009 millones de filas | Diferido (Post-MVP) |

### 4.3 Variables Agregadas por `(cell_id, fecha)`
- `N` ($N^{(s)}_{g,t}$): Conteo de píxeles activos del sensor $s$ en la celda $g$ el día $t$.
- `phi` ($\Phi^{(s)}_{g,t}$): FRP diario acumulado (MW).
- `phi_max` ($\Phi^{(s)}_{\max,g,t}$): FRP máximo individual del día (diferencia fuegos violentos de quemas dispersas).
- `theta_bar` ($\bar{\theta}^{(s)}_{g,t}$): Ángulo cenital medio de barrido (corrige degradación periférica / *Bow-Tie*).
- `f_nube` ($f_{\text{nube},g,t}$): Fracción de nubosidad diaria extraída de QA flags (discrimina "cero real" de "cero por oclusión"; celdas con $>80\%$ de nubes se marcan en UI como visibilidad ocluida).
- **Ventanas móviles retrospectivas:** Sumas móviles a **3 días**, **7 días** (memoria térmica ante tormentas de 24–48h) y **30 días** (severidad mensual acumulada) para `N` y `phi`.

### 4.4 Patrón de Integración DuckDB + Parquet + FastAPI
**DuckDB** actúa como motor OLAP embebido *in-process* (on-demand, sin servidor dedicado) ejecutando SQL columnar directamente sobre archivos `.parquet` de San Luis en disco:

```python
import duckdb

con = duckdb.connect(database=":memory:")

# Ejemplo 1: Agregación diaria con imputación explícita de ceros para San Luis (ETL)
query_etl = """
WITH calendario AS (
    SELECT UNNEST(GENERATE_SERIES(DATE '2012-01-01', DATE '2024-12-31', INTERVAL 1 DAY))::DATE AS fecha
),
grilla_base AS (
    SELECT c.cell_id, cal.fecha
    FROM read_parquet('data/dim_celdas_san_luis.parquet') c
    CROSS JOIN calendario cal
),
firms_diario AS (
    SELECT
        cell_id,
        fecha_local AS fecha,
        COUNT(*)::INT32 AS n_modis,
        SUM(frp)::FLOAT AS phi_modis,
        MAX(frp)::FLOAT AS phi_max_modis,
        AVG(scan_angle)::FLOAT AS theta_bar
    FROM read_parquet('data/raw/modis_san_luis_*.parquet')
    GROUP BY cell_id, fecha_local
)
SELECT
    g.cell_id,
    g.fecha,
    COALESCE(f.n_modis, 0) AS n_modis,
    COALESCE(f.phi_modis, 0.0) AS phi_modis,
    COALESCE(f.phi_max_modis, 0.0) AS phi_max_modis,
    f.theta_bar,
    SUM(COALESCE(f.phi_modis, 0.0)) OVER (
        PARTITION BY g.cell_id ORDER BY g.fecha ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS phi_modis_3d,
    SUM(COALESCE(f.phi_modis, 0.0)) OVER (
        PARTITION BY g.cell_id ORDER BY g.fecha ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS phi_modis_7d
FROM grilla_base g
LEFT JOIN firms_diario f ON g.cell_id = f.cell_id AND g.fecha = f.fecha
"""

# Ejemplo 2: Consulta de Serving en FastAPI (MODIS crudo vs Armonizado vs VIIRS real)
query_serving = """
SELECT
    cell_id,
    SUM(phi_modis) AS frp_modis_crudo,
    SUM(phi_harmonized) AS frp_armonizado,
    SUM(phi_viirs) AS frp_viirs_real
FROM read_parquet('data/serving/san_luis_harmonized.parquet')
WHERE fecha BETWEEN ? AND ?
GROUP BY cell_id
"""
```

---

## 5. Arquitectura del Modelo Hurdle GBDT (LightGBM)

Los datos diarios presentan **inflación masiva de ceros (>98,5% de filas con `FRP = 0`)** y una **distribución de cola pesada** en los casos positivos (1 MW a >2.500 MW). Se implementa un modelo **Hurdle en dos etapas desacopladas**:

$$p(y_{g,t} \mid x_{g,t}) = \begin{cases} 1 - \pi(x_{g,t}), & y_{g,t} = 0 \\ \pi(x_{g,t}) \cdot f_{\text{Tweedie}}(y_{g,t} \mid \mu(x_{g,t}), \phi, p), & y_{g,t} > 0 \end{cases}$$

### 5.1 Etapa 1a: Clasificador de Ocurrencia (`Focal Loss`)
Predice la probabilidad $\pi_{g,t} = \Pr(Y_{g,t} > 0 \mid x_{g,t}) = \sigma(F_{\text{occ}}(x_{g,t}))$. Para evitar que los millones de ceros fáciles saturen el gradiente, se utiliza **Binary Focal Loss** con $\gamma = 2,0$ y $\alpha = 0,75$:

$$\mathcal{L}_{\text{Focal}}(\pi, y) = -\alpha_t (1 - p_t)^\gamma \log(p_t), \quad p_t = \begin{cases} \pi & \text{si } y = 1 \\ 1 - \pi & \text{si } y = 0 \end{cases}$$

*(Nota de implementación en LightGBM: requiere función objetivo personalizada `fobj` que retorne gradiente y hessiano analíticos o numéricos de $\mathcal{L}_{\text{Focal}}$, dado que LightGBM no incluye Focal Loss nativa).*

### 5.2 Etapa 1b: Regresor de Intensidad (`Tweedie Deviance`)
Entrenado **únicamente sobre registros positivos ($Y_{g,t} > 0$)** mediante una distribución compuesta **Poisson-Gamma de Tweedie** ($\text{Var}(Y) = \phi \mu^p$, con $1 < p < 2$). Utiliza enlace logarítmico $\mu(x_{g,t}) = \exp(F_{\text{int}}(x_{g,t}))$, garantizando físicamente $\mu > 0$:

$$\mathcal{L}_{\text{Tweedie}}(y, \mu; p) = 2 \left( \frac{y^{2-p}}{(1-p)(2-p)} - \frac{y\mu^{1-p}}{1-p} + \frac{\mu^{2-p}}{2-p} \right)$$

- **Potencia de varianza óptima (`tweedie_variance_power`):**
  - $p = 1,45$ para conteo de focos ($N$).
  - $p = 1,62$ para energía radiativa acumulada ($\Phi$, FRP en MW).
- En LightGBM: `objective="tweedie"`, `tweedie_variance_power=1.62` (para FRP).

### 5.3 Inferencia Combinada
$$\hat{y}_{\text{VIIRS-equiv}} = \hat{\pi}_{g,t} \cdot \hat{\mu}_{g,t}$$

---

## 6. Protocolo de Validación Espacial (LOBBO), Benchmark Delta del Paraná y Criterios de Aceptación

Una división aleatoria (`train_test_split`) provoca **fuga de información espacial (*data leakage*)** porque celdas vecinas comparten vegetación, humedad y meteorología casi idénticas.

### 6.1 Validación Primaria: Estrategia LOBBO en San Luis
1. **Partición:** El dominio de San Luis se divide en bloques espaciales/ecorregionales independientes (p. ej. bloques Norte–Sur: Sierras Centrales / Chaco Seco vs. llanuras del Sur), manteniendo la compatibilidad con el protocolo nacional de 12 ecorregiones de la APN.
2. **Buffer de Exclusión de 20 km:** Se aplica una franja muerta de **20 km (4 celdas de 5 km)** alrededor del bloque de validación. Ninguna celda dentro del buffer participa en entrenamiento, suprimiendo la autocorrelación espacial transfronteriza.
3. **Evaluación:** Entrenar en $K - 1$ bloques y evaluar en el bloque retenido durante la ventana 2012–2024.

### 6.2 Validación Secundaria: Benchmark Ecológico en el Delta del Paraná
- **Alcance y momento de ejecución:** Una vez finalizada y aprobada la validación primaria en San Luis, se ejecutarán pruebas secundarias sobre la región del **Delta del Río Paraná**.
- **Justificación técnica:** El modelo entrenado exclusivamente en San Luis prioriza la precisión local en topografía serrana e interfaz urbano-forestal, por lo que no generaliza directamente a otros biomas sin ajuste de parámetros o reentrenamiento local. El Delta del Paraná actúa como **benchmark ecológico de estrés** para densidad vegetal extrema (pajonales) y estrés hídrico/pulsos hidrológicos, donde VIIRS detecta históricamente ~6,2× más focos que MODIS, permitiendo validar que la arquitectura Hurdle GBDT corrige el sesgo instrumental en ecosistemas contrastantes.

### 6.3 Umbrales Mínimos de Aceptación (Anti-Sesgo y Anti-Alucinación)
1. **Desempeño LOBBO:** $R^2 \ge 0,85$ y $\text{MBE} \in [-0,10, +0,10]$ focos/celda.
2. **Prueba de Cero-Inflación (Días Basales):** En jornadas históricas con cielo despejado y sin fuego reportado, el modelo debe predecir `0` en $\ge 98\%$ de las celdas (tasa de falsos positivos $< 2\%$).
3. **Auditoría de Continuidad Multidecadal (2000–2024):** La serie anual agregada de San Luis no debe presentar un escalón estadístico artificial en 2012 al transicionar de MODIS armonizado a VIIRS.

---

## 7. Script de Benchmark Sintético en CPU (San Luis vs. Cuyo)

El siguiente script replica exactamente la **forma**, el **tipo de dato (`float32`)** y la **dispersión estadística (98,5% de ceros)** para evaluar rendimiento y RAM en **CPU (Ryzen 7)** a distintas escalas (p. ej. **San Luis** ~3.070 celdas vs. **Cuyo** ~12.000–14.000 celdas), evitando desbordamientos de VRAM en GPU.

```python
import time
import numpy as np
import lightgbm as lgb

def generate_synthetic_fire_tensor(n_rows: int = 14_570_000, n_features: int = 18, zero_rate: float = 0.985, seed: int = 42):
    """Genera matriz sintética float32 replicando San Luis (2012-2024) con 98.5% de ceros."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_features), dtype=np.float32)

    # Máscara de ocurrencia (1.5% positivos)
    y_occ = (rng.random(n_rows, dtype=np.float32) > zero_rate).astype(np.int32)

    # FRP positivo de cola pesada (1 MW a >2500 MW) solo donde y_occ == 1
    y_frp = np.zeros(n_rows, dtype=np.float32)
    pos_idx = np.where(y_occ == 1)[0]
    y_frp[pos_idx] = np.clip(rng.gamma(shape=1.2, scale=45.0, size=len(pos_idx)), 0.5, 3000.0).astype(np.float32)
    return X, y_occ, y_frp

def run_benchmark(device: str = "cpu", n_rows: int = 2_000_000):
    """Ejecuta benchmark de Etapa 1a (Clasificación) y Etapa 1b (Tweedie) en CPU."""
    print(f"\n--- Benchmark LightGBM [{device.upper()}] | Filas: {n_rows:,} ---")
    t0 = time.perf_counter()
    X, y_occ, y_frp = generate_synthetic_fire_tensor(n_rows=n_rows)
    print(f"Generación de tensor float32: {time.perf_counter() - t0:.2f}s | RAM X: {X.nbytes / 1e9:.2f} GB")

    common_params = {
        "device": device,
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_bin": 255,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Etapa 1a: Clasificador de Ocurrencia (proxy de carga computacional para Focal Loss)
    t1 = time.perf_counter()
    clf = lgb.LGBMClassifier(objective="binary", **common_params)
    clf.fit(X, y_occ)
    t_clf = time.perf_counter() - t1
    print(f"Etapa 1a (Clasificador {n_rows:,} filas): {t_clf:.2f}s")

    # Etapa 1b: Regresor Tweedie (solo registros positivos Y > 0)
    pos_mask = y_occ == 1
    X_pos, y_pos = X[pos_mask], y_frp[pos_mask]
    t2 = time.perf_counter()
    reg = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.62, **common_params)
    reg.fit(X_pos, y_pos)
    t_reg = time.perf_counter() - t2
    print(f"Etapa 1b (Tweedie p=1.62 sobre {len(y_pos):,} positivos): {t_reg:.2f}s")

if __name__ == "__main__":
    # Muestra inicial rápida (~2M filas) -> escalar a 14_570_000 (San Luis) o ~57M-66M (Cuyo) según RAM
    run_benchmark(device="cpu", n_rows=2_000_000)
```

---

## 8. Stack Tecnológico, Infraestructura y Arquitectura de Despliegue

| Capa | Tecnología Confirmada | Entorno de Ejecución | Notas de Implementación |
| :--- | :--- | :--- | :--- |
| **ETL & Normalización Histórica** | Python 3.11+, `polars`/`pandas`, `pyproj`, `shapely` | **Local (CPU Ryzen 7)** | Ejecución única histórica (`EPSG:4326` → Argentina Albers `+proj=aea`, `float32`). |
| **Machine Learning (Entrenamiento)** | `lightgbm`, `scikit-learn` | **Local (CPU Ryzen 7)** | Hurdle GBDT (Focal Loss custom + Tweedie `p=1.62`). |
| **Base de Datos Analítica** | **DuckDB** sobre **Parquet** | **DonWeb Cloud (4 GB RAM / 2 vCPU)** | Consultas *on-demand* por lotes directo en disco para mantener RAM baja. |
| **Ingesta & Inferencia Diaria** | Python + DuckDB + `lightgbm` (inferencia) | **DonWeb Cloud (4 GB RAM / 2 vCPU)** | Procesamiento diario incremental; inferencia ligera en 2 núcleos. |
| **Backend API** | **FastAPI** (Python, Uvicorn) | **DonWeb Cloud (4 GB RAM / 2 vCPU)** | Endpoints REST/GeoJSON conectados a DuckDB/Parquet. |
| **Frontend / Interfaz Web** | **React** + **MapLibre GL** | Cliente Web | Mapa de las 3.070 celdas de San Luis, conmutador *MODIS Crudo vs. Armonizado vs. VIIRS*, indicador de nubes $>80\%$ y series 2000–2024. |
