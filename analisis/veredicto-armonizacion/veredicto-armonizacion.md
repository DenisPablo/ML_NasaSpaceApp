# Veredicto científico de la armonización MODIS → VIIRS

**Veredicto: sólido con reservas.**

> **Revisado el 2026-11-15** contra `main` de la API (18d6998) y su `data/current/`. H2 ya estaba corregido en `main` y quedó descartado. Una segunda medición independiente confirmó las medianas MODIS de H1 (75.385 / 50.362) y la razón S-NPP/NOAA-20 de H7 (1,02). Para los días parciales encontró 37 días y ~7.230 focos con otro filtro de días. Con el filtro de `m4b_gaps.py` (los dos satélites operativos, N20 ≥ 50 y S-NPP < 0,5 × N20) salen 31 días y 7.054 focos. El orden de magnitud no cambia.

El factor V/M es reproducible y no muestra tendencia en 2012–2025. Además, el error nacional no crece al alejarse del período de calibración. Las reservas están en tres lugares: la cobertura real de los rangos fuera de muestra, el criterio de "día operativo" y la anomalía, que compara meses de dos eras con niveles distintos.

Fecha: 2026-11-15 · Versión de datos: `20261114T174945` (`pipeline_commit = dcf5b18`, `factores_version = 2026.11-b1`).

---

## 1. Qué se midió y cómo

- **Datos.** `serving_api/20261114T174945/` de este repo, idéntico byte a byte a `data/current/` de FireHistoryInterfaceAPI (ver §6).
- **Reproducción.** Los scripts importan `classification/` de este repo, que es igual al de `dcf5b18`. Reproduce los factores publicados con diferencia máxima 0,0 (`modis` y `terra`), los 6 tramos de `rangos.parquet` y la cobertura de 83,1 % de `validacion.json`. Todo lo que sigue parte de esa base reproducida.
- **Validación.** La unidad es provincia × mes. La validación principal deja un año afuera (LOYO, la misma que usa el pipeline para los rangos). También usé:
  - una prueba hacia adelante: calibrar en 2012–2019 y probar en 2020–2025;
  - leave-one-province-out (LOPO);
  - una prueba de distancia temporal: calibrar con un año y predecir cada uno de los otros.
- **Lógica de la API.** Para evaluar la confianza y la anomalía repliqué el código de `app/services/armonizacion.py` y `anomalia.py` a escala nacional. No levanté la API.
- **Scripts.** `m0`–`m9` en `scripts/`; las salidas completas están en `resultados/` (§7).

---

## 2. Tabla de hallazgos

Severidad: **Alta** = puede cambiar una conclusión que ve el usuario. **Media** = sesgo acotado o medido pero sin documentar. **Baja** = de documentación o cosmética.

| # | Tema | Hallazgo | Evidencia medida | Sev. | Recomendación | Resp. |
|---|---|---|---|---|---|---|
| H1 | Anomalía | La muestra de referencia mezcla dos eras con niveles distintos: 2001–2011, estimada a partir de MODIS, y 2012–2025, observada. El salto ya está en **MODIS crudo**, así que el factor no lo genera. Pero la anomalía lo hereda. | MODIS crudo (mediana anual): 75.385 en 2003–2011 contra 50.362 en 2012–2025 (razón 0,67). Armonizado nacional: tendencia de −2,65 %/año (Kendall τ = −0,44, p = 0,002). Mediana 2003–2011 = 245.761 contra 162.068 en 2012–2025 (Mann-Whitney p = 0,003). En 2012+, el 35 % de los meses sale "por debajo" o "muy por debajo" (lo esperado es 25 %) y solo el 17 % sale "por encima" o "muy por encima". Si la referencia fueran solo años observados, cambiaría la categoría del 32 % de los meses 2012–2026 (57 de 178). | **Alta** | Mostrar la anomalía contra la referencia observada 2012+, que tiene 13–14 años y supera el mínimo de 10. Alternativa: informar las dos referencias y el año de corte. Documentar que el descenso 2003→2012 no es atribuible al método, pero tampoco está verificado como real (ver H9). | API (y documentación) |
| H2 | Anomalía | ~~El mes parcial entra en la muestra de referencia.~~ **Descartado en la revisión:** en `main` (18d6998) la API ya lo excluye. | `app/services/anomalia.py:39` filtra `periodo != mes and periodo[5:] == mes[5:] and not datos.parcial`. El análisis leyó una versión anterior del código. | — | Ninguna. Ya hay prueba en `tests/`. | — |
| H3 | Rangos 80 % | La cobertura publicada (83,1 %) es **dentro de muestra**: aplica factores ajustados con todos los años. Por tramo, la cobertura LOYO es de alrededor de 80 % **por construcción**, porque los multiplicadores salen de esas mismas razones. Fuera de muestra real, la cobertura baja, y cae más en los meses grandes. | LOYO con los rangos publicados: 81,0 % global y entre 79,6 % y 80,0 % por tramo (tautológico). Hacia adelante (factores y rangos de 2012–19, prueba en 2020–25): **77,1 %** global y **64,0 % en el tramo > 500 focos MODIS**. Fallas repartidas: 9,0 % por debajo y 10,0 % por encima. | **Alta** | Informar la cobertura hacia adelante como métrica principal. Recalibrar los rangos con validación hacia adelante o por bloques de años. Ensanchar el tramo > 500 (es el de los meses de incendios grandes). | Pipeline |
| H4 | Rangos 80 % | La cobertura **varía por provincia**, pero los rangos son únicos para todo el país. | LOYO: de 66,5 % (Salta) y 67,7 % (Neuquén) a 89,2 % (Santa Fe) y 87,4 % (Córdoba, Formosa). Con test binomial contra 80 %, 9 de 24 provincias difieren con p < 0,05 y 6 siguen difiriendo con Bonferroni (02, 30, 58, 66, 82, 94). | Media | Usar rangos por región, o escalarlos con la dispersión de cada provincia. Como mínimo, publicar la cobertura por provincia. | Pipeline |
| H5 | Rangos 80 % | Sumar los límites provincia por provincia hace que el rango nacional sea **mucho más ancho de lo necesario**. Ya está documentado como "conservador", pero no estaba cuantificado. | Cobertura del rango nacional mensual (suma de límites): **98,8 %**, con ancho relativo mediano de 0,52. Si la zona es grande, el "80 %" que ve el usuario es en realidad cerca de 99 %. | Media | Calibrar los rangos directamente sobre agregados (región o país), o combinar las provincias con suma en cuadratura en lugar de sumar los límites. En la API, rotular el rango como "≥ 80 %" en zonas multiprovincia. | Pipeline + API |
| H6 | Rangos (solo Terra) | Los rangos se calibran con Terra + Aqua y se aplican también a los meses solo Terra (2000-11 → 2002-07). | LOYO solo Terra 2012–2019 con los rangos publicados: cobertura **73,3 %** (66,2 % en el tramo M = 0) y error provincia-mes de 23,3 %. La documentación del equipo informa 67,8 % para 2020–2025. La confianza de esos meses ya es "baja". | Media | Publicar rangos propios para `tipo = terra`. Cambia el contrato de `rangos.parquet`, porque hace falta la columna `tipo`. | Pipeline (+ contrato) |
| H7 | Día operativo | El criterio "≥ 1 foco nacional" **no detecta coberturas parciales**: un día con unos pocos gránulos cuenta como observado completo. Hay que comparar S-NPP contra NOAA-20, que mide lo mismo, para verlo. | Razón S-NPP/NOAA-20: **1,024** (la documentación dice 0,99). Hay 31 días con S-NPP "operativo" y menos de 0,5 × NOAA-20 (con N20 ≥ 50), y suman **7.054 focos faltantes** (9 días en 2025). Ejemplos: 2025-07-04 a 07 (razones 0,09–0,18), 2026-08-04 (0,27) y 2026-11-02 (1 foco contra 98). Efecto en los factores al excluir esos días: nacional +0,01 %, provincia-mes con mediana de 1,25 % y máximo de 8,1 %. | **Alta** (conteos) / Baja (factores) | Definir "operativo" en términos relativos: n_ref ≥ x % de lo esperado según otro sensor del mismo día (NOAA-20 o N21 para S-NPP, y Aqua para Terra). Sino, marcar el día como "parcial" y estimarlo. En la API, tratar esos días como estimados. | Pipeline (definición) + API (aplicación) |
| H8 | Día operativo | El mismo criterio confunde **cero fuego real** con apagón. Además, la calibración descarta días con un sesgo estacional. | De 515 días con Terra "caído" (n = 0) desde 2003, en 175 Aqua vio 3 focos o menos (probable cero real) y en 131 vio 20 o más (apagón real). La regla "todos operativos" excluye 450 de 5.083 días (8,9 %), concentrados en abril–julio (mayo: 101 días), es decir, días de poco fuego. | Media | Usar una señal de disponibilidad independiente del fuego: metadatos de gránulos de LAADS/FIRMS, o la presencia de detecciones de baja confianza o de agua. Repetir la calibración con y sin esos días y documentar el efecto en los meses de invierno. | Pipeline |
| H9 | Ceros silenciosos | La API estima solo cuando la **referencia** está caída y **no mira si MODIS lo está**. Cuando cae MODIS, el estimado es 0 sin avisar, lo que contradice "`null` nunca es 0". | En 2000-11 → 2002-07 hay 72 días con Terra caído, y entonces estimado = 0. En 2002-07 → 2012-01 hay 201 días con Terra o Aqua caído (14 con ambos). Faltan unos **3.817 focos MODIS ≈ 12.700 VIIRS-equivalentes** (ajustado por la razón entre sensores del mes). En 2012–2026 hay 5 días con S-NPP caído y Terra o Aqua caído. Los más afectados: 2005-06 (14 días), 2003-12, 2009-12 y 2010-05 (8 cada uno). | Media | (a) En el pipeline, publicar un estimado con un solo sensor MODIS (factor Aqua o Terra) o marcar el día "sin datos". (b) En la API, contar esos días en `frac_estimada`, bajar la confianza y no sumar 0 como si fuera dato. | Pipeline + API |
| H10 | Factor: estabilidad | **Fortaleza:** el factor nacional no tiene tendencia y las provincias no derivan, salvo una. | V/M nacional 2012–2025: entre 3,02 y 3,83 (2022), con CV anual de 6,0 %. Theil-Sen: +0,005/año. Spearman ρ = 0,09 (p = 0,77). Por provincia, 2 de 21 con p < 0,05 y solo Catamarca (−2,7 %/año, p = 0,002) sobrevive a Bonferroni. Error nacional al predecir un año con otro: sin relación con la distancia (ρ = −0,03, p = 0,71). Entre provincia-mes sube de 17,0 % (d = 1) a ~19 % (d = 10–13). | — | Mantener. Documentar el aumento leve del error provincial con la distancia, porque los años 2001–2011 están a 1–24 años de la calibración. | — |
| H11 | Factor: heterogeneidad | La variabilidad interanual del factor es **~5 veces la de Poisson**. El factor provincia × mes describe bien el promedio pero no un año puntual. | CV interanual mediano por provincia-mes (M ≥ 100 y ≥ 5 años): 0,164, contra 0,035 esperado por Poisson (razón 4,9). GLM provincia × año: χ²/gl = 118 (nacional anual: 720). Factores agregados por provincia: de 2,85 (Santiago del Estero) a 5,23 (Neuquén). Por mes: de 3,02 (agosto) a 3,89 (febrero). | Media | La sobredispersión justifica factores por provincia y mes, y rangos anchos. No justifica estrecharlos. Citar esta cifra en la defensa. | Pipeline (documentación) |
| H12 | Shrinkage K = 200 | K = 200 **no hace daño, pero no está derivado**. El error es plano entre K = 0 y K = 1.600. Lo que sí importa es tener el nivel provincial. | Error provincia-mes LOYO: 14,95 % (K≈0), 14,84 % (K = 200), 14,81 % (400–800), 14,97 % (5.000) y **18,04 %** (K→∞, factor nacional por mes). La documentación del equipo dice "cambió poco el error". | Baja | Justificar K con el mínimo de LOYO (400–800 da 0,03 pp menos, irrelevante) o con un estimador empírico-bayesiano. Aclarar que `nivel_origen` usa M ≥ K como umbral y que por eso la etiqueta depende de K. | Pipeline |
| H13 | Respaldo regional (LOPO) | Cuando se usa el factor de la región en lugar del de la provincia, el sesgo puede ser grande. | LOPO, con la provincia excluida y el factor regional de las demás: mediana \|sesgo\| 7,7 %. Peores casos: Neuquén −29,7 %, Río Negro +19,6 %, La Pampa +17,0 %, Salta −15,2 %, Santiago del Estero +14,9 %. Error mediano LOPO 20,5 % contra 18,7 % LOYO. En Santiago del Estero, 20,5 % contra 10,5 %. | Media | En la API, cuando `nivel_origen ∈ {region, pais}`, bajar la confianza un nivel. Hoy no la afecta. Publicar la tabla LOPO como validación espacial. | API + Pipeline |
| H14 | Confianza "alta" (frac ≤ 0,1) | **Defendible.** Simulé, en meses observados, que una fracción aleatoria de días se estimaba. | Nacional (V ≥ 200): con frac 0,05 / 0,10 / 0,20 / 1,0, el error mediano es 0,7 / **1,3** / 2,1 / 5,8 % y el p90 es 2,8 / **4,1** / 5,7 / 13,9 %. Provincia-mes (V ≥ 50), con frac 0,1: mediana 3,1 % y p90 11,7 %. Los factores son dentro de muestra, así que estas cifras son optimistas. | Baja | Mantener. Documentar el error esperado. Como `frac` se mide en días y no en focos, conviene ponderar por actividad si se quiere cubrir picos. | API (documentación) |
| H15 | Confianza "media" (≥ 21) | El umbral 21 **coincide con un borde de tramo, pero no con un quiebre del error**. Además, la API lo aplica al total de la zona, mientras que el error depende de cada provincia. | Error relativo mediano LOYO por M: 28,2 % (16–20), 23,3 % (21–25), 22,2 % (26–30) y 20,6 % (31–50). La curva es suave. M ≥ 21: mediana 14,8 % y p90 48 %. M 6–20: 32,1 %. Nacional 100 % estimado: mediana 6,6 % y p90 15,7 %. | Baja–Media | Atar la confianza al **ancho relativo del rango** (por ejemplo, media si (max−min)/armonizado < 0,6), que ya se calcula. Así el criterio queda sustentado y vale para zonas multiprovincia. Considerar también `nivel_origen` (H13). | API |
| H16 | Terra → Terra + Aqua | **Fortaleza:** el cambio de fuente en 2002-07 no deja escalón. | En 2002-07 → 2012-01, el estimado solo con Terra dividido por el estimado con Terra + Aqua da **0,981** en total (0,91–1,04 por año). Aqua/Terra estable 2003–2023 (1,59–1,96, Kendall p = 0,40). | — | Mantener. Citarlo en la defensa. | — |
| H17 | Deriva orbital Terra/Aqua | La deriva es **medible** desde 2022 y el factor `modis` incluye esos años. El combinado no se mueve, pero cada sensor sí. | Hora local mediana de los focos diurnos: Terra 11,3 h (2000–2021) → 10,92 (2023) → 10,27 (2025) → 9,67 (2026). Aqua 15,0 h → 15,55 (2024) → 16,13 (2025) → 16,82 (2026). Aqua/Terra: 2,23 (2024) y 2,46 (2025). V/T 2025 = 11,44 (mediana 2012–2019: ~9,2). El factor `terra` (2012–19) aplicado a 2022 y 2025 subestima −23,8 % y −24,2 %. El factor `modis` combinado no tiene tendencia (V/T y V/A: Kendall p ≈ 0,28). | Media | Cerrar la ventana de calibración de `modis` en 2021, como ya se hace con `terra` en 2019, o comprobar que los factores no cambian. Documentar la deriva. Revisar el NRT de 2026, porque la deriva sigue. | Pipeline |
| H18 | Hora de paso y ciclo diurno | Los factores por mes absorben parte del efecto del ciclo diurno, pero no todo. | Factor nacional por mes: de 3,02 (agosto) a 3,89 (febrero). V/M anual contra la fracción de focos MODIS con FRP < 10 MW: Spearman ρ = −0,56 (p = 0,037, n = 14, sin corregir por multiplicidad). Es evidencia débil de que el tamaño de los fuegos modula el factor. | Baja | Documentar como confusor residual. Si hay tiempo, probar un factor estratificado por día/noche o por FRP. | Pipeline |
| H19 | Uso del suelo / composición | El cambio en la mezcla de provincias **no** explica el salto de era. | Factor nacional implícito con factores de 2012–25: 3,26 con la composición MODIS de 2003–2011 y 3,33 con la de 2012–2025 (−2 %). No medí cambios de cobertura del suelo dentro de cada provincia (no hay datos de uso del suelo en el paquete). | Baja (lo medido) | Para la defensa: estratificar por cobertura (por ejemplo, MapBiomas Argentina) y probar la estabilidad del factor por clase. | Pipeline |
| H20 | Documentación | El fundamento científico citado no está y hay valores del contrato desactualizados. `docs/02` y `docs/04` citan `propuesta_B_calendario_focos_armonizado_ES.md`, que no está en el disco ni en ninguna rama de los dos repos de GitHub (revisado el 2026-11-15). En `main` de la API (18d6998) siguen: `docs/02` con mult_alto del tramo 0 = 10 (dato real: 9,0) y cobertura 77,9 % (validacion: 83,1 %, en muestra); `docs/04` con "por defecto 10 focos" y S-NPP/N20 = 0,99 (medido: 1,024). El límite del factor `terra` (< 40) ya está corregido en `main`. El criterio de K y los umbrales 0,1 y 21 no tienen justificación escrita. | Baja | Restaurar o enlazar la propuesta B. Actualizar `docs/02` y `docs/04` con los valores reales y con estas validaciones. | Pipeline + API (documentación) |

---

## 3. Respuestas por pregunta

1. **Validez del factor.** Es estable en el tiempo (H10) y no cambia con la composición (H19) ni con el cambio de fuente (H16). Entre provincias varía de forma real, de 2,85 a 5,23, lo que justifica el nivel provincial (18,0 % → 14,8 % de error). K = 200 es inocuo pero arbitrario (H12). El factor representa bien el promedio, no un año puntual (H11).
2. **Rangos del 80 %.** Por tramo dan 80 % por construcción. Fuera de muestra real cubren 77 % (64 % en meses grandes), varían por provincia de 66 % a 89 % y, sumados para el país, cubren 99 % (H3–H6).
3. **Día operativo.** El sesgo en los factores es despreciable (+0,01 % nacional). En los conteos no: hay días parciales tomados como observados (7.054 focos), ceros silenciosos cuando falta MODIS (≈ 12.700 VIIRS-equivalentes antes de 2012) y una calibración que excluye días de poco fuego (H7–H9).
4. **Confianza.** El umbral 0,1 se sostiene (error p90 de 4 % nacional). El 21 es una convención: no está mal, pero no está derivado. Falta incorporar `nivel_origen` (H13–H15).
5. **Anomalía.** El mínimo de 10 años se cumple con holgura (26 años por mes). El problema es la mezcla de eras (H1). La exclusión del mes parcial ya está resuelta en la API (H2 descartado).
6. **Confusores.** La deriva orbital es real y medible desde 2022 (H17). El cambio Terra → Terra + Aqua es consistente (H16). El ciclo diurno y el tamaño del fuego son residuales (H18). El uso del suelo no se midió (H19).
7. **Validación que falta:** ver §5.

---

## 4. Qué le toca al pipeline y qué a la API

| Pipeline (`ML_NasaSpaceApp`) | API (`FireHistoryInterfaceAPI`) |
|---|---|
| H3: cobertura hacia adelante como métrica y recalibrar los rangos (tramo > 500) | H1: referencia de anomalía solo con años observados, o informar las dos |
| H4: rangos por región o provincia | ~~H2~~ (ya resuelto en `main`) |
| H6: rangos para `tipo = terra` (cambia el contrato) | H5: rotular el rango en zonas multiprovincia |
| H7 y H8: definición relativa de "operativo" o bandera de día parcial | H7: tratar los días parciales como estimados (cuando el pipeline los marque) |
| H9a: estimado con un solo MODIS o bandera "sin datos" | H9b: no sumar 0 cuando MODIS está caído; reflejarlo en `frac_estimada` y en la confianza |
| H12: justificar K | H13 y H15: confianza según el ancho del rango y `nivel_origen` |
| H17: ventana de calibración `modis` hasta 2021 y monitorear 2026 | H14: documentar el error esperado de "alta" |
| H11, H18, H19 y H20: documentación y validaciones extra | H20: actualizar `docs/02` y `docs/04` |

---

## 5. Validación que agregaría para la defensa (en orden de impacto)

1. **Cobertura hacia adelante por tramo y por provincia** (ya medida acá: 77,1 % y 64 % en > 500). Es la cifra honesta para publicar en lugar de 83,1 %.
2. **LOPO como validación espacial** (ya medida: sesgo mediano 7,7 %, peor −29,7 %). Muestra cuánto se pierde con el respaldo regional.
3. **S-NPP contra NOAA-20 como control de calidad diario** (2018+). Es un validador independiente que ya está en los datos y detecta coberturas parciales.
4. **Prueba de distancia temporal** (ya medida: sin efecto nacional, +2 pp por provincia en 12 años). Es el mejor argumento disponible para extrapolar a 2001–2011.
5. **Pseudo-retroproyección por era.** Calibrar en 2020–2025 y probar en 2012–2014, y lo inverso, separando años con y sin deriva orbital.
6. **Validación externa del descenso 2003→2012** (H1): comparar contra otro producto independiente de MODIS (por ejemplo, área quemada MCD64A1 o registros provinciales de incendios). Sin eso, el escalón no se puede atribuir a un cambio real de los incendios.
7. **Estratificación por cobertura del suelo y por día/noche** para el factor (H18–H19).
8. **Calibración de los rangos a nivel agregado** (región o país), para que el "80 %" de una zona grande sea un 80 % real (H5).

---

## 6. Limitaciones de este análisis

- **Datos y código verificados (2026-11-15).**
  - `serving_api/20261114T174945/` de este repo es **idéntico byte a byte** a `data/current/` de `main` de FireHistoryInterfaceAPI (18d6998), comparado con `diff -rq`.
  - `classification/` en `main` de este repo (f9ffbc2) es igual a `dcf5b18`, el commit que figura en `metadata.json`.
  - La lógica de la API se leyó primero en la rama `codex/feat/f7-recarga` (99006c5). Contra `main` solo cambian H2 (ya resuelto) y la regla de confianza: ahora "pocos datos" no baja un mes observado. Ningún otro hallazgo cambia.
  - `m6_anom.py` todavía calcula el efecto del mes parcial con la lógica anterior: sirve para documentar H2, no para describir `main`.
  - La revisión geoespacial previa (`revision-geomaster-backend.md`) y la propuesta B no están en ninguno de los dos repos.
- **No levanté la API.** Repliqué su lógica (confianza, `frac_estimada`, anomalía) a escala nacional en scripts propios. Puede haber diferencias en zonas parciales o en polígonos.
- **Heurísticas propias.** Para los días parciales usé umbrales elegidos por mí: < 0,5 × NOAA-20 con N20 ≥ 50, y < 20 % de lo esperado. Las cifras de H7 y H8 dependen de esos umbrales y son una cota de orden de magnitud, no un recuento exacto.
- **Simulación de enmascarado (H14).** Usa factores dentro de muestra, así que el error real es algo mayor. Además, los días se enmascaran al azar y no en bloques, como ocurren los apagones reales (por ejemplo, 15 días en 2022-07/08).
- **Dependencia entre unidades.** LOYO trata cada año como independiente. Hay autocorrelación entre provincias vecinas y entre meses de un mismo año. Los p-valores provinciales (H4, H10) son orientativos. Apliqué Bonferroni solo donde lo indico.
- **Muestras chicas.** Las tendencias anuales usan 14 puntos (2012–2025). H18 (ρ = −0,56, p = 0,037) no sobrevive a una corrección por las varias pruebas que hice.
- **No pude separar causas en el salto de era (H1)**, entre un cambio real de los incendios y una pérdida de sensibilidad de MODIS antes de 2012. Lo único que puedo afirmar es que la relación V/M no cambia en 2012–2025.
- **Uso del suelo** no medido directamente (no hay capa de cobertura en el paquete). Usé como sustitutos la composición provincial y el FRP.
- No revisé la geometría ni el horario de verano: ya están en la revisión geoespacial previa (1.467 focos de Terra) y no los repetí.

---

## 7. Reproducibilidad

Todo está en `analisis/veredicto-armonizacion/` de este repo. El paso a paso para replicar y el mapa de cada cifra con su archivo de resultados están en el [README](README.md). En resumen, desde la raíz del repo:

```bash
uv venv --python 3.12 .venv-analisis
uv pip install --python .venv-analisis/bin/python -r analisis/veredicto-armonizacion/requirements.txt
bash analisis/veredicto-armonizacion/run_all.sh .venv-analisis/bin/python
```

`run_all.sh` corre los scripts en orden y guarda cada salida en `resultados/<script>.txt`. Tarda ~30 s en una Mac M1. Los archivos intermedios (`mm.parquet`, `mt.parquet`, `loyo200.parquet`, `nat.parquet`) quedan en `_trabajo/`, que no se versiona.

| Script | Mide |
|---|---|
| `m0_inspect.py` | Esquemas, factores, rangos y días caídos por satélite |
| `m1_repro.py` | Reproducción exacta de factores, rangos y cobertura |
| `m2_factor.py` | Estabilidad temporal y espacial del factor y sobredispersión (H10, H11) |
| `m3_cv.py` | K por LOYO, cobertura por tramo, provincia, año y hacia adelante, y LOPO (H3–H5, H12, H13) |
| `m4_oper.py`, `m4b_gaps.py` | Día operativo, gaps parciales contra NOAA-20 y ceros silenciosos (H7–H9) |
| `m5_conf.py` | Simulación de `frac_estimada` y error por M (H14, H15) |
| `m6_anom.py`, `m7_conf.py` | Serie nacional, anomalía, eras, Aqua/Terra y cambio de fuente (H1, H2, H16, H17) |
| `m8_orbit.py` | Hora de paso, FRP, composición y distancia temporal (H10, H17–H19) |
| `m9_extra.py` | Rangos solo Terra y error por clase de confianza (H6, H15) |
