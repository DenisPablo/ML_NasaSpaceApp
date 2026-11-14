"""Etapa de clasificación y armonización de conteos (Propuesta B).

Lee el Parquet limpio de la Fase 1 (``clean_firms.py``) y produce los archivos que
consume FireHistoryInterfaceAPI (ver ``CLASIFICACION.md``):

- ``focos/``, ``conteos_celda_dia/``: detecciones y conteos diarios por celda y satélite
- ``dim_celdas``, ``dim_provincias``: máscara nacional y provincias
- ``factores_provincia_mes``, ``rangos``: armonización MODIS -> VIIRS (S-NPP / NOAA-20)
- ``satelite_operativo``, ``puntos_industriales``, ``metadata.json``

Los módulos de lógica (``geo``, ``static_sources``, ``counts``, ``calibrate``) usan solo
numpy/pandas para que se puedan probar sin datos reales. La lectura y escritura de
Parquet vive en ``publish``.
"""
