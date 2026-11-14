"""Constantes de la etapa de clasificación (Propuesta B)."""

from __future__ import annotations

from datetime import date

SCHEMA_VERSION = "1.0.0"

# --- Satélites -----------------------------------------------------------------------
MODIS_SATS = ("Terra", "Aqua")
VIIRS_SATS = ("SNPP", "N20", "N21")
ALL_SATS = MODIS_SATS + VIIRS_SATS
SAT_COLUMN = {"Terra": "n_terra", "Aqua": "n_aqua", "SNPP": "n_snpp", "N20": "n_n20", "N21": "n_n21"}

# Ventanas de misión en los archivos FIRMS (para la bandera de satélite operativo)
MISSION_START = {
    "Terra": date(2000, 11, 1),
    "Aqua": date(2002, 7, 4),
    "SNPP": date(2012, 1, 20),
    "N20": date(2018, 4, 1),
    "N21": date(2023, 3, 1),
}
MISSION_END = {"SNPP": date(2026, 11, 2)}

# Referencia VIIRS "observado": un solo satélite por fecha (nunca se suma NOAA-21)
VIIRS_START = date(2012, 1, 20)
SNPP_END = date(2026, 11, 2)
AQUA_START = MISSION_START["Aqua"]

# --- Regiones (códigos INDEC de provincia) ------------------------------------------------
PROVINCE_REGION = {
    "02": "Pampeana", "06": "Pampeana", "14": "Pampeana", "30": "Pampeana", "42": "Pampeana", "82": "Pampeana",
    "10": "NOA", "38": "NOA", "66": "NOA", "86": "NOA", "90": "NOA",
    "18": "NEA", "22": "NEA", "34": "NEA", "54": "NEA",
    "46": "Cuyo", "50": "Cuyo", "70": "Cuyo", "74": "Cuyo",
    "26": "Patagonia", "58": "Patagonia", "62": "Patagonia", "78": "Patagonia", "94": "Patagonia",
}
LOW_DATA_PROVINCES = ("02", "78", "94")  # CABA, Santa Cruz, Tierra del Fuego

# --- Calibración --------------------------------------------------------------------------
SHRINK_K = 200                       # "colchón" (focos MODIS)
CALIB_START = date(2012, 2, 1)       # primer mes completo con S-NPP
CALIB_END = date(2025, 12, 31)       # último año cerrado con producto científico (SP)
BACKTEST_SPLIT = ((2015, 2022), (2012, 2014))   # (calibración, prueba) hacia atrás
FORWARD_SPLIT = ((2012, 2019), (2020, 2025))    # (calibración, prueba) hacia adelante
# Validación del contrato de la API. Terra sola ve muchos menos focos en invierno: factores
# de hasta ~26 en julio con datos reales 2012–2019, por eso su techo es más alto.
FACTOR_BOUNDS = {"modis": (0.5, 20.0), "terra": (0.5, 40.0)}
TERRA_CALIB_YEARS = (2012, 2019)     # antes de la deriva orbital de Terra
RANGE_BINS = ((0, 0), (1, 5), (6, 20), (21, 100), (101, 500), (501, 2_147_483_647))
RANGE_QUANTILES = (0.10, 0.90)

# --- Fuentes fijas industriales -----------------------------------------------------------
PERSIST_MIN_DAYS = 30
PERSIST_MIN_YEARS = 3
PERSIST_MIN_MONTHS = 6
RULE_CONTINUOUS_NIGHT = 0.60   # y presente en >= 10 meses
RULE_CONTINUOUS_MONTHS = 10
RULE_TYPE2_SHARE = 0.30
RULE_SEASONAL_NIGHT = 0.80     # y presente en >= 5 años
RULE_SEASONAL_YEARS = 5

# --- Grilla (igual que clean_firms.py) ----------------------------------------------------
GRID_X_MIN = 1_400_000.0
GRID_Y_MIN = -2_100_000.0
GRID_CELL_SIZE_M = 5_000.0
GRID_N_COLS = 460
GRID_N_ROWS = 780
