"""Utilidades compartidas por los scripts del veredicto de armonización.

Usa el código de calibración del propio repo (``classification/``) y los datos publicados en
``serving_api/<versión>/``. Otra versión de datos: ``FH_DATA=/ruta/al/paquete``.
"""
import os, sys
from pathlib import Path
import duckdb, numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
D = os.environ.get("FH_DATA", str(REPO / "serving_api" / "20261114T174945"))
from classification import calibrate as CAL, config as C  # noqa: E402

con = duckdb.connect()

def _dates(s): return pd.to_datetime(s).dt.date

def load():
    """Conteos provincia × día por satélite y la tabla ``satelite_operativo``."""
    pd_ = con.sql(f"""SELECT fecha, provincia_id, sum(n_terra)::BIGINT n_terra, sum(n_aqua)::BIGINT n_aqua,
        sum(n_snpp)::BIGINT n_snpp, sum(n_n20)::BIGINT n_n20, sum(n_n21)::BIGINT n_n21
        FROM read_parquet('{D}/conteos_celda_dia/*/*.parquet', hive_partitioning=1) GROUP BY 1,2""").df()
    pd_["fecha"] = _dates(pd_["fecha"])
    op = con.sql(f"SELECT * FROM '{D}/satelite_operativo.parquet'").df()
    op["fecha"] = _dates(op["fecha"])
    return pd_, op

def pub_factors():
    return con.sql(f"SELECT * FROM '{D}/factores_provincia_mes.parquet'").df()

def rangos():
    return con.sql(f"SELECT * FROM '{D}/rangos.parquet'").df()

def cover(m, v, pred, rg):
    """Aplica los rangos publicados: devuelve (dentro, mínimo, máximo)."""
    lo = np.full(len(m), np.nan); hi = np.full(len(m), np.nan)
    for _, b in rg.iterrows():
        s = (m >= b.modis_min) & (m <= b.modis_max)
        if b.modis_min == 0: lo[s], hi[s] = b.mult_bajo, b.mult_alto
        else: lo[s], hi[s] = pred[s]*b.mult_bajo, pred[s]*b.mult_alto
    return (v >= lo) & (v <= hi), lo, hi
