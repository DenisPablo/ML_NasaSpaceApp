"""Detección de fuentes de calor fijas industriales (antorchas, refinerías, ingenios).

Un punto de ~1 km es *persistente* si tiene focos en >= 30 días, >= 3 años y >= 6 meses
calendario distintos. Un punto persistente es *industrial* si cumple alguna regla:

1. Nocturno todo el año: >= 60 % de focos nocturnos y presente en >= 10 meses calendario.
2. Marcado por FIRMS: >= 30 % de sus detecciones científicas (SP) con ``type = 2``.
3. Nocturno estacional: >= 80 % nocturnos durante >= 5 años (p. ej. ingenios en zafra).

Las quemas recurrentes **diurnas** (agricultura, humedales) NO se excluyen: son fuego real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def point_bin(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Identificador de la celda de ~1 km (redondeo a 0,01°)."""
    la = np.round(np.asarray(lat, dtype=np.float64) * 100).astype(np.int64)
    lo = np.round(np.asarray(lon, dtype=np.float64) * 100).astype(np.int64)
    return la * 100_000 + lo


def type2_share_from_raw(raw: pd.DataFrame) -> pd.Series:
    """Proporción de detecciones SP con ``type = 2`` por punto (``raw``: lat, lon, type)."""
    sp = raw.dropna(subset=["type"])
    if sp.empty:
        return pd.Series(dtype="float64", name="t2")
    bins = point_bin(sp["lat"].to_numpy(), sp["lon"].to_numpy())
    is2 = pd.to_numeric(sp["type"], errors="coerce").to_numpy() == 2
    return pd.Series(is2, index=bins).groupby(level=0).mean().rename("t2")


def detect_static_sources(det: pd.DataFrame, type2_share: pd.Series | None = None) -> pd.DataFrame:
    """Devuelve la tabla ``puntos_industriales``.

    ``det`` necesita las columnas: ``lat``, ``lon``, ``fecha`` (date) y ``dia_noche`` (D/N).
    """
    if det.empty:
        return _empty()
    d = det[["lat", "lon", "fecha", "dia_noche"]].copy()
    d["punto_id"] = point_bin(d["lat"].to_numpy(), d["lon"].to_numpy())
    fecha = pd.to_datetime(d["fecha"])
    d["anio"] = fecha.dt.year
    d["mes"] = fecha.dt.month
    d["noche"] = (d["dia_noche"] == "N").astype("float64")
    g = d.groupby("punto_id").agg(
        dias=("fecha", "nunique"),
        anios=("anio", "nunique"),
        meses=("mes", "nunique"),
        n_focos=("lat", "size"),
        frac_nocturna=("noche", "mean"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
    )
    pers = g[(g.dias >= C.PERSIST_MIN_DAYS) & (g.anios >= C.PERSIST_MIN_YEARS) & (g.meses >= C.PERSIST_MIN_MONTHS)].copy()
    if pers.empty:
        return _empty()
    t2 = type2_share.reindex(pers.index).fillna(0.0) if type2_share is not None else pd.Series(0.0, index=pers.index)
    r1 = (pers.frac_nocturna >= C.RULE_CONTINUOUS_NIGHT) & (pers.meses >= C.RULE_CONTINUOUS_MONTHS)
    r2 = t2 >= C.RULE_TYPE2_SHARE
    r3 = (pers.frac_nocturna >= C.RULE_SEASONAL_NIGHT) & (pers.anios >= C.RULE_SEASONAL_YEARS)
    pers["motivo"] = np.select(
        [r1, r2, r3],
        ["nocturno todo el año", "marcado industrial por FIRMS", "nocturno estacional"],
        default="",
    )
    out = pers[r1 | r2 | r3].reset_index()
    out = out.astype({"punto_id": "int64", "dias": "int16", "anios": "int16", "meses": "int16", "n_focos": "int32"})
    out["frac_nocturna"] = out["frac_nocturna"].astype("float32")
    out["lat"] = out["lat"].astype("float32")
    out["lon"] = out["lon"].astype("float32")
    return out[["punto_id", "lat", "lon", "motivo", "dias", "anios", "meses", "frac_nocturna", "n_focos"]]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "punto_id": pd.Series(dtype="int64"),
            "lat": pd.Series(dtype="float32"),
            "lon": pd.Series(dtype="float32"),
            "motivo": pd.Series(dtype="object"),
            "dias": pd.Series(dtype="int16"),
            "anios": pd.Series(dtype="int16"),
            "meses": pd.Series(dtype="int16"),
            "frac_nocturna": pd.Series(dtype="float32"),
            "n_focos": pd.Series(dtype="int32"),
        }
    )
