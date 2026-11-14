"""De detecciones limpias (Fase 1) a las tablas ``focos``, ``conteos_celda_dia`` y ``satelite_operativo``."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import config as C
from .static_sources import point_bin

SAT_ALIASES = {
    "terra": "Terra", "t": "Terra",
    "aqua": "Aqua", "a": "Aqua",
    "snpp": "SNPP", "n": "SNPP", "suomi-npp": "SNPP", "npp": "SNPP",
    "n20": "N20", "noaa-20": "N20", "noaa20": "N20", "j1": "N20", "1": "N20",
    "n21": "N21", "noaa-21": "N21", "noaa21": "N21", "j2": "N21",
}


def normalize_satellite(s: pd.Series) -> pd.Series:
    """Lleva los códigos de satélite de FIRMS a ``Terra/Aqua/SNPP/N20/N21``."""
    return s.astype(str).str.strip().str.lower().map(SAT_ALIASES)


def prepare_detections(clean: pd.DataFrame) -> pd.DataFrame:
    """Renombra el esquema de la Fase 1 al vocabulario del contrato y descarta lo no reconocido."""
    d = pd.DataFrame(
        {
            "fecha": pd.to_datetime(clean["fecha_local"]).dt.date,
            "hora_utc": clean["acq_time"].astype(str).str.zfill(4).str[-4:],
            "satelite": normalize_satellite(clean["satellite"]),
            "lat": clean["latitude"].astype("float64"),
            "lon": clean["longitude"].astype("float64"),
            "confianza": clean["confidence"].astype(str).str.strip().str.lower(),
            "frp_mw": clean["frp"].astype("float32"),
            "dia_noche": clean["daynight"].astype(str).str.upper().str[:1],
            "fuente": clean.get("source_quality", pd.Series("SP", index=clean.index)).astype(str).str.upper(),
            "cell_id": clean["cell_id"].astype("int64"),
        }
    )
    d = d[d["satelite"].notna()].copy()
    d["sensor"] = np.where(d["satelite"].isin(C.MODIS_SATS), "MODIS", "VIIRS")
    return d.reset_index(drop=True)


LOCAL_TZ = timezone(timedelta(hours=-3))


def trim_to_missions(det: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Descarta detecciones con ``fecha`` local fuera de la misión de su satélite.

    Al pasar de UTC a hora local (UTC−3), los focos de las primeras horas UTC del día de
    inicio caen el día anterior (p. ej. Terra 2000-11-01 → 2000-10-31), un día que no tiene
    fila en ``satelite_operativo``. Devuelve (detecciones, focos descartados por satélite).
    """
    start = det["satelite"].map(C.MISSION_START)
    end = det["satelite"].map(C.MISSION_END)
    fecha = pd.to_datetime(det["fecha"])
    out = (fecha < pd.to_datetime(start)) | (end.notna() & (fecha > pd.to_datetime(end)))
    dropped = det.loc[out, "satelite"].value_counts()
    counts = {sat: int(dropped.get(sat, 0)) for sat in C.ALL_SATS}
    return det.loc[~out].reset_index(drop=True), counts


def default_fecha_max(data_max: date, today: date | None = None) -> date:
    """Último día completo: ``min(fecha máxima de los datos, ayer en hora local UTC−3)``."""
    today = today or datetime.now(LOCAL_TZ).date()
    return min(data_max, today - timedelta(days=1))


def resolve_fecha_max(data_max: date, pedida: date | None = None, today: date | None = None) -> date:
    """``fecha_max`` a publicar: la pedida (``--fecha-max``), pero nunca después del último día completo.

    Así un ``--fecha-max`` futuro o posterior a los datos no genera días vacíos en
    ``satelite_operativo`` ni una ``fecha_max`` futura en ``metadata.json``.
    """
    limite = default_fecha_max(data_max, today)
    return limite if pedida is None else min(pedida, limite)


def exclude_static(det: pd.DataFrame, puntos: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Quita los focos ubicados en puntos industriales. Devuelve (focos, cantidad excluida)."""
    if puntos.empty or det.empty:
        return det, 0
    bins = point_bin(det["lat"].to_numpy(), det["lon"].to_numpy())
    drop = np.isin(bins, puntos["punto_id"].to_numpy())
    return det.loc[~drop].reset_index(drop=True), int(drop.sum())


def build_focos(det: pd.DataFrame, cell_province: pd.Series) -> pd.DataFrame:
    """Tabla ``focos`` con ``foco_id`` estable (yyyymmdd · 10^6 + secuencia del día)."""
    if det.empty:
        return _empty_focos()
    d = det.sort_values(["fecha", "hora_utc", "satelite", "lat", "lon"], kind="mergesort").reset_index(drop=True)
    ymd = pd.to_datetime(d["fecha"]).dt.strftime("%Y%m%d").astype("int64")
    seq = d.groupby("fecha").cumcount().astype("int64")
    if (seq >= 1_000_000).any():
        raise ValueError("Más de 10^6 focos en un día: el esquema de foco_id no alcanza")
    out = pd.DataFrame(
        {
            "foco_id": ymd * 1_000_000 + seq,
            "fecha": d["fecha"],
            "hora_utc": d["hora_utc"],
            "satelite": d["satelite"],
            "sensor": d["sensor"],
            "lat": d["lat"].astype("float32"),
            "lon": d["lon"].astype("float32"),
            "confianza": d["confianza"],
            "frp_mw": d["frp_mw"].astype("float32"),
            "dia_noche": d["dia_noche"],
            "fuente": d["fuente"],
            "cell_id": d["cell_id"].astype("int32"),
            "provincia_id": d["cell_id"].map(cell_province),
            "anio": pd.to_datetime(d["fecha"]).dt.year.astype("int16"),
        }
    )
    return out


def build_conteos(det: pd.DataFrame, cell_province: pd.Series) -> pd.DataFrame:
    """Matriz dispersa celda × día con un conteo por satélite (solo celdas de la máscara)."""
    cols = list(C.SAT_COLUMN.values())
    if det.empty:
        return pd.DataFrame(columns=["cell_id", "fecha", "provincia_id", *cols, "anio"])
    g = det.groupby(["cell_id", "fecha", "satelite"]).size().unstack("satelite", fill_value=0)
    g = g.reindex(columns=list(C.SAT_COLUMN), fill_value=0).rename(columns=C.SAT_COLUMN)
    if (g.to_numpy() > np.iinfo(np.int16).max).any():
        raise ValueError("Conteo celda-día fuera de rango INT16")
    g = g.astype("int16").reset_index()
    g["provincia_id"] = g["cell_id"].map(cell_province)
    g = g[g["provincia_id"].notna()].copy()
    g["cell_id"] = g["cell_id"].astype("int32")
    g["anio"] = pd.to_datetime(g["fecha"]).dt.year.astype("int16")
    return g[["cell_id", "fecha", "provincia_id", *cols, "anio"]].sort_values(["fecha", "cell_id"]).reset_index(drop=True)


def build_satelite_operativo(det: pd.DataFrame, fecha_max: date) -> pd.DataFrame:
    """Un registro por día × satélite desde el inicio de cada misión hasta ``fecha_max``.

    ``operativo = False`` si el satélite no tuvo ningún foco en el país ese día o si la fecha
    está fuera de la misión. Así un apagón no se confunde con "cero fuego".
    """
    per_day = det.groupby(["fecha", "satelite"]).size() if not det.empty else pd.Series(dtype="int64")
    frames = []
    for sat in C.ALL_SATS:
        start = C.MISSION_START[sat]
        if start > fecha_max:
            continue
        days = pd.date_range(start, fecha_max, freq="D").date
        n = np.zeros(len(days), dtype="int32")
        if not per_day.empty and sat in per_day.index.get_level_values("satelite"):
            s = per_day.xs(sat, level="satelite")
            n = pd.Series(days).map(s).fillna(0).astype("int32").to_numpy()
        end = C.MISSION_END.get(sat)
        in_mission = np.array([end is None or d <= end for d in days])
        frames.append(
            pd.DataFrame({"fecha": days, "satelite": sat, "operativo": (n > 0) & in_mission, "n_nacional": n})
        )
    return pd.concat(frames, ignore_index=True)


def _empty_focos() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "foco_id", "fecha", "hora_utc", "satelite", "sensor", "lat", "lon", "confianza",
            "frp_mw", "dia_noche", "fuente", "cell_id", "provincia_id", "anio",
        ]
    )
