"""Calibración de la Propuesta B: factores jerárquicos provincia × mes, rangos del 80 % y métricas.

Fórmula (encogimiento hacia el nivel superior, ``K`` focos MODIS de "colchón")::

    f_pais          = ΣV / ΣM
    f_pais,m        = (V_m   + K · f_pais)        / (M_m   + K)
    f_region,m      = (V_r,m + K · f_pais,m)      / (M_r,m + K)
    f_provincia,m   = (V_p,m + K · f_region,m)    / (M_p,m + K)

``V`` = focos VIIRS de referencia (S-NPP) y ``M`` = focos MODIS (Terra + Aqua, o solo Terra),
sumados solo en días en que todos los satélites involucrados estaban operativos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

TIPO_INPUT = {"modis": ("n_terra", "n_aqua"), "terra": ("n_terra",)}
TIPO_SATS = {"modis": ("SNPP", "Terra", "Aqua"), "terra": ("SNPP", "Terra")}


def valid_days(oper: pd.DataFrame, sats: tuple[str, ...]) -> set:
    """Días en que todos los satélites indicados estaban operativos."""
    o = oper[oper["satelite"].isin(sats)]
    ok = o.groupby("fecha")["operativo"].agg(["all", "size"])
    return set(ok.index[(ok["all"]) & (ok["size"] == len(sats))])


def monthly_counts(
    conteos: pd.DataFrame,
    oper: pd.DataFrame,
    tipo: str,
    years: tuple[int, int],
    provinces: list[str] | None = None,
) -> pd.DataFrame:
    """Tabla provincia × año × mes con ``V`` (S-NPP) y ``M`` (MODIS) en días válidos.

    Completa con ceros las combinaciones sin focos para que los meses vacíos cuenten.
    """
    days = valid_days(oper, TIPO_SATS[tipo])
    c = conteos[conteos["fecha"].isin(days)].copy()
    f = pd.to_datetime(c["fecha"])
    c["anio_c"], c["mes"] = f.dt.year, f.dt.month
    c = c[(c["anio_c"] >= years[0]) & (c["anio_c"] <= years[1])]
    c = c[pd.to_datetime(c["fecha"]) >= pd.Timestamp(C.CALIB_START)]
    c["V"] = c["n_snpp"].astype("int64")
    c["M"] = c[list(TIPO_INPUT[tipo])].astype("int64").sum(axis=1)
    g = c.groupby(["provincia_id", "anio_c", "mes"])[["V", "M"]].sum()

    provs = provinces or sorted(C.PROVINCE_REGION)
    valid_months = sorted({(d.year, d.month) for d in days if years[0] <= d.year <= years[1] and d >= C.CALIB_START})
    idx = pd.MultiIndex.from_tuples(
        [(p, y, m) for p in provs for (y, m) in valid_months], names=["provincia_id", "anio_c", "mes"]
    )
    g = g.reindex(idx, fill_value=0).reset_index().rename(columns={"anio_c": "anio"})
    g["region"] = g["provincia_id"].map(C.PROVINCE_REGION)
    return g


@dataclass
class Fit:
    nat: float
    nat_m: pd.Series                 # índice: mes
    region_m: pd.Series              # índice: (region, mes)
    prov_m: pd.DataFrame             # columnas: provincia_id, mes, factor, nivel_origen, n_modis_calib


def fit_hierarchical(monthly: pd.DataFrame, k: float = C.SHRINK_K) -> Fit:
    """Ajusta los factores jerárquicos sobre la tabla de ``monthly_counts``."""
    tv, tm = monthly["V"].sum(), monthly["M"].sum()
    if tm <= 0:
        raise ValueError("Sin focos MODIS en la ventana de calibración")
    nat = tv / tm

    months = pd.Index(range(1, 13), name="mes")
    by_m = monthly.groupby("mes")[["V", "M"]].sum().reindex(months, fill_value=0)
    nat_m = (by_m["V"] + k * nat) / (by_m["M"] + k)

    regions = sorted(set(C.PROVINCE_REGION.values()))
    ridx = pd.MultiIndex.from_product([regions, months], names=["region", "mes"])
    by_rm = monthly.groupby(["region", "mes"])[["V", "M"]].sum().reindex(ridx, fill_value=0)
    prior_r = nat_m.reindex(ridx.get_level_values("mes")).to_numpy()
    region_m = pd.Series((by_rm["V"].to_numpy() + k * prior_r) / (by_rm["M"].to_numpy() + k), index=ridx)

    provs = sorted(C.PROVINCE_REGION)
    pidx = pd.MultiIndex.from_product([provs, months], names=["provincia_id", "mes"])
    by_pm = monthly.groupby(["provincia_id", "mes"])[["V", "M"]].sum().reindex(pidx, fill_value=0)
    p_region = [C.PROVINCE_REGION[p] for p in pidx.get_level_values("provincia_id")]
    prior_p = region_m.reindex(pd.MultiIndex.from_arrays([p_region, pidx.get_level_values("mes")])).to_numpy()
    m_pm = by_pm["M"].to_numpy()
    m_rm = by_rm.reindex(pd.MultiIndex.from_arrays([p_region, pidx.get_level_values("mes")]))["M"].to_numpy()
    factor = (by_pm["V"].to_numpy() + k * prior_p) / (m_pm + k)
    nivel = np.where(m_pm >= k, "provincia", np.where(m_rm >= k, "region", "pais"))

    low = np.isin(pidx.get_level_values("provincia_id"), C.LOW_DATA_PROVINCES)
    factor = np.where(low, prior_p, factor)
    nivel = np.where(low, np.where(m_rm >= k, "region", "pais"), nivel)

    prov_m = pd.DataFrame(
        {
            "provincia_id": pidx.get_level_values("provincia_id"),
            "mes": pidx.get_level_values("mes").astype("int8"),
            "factor": factor.astype("float32"),
            "nivel_origen": nivel,
            "n_modis_calib": m_pm.astype("int32"),
        }
    )
    return Fit(nat=float(nat), nat_m=nat_m, region_m=region_m, prov_m=prov_m)


def factors_table(fits: dict[str, Fit], version: str) -> pd.DataFrame:
    """Une los ajustes ``modis`` y ``terra`` en la tabla ``factores_provincia_mes``."""
    frames = []
    for tipo, fit in fits.items():
        t = fit.prov_m.copy()
        t.insert(2, "tipo", tipo)
        t["factores_version"] = version
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def predict(monthly: pd.DataFrame, fit: Fit) -> np.ndarray:
    """Focos VIIRS estimados = factor(provincia, mes) × focos MODIS."""
    f = fit.prov_m.set_index(["provincia_id", "mes"])["factor"]
    key = pd.MultiIndex.from_arrays([monthly["provincia_id"], monthly["mes"].astype("int8")])
    return f.reindex(key).to_numpy(dtype="float64") * monthly["M"].to_numpy(dtype="float64")


def score(monthly: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    """Error mensual por provincia (ponderado) y sesgo total, en %."""
    v = monthly["V"].to_numpy(dtype="float64")
    tot = v.sum()
    if tot == 0:
        return {"error_pct": float("nan"), "sesgo_pct": float("nan"), "n_meses": int(len(v))}
    return {
        "error_pct": float(100 * np.abs(pred - v).sum() / tot),
        "sesgo_pct": float(100 * (pred.sum() - tot) / tot),
        "n_meses": int(len(v)),
    }


def evaluate(conteos: pd.DataFrame, oper: pd.DataFrame, k: float = C.SHRINK_K) -> dict[str, dict]:
    """Pruebas hacia atrás y hacia adelante + línea base sin armonizar (MODIS crudo × factor nacional 1)."""
    out: dict[str, dict] = {}
    for name, (cal, test) in {"hacia_atras": C.BACKTEST_SPLIT, "hacia_adelante": C.FORWARD_SPLIT}.items():
        tr = monthly_counts(conteos, oper, "modis", cal)
        te = monthly_counts(conteos, oper, "modis", test)
        if tr["M"].sum() == 0 or te["V"].sum() == 0:
            out[name] = {"error_pct": float("nan"), "sesgo_pct": float("nan"), "n_meses": 0}
            continue
        fit = fit_hierarchical(tr, k)
        res = score(te, predict(te, fit))
        res["error_modis_crudo_pct"] = score(te, te["M"].to_numpy(dtype="float64"))["error_pct"]
        res["calibracion"], res["prueba"] = list(cal), list(test)
        out[name] = res
    return out


def calibrate_ranges(monthly: pd.DataFrame, k: float = C.SHRINK_K) -> pd.DataFrame:
    """Rangos del 80 % por tramo de focos MODIS del mes, con validación dejando un año afuera.

    Para cada año se ajusta con los demás, se predice ese año y se mide ``V / predicho``.
    En el tramo ``M = 0`` los límites son valores absolutos de focos VIIRS.
    """
    lo_q, hi_q = C.RANGE_QUANTILES
    ratios: list[pd.DataFrame] = []
    for y in sorted(monthly["anio"].unique()):
        tr, te = monthly[monthly["anio"] != y], monthly[monthly["anio"] == y]
        if tr["M"].sum() == 0 or te.empty:
            continue
        p = predict(te, fit_hierarchical(tr, k))
        ratios.append(pd.DataFrame({"M": te["M"].to_numpy(), "V": te["V"].to_numpy(), "pred": p}))
    if not ratios:
        raise ValueError("Datos insuficientes para calibrar los rangos")
    r = pd.concat(ratios, ignore_index=True)
    rows = []
    for mn, mx in C.RANGE_BINS:
        sel = r[(r["M"] >= mn) & (r["M"] <= mx)]
        if mn == 0:
            vals = sel["V"].to_numpy(dtype="float64")
            lo, hi = (np.quantile(vals, [lo_q, hi_q]) if len(vals) else (0.0, 0.0))
        else:
            sel = sel[sel["pred"] > 0]
            q = (sel["V"] / sel["pred"]).to_numpy(dtype="float64")
            lo, hi = (np.quantile(q, [lo_q, hi_q]) if len(q) else (np.nan, np.nan))
        rows.append({"modis_min": mn, "modis_max": mx, "mult_bajo": float(lo), "mult_alto": float(hi), "n_meses": int(len(sel))})
    out = pd.DataFrame(rows)
    out[["modis_min", "modis_max"]] = out[["modis_min", "modis_max"]].astype("int32")
    out[["mult_bajo", "mult_alto"]] = out[["mult_bajo", "mult_alto"]].astype("float32")
    return out


def range_coverage(monthly: pd.DataFrame, fit: Fit, rangos: pd.DataFrame) -> float:
    """Porcentaje de meses-provincia cuyo valor real cae dentro del rango."""
    p = predict(monthly, fit)
    m, v = monthly["M"].to_numpy(), monthly["V"].to_numpy()
    lo = np.empty(len(m)); hi = np.empty(len(m))
    for _, b in rangos.iterrows():
        sel = (m >= b.modis_min) & (m <= b.modis_max)
        if b.modis_min == 0:
            lo[sel], hi[sel] = b.mult_bajo, b.mult_alto
        else:
            lo[sel], hi[sel] = p[sel] * b.mult_bajo, p[sel] * b.mult_alto
    return float(100 * np.mean((v >= lo) & (v <= hi))) if len(v) else float("nan")
