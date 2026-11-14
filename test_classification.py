"""Pruebas de la etapa de clasificación (Propuesta B) con datos sintéticos.

Solo usan numpy/pandas: la proyección se reemplaza por una transformación lineal falsa.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from classification import config as C
from classification.calibrate import (
    calibrate_ranges,
    fit_hierarchical,
    monthly_counts,
    predict,
    range_coverage,
    region_factors_table,
    score,
)
from classification.counts import (
    build_conteos,
    build_focos,
    LOCAL_TZ,
    build_satelite_operativo,
    default_fecha_max,
    exclude_static,
    normalize_satellite,
    resolve_fecha_max,
    trim_to_missions,
)
from classification.geo import Province, assign_province, build_cell_mask, points_in_ring, provinces_table
from classification.publish import build_metadata, conform, validate, validate_metadata
from classification.static_sources import detect_static_sources, point_bin

# ---------------------------------------------------------------- geometría


def fake_fwd(lon, lat):
    return 2_500_000 + (np.asarray(lon) + 60) * 100_000, (np.asarray(lat) + 37) * 100_000


def fake_inv(x, y):
    return (np.asarray(x) - 2_500_000) / 100_000 - 60, np.asarray(y) / 100_000 - 37


def square(w, s, e, n):
    return np.array([[w, s], [e, s], [e, n], [w, n], [w, s]], dtype=float)


def prov(pid, w, s, e, n, holes=()):
    return Province(pid, f"P{pid}", C.PROVINCE_REGION[pid], [[square(w, s, e, n), *holes]], (w, s, e, n))


def test_points_in_ring_and_holes():
    p = prov("74", 0, 0, 1, 1, holes=(square(0.4, 0.4, 0.6, 0.6),))
    lon = np.array([0.1, 0.5, 1.5, 0.9])
    lat = np.array([0.1, 0.5, 0.5, 0.9])
    assert points_in_ring(lon, lat, square(0, 0, 1, 1)).tolist() == [True, True, False, True]
    assert p.contains(lon, lat).tolist() == [True, False, False, True]


def test_assign_province_returns_none_outside():
    ps = [prov("74", -66, -34, -65, -33), prov("70", -66, -33, -65, -32)]
    out = assign_province(np.array([-65.5, -65.5, -60.0]), np.array([-33.5, -32.5, -33.0]), ps)
    assert out.tolist() == ["74", "70", None]


def test_build_cell_mask_fractions_and_province():
    # Celdas de 0,05° con la proyección falsa. San Luis ocupa 0,1° × 0,1° = 4 celdas enteras.
    ps = [prov("74", -66.0, -34.0, -65.9, -33.9), prov("70", -66.0, -33.9, -65.9, -33.825)]
    m = build_cell_mask(ps, inv=fake_inv, fwd=fake_fwd, subsample=4)
    sl = m[m.provincia_id == "74"]
    assert len(sl) == 4 and np.allclose(sl.frac_dentro, 1.0)
    sj = m[m.provincia_id == "70"]
    assert sorted(np.round(sj.frac_dentro, 2).tolist()) == [0.5, 0.5, 1.0, 1.0]
    assert set(m.region) == {"Cuyo"}
    assert m.cell_id.is_unique
    row, col = np.divmod(m.cell_id.to_numpy(), C.GRID_N_COLS)
    assert (row == m.grid_row).all() and (col == m.grid_col).all()


def test_provinces_table_bbox_format():
    t = provinces_table([prov("74", -67.38, -36.0, -64.88, -31.83)])
    assert t.loc[0, "bbox"] == "-67.38,-36.00,-64.88,-31.83"
    assert '"MultiPolygon"' in t.loc[0, "geom_geojson"]


# ---------------------------------------------------------------- fuentes fijas


def _detections(lat, lon, dates, night):
    return pd.DataFrame({"lat": lat, "lon": lon, "fecha": dates, "dia_noche": ["N" if night else "D"] * len(dates)})


def _spread_dates(n_years=4, per_month=1):
    return [date(2015 + y, m, d) for y in range(n_years) for m in range(1, 13) for d in range(1, per_month + 1)]


def test_static_sources_flags_nightly_flare_but_not_daytime_burning():
    ds = _spread_dates()
    flare = _detections(-38.0, -68.0, ds, night=True)
    crops = _detections(-33.0, -62.0, ds, night=False)
    out = detect_static_sources(pd.concat([flare, crops], ignore_index=True))
    assert len(out) == 1
    assert out.iloc[0].motivo == "nocturno todo el año"
    assert out.iloc[0].punto_id == point_bin(np.array([-38.0]), np.array([-68.0]))[0]


def test_static_sources_type2_rule_and_persistence_threshold():
    ds = _spread_dates()
    day = _detections(-33.0, -62.0, ds, night=False)
    t2 = pd.Series({int(point_bin(np.array([-33.0]), np.array([-62.0]))[0]): 0.5})
    assert detect_static_sources(day, t2).iloc[0].motivo == "marcado industrial por FIRMS"
    short = _detections(-38.0, -68.0, ds[:20], night=True)  # < 30 días
    assert detect_static_sources(short).empty


def test_exclude_static_removes_only_matching_bins():
    det = pd.DataFrame({"lat": [-38.0, -38.001, -33.0], "lon": [-68.0, -68.0, -62.0]})
    puntos = pd.DataFrame({"punto_id": point_bin(np.array([-38.0]), np.array([-68.0]))})
    kept, n = exclude_static(det, puntos)
    assert n == 2 and kept.lat.tolist() == [-33.0]


# ---------------------------------------------------------------- conteos


def _det(rows):
    d = pd.DataFrame(rows, columns=["fecha", "satelite", "cell_id"])
    d["hora_utc"] = "0425"
    d["sensor"] = np.where(d.satelite.isin(C.MODIS_SATS), "MODIS", "VIIRS")
    d["lat"], d["lon"] = -33.3, -66.3
    d["confianza"], d["frp_mw"], d["dia_noche"], d["fuente"] = "n", 5.0, "D", "SP"
    return d


def test_normalize_satellite_codes():
    s = normalize_satellite(pd.Series(["Terra", "Aqua", "N", "SNPP", "N20", "1", "N21", "xx"]))
    assert s.tolist()[:7] == ["Terra", "Aqua", "SNPP", "SNPP", "N20", "N20", "N21"]
    assert pd.isna(s.iloc[7])


def test_build_conteos_sparse_int16_and_mask():
    d1 = date(2020, 8, 1)
    det = _det([(d1, "Terra", 10), (d1, "SNPP", 10), (d1, "SNPP", 10), (d1, "N20", 11), (d1, "Aqua", 999)])
    cp = pd.Series({10: "74", 11: "74"})
    c = build_conteos(det, cp)
    assert len(c) == 2  # la celda 999 está fuera de la máscara
    r10 = c[c.cell_id == 10].iloc[0]
    assert (r10.n_terra, r10.n_aqua, r10.n_snpp, r10.n_n20, r10.n_n21) == (1, 0, 2, 0, 0)
    assert str(c.n_snpp.dtype) == "int16" and str(c.anio.dtype) == "int16"


def test_build_focos_ids_unique_and_province_nullable():
    d1, d2 = date(2020, 8, 1), date(2020, 8, 2)
    f = build_focos(_det([(d1, "Terra", 10), (d1, "SNPP", 999), (d2, "Aqua", 10)]), pd.Series({10: "74"}))
    assert f.foco_id.is_unique
    assert set(f.foco_id // 1_000_000) == {20200801, 20200802}
    assert f.provincia_id.isna().sum() == 1


def test_satelite_operativo_marks_outage_and_mission_end():
    start = C.MISSION_START["SNPP"]
    det = _det([(start, "SNPP", 1), (start + timedelta(days=2), "SNPP", 1)])
    o = build_satelite_operativo(det, start + timedelta(days=2))
    s = o[o.satelite == "SNPP"].set_index("fecha")
    assert s.operativo.tolist() == [True, False, True]
    assert s.n_nacional.tolist() == [1, 0, 1]
    assert "N20" not in set(o.satelite)  # misión posterior a fecha_max

    late = _det([(C.SNPP_END + timedelta(days=1), "SNPP", 1)])
    o2 = build_satelite_operativo(late, C.SNPP_END + timedelta(days=1))
    assert not o2[(o2.satelite == "SNPP")].iloc[-1].operativo


def test_trim_to_missions_drops_days_outside_each_mission():
    t0, s_end = C.MISSION_START["Terra"], C.MISSION_END["SNPP"]
    det = _det(
        [
            (date(2000, 10, 31), "Terra", 1),          # UTC 2000-11-01 → local día anterior
            (t0, "Terra", 1),
            (C.MISSION_START["Aqua"] - timedelta(days=1), "Aqua", 1),
            (s_end, "SNPP", 1),
            (s_end + timedelta(days=1), "SNPP", 1),    # después del fin de S-NPP
            (s_end + timedelta(days=1), "N20", 1),     # N20 no tiene fin de misión
        ]
    )
    kept, dropped = trim_to_missions(det)
    assert sorted(zip(kept.fecha, kept.satelite)) == sorted(
        [(t0, "Terra"), (s_end, "SNPP"), (s_end + timedelta(days=1), "N20")]
    )
    assert dropped == {"Terra": 1, "Aqua": 1, "SNPP": 1, "N20": 0, "N21": 0}
    # Todo día con focos queda cubierto por satelite_operativo.
    oper = build_satelite_operativo(kept, max(kept.fecha))
    keys = set(zip(oper.fecha, oper.satelite))
    assert all(k in keys for k in zip(kept.fecha, kept.satelite))


def test_default_fecha_max_is_last_complete_day():
    today = date(2026, 11, 14)
    assert default_fecha_max(date(2026, 11, 14), today=today) == date(2026, 11, 13)
    assert default_fecha_max(date(2026, 8, 31), today=today) == date(2026, 8, 31)
    real_today = datetime.now(LOCAL_TZ).date()
    assert default_fecha_max(real_today) < real_today


def test_resolve_fecha_max_never_goes_past_last_complete_day():
    today = date(2026, 11, 14)
    data_max = date(2026, 11, 14)
    assert resolve_fecha_max(data_max, None, today) == date(2026, 11, 13)
    assert resolve_fecha_max(data_max, date(2027, 1, 1), today) == date(2026, 11, 13)  # futura
    assert resolve_fecha_max(date(2026, 11, 10), date(2026, 11, 12), today) == date(2026, 11, 10)  # sin datos
    assert resolve_fecha_max(data_max, date(2025, 12, 31), today) == date(2025, 12, 31)  # corte anterior


# ---------------------------------------------------------------- calibración


def _synthetic_conteos(ratio_by_prov, years=(2012, 2016), m_per_day=2, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    d = date(years[0], 2, 1)
    end = date(years[1], 12, 31)
    while d <= end:
        for i, (pid, (r, mpd)) in enumerate(ratio_by_prov.items()):
            m = mpd
            v = int(round(r * m * (1 + noise * rng.standard_normal()))) if noise else int(r * m)
            rows.append((100 + i, d, pid, m, 0, max(v, 0), 0, 0))
        d += timedelta(days=1)
    c = pd.DataFrame(rows, columns=["cell_id", "fecha", "provincia_id", "n_terra", "n_aqua", "n_snpp", "n_n20", "n_n21"])
    c["anio"] = pd.to_datetime(c.fecha).dt.year
    days = pd.date_range(date(years[0], 1, 1), end).date
    oper = pd.concat(
        [pd.DataFrame({"fecha": days, "satelite": s, "operativo": True, "n_nacional": 1}) for s in ("Terra", "Aqua", "SNPP")]
    )
    return c, oper


def test_monthly_counts_skips_outage_days_and_fills_zeros():
    c, oper = _synthetic_conteos({"74": (3.0, 2)}, years=(2012, 2012))
    bad = date(2012, 3, 5)
    oper.loc[(oper.fecha == bad) & (oper.satelite == "SNPP"), "operativo"] = False
    m = monthly_counts(c, oper, "modis", (2012, 2012))
    sl_mar = m[(m.provincia_id == "74") & (m.mes == 3)].iloc[0]
    assert sl_mar.M == 2 * 30 and sl_mar.V == 6 * 30
    assert len(m) == len(C.PROVINCE_REGION) * 11  # feb–dic, todas las provincias
    assert m[m.provincia_id == "70"].M.sum() == 0


def test_fit_recovers_ratio_and_shrinks_sparse_province():
    c, oper = _synthetic_conteos({"74": (3.0, 20), "70": (6.0, 20), "46": (10.0, 0)})
    c.loc[(c.provincia_id == "46") & (pd.to_datetime(c.fecha).dt.month == 1) & (c.anio == 2013), ["n_terra", "n_snpp"]] = [1, 10]
    m = monthly_counts(c, oper, "modis", (2012, 2016))
    fit = fit_hierarchical(m, k=200)
    f = fit.prov_m.set_index(["provincia_id", "mes"])
    assert f.loc[("74", 7), "factor"] == pytest.approx(3.0, rel=0.05)
    assert f.loc[("70", 7), "factor"] == pytest.approx(6.0, rel=0.05)
    assert f.loc[("74", 7), "nivel_origen"] == "provincia"
    # La Rioja (46) casi sin datos: hereda el de Cuyo (≈ promedio de 74 y 70), no su 10×.
    assert f.loc[("46", 1), "nivel_origen"] == "region"
    assert 3.0 < f.loc[("46", 1), "factor"] < 6.0
    # Provincia sin datos fuera de Cuyo: nivel país.
    assert f.loc[("62", 1), "nivel_origen"] == "pais"
    assert fit.nat == pytest.approx(4.5, rel=0.01)
    assert len(fit.prov_m) == len(C.PROVINCE_REGION) * 12


def test_low_data_provinces_use_region_factor():
    c, oper = _synthetic_conteos({"06": (3.0, 20), "02": (50.0, 20)})
    fit = fit_hierarchical(monthly_counts(c, oper, "modis", (2012, 2016)))
    f = fit.prov_m.set_index(["provincia_id", "mes"])["factor"]
    assert f[("02", 5)] == pytest.approx(fit.region_m[("Pampeana", 5)], rel=1e-6)


def test_region_factors_table_exports_fit_levels():
    c, oper = _synthetic_conteos({"74": (3.0, 20), "70": (6.0, 20), "06": (2.0, 10)})
    m = monthly_counts(c, oper, "modis", (2012, 2016))
    fits = {"modis": fit_hierarchical(m), "terra": fit_hierarchical(m)}
    t = region_factors_table(fits, "v1")
    regs = set(C.PROVINCE_REGION.values()) | {C.NATIONAL_REGION}
    assert len(t) == len(regs) * 12 * 2
    assert set(t.region) == regs and not t.duplicated(["region", "mes", "tipo"]).any()
    assert list(t.columns) == ["region", "mes", "tipo", "factor", "n_modis_calib", "factores_version"]
    i = t[t.tipo == "modis"].set_index(["region", "mes"])
    fit = fits["modis"]
    assert i.loc[("Cuyo", 7), "factor"] == pytest.approx(fit.region_m[("Cuyo", 7)], rel=1e-6)
    assert i.loc[(C.NATIONAL_REGION, 7), "factor"] == pytest.approx(fit.nat_m[7], rel=1e-6)
    # n_modis_calib: suma de M de las provincias de la región / del país en ese mes.
    mm = m[m.mes == 7]
    assert i.loc[("Cuyo", 7), "n_modis_calib"] == mm[mm.region == "Cuyo"].M.sum()
    assert i.loc[(C.NATIONAL_REGION, 7), "n_modis_calib"] == mm.M.sum()
    assert i.loc[("NOA", 7), "n_modis_calib"] == 0
    # Región sin datos: hereda el factor nacional del mes (encogimiento completo).
    assert i.loc[("NOA", 7), "factor"] == pytest.approx(fit.nat_m[7], rel=1e-6)


def test_predict_score_and_ranges():
    c, oper = _synthetic_conteos({"74": (3.0, 4), "70": (5.0, 1)}, noise=0.3, seed=1)
    m = monthly_counts(c, oper, "modis", (2012, 2016))
    fit = fit_hierarchical(m)
    s = score(m, predict(m, fit))
    assert s["error_pct"] < 20 and abs(s["sesgo_pct"]) < 5
    r = calibrate_ranges(m)
    assert r[["modis_min", "modis_max"]].values.tolist() == [list(b) for b in C.RANGE_BINS]
    active = r[r.n_meses > 0]
    assert (active.mult_bajo <= active.mult_alto).all()
    cov = range_coverage(m, fit, r.fillna(1.0))
    assert 60 <= cov <= 100


# ---------------------------------------------------------------- contrato


def _valid_tables():
    d1 = date(2020, 8, 1)
    cp = pd.Series({10: "74"})
    det = _det([(d1, "Terra", 10), (d1, "SNPP", 10)])
    provs = [prov("74", -67, -36, -65, -32)]
    dim_c = pd.DataFrame(
        {"cell_id": [10], "grid_row": [0], "grid_col": [10], "provincia_id": ["74"], "region": ["Cuyo"],
         "frac_dentro": [1.0], "lat_centro": [-33.0], "lon_centro": [-66.0], "geom_geojson": ["{}"]}
    )
    fac = pd.DataFrame(
        [("74", m, t, 3.0, "provincia", 100, "v") for t in ("modis", "terra") for m in range(1, 13)],
        columns=["provincia_id", "mes", "tipo", "factor", "nivel_origen", "n_modis_calib", "factores_version"],
    )
    regs = sorted(set(C.PROVINCE_REGION.values())) + [C.NATIONAL_REGION]
    fac_reg = pd.DataFrame(
        [(r, m, t, 3.0, 1000, "v") for t in ("modis", "terra") for r in regs for m in range(1, 13)],
        columns=["region", "mes", "tipo", "factor", "n_modis_calib", "factores_version"],
    )
    rangos = pd.DataFrame({"modis_min": [0, 1], "modis_max": [0, 2_147_483_647], "mult_bajo": [0.0, 0.5], "mult_alto": [10.0, 2.0]})
    puntos = detect_static_sources(det)
    puntos["provincia_id"] = None
    return {
        "focos": build_focos(det, cp),
        "conteos_celda_dia": build_conteos(det, cp),
        "dim_celdas": dim_c,
        "dim_provincias": provinces_table(provs),
        "factores_provincia_mes": fac,
        "factores_region_mes": fac_reg,
        "rangos": rangos,
        "satelite_operativo": build_satelite_operativo(det, d1),
        "puntos_industriales": puntos,
    }


def test_validate_accepts_valid_tables_and_conform_casts():
    t = _valid_tables()
    assert validate(t, date(2020, 8, 1)) == []
    out = conform("conteos_celda_dia", t["conteos_celda_dia"])
    assert list(out.columns) == ["cell_id", "fecha", "provincia_id", "n_terra", "n_aqua", "n_snpp", "n_n20", "n_n21", "anio"]
    assert str(out.cell_id.dtype) == "int32"


def test_validate_detects_contract_violations():
    t = _valid_tables()
    t["factores_provincia_mes"].loc[0, "factor"] = 25.0
    t["conteos_celda_dia"].loc[0, "cell_id"] = 12345
    errs = validate(t, date(2020, 7, 31))
    assert any("factor fuera" in e for e in errs)
    assert any("cell_id fuera" in e for e in errs)
    assert any("posteriores a fecha_max" in e for e in errs)
    t3 = _valid_tables()
    t3["factores_region_mes"] = t3["factores_region_mes"].iloc[1:]
    t3["factores_region_mes"].loc[5, "factor"] = 0.1
    errs3 = validate(t3, date(2020, 8, 1))
    assert any("factores_region_mes[modis]: se esperaban 6 × 12" in e for e in errs3)
    assert any("factores_region_mes[modis]: factor fuera" in e for e in errs3)
    t2 = _valid_tables()
    del t2["rangos"]
    assert validate(t2, date(2020, 8, 1)) == ["rangos: tabla ausente"]


def test_validate_rejects_dates_before_data_start():
    # docs/04 (API): antes del 2000-11-01 no hay datos.
    t = _valid_tables()
    t["focos"].loc[0, "fecha"] = date(2000, 10, 31)
    t["conteos_celda_dia"].loc[0, "fecha"] = date(2000, 10, 31)
    errs = validate(t, date(2020, 8, 1))
    assert any("focos: hay fechas anteriores a" in e for e in errs)
    assert any("conteos_celda_dia: hay fechas anteriores a" in e for e in errs)


def test_metadata_schema_version_with_factores_region():
    meta = build_metadata(date(2000, 11, 1), date(2026, 11, 13), date(2026, 8, 31), "v", "abc")
    assert meta["schema_version"] == "1.1.0"


def test_validate_rejects_future_fecha_max():
    t = _valid_tables()
    errs = validate(t, date(2020, 8, 2), today=date(2020, 8, 1))
    assert any("fecha futura" in e for e in errs)
    assert validate(t, date(2020, 8, 1), today=date(2020, 8, 1)) == []


def test_validate_metadata_requires_fecha_ultimo_sp_and_contract_fields():
    ok = build_metadata(date(2000, 11, 1), date(2026, 11, 13), date(2026, 8, 31), "v", "abc")
    assert validate_metadata(ok) == []
    assert set(ok) == {
        "schema_version", "data_version", "factores_version", "fecha_min", "fecha_max",
        "fecha_ultimo_sp", "referencia_viirs", "fuentes", "pipeline_commit",
    }
    sin_sp = build_metadata(date(2000, 11, 1), date(2026, 11, 13), None, "v", "abc")
    assert any("fecha_ultimo_sp" in e for e in validate_metadata(sin_sp))
    temprana = build_metadata(date(2000, 10, 31), date(2026, 11, 13), date(2026, 8, 31), "v", "abc")
    assert any("fecha_min" in e for e in validate_metadata(temprana))
    ref = dict(ok, referencia_viirs=[{"satelite": "N21", "desde": "2023-03-01", "hasta": None}])
    assert any("referencia_viirs" in e for e in validate_metadata(ref))
