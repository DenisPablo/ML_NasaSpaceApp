"""Validación del contrato de datos y escritura de los artefactos para la API.

La validación es pandas puro (se prueba sin pyarrow). La escritura usa pyarrow (ZSTD,
particionado Hive por ``anio``) y se publica en un directorio temporal que luego se renombra,
para que la API nunca lea una versión a medio escribir.
"""

from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import config as C

SCHEMAS: dict[str, dict[str, str]] = {
    "focos": {
        "foco_id": "int64", "fecha": "date", "hora_utc": "string", "satelite": "string", "sensor": "string",
        "lat": "float32", "lon": "float32", "confianza": "string", "frp_mw": "float32", "dia_noche": "string",
        "fuente": "string", "cell_id": "int32", "provincia_id": "string?", "anio": "int16",
    },
    "conteos_celda_dia": {
        "cell_id": "int32", "fecha": "date", "provincia_id": "string",
        "n_terra": "int16", "n_aqua": "int16", "n_snpp": "int16", "n_n20": "int16", "n_n21": "int16", "anio": "int16",
    },
    "dim_celdas": {
        "cell_id": "int32", "grid_row": "int16", "grid_col": "int16", "provincia_id": "string", "region": "string",
        "frac_dentro": "float32", "lat_centro": "float32", "lon_centro": "float32", "geom_geojson": "string",
    },
    "dim_provincias": {"provincia_id": "string", "nombre": "string", "region": "string", "geom_geojson": "string", "bbox": "string"},
    "factores_provincia_mes": {
        "provincia_id": "string", "mes": "int8", "tipo": "string", "factor": "float32",
        "nivel_origen": "string", "n_modis_calib": "int32", "factores_version": "string",
    },
    "factores_region_mes": {
        "region": "string", "mes": "int8", "tipo": "string", "factor": "float32",
        "n_modis_calib": "int32", "factores_version": "string",
    },
    "rangos": {"modis_min": "int32", "modis_max": "int32", "mult_bajo": "float32", "mult_alto": "float32"},
    "satelite_operativo": {"fecha": "date", "satelite": "string", "operativo": "bool", "n_nacional": "int32"},
    "puntos_industriales": {
        "punto_id": "int64", "lat": "float32", "lon": "float32", "provincia_id": "string?", "motivo": "string",
        "dias": "int16", "anios": "int16", "meses": "int16", "frac_nocturna": "float32", "n_focos": "int32",
    },
}
PARTITIONED = {"focos", "conteos_celda_dia"}


def conform(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Ordena columnas y castea tipos según el contrato (falla si falta una columna)."""
    schema = SCHEMAS[name]
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: faltan columnas {missing}")
    out = df[list(schema)].copy()
    for col, t in schema.items():
        base = t.rstrip("?")
        if base == "date":
            out[col] = pd.to_datetime(out[col]).dt.date
        elif base == "string":
            out[col] = out[col].astype("object").where(out[col].notna(), None)
        else:
            out[col] = out[col].astype(base)
    return out


def validate(tables: dict[str, pd.DataFrame], fecha_max: date, today: date | None = None) -> list[str]:
    """Chequeos obligatorios de ``docs/02-contratos-de-datos.md`` (API). Devuelve la lista de errores."""
    err: list[str] = []
    today = today or datetime.now(timezone(timedelta(hours=-3))).date()
    if fecha_max > today:
        err.append(f"fecha_max {fecha_max} es una fecha futura (hoy es {today} en UTC−3)")
    for name, schema in SCHEMAS.items():
        if name not in tables:
            err.append(f"{name}: tabla ausente")
            continue
        df = tables[name]
        for col, t in schema.items():
            if col not in df.columns:
                err.append(f"{name}.{col}: columna ausente")
            elif not t.endswith("?") and df[col].isna().any():
                err.append(f"{name}.{col}: tiene nulos y el contrato no los admite")
    if err:
        return err

    cells, provs = tables["dim_celdas"], tables["dim_provincias"]
    prov_ids = set(provs["provincia_id"])
    for name in ("focos", "conteos_celda_dia", "satelite_operativo"):
        f = pd.to_datetime(tables[name]["fecha"])
        if len(f) and f.max().date() > fecha_max:
            err.append(f"{name}: hay fechas posteriores a fecha_max {fecha_max}")
        if len(f) and f.min().date() < C.DATA_START:
            err.append(f"{name}: hay fechas anteriores a {C.DATA_START} (inicio de los datos)")
    for name in ("focos", "conteos_celda_dia"):
        bad = set(tables[name]["satelite"]) - set(C.ALL_SATS) if "satelite" in tables[name] else set()
        if bad:
            err.append(f"{name}: satélites no válidos {sorted(bad)}")
    cnt = tables["conteos_celda_dia"]
    ncols = list(C.SAT_COLUMN.values())
    if len(cnt):
        if (cnt[ncols] < 0).any().any():
            err.append("conteos_celda_dia: hay conteos negativos")
        if not (cnt[ncols].sum(axis=1) > 0).all():
            err.append("conteos_celda_dia: hay filas sin ningún foco (la tabla es dispersa)")
        if cnt.duplicated(["cell_id", "fecha"]).any():
            err.append("conteos_celda_dia: (cell_id, fecha) duplicado")
        if not set(cnt["cell_id"]).issubset(set(cells["cell_id"])):
            err.append("conteos_celda_dia: cell_id fuera de dim_celdas")
        if not set(cnt["provincia_id"]).issubset(prov_ids):
            err.append("conteos_celda_dia: provincia_id fuera de dim_provincias")
    if tables["focos"]["foco_id"].duplicated().any():
        err.append("focos: foco_id duplicado")
    if not set(cells["provincia_id"]).issubset(prov_ids):
        err.append("dim_celdas: provincia_id fuera de dim_provincias")
    fac = tables["factores_provincia_mes"]
    for tipo, g in fac.groupby("tipo"):
        if len(g) != len(prov_ids) * 12 or g.duplicated(["provincia_id", "mes"]).any():
            err.append(f"factores_provincia_mes[{tipo}]: se esperaban {len(prov_ids)} × 12 filas únicas")
    for tipo, g in fac.groupby("tipo"):
        lo, hi = C.FACTOR_BOUNDS.get(tipo, (0.5, 20.0))
        if not g["factor"].between(lo, hi, inclusive="neither").all():
            err.append(f"factores_provincia_mes[{tipo}]: factor fuera de ({lo}, {hi})")
    freg = tables["factores_region_mes"]
    reg_ids = set(C.PROVINCE_REGION.values()) | {C.NATIONAL_REGION}
    if not set(freg["region"]).issubset(reg_ids):
        err.append(f"factores_region_mes: regiones no válidas {sorted(set(freg['region']) - reg_ids)}")
    for tipo, g in freg.groupby("tipo"):
        if len(g) != len(reg_ids) * 12 or g.duplicated(["region", "mes"]).any():
            err.append(f"factores_region_mes[{tipo}]: se esperaban {len(reg_ids)} × 12 filas únicas")
        lo, hi = C.FACTOR_BOUNDS.get(tipo, (0.5, 20.0))
        if not g["factor"].between(lo, hi, inclusive="neither").all():
            err.append(f"factores_region_mes[{tipo}]: factor fuera de ({lo}, {hi})")
    if set(freg["tipo"]) != set(fac["tipo"]):
        err.append("factores_region_mes: los tipos no coinciden con factores_provincia_mes")
    if tables["rangos"][["mult_bajo", "mult_alto"]].isna().any().any():
        err.append("rangos: tramo sin datos suficientes")
    return err


def build_metadata(
    fecha_min: date,
    fecha_max: date,
    fecha_ultimo_sp: date | None,
    factores_version: str,
    pipeline_commit: str,
) -> dict:
    tz = timezone(timedelta(hours=-3))
    return {
        "schema_version": C.SCHEMA_VERSION,
        "data_version": datetime.now(tz).replace(microsecond=0).isoformat(),
        "factores_version": factores_version,
        "fecha_min": fecha_min.isoformat(),
        "fecha_max": fecha_max.isoformat(),
        "fecha_ultimo_sp": fecha_ultimo_sp.isoformat() if fecha_ultimo_sp else None,
        "referencia_viirs": [
            {"satelite": "SNPP", "desde": C.VIIRS_START.isoformat(), "hasta": C.SNPP_END.isoformat()},
            {"satelite": "N20", "desde": (C.SNPP_END + timedelta(days=1)).isoformat(), "hasta": None},
        ],
        "fuentes": ["NASA FIRMS MODIS C6.1", "NASA FIRMS VIIRS 375m", "IGN via API Georef"],
        "pipeline_commit": pipeline_commit,
    }


def write_all(tables: dict[str, pd.DataFrame], metadata: dict, out_root: Path, extra: dict[str, dict] | None = None) -> Path:
    """Escribe en ``out_root/<data_version>/`` y apunta ``out_root/current`` a esa versión."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    version = metadata["data_version"].replace(":", "").replace("-", "")[:15]
    final = Path(out_root) / version
    tmp = Path(out_root) / f".tmp-{version}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for name, df in tables.items():
        tbl = pa.Table.from_pandas(conform(name, df), preserve_index=False)
        if name in PARTITIONED:
            for anio in sorted(df["anio"].unique()):
                part = tmp / name / f"anio={int(anio)}"
                part.mkdir(parents=True, exist_ok=True)
                sub = tbl.filter(pc.equal(tbl["anio"], pa.scalar(int(anio), pa.int16()))).drop_columns(["anio"])
                pq.write_table(sub, part / "part-0.parquet", compression="zstd")
        else:
            pq.write_table(tbl, tmp / f"{name}.parquet", compression="zstd")
    (tmp / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    for fname, content in (extra or {}).items():
        (tmp / fname).write_text(json.dumps(content, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    link = Path(out_root) / "current"
    tmp_link = Path(out_root) / ".current-new"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    tmp_link.symlink_to(final.name)
    os.replace(tmp_link, link)
    return final


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    raise TypeError(type(o))
