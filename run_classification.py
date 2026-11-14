"""Etapa de clasificación y armonización (Propuesta B) → artefactos para FireHistoryInterfaceAPI.

Lee el Parquet limpio de la Fase 1 (``clean_firms.py --region argentina``) y produce, en
``<out>/<versión>/`` con un enlace ``<out>/current``, los archivos definidos en
``FireHistoryInterfaceAPI/docs/02-contratos-de-datos.md``.

Uso::

    python run_classification.py \\
        --clean data/clean/firms_argentina_clean.parquet \\
        --provinces-geojson data/ref/provincias.geojson \\
        --raw-dir data/raw \\
        --out data/serving_api

No modifica ni reemplaza las fases 2–4 existentes.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from classification import config as C
from classification.calibrate import (
    calibrate_ranges,
    evaluate,
    factors_table,
    region_factors_table,
    fit_hierarchical,
    monthly_counts,
    range_coverage,
)
from classification.counts import (
    build_conteos,
    build_focos,
    build_satelite_operativo,
    exclude_static,
    prepare_detections,
    resolve_fecha_max,
    trim_to_missions,
)
from classification.geo import assign_province, build_cell_mask, load_provinces_geojson, provinces_table
from classification.static_sources import detect_static_sources, type2_share_from_raw

CLEAN_COLUMNS = [
    "cell_id", "fecha_local", "acq_time", "satellite", "latitude", "longitude",
    "confidence", "frp", "daynight", "source_quality",
]


def read_raw_type(raw_dir: Path) -> pd.DataFrame:
    """Lee ``latitude, longitude, type`` de los CSV científicos (SP) de FIRMS."""
    frames = []
    for f in sorted(Path(raw_dir).rglob("*.csv")):
        if "nrt" in f.name.lower():
            continue
        head = pd.read_csv(f, nrows=0).columns
        if "type" not in head:
            continue
        df = pd.read_csv(f, usecols=["latitude", "longitude", "type"])
        frames.append(df.rename(columns={"latitude": "lat", "longitude": "lon"}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["lat", "lon", "type"])


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "desconocido"


def run(args: argparse.Namespace) -> int:
    log = print
    log(f"[1/7] Leyendo Parquet limpio: {args.clean}")
    clean = pd.read_parquet(args.clean, columns=CLEAN_COLUMNS)
    det = prepare_detections(clean)
    del clean
    det, recorte = trim_to_missions(det)
    log("      recorte a la ventana de cada misión: " + ", ".join(f"{s} {n:,}" for s, n in recorte.items()))
    pedida = date.fromisoformat(args.fecha_max) if args.fecha_max else None
    fecha_max = resolve_fecha_max(max(det["fecha"]), pedida)
    if pedida is None:
        log(f"      fecha_max por defecto (último día completo): {fecha_max}")
    elif fecha_max != pedida:
        log(f"      --fecha-max {pedida} es posterior al último día completo: se usa {fecha_max}")
    det = det[det["fecha"] <= fecha_max].reset_index(drop=True)

    log("[2/7] Provincias y máscara nacional de celdas")
    from classification.geo import make_albers_transforms

    provinces = load_provinces_geojson(args.provinces_geojson)
    fwd, inv = make_albers_transforms()
    dim_celdas = build_cell_mask(provinces, inv=inv, fwd=fwd)
    dim_provincias = provinces_table(provinces)
    cell_province = dim_celdas.set_index("cell_id")["provincia_id"]

    log("[3/7] Fuentes fijas industriales")
    t2 = type2_share_from_raw(read_raw_type(args.raw_dir)) if args.raw_dir else None
    if t2 is None:
        log("      (sin --raw-dir: la regla 'marcado industrial por FIRMS' queda desactivada)")
    puntos = detect_static_sources(det, t2)
    puntos["provincia_id"] = assign_province(puntos["lon"].to_numpy(), puntos["lat"].to_numpy(), provinces)
    det_ok, n_excl = exclude_static(det, puntos)
    log(f"      {len(puntos)} puntos · {n_excl:,} focos excluidos")

    log("[4/7] Conteos celda-día, focos y satélites operativos")
    oper = build_satelite_operativo(det, fecha_max)
    conteos = build_conteos(det_ok, cell_province)
    focos = build_focos(det_ok, cell_province)

    log("[5/7] Calibración de factores (Propuesta B)")
    mm = monthly_counts(conteos, oper, "modis", (C.CALIB_START.year, C.CALIB_END.year))
    mt = monthly_counts(conteos, oper, "terra", C.TERRA_CALIB_YEARS)
    fits = {"modis": fit_hierarchical(mm, args.k), "terra": fit_hierarchical(mt, args.k)}
    version = args.factores_version or f"{fecha_max:%Y.%m}-b1"
    factores = factors_table(fits, version)
    factores_region = region_factors_table(fits, version)
    rangos_full = calibrate_ranges(mm, args.k)
    rangos = rangos_full.drop(columns=["n_meses"])

    log("[6/7] Validación (pruebas hacia atrás / adelante y cobertura de rangos)")
    metrics = evaluate(conteos, oper, args.k)
    metrics["cobertura_rangos_pct"] = range_coverage(mm, fits["modis"], rangos)
    metrics["factor_nacional"] = {t: f.nat for t, f in fits.items()}
    metrics["rangos"] = rangos_full.to_dict(orient="records")
    metrics["puntos_industriales"] = {"n_puntos": int(len(puntos)), "focos_excluidos": n_excl}
    metrics["recorte_misiones"] = recorte
    metrics["filas"] = {"focos": int(len(focos)), "conteos_celda_dia": int(len(conteos)), "dim_celdas": int(len(dim_celdas))}
    log(json.dumps({k: v for k, v in metrics.items() if k != "rangos"}, ensure_ascii=False, indent=2, default=str))

    tables = {
        "focos": focos,
        "conteos_celda_dia": conteos,
        "dim_celdas": dim_celdas,
        "dim_provincias": dim_provincias,
        "factores_provincia_mes": factores,
        "factores_region_mes": factores_region,
        "rangos": rangos,
        "satelite_operativo": oper,
        "puntos_industriales": puntos,
    }
    from classification.publish import build_metadata, validate, write_all

    errors = validate(tables, fecha_max)
    if errors:
        log("ERROR: el contrato de datos no se cumple:\n  - " + "\n  - ".join(errors))
        return 2

    log(f"[7/7] Publicando en {args.out}")
    sp = det.loc[det["fuente"] == "SP", "fecha"]
    meta = build_metadata(
        fecha_min=min(det["fecha"]),
        fecha_max=fecha_max,
        fecha_ultimo_sp=max(sp) if len(sp) else None,
        factores_version=version,
        pipeline_commit=git_commit(),
    )
    final = write_all(tables, meta, args.out, extra={"validacion.json": metrics})
    log(f"Listo: {final} (enlace {Path(args.out) / 'current'})")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clasificación y armonización Propuesta B → datos para la API.")
    p.add_argument("--clean", type=Path, default=Path("data/clean/firms_argentina_clean.parquet"))
    p.add_argument("--provinces-geojson", type=Path, default=Path("data/ref/provincias.geojson"))
    p.add_argument("--raw-dir", type=Path, default=None, help="CSV crudos de FIRMS (para la proporción type=2).")
    p.add_argument("--out", type=Path, default=Path("data/serving_api"))
    p.add_argument("--k", type=float, default=C.SHRINK_K, help="Colchón de encogimiento (focos MODIS).")
    p.add_argument(
        "--fecha-max",
        default=None,
        help="Fecha local máxima a publicar (YYYY-MM-DD). Nunca pasa del último día completo: "
        "min(fecha máxima de los datos, ayer en hora local UTC−3), que es también el valor por defecto.",
    )
    p.add_argument("--factores-version", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
