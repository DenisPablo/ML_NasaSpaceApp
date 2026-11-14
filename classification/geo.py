"""Geometría sin dependencias pesadas: punto en polígono, provincias y máscara de celdas.

Las funciones de proyección se inyectan (``fwd`` / ``inv``) para poder probar la lógica sin
``pyproj``. En producción se usa ``make_albers_transforms()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import config as C

Transform = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class Province:
    """Provincia con anillos en lon/lat (cada polígono: exterior + huecos)."""

    provincia_id: str
    nombre: str
    region: str
    polygons: list[list[np.ndarray]] = field(default_factory=list)
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def contains(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Máscara booleana de puntos dentro de la provincia (regla par-impar con huecos)."""
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        out = np.zeros(lon.shape, dtype=bool)
        w, s, e, n = self.bbox
        cand = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
        if not cand.any():
            return out
        lo, la = lon[cand], lat[cand]
        inside = np.zeros(lo.shape, dtype=bool)
        for rings in self.polygons:
            in_poly = points_in_ring(lo, la, rings[0])
            for hole in rings[1:]:
                in_poly &= ~points_in_ring(lo, la, hole)
            inside |= in_poly
        out[cand] = inside
        return out


def points_in_ring(lon: np.ndarray, lat: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Ray casting vectorizado sobre los puntos (bucle sobre las aristas del anillo)."""
    ring = np.asarray(ring, dtype=np.float64)
    if ring.shape[0] < 3:
        return np.zeros(lon.shape, dtype=bool)
    xs, ys = ring[:, 0], ring[:, 1]
    xe, ye = np.roll(xs, -1), np.roll(ys, -1)
    inside = np.zeros(lon.shape, dtype=bool)
    for x0, y0, x1, y1 in zip(xs, ys, xe, ye):
        if y0 == y1:
            continue
        crosses = (y0 > lat) != (y1 > lat)
        if not crosses.any():
            continue
        x_int = x0 + (lat - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (lon < x_int)
    return inside


def load_provinces_geojson(path: Path, continental_only: bool = True) -> list[Province]:
    """Lee el GeoJSON de provincias de la API Georef (IGN).

    Fuente: https://apis.datos.gob.ar/georef/api/v2.0/provincias.geojson

    Con ``continental_only`` se descartan la Antártida y las islas del Atlántico Sur
    (Malvinas, Georgias, Sandwich), que no forman parte del alcance del calendario.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    provinces: list[Province] = []
    for feat in data["features"]:
        props = feat.get("properties", {})
        pid = str(props.get("id", "")).zfill(2)
        geom = feat["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        rings = [[np.asarray(r, dtype=np.float64)[:, :2] for r in poly] for poly in polys]
        if continental_only:
            rings = [r for r in rings if _is_continental(r[0])]
        if not rings:
            continue
        allpts = np.vstack([r[0] for r in rings])
        bbox = (float(allpts[:, 0].min()), float(allpts[:, 1].min()), float(allpts[:, 0].max()), float(allpts[:, 1].max()))
        provinces.append(
            Province(
                provincia_id=pid,
                nombre=str(props.get("nombre", pid)),
                region=C.PROVINCE_REGION.get(pid, "Desconocida"),
                polygons=rings,
                bbox=bbox,
            )
        )
    return provinces


def _is_continental(ring: np.ndarray) -> bool:
    lon, lat = float(ring[:, 0].mean()), float(ring[:, 1].mean())
    if lat < -60:
        return False
    return not (lat < -50 and lon > -63.5)


def assign_province(lon: np.ndarray, lat: np.ndarray, provinces: list[Province]) -> np.ndarray:
    """Código de provincia para cada punto (``None`` si cae fuera de todas)."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    out = np.full(lon.shape, None, dtype=object)
    pending = np.ones(lon.shape, dtype=bool)
    for prov in provinces:
        if not pending.any():
            break
        idx = np.flatnonzero(pending)
        hit = prov.contains(lon[idx], lat[idx])
        out[idx[hit]] = prov.provincia_id
        pending[idx[hit]] = False
    return out


def cell_xy(cell_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(fila, columna, x0, y0) de la esquina inferior izquierda de cada celda."""
    cell_id = np.asarray(cell_id, dtype=np.int64)
    row, col = np.divmod(cell_id, C.GRID_N_COLS)
    x0 = C.GRID_X_MIN + col * C.GRID_CELL_SIZE_M
    y0 = C.GRID_Y_MIN + row * C.GRID_CELL_SIZE_M
    return row, col, x0, y0


def build_cell_mask(
    provinces: list[Province],
    inv: Transform,
    fwd: Transform,
    subsample: int = 5,
) -> pd.DataFrame:
    """Máscara nacional: una fila por celda 5 × 5 km que toca alguna provincia.

    - Celdas cuyas 4 esquinas y centro caen en la misma provincia → fracción 1.
    - Celdas de borde → muestreo ``subsample × subsample`` para la fracción y la provincia principal.
    """
    w = min(p.bbox[0] for p in provinces)
    s = min(p.bbox[1] for p in provinces)
    e = max(p.bbox[2] for p in provinces)
    n = max(p.bbox[3] for p in provinces)
    gx, gy = fwd(np.array([w, e, w, e]), np.array([s, s, n, n]))
    c0 = max(int((np.min(gx) - C.GRID_X_MIN) // C.GRID_CELL_SIZE_M) - 1, 0)
    c1 = min(int((np.max(gx) - C.GRID_X_MIN) // C.GRID_CELL_SIZE_M) + 1, C.GRID_N_COLS - 1)
    r0 = max(int((np.min(gy) - C.GRID_Y_MIN) // C.GRID_CELL_SIZE_M) - 1, 0)
    r1 = min(int((np.max(gy) - C.GRID_Y_MIN) // C.GRID_CELL_SIZE_M) + 1, C.GRID_N_ROWS - 1)
    rows, cols = np.meshgrid(np.arange(r0, r1 + 1), np.arange(c0, c1 + 1), indexing="ij")
    cell_ids = (rows * C.GRID_N_COLS + cols).ravel()
    _, _, x0, y0 = cell_xy(cell_ids)
    size = C.GRID_CELL_SIZE_M

    # Centro + 4 esquinas
    offsets = [(0.5, 0.5), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    probes = []
    for fx, fy in offsets:
        lo, la = inv(x0 + fx * size, y0 + fy * size)
        probes.append(assign_province(lo, la, provinces))
    probes_arr = np.vstack(probes)  # (5, n_cells)
    center = probes_arr[0]
    same = np.all(probes_arr == center, axis=0)
    touched = np.any(probes_arr != None, axis=0)  # noqa: E711

    prov_main = center.copy()
    frac = np.where(same & (center != None), 1.0, 0.0)  # noqa: E711

    border = np.flatnonzero(~same | (center == None))  # noqa: E711
    if border.size:
        k = subsample
        f = (np.arange(k) + 0.5) / k
        fx, fy = np.meshgrid(f, f)
        fx, fy = fx.ravel(), fy.ravel()
        bx = (x0[border][:, None] + fx[None, :] * size).ravel()
        by = (y0[border][:, None] + fy[None, :] * size).ravel()
        lo, la = inv(bx, by)
        pv = assign_province(lo, la, provinces).reshape(border.size, k * k)
        for i, ci in enumerate(border):
            vals = [v for v in pv[i] if v is not None]
            if not vals:
                continue
            ids, counts = np.unique(np.asarray(vals, dtype=object).astype(str), return_counts=True)
            prov_main[ci] = ids[np.argmax(counts)]
            frac[ci] = len(vals) / (k * k)
            touched[ci] = True

    keep = touched & (prov_main != None) & (frac > 0)  # noqa: E711
    cid = cell_ids[keep]
    row, col, x0k, y0k = cell_xy(cid)
    lon_c, lat_c = inv(x0k + size / 2, y0k + size / 2)
    corners = [inv(x0k + a * size, y0k + b * size) for a, b in [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]
    geoms = [
        json.dumps({"type": "Polygon", "coordinates": [[[round(float(c[0][i]), 5), round(float(c[1][i]), 5)] for c in corners]]})
        for i in range(cid.size)
    ]
    region_by_prov = {p.provincia_id: p.region for p in provinces}
    df = pd.DataFrame(
        {
            "cell_id": cid.astype("int32"),
            "grid_row": row.astype("int16"),
            "grid_col": col.astype("int16"),
            "provincia_id": prov_main[keep].astype(str),
            "frac_dentro": frac[keep].astype("float32"),
            "lat_centro": np.asarray(lat_c, dtype="float32"),
            "lon_centro": np.asarray(lon_c, dtype="float32"),
            "geom_geojson": geoms,
        }
    )
    df.insert(4, "region", df["provincia_id"].map(region_by_prov).fillna("Desconocida"))
    return df.sort_values("cell_id").reset_index(drop=True)


def provinces_table(provinces: list[Province], simplify_decimals: int = 2) -> pd.DataFrame:
    """Tabla ``dim_provincias`` con geometría simplificada para el mapa."""
    rows = []
    for p in provinces:
        polys = []
        for rings in p.polygons:
            ext = np.round(rings[0], simplify_decimals)
            keep = np.r_[True, np.any(np.diff(ext, axis=0) != 0, axis=1)]
            polys.append([ext[keep].tolist()])
        geom = {"type": "MultiPolygon", "coordinates": polys}
        rows.append(
            {
                "provincia_id": p.provincia_id,
                "nombre": p.nombre,
                "region": p.region,
                "geom_geojson": json.dumps(geom),
                "bbox": ",".join(f"{v:.2f}" for v in p.bbox),
            }
        )
    return pd.DataFrame(rows).sort_values("provincia_id").reset_index(drop=True)


def make_albers_transforms() -> tuple[Transform, Transform]:
    """Transformaciones EPSG:4326 <-> Albers Argentina con pyproj (producción)."""
    from pyproj import Transformer

    from clean_firms import ARGENTINA_ALBERS_PROJ

    fwd_t = Transformer.from_crs("EPSG:4326", ARGENTINA_ALBERS_PROJ, always_xy=True)
    inv_t = Transformer.from_crs(ARGENTINA_ALBERS_PROJ, "EPSG:4326", always_xy=True)

    def fwd(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = fwd_t.transform(lon, lat)
        return np.asarray(x), np.asarray(y)

    def inv(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo, la = inv_t.transform(x, y)
        return np.asarray(lo), np.asarray(la)

    return fwd, inv
