# /// script
# requires-python = ">=3.12"
# dependencies = ["duckdb", "numpy", "pandas", "pyarrow", "shapely", "rasterio"]
# ///
"""Overture buildings and roads drawn onto each lab window's 256 x 256 grid (the AEF chunk grid).

Per window, data/fuel_overture/<cy>_<cx>.npz:
  bld   1 where the pixel center falls inside a building footprint
  road  road class code where the pixel center is within half the road's width of a segment, else 0
        (1 motorway / trunk, 2 primary / secondary, 3 tertiary / residential / unclassified,
         4 service / living street, 5 track / path / other road, 6 rail)
Widths in meters are typical paved widths, not measured: 1: 24, 2: 14, 3: 8, 4: 5, 5: 3, 6: 4.

This is a sharper "pavement or roof" truth than LANDFIRE's 30 m cells, and a test of drawing mapped roads and
buildings low directly in the app instead of asking the head.

Release 2026-10-21.0 on s3://overturemaps-us-west-2, read with DuckDB by bounding box (about 15 s per window).

    uv run scripts/sample_fuel_overture.py [--workers 3]
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import shapely
from rasterio.features import rasterize
from rasterio.transform import from_origin

from fuel_common import CHUNK, MOS_RES, MOS_X0, MOS_Y0

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "fuel_overture"
REL = "s3://overturemaps-us-west-2/release/2026-10-21.0"
ROAD_CLASS = {"motorway": 1, "trunk": 1, "primary": 2, "secondary": 2, "tertiary": 3, "residential": 3, "unclassified": 3,
              "service": 4, "living_street": 4}
WIDTH_M = {1: 24, 2: 14, 3: 8, 4: 5, 5: 3, 6: 4}


def connect():
    c = duckdb.connect()
    c.execute("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2'; "
              "SET enable_object_cache=true; SET enable_progress_bar=false;")
    return c


def query(c, t, cols, x0, y0, x1, y1):
    return c.execute(f"select {cols}, ST_AsWKB(geometry) as g from read_parquet('{REL}/{t}/*') "
                     f"where bbox.xmin < {x1} and bbox.xmax > {x0} and bbox.ymin < {y1} and bbox.ymax > {y0}").df()


def window(c, cy, cx):
    lon0, lat0 = MOS_X0 + cx * CHUNK * MOS_RES, MOS_Y0 - cy * CHUNK * MOS_RES
    lon1, lat1 = lon0 + CHUNK * MOS_RES, lat0 - CHUNK * MOS_RES
    pad = 0.0003
    tr = from_origin(lon0, lat0, MOS_RES, MOS_RES)
    b = query(c, "theme=buildings/type=building", "id", lon0 - pad, lat1 - pad, lon1 + pad, lat0 + pad)
    geoms = [g for g in shapely.from_wkb([bytes(x) for x in b.g]) if g is not None and not g.is_empty] if len(b) else []
    bld = rasterize(((g, 1) for g in geoms), out_shape=(CHUNK, CHUNK), transform=tr, dtype="uint8") if geoms else \
        np.zeros((CHUNK, CHUNK), np.uint8)
    s = query(c, "theme=transportation/type=segment", "subtype, class", lon0 - pad, lat1 - pad, lon1 + pad, lat0 + pad)
    road = np.zeros((CHUNK, CHUNK), np.uint8)
    if len(s):
        code = np.where(s.subtype == "rail", 6, s["class"].map(ROAD_CLASS).fillna(5).astype(int))
        code = np.where(s.subtype.isin(["road", "rail"]), code, 0)
        lines = shapely.from_wkb([bytes(x) for x in s.g])
        m_per_deg_lon = 111320 * np.cos(np.radians((lat0 + lat1) / 2))
        for k in sorted(WIDTH_M, reverse=True):  # wide classes drawn last win
            sel = [ln for ln, cc in zip(lines, code) if cc == k and ln is not None]
            if not sel:
                continue
            # buffer in degrees: half width over meters per degree of longitude (close enough at this scale)
            buf = shapely.buffer(np.array(sel), WIDTH_M[k] / 2 / m_per_deg_lon, cap_style="flat")
            mask = rasterize(((g, 1) for g in buf if not g.is_empty), out_shape=(CHUNK, CHUNK), transform=tr, dtype="uint8")
            road[mask == 1] = k
    return bld, road, len(b), len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    wins = pd.read_parquet(ROOT / "data" / "fuel_ca_windows.parquet")
    todo = [(int(w.cy), int(w.cx)) for w in wins.itertuples() if not (OUT / f"{w.cy}_{w.cx}.npz").exists()]
    print(f"{len(todo)} windows to draw ({len(wins) - len(todo)} done)", flush=True)
    conns = {}

    def job(k):
        import threading
        c = conns.setdefault(threading.get_ident(), connect())
        for attempt in range(4):
            try:
                bld, road, nb, ns = window(c, *k)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5 + 5 * attempt)
        np.savez_compressed(OUT / f"{k[0]}_{k[1]}.npz", bld=bld, road=road)
        return k, nb, ns, bld.mean(), (road > 0).mean()

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for n, (k, nb, ns, fb, fr) in enumerate(ex.map(job, todo)):
            el = time.time() - t0
            print(f"{n + 1}/{len(todo)} {k}: {nb} buildings ({fb:.1%} px), {ns} segments ({fr:.1%} px road), "
                  f"{el / (n + 1):.1f} s/window, eta {el / (n + 1) * (len(todo) - n - 1) / 60:.0f} min", flush=True)
    print(f"done: {len(list(OUT.glob('*.npz')))} windows in {OUT}")


if __name__ == "__main__":
    main()
