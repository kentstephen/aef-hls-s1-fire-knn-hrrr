# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "rasterio", "scipy", "shapely", "tabulate", "pmtiles", "pillow", "requests", "psutil",
#                 "zarr>=3", "obstore"]
# ///
"""Backtest of the risk map's value (web/src/risk/): does ground that burned score higher than ground that did not,
using only what was known before the fire? And does the value beat fuel alone?

The value per 10 m pixel, as the map has it (score.ts, factors.ts, terrain.ts, history.ts):

    risk = fuel x (continuity + slope + facing + time) / 4

  fuel        P(fuel) from the embeddings of the year before the fire (the grids of fire_test_grids.py)
  continuity  share of fuel within 500 m, from 8 x 8 pixel cells over the chunk and its 8 neighbors
  slope       Mapterhorn z12, median then Gaussian cleaned, Horn; degrees / 30, capped at 1
  facing      south-west most, in step with the slope up to 10 degrees
  time        years since the last fire on record before the fire's year, / 20, capped at 1; 1 where none

Fires: the 200 NIFC WFIGS perimeters and their edge chunks of the fire test (data/fire_test/edges.parquet).
Controls, all the fire's own year (ground only differs, the skill event-vs-matched-controls):
  outside     the edge chunk's own unburned side (0 to ~2 km from the line)
  5-10 km, 20-30 km   one chunk each per fire, no perimeter of that year touching it

Steps (each resumable, data in data/risk_backtest/):
  need     control chunks, and every fuel grid the 3 x 3 neighborhoods need -> grids_needed.parquet
           (then: uv run scripts/fire_test_grids.py --need data/risk_backtest/grids_needed.parquet --no-lf)
  terrain  slope and aspect per chunk
  history  NIFC history (< 2020) per 0.25 degree cell, cached
  score    the parts per pixel, AUC per part and control kind -> results/risk_backtest.json
  need-starts, score-starts   the same for the WFIGS starts of the fire test (starts.parquet), as that test scored
           them (at the pixel, within 30 m and 100 m) -> results/risk_backtest_starts.json
  check-need, check   the Python value now at a few spots, to set beside the map's cards (check-need lists the
           2025 grids; make them with fire_test_grids.py --need data/risk_backtest/check_needed.parquet --no-lf)

RAM: under 2 GB for every step (one chunk neighborhood at a time; tiles and history cached on disk).

    uv run scripts/risk_backtest.py need|terrain|history|score|check-need|check
"""
import argparse
import io
import json
import math
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuel_common import CHUNK, MOS_RES, MOS_X0, MOS_Y0  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FT = ROOT / "data" / "fire_test"
D = ROOT / "data" / "risk_backtest"
RINGS = {"5-10 km": (5, 10), "20-30 km": (20, 30)}


def chunk_box(cy, cx):
    lon0, lat1 = MOS_X0 + cx * CHUNK * MOS_RES, MOS_Y0 - cy * CHUNK * MOS_RES
    return lon0, lat1 - CHUNK * MOS_RES, lon0 + CHUNK * MOS_RES, lat1


def chunk_of(lon, lat):
    return int((MOS_Y0 - lat) / MOS_RES // CHUNK), int((lon - MOS_X0) / MOS_RES // CHUNK)


def offset(lon, lat, km, bearing):
    dlat = km * math.cos(bearing) / 111.32
    dlon = km * math.sin(bearing) / (111.32 * math.cos(math.radians(lat)))
    return lon + dlon, lat + dlat


# ---------------------------------------------------------------- need

def need(seed=0):
    rng = np.random.default_rng(seed)
    D.mkdir(parents=True, exist_ok=True)
    edges = pd.read_parquet(FT / "edges.parquet")
    per = pd.read_parquet(FT / "perimeters.parquet")
    geoms = shapely.from_wkb(per.geom.to_numpy())
    tree = shapely.STRtree(geoms)
    win = pd.read_parquet(ROOT / "data" / "fuel_ca_windows.parquet")
    train = set(zip(win.cy.astype(int), win.cx.astype(int)))
    used = set(zip(edges.cy, edges.cx))
    rows = []
    for fid, e in edges.groupby("fire_id"):
        year = int(e.year.iloc[0])
        c = shapely.centroid(geoms[per.index[per.fire_id == fid][0]])
        for ring, (lo, hi) in RINGS.items():
            for _ in range(50):
                lon, lat = offset(c.x, c.y, rng.uniform(lo, hi), rng.uniform(0, 2 * np.pi))
                cy, cx = chunk_of(lon, lat)
                if (cy, cx) in train or (cy, cx) in used:
                    continue
                # no perimeter of the fire's year touching the chunk
                hits = tree.query(shapely.box(*chunk_box(cy, cx)), predicate="intersects")
                if (per.year.to_numpy()[hits] == year).any():
                    continue
                rows.append(dict(fire_id=fid, year=year, ring=ring, cy=cy, cx=cx))
                used.add((cy, cx))
                break
    ctrl = pd.DataFrame(rows)
    ctrl.to_parquet(D / "controls.parquet", index=False)
    chunks = pd.concat([edges[["cy", "cx", "year"]], ctrl[["cy", "cx", "year"]]])
    need = {(cy + dy, cx + dx, y - 1) for cy, cx, y in chunks.itertuples(index=False) for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
    n = pd.DataFrame(sorted(need), columns=["cy", "cx", "emb_year"])
    n.to_parquet(D / "grids_needed.parquet", index=False)
    have = sum((FT / "grids" / f"{a}_{b}_{c}.npz").exists() for a, b, c in need)
    print(f"controls {len(ctrl)} ({ctrl.ring.value_counts().to_dict()}) for {ctrl.fire_id.nunique()} fires")
    print(f"grids {len(n)}, {have} on disk, {len(n) - have} to make")


def need_starts():
    """The WFIGS starts of the fire test (starts.parquet): their chunk and their far control chunk, 3 x 3 each."""
    st = pd.read_parquet(FT / "starts.parquet")
    chunks = pd.concat([st[["cy", "cx", "year"]], st[["ctrl_cy", "ctrl_cx", "year"]].rename(columns={"ctrl_cy": "cy", "ctrl_cx": "cx"})])
    need = {(cy + dy, cx + dx, y - 1) for cy, cx, y in chunks.itertuples(index=False) for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
    n = pd.DataFrame(sorted(need), columns=["cy", "cx", "emb_year"])
    n.to_parquet(D / "grids_needed_starts.parquet", index=False)
    have = sum((FT / "grids" / f"{a}_{b}_{c}.npz").exists() for a, b, c in need)
    print(f"starts {len(st)}: grids {len(n)}, {have} on disk, {len(n) - have} to make")


# ---------------------------------------------------------------- the older years (embeddings 2017 on)

HIST = ["fuel_mean", "fuel_min", "fuel_trend", "chg_max", "yrs_since_chg", "chg_total", "chg_recent3"]
HIST_NOTE = {
    "fuel_mean": "mean P(fuel) over the years 2019 to the year before the fire",
    "fuel_min": "lowest P(fuel) over those years",
    "fuel_trend": "P(fuel) per year, least-squares slope (needs 2 years: none for 2020 fires)",
    "chg_max": "largest year-to-year embedding change (cosine distance), 2017 to the year before the fire",
    "yrs_since_chg": "years from that largest change to the fire year",
    "chg_total": "embedding change from 2017 to the year before the fire",
    "chg_recent3": "largest year-to-year change within the last 3 years before the fire (a window every fire has)",
}


def hist_chunks():
    """(cy, cx, fire year) of every scored chunk: fire areas, their controls, starts and their far chunks."""
    a = scored_chunks()[["cy", "cx", "year"]]
    st = pd.read_parquet(FT / "starts.parquet")
    b = pd.concat([st[["cy", "cx", "year"]], st[["ctrl_cy", "ctrl_cx", "year"]].rename(columns={"ctrl_cy": "cy", "ctrl_cx": "cx"})])
    return sorted(set(map(tuple, pd.concat([a, b]).itertuples(index=False))))


def hist_one(arr, head, cy, cx, year):
    """The older-years features for one chunk, from the embeddings of 2017 to year - 1; per-year P(fuel) grids kept."""
    from fire_test_grids import YEAR0, dequant, p_fuel_grid
    last = year - 1
    years = list(range(YEAR0, last + 1))
    blocks = {}
    for y in years:  # each year read once
        for attempt in range(4):
            try:
                blocks[y] = arr[y - YEAR0, :, cy * CHUNK:(cy + 1) * CHUNK, cx * CHUNK:(cx + 1) * CHUNK]
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 + 3 * attempt)
    # P(fuel) per year, the head's own three-year input; saved to the shared grid cache when missing
    ps = []
    for y in range(YEAR0 + 2, last + 1):
        f = FT / "grids" / f"{cy}_{cx}_{y}.npz"
        if f.exists():
            q = np.load(f)["p"]
            p = np.where(q == 255, np.nan, q / 254.0).astype(np.float32)
        else:
            p = p_fuel_grid(head, [blocks[y - 2], blocks[y - 1], blocks[y]])
            p = np.where(np.isnan(p), np.nan, np.round(np.nan_to_num(p) * 254) / 254).astype(np.float32)
            q = np.where(np.isnan(p), 255, np.round(np.nan_to_num(p) * 254)).astype(np.uint8)
            tmp = f.with_name(f.stem + f".tmp{os.getpid()}.npz")
            np.savez_compressed(tmp, p=q)
            os.replace(tmp, f)
        ps.append(p)
    ps = np.stack(ps)  # (n years, 256, 256)
    import warnings
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["fuel_mean"] = np.nanmean(ps, 0)
        out["fuel_min"] = np.nanmin(ps, 0)
        if len(ps) >= 2:
            t = np.arange(len(ps), dtype=np.float32) - (len(ps) - 1) / 2
            ok = np.isfinite(ps).all(0)
            out["fuel_trend"] = np.where(ok, np.tensordot(t, np.nan_to_num(ps), 1) / (t * t).sum(), np.nan)
        else:
            out["fuel_trend"] = np.full((CHUNK, CHUNK), np.nan, np.float32)

        # unit embeddings per year, NaN where no data
        def unit(b):
            v = dequant(b).reshape(64, -1)
            n = np.linalg.norm(v, axis=0)
            v = v / np.where(n > 0, n, 1)
            v[:, (b == -128).all(0).ravel()] = np.nan
            return v

        u = {y: unit(blocks[y]) for y in years}
        d = np.stack([1 - (u[y] * u[y + 1]).sum(0) for y in years[:-1]])  # (pairs, 65536): change into year y + 1
        k = np.nanargmax(np.where(np.isfinite(d), d, -1), 0)
        out["chg_max"] = np.nanmax(d, 0).reshape(CHUNK, CHUNK)
        out["yrs_since_chg"] = (year - (np.array(years[1:])[k])).reshape(CHUNK, CHUNK).astype(np.float32)
        out["yrs_since_chg"][~np.isfinite(out["chg_max"])] = np.nan
        out["chg_total"] = (1 - (u[years[0]] * u[last]).sum(0)).reshape(CHUNK, CHUNK)
        out["chg_recent3"] = np.nanmax(d[-2:], 0).reshape(CHUNK, CHUNK)  # pairs into the last two years: the last 3 years
    f = D / "hist" / f"{cy}_{cx}_{year}.npz"
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.stem + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **{k: v.astype(np.float16) for k, v in out.items()})
    os.replace(tmp, f)


def hist_all(workers=6, max_gb=5.0):
    import psutil
    from concurrent.futures import ThreadPoolExecutor
    from fire_test_grids import Head, open_aef
    todo = [c for c in hist_chunks() if not (D / "hist" / f"{c[0]}_{c[1]}_{c[2]}.npz").exists()]
    print(f"{len(todo)} chunks to read (all years up to the one before the fire)", flush=True)
    arr, head = open_aef(), Head()
    proc = psutil.Process(os.getpid())
    t0, peak = time.time(), 0.0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, _ in enumerate(ex.map(lambda c: hist_one(arr, head, *c), todo)):
            gb = proc.memory_info().rss / 1e9
            peak = max(peak, gb)
            if gb > max_gb:
                print(f"STOP: {gb:.1f} GB > {max_gb}; rerun to resume", flush=True)
                os._exit(2)
            if n % 20 == 0 or n == len(todo) - 1:
                el = time.time() - t0
                print(f"{n + 1}/{len(todo)} {el / (n + 1):.2f} s/chunk, eta {el / (n + 1) * (len(todo) - n - 1) / 60:.1f} min, "
                      f"rss {gb:.2f} GB (peak {peak:.2f})", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min, peak rss {peak:.2f} GB")


def hist_of(cy, cx, year):
    f = D / "hist" / f"{cy}_{cx}_{year}.npz"
    if not f.exists():
        return {k: np.full((CHUNK, CHUNK), np.nan, np.float32) for k in HIST}
    z = np.load(f)
    return {k: z[k].astype(np.float32) for k in HIST}


# ---------------------------------------------------------------- terrain (terrain.ts)

MAPTERHORN = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/mapterhorn/mapterhorn/planet.pmtiles"
TZ = 12
TN = 2 ** TZ
EARTH = 40075016.686
MEDIAN, SIGMA = 5, 2
GR = 3 * SIGMA
MARGIN = (MEDIAN - 1) // 2 + GR + 2


def _fx(lon):
    return (lon + 180) / 360 * TN


def _fy(lat):
    r = math.radians(lat)
    return (1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2 * TN


_session = None


@lru_cache(maxsize=512)
def _range(offset, length):
    import requests
    global _session
    _session = _session or requests.Session()
    for attempt in range(5):
        try:
            r = _session.get(MAPTERHORN, headers={"Range": f"bytes={offset}-{offset + length - 1}"}, timeout=60)
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 + 3 * attempt)


_reader = None


def mapterhorn_tile(x, y):
    """Elevation (512, 512) float32 of a z12 tile, or None; the WebP bytes cached on disk."""
    from PIL import Image
    from pmtiles.reader import Reader
    global _reader
    f = D / "mapterhorn" / f"{x}_{y}.webp"
    if f.exists():
        b = f.read_bytes()
    else:
        _reader = _reader or Reader(_range)
        b = _reader.get(TZ, x, y) or b""
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b)
    if not b:
        return None
    px = np.asarray(Image.open(io.BytesIO(b)).convert("RGB")).astype(np.float64)
    return (px[..., 0] * 256 + px[..., 1] + px[..., 2] / 256 - 32768).astype(np.float32)


def _nan_window(a, k, axis=None):
    """Sliding windows of size k (odd) centered on each cell, NaN past the edges: (H, W, k*k) or along one axis."""
    from numpy.lib.stride_tricks import sliding_window_view
    h = k // 2
    if axis is None:
        p = np.pad(a, h, constant_values=np.nan)
        return sliding_window_view(p, (k, k)).reshape(a.shape[0], a.shape[1], k * k)
    pad = [(0, 0), (0, 0)]
    pad[axis] = (h, h)
    return sliding_window_view(np.pad(a, pad, constant_values=np.nan), k, axis=axis)


def chunk_terrain(cy, cx):
    """slope (degrees, NaN where no terrain) and aspect (degrees from north the ground faces, NaN on flat ground)."""
    lon0, lat0, lon1, lat1 = chunk_box(cy, cx)
    size = 512
    gx0, gx1 = math.floor(_fx(lon0) * size) - MARGIN, math.ceil(_fx(lon1) * size) + MARGIN
    gy0, gy1 = math.floor(_fy(lat1) * size) - MARGIN, math.ceil(_fy(lat0) * size) + MARGIN
    W, H = gx1 - gx0 + 1, gy1 - gy0 + 1
    raw = np.full((H, W), np.nan, np.float32)
    for ty in range(gy0 // size, gy1 // size + 1):
        for tx in range(gx0 // size, gx1 // size + 1):
            t = mapterhorn_tile(tx, ty)
            if t is None:
                continue
            ya, yb = max(gy0, ty * size), min(gy1, ty * size + size - 1)
            xa, xb = max(gx0, tx * size), min(gx1, tx * size + size - 1)
            raw[ya - gy0:yb - gy0 + 1, xa - gx0:xb - gx0 + 1] = t[ya - ty * size:yb - ty * size + 1, xa - tx * size:xb - tx * size + 1]
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(_nan_window(raw, MEDIAN), axis=2).astype(np.float32)
        kern = np.exp(-((np.arange(2 * GR + 1) - GR) ** 2) / (2 * SIGMA * SIGMA)).astype(np.float32)

        def blur(src, axis):
            w = _nan_window(src, 2 * GR + 1, axis=1 if axis == 0 else 0)
            ok = np.isfinite(w)
            ws = (ok * kern).sum(-1)
            return np.where(ws > 0, (np.where(ok, w, 0) * kern).sum(-1) / np.where(ws > 0, ws, 1), np.nan).astype(np.float32)

        elev = blur(blur(med, 0), 1)  # across, then down
    cell = EARTH / (TN * size) * math.cos(math.radians((lat0 + lat1) / 2))
    e = lambda dx, dy: elev[1 + dy:H - 1 + dy, 1 + dx:W - 1 + dx]  # noqa: E731
    ex = np.full((H, W), np.nan, np.float32)
    es = np.full((H, W), np.nan, np.float32)
    ex[1:-1, 1:-1] = (e(1, -1) + 2 * e(1, 0) + e(1, 1) - (e(-1, -1) + 2 * e(-1, 0) + e(-1, 1))) / (8 * cell)
    es[1:-1, 1:-1] = (e(-1, 1) + 2 * e(0, 1) + e(1, 1) - (e(-1, -1) + 2 * e(0, -1) + e(1, -1))) / (8 * cell)
    u = np.array([_fx(lon0 + (c + 0.5) * MOS_RES) * size - 0.5 - gx0 for c in range(CHUNK)])
    v = np.array([_fy(lat1 - (r + 0.5) * MOS_RES) * size - 0.5 - gy0 for r in range(CHUNK)])
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = (u - x0)[None, :], (v - y0)[:, None]
    Y0, X0 = y0[:, None], x0[None, :]

    def lerp(a):
        return (a[Y0, X0] * (1 - fu) + a[Y0, X0 + 1] * fu) * (1 - fv) + (a[Y0 + 1, X0] * (1 - fu) + a[Y0 + 1, X0 + 1] * fu) * fv

    dzdx, dzds = lerp(ex), lerp(es)
    g = np.hypot(dzdx, dzds)
    slope = np.degrees(np.arctan(g)).astype(np.float32)
    with np.errstate(invalid="ignore"):
        aspect = np.where((g < 1e-3) | np.isnan(g), np.nan, (np.degrees(np.arctan2(-dzdx, dzds)) + 360) % 360).astype(np.float32)
    return slope, aspect


def scored_chunks():
    edges = pd.read_parquet(FT / "edges.parquet").assign(kind="edge")
    ctrl = pd.read_parquet(D / "controls.parquet").rename(columns={"ring": "kind"})
    return pd.concat([edges[["fire_id", "year", "cy", "cx", "kind"]], ctrl[["fire_id", "year", "cy", "cx", "kind"]]], ignore_index=True)


def terrain_of(cy, cx):
    f = D / "terrain" / f"{cy}_{cx}.npz"
    if not f.exists():
        s, a = chunk_terrain(cy, cx)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, slope=s, aspect=a)
    z = np.load(f)
    return z["slope"], z["aspect"]


def terrain_all(chunks):
    todo = [(cy, cx) for cy, cx in chunks if not (D / "terrain" / f"{cy}_{cx}.npz").exists()]
    t0 = time.time()
    for n, (cy, cx) in enumerate(todo):
        terrain_of(cy, cx)
        if n % 50 == 0 or n == len(todo) - 1:
            print(f"terrain {n + 1}/{len(todo)} {(time.time() - t0) / (n + 1):.2f} s/chunk", flush=True)


# ---------------------------------------------------------------- history (history.ts)

HISTORY = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/InterAgencyFirePerimeterHistory_All_Years_View/"
           "FeatureServer/0/query")
HCELL = 0.25
WFIGS_FROM = 2020


def history_cell(i, j):
    """The history service's fires (< 2020, >= 100 ac) in one 0.25 degree cell, as the map reads them; cached."""
    import requests
    f = D / "history" / f"{i}_{j}.json"
    if f.exists():
        return json.loads(f.read_text())
    xmin, ymin = -180 + i * HCELL, -90 + j * HCELL
    out, offset = [], 0
    while True:
        q = dict(where=f"GIS_ACRES >= 100 AND FIRE_YEAR_INT < {WFIGS_FROM}",
                 geometry=json.dumps(dict(xmin=xmin, ymin=ymin, xmax=xmin + HCELL, ymax=ymin + HCELL, spatialReference=dict(wkid=4326))),
                 geometryType="esriGeometryEnvelope", inSR="4326", spatialRel="esriSpatialRelIntersects",
                 outFields="OBJECTID,INCIDENT,FIRE_YEAR_INT,GIS_ACRES,UNQE_FIRE_ID", outSR="4326", maxAllowableOffset="0.0003",
                 geometryPrecision="5", resultOffset=str(offset), resultRecordCount="1000", f="geojson")
        for attempt in range(5):
            try:
                r = requests.post(HISTORY, data=q, timeout=180)
                r.raise_for_status()
                fc = r.json()
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(3 + 5 * attempt)
        out += fc["features"]
        if not ((fc.get("properties") or {}).get("exceededTransferLimit") or fc.get("exceededTransferLimit")) or not fc["features"]:
            break
        offset += len(fc["features"])
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    return out


def cells_of(chunks):
    cells = set()
    for cy, cx in chunks:
        lon0, lat0, lon1, lat1 = chunk_box(cy, cx)
        for i in range(int((lon0 + 180) // HCELL), int((lon1 + 180) // HCELL) + 1):
            for j in range(int((lat0 + 90) // HCELL), int((lat1 + 90) // HCELL) + 1):
                cells.add((i, j))
    return sorted(cells)


class Fires:
    """Every fire on record (history < 2020 + WFIGS 2020 on), deduplicated as history.ts does, with a tree."""

    repaired = 0

    def valid(self, g):
        """make_valid, falling back to the 'structure' method, then buffer(0), for outlines the default rejects
        (GEOS raises "Overlay input is mixed-dimension" on a few NIFC history records)."""
        try:
            return shapely.make_valid(g)
        except shapely.errors.GEOSException:
            self.repaired += 1
            try:
                return shapely.make_valid(g, method="structure")
            except shapely.errors.GEOSException:
                return g.buffer(0)

    def __init__(self, cells):
        from shapely.geometry import shape
        names, years, geoms, seen = [], [], [], set()
        per = pd.read_parquet(FT / "perimeters.parquet")
        for r in per.itertuples():
            names.append(r.name or "unnamed"), years.append(int(r.year)), geoms.append(shapely.from_wkb(r.geom))
        for i, j in cells:
            for f in history_cell(i, j):
                p = f["properties"]
                fid = p.get("UNQE_FIRE_ID") or f"{(p.get('INCIDENT') or '').upper()}|{p.get('FIRE_YEAR_INT')}|{round(p.get('GIS_ACRES') or 0)}"
                if not f.get("geometry") or p.get("FIRE_YEAR_INT") is None or fid in seen:
                    continue
                seen.add(fid)
                names.append(p.get("INCIDENT") or "unnamed"), years.append(int(p["FIRE_YEAR_INT"]))
                geoms.append(self.valid(shape(f["geometry"])))
        if self.repaired:
            print(f"history: {self.repaired} outlines GEOS's default make_valid rejects, repaired another way")
        self.names, self.years, self.geoms = names, np.array(years), np.array(geoms, dtype=object)
        self.tree = shapely.STRtree(self.geoms)

    def last(self, cy, cx, before):
        """Last burn year per pixel from fires of years < before (0 = none), and the years used (the leak check)."""
        from rasterio.features import rasterize
        from rasterio.transform import from_origin
        lon0, lat0, lon1, lat1 = chunk_box(cy, cx)
        hits = self.tree.query(shapely.box(lon0, lat0, lon1, lat1), predicate="intersects")
        hits = hits[self.years[hits] < before]
        out = np.zeros((CHUNK, CHUNK), np.int16)
        tr = from_origin(lon0, lat1, MOS_RES, MOS_RES)
        for k in hits[np.argsort(self.years[hits])]:
            m = rasterize([(self.geoms[k], 1)], out_shape=(CHUNK, CHUNK), transform=tr, fill=0).astype(bool)
            out[m] = self.years[k]
        return out, set(self.years[hits].tolist())


# ---------------------------------------------------------------- the value (factors.ts, score.ts)

COARSE = 8
NC = CHUNK // COARSE
M_LAT = 111_320
CONT_RADIUS_M = 500
SLOPE_FULL, TIME_FULL, FACE_MOST, FACE_SLOPE = 30, 20, 225, 10


def grid(cy, cx, emb_year):
    f = FT / "grids" / f"{cy}_{cx}_{emb_year}.npz"
    if not f.exists():
        return None
    q = np.load(f)["p"]
    return np.where(q == 255, np.nan, q / 254.0).astype(np.float32)


def coarse_of(g):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(g.reshape(NC, COARSE, NC, COARSE), axis=(1, 3))


def continuity(cy, cx, emb_year):
    """Share of fuel within 500 m per pixel of the chunk, from its 3 x 3 neighborhood, and the chunk's own P(fuel)."""
    from scipy.ndimage import correlate
    mos = np.full((3 * NC, 3 * NC), np.nan, np.float32)
    center = None
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            g = grid(cy + dy, cx + dx, emb_year)
            if g is None:
                continue
            if dy == 0 and dx == 0:
                center = g
            mos[(dy + 1) * NC:(dy + 2) * NC, (dx + 1) * NC:(dx + 2) * NC] = coarse_of(g)
    if center is None:
        return None, None
    lat = (chunk_box(cy, cx)[1] + chunk_box(cy, cx)[3]) / 2
    cell_h = COARSE * MOS_RES * M_LAT
    cell_w = cell_h * math.cos(math.radians(lat))
    ri, rj = math.ceil(CONT_RADIUS_M / cell_h), math.ceil(CONT_RADIUS_M / cell_w)
    di, dj = np.mgrid[-ri:ri + 1, -rj:rj + 1]
    k = (np.hypot(di * cell_h, dj * cell_w) <= CONT_RADIUS_M).astype(np.float32)
    ok = np.isfinite(mos)
    s = correlate(np.where(ok, mos, 0), k, mode="constant", cval=0)
    n = correlate(ok.astype(np.float32), k, mode="constant", cval=0)
    near = np.where(n > 0, s / np.where(n > 0, n, 1), np.nan)
    G = near[NC - 1:2 * NC + 1, NC - 1:2 * NC + 1]  # the chunk's cells with a one-cell margin
    u = (np.arange(CHUNK) + 0.5) / COARSE - 0.5 + 1
    i0 = np.floor(u).astype(int)
    f = u - i0
    s = np.zeros((CHUNK, CHUNK))
    w = np.zeros((CHUNK, CHUNK))
    for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
        x = G[(i0 + a)[:, None], (i0 + b)[None, :]]
        wt = (f if a else 1 - f)[:, None] * (f if b else 1 - f)[None, :]
        okx = np.isfinite(x) & (wt != 0)
        s += np.where(okx, x * wt, 0)
        w += np.where(okx, wt, 0)
    cont = np.where(w > 0, s / np.where(w > 0, w, 1), np.nan).astype(np.float32)
    return center, cont


def parts(fuel, cont, slope_deg, aspect_deg, last, now):
    c = np.clip(cont, 0, 1)
    sl = np.where(np.isnan(slope_deg), 0, np.clip(slope_deg / SLOPE_FULL, 0, 1))
    face = np.where(np.isnan(aspect_deg), 0.5, (1 + np.cos(np.radians(aspect_deg - FACE_MOST))) / 2)
    facing = 0.5 + (face - 0.5) * np.where(np.isnan(slope_deg), 0, np.clip(slope_deg / FACE_SLOPE, 0, 1))
    t = np.where(last > 0, np.clip((now - last) / TIME_FULL, 0, 1), 1.0)
    avg = (c + sl + facing + t) / 4
    ok = np.isfinite(fuel) & np.isfinite(cont)
    nan = lambda a: np.where(ok, a, np.nan).astype(np.float32)  # noqa: E731
    return dict(risk=nan(fuel * avg), fuel=nan(fuel), continuity=nan(c), slope=nan(sl), facing=nan(facing), time=nan(t), avg4=nan(avg))


FEATURES = ["risk", "fuel", "continuity", "slope", "facing", "time", "avg4"]


def with_hist(P, cy, cx, year):
    """The value's parts and the older-years features (hist step), where the value is defined."""
    ok = np.isfinite(P["risk"])
    return {**P, **{k: np.where(ok, v, np.nan).astype(np.float32) for k, v in hist_of(cy, cx, year).items()}}


# ---------------------------------------------------------------- score

BOOT = 1000
N_SAMPLE = 3000


def auc(pos, neg):
    """Mann-Whitney AUC, ties counted half."""
    from scipy.stats import rankdata
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot_ci(v, rng):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, np.nan, np.nan, 0
    b = [rng.choice(v, len(v)).mean() for _ in range(BOOT)]
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(len(v))


def score():
    import psutil
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from tabulate import tabulate
    rng = np.random.default_rng(0)
    ch = scored_chunks()
    per = pd.read_parquet(FT / "perimeters.parquet").set_index("fire_id")
    fires = Fires(cells_of(set(zip(ch.cy, ch.cx))))
    proc = psutil.Process(os.getpid())
    rows, case_px, ctrl_px, leaks = [], {}, {}, 0
    t0 = time.time()
    for n, r in enumerate(ch.itertuples()):
        fuel, cont = continuity(r.cy, r.cx, r.year - 1)
        if fuel is None:
            continue
        slope, aspect = terrain_of(r.cy, r.cx)
        last, used = fires.last(r.cy, r.cx, before=r.year)
        if used and max(used) >= r.year:
            leaks += 1
        P = with_hist(parts(fuel, cont, slope, aspect, last, now=r.year), r.cy, r.cx, r.year)
        ok = np.isfinite(P["risk"])
        d = dict(fire_id=r.fire_id, year=r.year, cy=r.cy, cx=r.cx, kind=r.kind, valid=int(ok.sum()),
                 burned_before=float((last[ok] > 0).mean()) if ok.any() else np.nan)
        if r.kind == "edge":
            lon0, lat0, lon1, lat1 = chunk_box(r.cy, r.cx)
            inside = rasterize([(shapely.from_wkb(per.loc[r.fire_id, "geom"]), 1)], out_shape=(CHUNK, CHUNK),
                               transform=from_origin(lon0, lat1, MOS_RES, MOS_RES), fill=0).astype(bool)
            if inside[ok].all() or not inside[ok].any():
                continue
            for f in FEATURES + HIST:
                d[f"auc_{f}"] = auc(P[f][inside & ok], P[f][~inside & ok])
                d[f"in_{f}"] = float(np.mean(P[f][inside & ok]))
                d[f"out_{f}"] = float(np.mean(P[f][~inside & ok]))
            idx = np.flatnonzero(inside & ok)
            pick = rng.choice(idx, min(N_SAMPLE, len(idx)), replace=False)
            case_px.setdefault(r.fire_id, []).append(np.stack([P[f].ravel()[pick] for f in FEATURES + HIST], 1))
        else:
            idx = np.flatnonzero(ok)
            if not len(idx):
                continue
            pick = rng.choice(idx, min(N_SAMPLE, len(idx)), replace=False)
            ctrl_px[(r.fire_id, r.kind)] = np.stack([P[f].ravel()[pick] for f in FEATURES + HIST], 1)
            for f in FEATURES + HIST:
                d[f"mean_{f}"] = float(np.mean(P[f][ok]))
        rows.append(d)
        if n % 50 == 0:
            print(f"score {n + 1}/{len(ch)} {(time.time() - t0) / (n + 1):.2f} s/chunk, rss {proc.memory_info().rss / 1e9:.2f} GB", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(D / "chunk_scores.parquet", index=False)
    out = dict(leaks=leaks, chunks=int(len(df)), notes=dict(avg4="the four parts' average without the fuel factor", **HIST_NOTE))
    print(f"\nleak check: {leaks} chunks used a fire of the test year or later (must be 0)")

    # 1. inside vs outside the line, per fire (mean over its chunks), bootstrap over fires
    e = df[df.kind == "edge"]
    pf = e.groupby("fire_id").mean(numeric_only=True)
    tab, out["inside_vs_outside"] = [], {}
    for f in FEATURES + HIST:
        m, lo, hi, nf = boot_ci(pf[f"auc_{f}"], rng)
        out["inside_vs_outside"][f] = dict(auc=m, lo=lo, hi=hi, fires=nf, mean_in=float(pf[f"in_{f}"].mean()), mean_out=float(pf[f"out_{f}"].mean()))
        tab.append([f, nf, f"{m:.3f}", f"[{lo:.3f}, {hi:.3f}]", f"{pf[f'in_{f}'].mean():.3f}", f"{pf[f'out_{f}'].mean():.3f}"])
    print("\n== burned vs unburned ground in the same chunk (fire line crosses it); AUC per fire, mean over fires")
    print(tabulate(tab, headers=["part", "fires", "AUC", "95 %", "mean inside", "mean outside"]))

    # 2. burned ground vs same-year chunks 5-10 km and 20-30 km away
    out["burned_vs_far"] = {}
    for ring in RINGS:
        tab, out["burned_vs_far"][ring] = [], {}
        per_fire = {f: [] for f in FEATURES + HIST}
        for (fid, kind), cpx in ctrl_px.items():
            if kind != ring or fid not in case_px:
                continue
            cs = np.concatenate(case_px[fid])
            for k, f in enumerate(FEATURES + HIST):
                per_fire[f].append(auc(cs[:, k], cpx[:, k]))
        for f in FEATURES + HIST:
            m, lo, hi, nf = boot_ci(per_fire[f], rng)
            out["burned_vs_far"][ring][f] = dict(auc=m, lo=lo, hi=hi, fires=nf)
            tab.append([f, nf, f"{m:.3f}", f"[{lo:.3f}, {hi:.3f}]"])
        print(f"\n== burned ground vs a same-year chunk {ring} away (no fire that year); AUC per fire, mean over fires")
        print(tabulate(tab, headers=["part", "fires", "AUC", "95 %"]))

    # 3. per fire year, inside vs outside: the value and fuel alone
    tab, out["by_year"] = [], {}
    for y, g in pf.groupby("year"):
        rv, fv = boot_ci(g.auc_risk, rng), boot_ci(g.auc_fuel, rng)
        out["by_year"][int(y)] = dict(risk=rv[:3], fuel=fv[:3], fires=rv[3])
        tab.append([int(y), rv[3], f"{rv[0]:.3f} [{rv[1]:.3f}, {rv[2]:.3f}]", f"{fv[0]:.3f} [{fv[1]:.3f}, {fv[2]:.3f}]"])
    print("\n== inside vs outside by fire year")
    print(tabulate(tab, headers=["year", "fires", "value", "fuel alone"]))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "risk_backtest.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote results/risk_backtest.json; peak rss {proc.memory_info().rss / 1e9:.2f} GB")


# ---------------------------------------------------------------- score the ignition sites

N_CTRL = 20
REACH_M = (0, 30, 100)


def start_chunks():
    st = pd.read_parquet(FT / "starts.parquet")
    return sorted(set(zip(st.cy, st.cx)) | set(zip(st.ctrl_cy, st.ctrl_cx)))


def score_starts():
    """Each WFIGS start against same-year controls (near: 300 m to ~2 km in its own chunk; its far chunk), as the
    fuel test scored them: the part at the pixel, and the highest within 30 m and 100 m (a box)."""
    import psutil
    from tabulate import tabulate
    rng = np.random.default_rng(0)
    st = pd.read_parquet(FT / "starts.parquet")
    fires = Fires(cells_of(set(start_chunks())))
    proc = psutil.Process(os.getpid())
    cache, leaks = {}, 0

    def chunk_parts(cy, cx, year):
        nonlocal leaks
        k = (cy, cx, year)
        if k not in cache:
            fuel, cont = continuity(cy, cx, year - 1)
            if fuel is None:
                cache[k] = None
            else:
                slope, aspect = terrain_of(cy, cx)
                last, used = fires.last(cy, cx, before=year)
                leaks += bool(used and max(used) >= year)
                cache[k] = with_hist(parts(fuel, cont, slope, aspect, last, now=year), cy, cx, year)
            if len(cache) > 64:
                cache.pop(next(iter(cache)))
        return cache[k]

    def reach(a, r, c, m, lat):
        if m == 0:
            return a[r, c]
        dy = MOS_RES * M_LAT
        dx = dy * math.cos(math.radians(lat))
        hy, hx = int(round(m / dy)), int(round(m / dx))
        w = a[max(0, r - hy):r + hy + 1, max(0, c - hx):c + hx + 1]
        return float(np.nanmax(w)) if np.isfinite(w).any() else np.nan

    rows = []
    t0 = time.time()
    for sid, r in enumerate(st.itertuples()):
        P = chunk_parts(r.cy, r.cx, r.year)
        if P is None:
            continue
        rr, cc = int((MOS_Y0 - r.lat) / MOS_RES) - r.cy * CHUNK, int((r.lon - MOS_X0) / MOS_RES) - r.cx * CHUNK
        base = dict(sid=sid, year=r.year, size=r.size, cause=r.cause)

        def add(kind, PP, i, j):
            d = dict(base, kind=kind)
            for f in FEATURES + HIST:
                for m in REACH_M:
                    d[f"{f}{m}"] = reach(PP[f], i, j, m, r.lat)
            rows.append(d)

        add("start", P, rr, cc)
        ok = np.isfinite(P["risk"])
        dy = MOS_RES * M_LAT
        ii, jj = np.mgrid[0:CHUNK, 0:CHUNK]
        cand = np.flatnonzero((np.hypot((ii - rr) * dy, (jj - cc) * dy * math.cos(math.radians(r.lat))) >= 300) & ok)
        for k in rng.choice(cand, min(N_CTRL, len(cand)), replace=False):
            add("near", P, *divmod(int(k), CHUNK))
        PC = chunk_parts(r.ctrl_cy, r.ctrl_cx, r.year)
        if PC is not None:
            cand = np.flatnonzero(np.isfinite(PC["risk"]))
            for k in rng.choice(cand, min(N_CTRL, len(cand)), replace=False):
                add(r.ring, PC, *divmod(int(k), CHUNK))
        if sid % 50 == 0:
            print(f"starts {sid + 1}/{len(st)} {(time.time() - t0) / (sid + 1):.2f} s/start, rss {proc.memory_info().rss / 1e9:.2f} GB", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(D / "start_scores.parquet", index=False)
    print(f"\nleak check: {leaks} chunks used a fire of the test year or later (must be 0)")

    def auc_ci(g, col, kind):
        case = g[g.kind == "start"].set_index("sid")[col]
        ctrl = g[g.kind == kind]
        sids = np.intersect1d(ctrl.sid.unique(), case.index)
        if not len(sids):
            return np.nan, np.nan, np.nan, 0
        by = {k: v[col].to_numpy() for k, v in ctrl.groupby("sid")}
        one = lambda pick: auc(case.loc[pick].to_numpy(float), np.concatenate([by[k] for k in pick]).astype(float))  # noqa: E731
        a = one(sids)
        b = [one(rng.choice(sids, len(sids))) for _ in range(300)]
        lo, hi = np.nanpercentile(b, [2.5, 97.5])
        return a, lo, hi, len(sids)

    out = dict(leaks=leaks, starts=int((df.kind == "start").sum()), notes=HIST_NOTE)
    groups = [("all", df)] + [(f"size {k}", df[df["size"] == k]) for k in df["size"].unique()]
    for gname, g in groups:
        out[gname] = {}
        tab = []
        for kind in ["near", "5-10 km", "20-30 km"]:
            for f in FEATURES + HIST if gname == "all" else ["risk", "fuel"]:
                for m in REACH_M:
                    a, lo, hi, n = auc_ci(g, f"{f}{m}", kind)
                    out[gname][f"{kind}|{f}|{m}"] = dict(auc=a, lo=lo, hi=hi, n=n)
                    tab.append([kind, f, m, n, f"{a:.3f}", f"[{lo:.3f}, {hi:.3f}]"])
        print(f"\n== starts ({gname}) vs same-year controls")
        print(tabulate(tab, headers=["control", "part", "reach m", "starts", "AUC", "95 %"]))
    (ROOT / "results" / "risk_backtest_starts.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote results/risk_backtest_starts.json; rss {proc.memory_info().rss / 1e9:.2f} GB")


# ---------------------------------------------------------------- check against the map

SPOTS = {"Paradise": (-121.60, 39.765), "Cohasset": (-121.82, 39.98), "Cohasset canyon": (-121.79, 39.965),
         "Paradise east wall": (-121.565, 39.77)}


def check_need():
    rows = {(cy + dy, cx + dx, 2025) for lon, lat in SPOTS.values() for cy, cx in [chunk_of(lon, lat)]
            for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
    pd.DataFrame(sorted(rows), columns=["cy", "cx", "emb_year"]).to_parquet(D / "check_needed.parquet", index=False)
    print(f"{len(rows)} grids -> data/risk_backtest/check_needed.parquet")


def check():
    """The value now (2025 embeddings, every fire on record, now = this year) at SPOTS, rounded as the map's card."""
    now = 2026
    chunks = {chunk_of(lon, lat) for lon, lat in SPOTS.values()}
    fires = Fires(cells_of(chunks))
    compass = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    for name, (lon, lat) in SPOTS.items():
        cy, cx = chunk_of(lon, lat)
        fuel, cont = continuity(cy, cx, 2025)
        slope, aspect = terrain_of(cy, cx)
        last, _ = fires.last(cy, cx, before=now + 1)
        P = parts(fuel, cont, slope, aspect, last, now)
        r, c = int((MOS_Y0 - lat) / MOS_RES) - cy * CHUNK, int((lon - MOS_X0) / MOS_RES) - cx * CHUNK
        a = aspect[r, c]
        print(f"{name:20s} risk {round(float(P['risk'][r, c]) * 100)}  fuel {round(float(fuel[r, c]) * 100)}%  "
              f"near {round(float(cont[r, c]) * 100)}%  slope {round(float(slope[r, c]))}  "
              f"facing {'flat' if np.isnan(a) else compass[round(a / 45) % 8]}  last {last[r, c] or 'none'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["need", "need-starts", "hist", "terrain", "history", "score", "score-starts", "check-need", "check"])
    a = ap.parse_args()
    if a.step == "need":
        need()
    elif a.step == "need-starts":
        need_starts()
    elif a.step == "hist":
        hist_all()
    elif a.step == "terrain":
        terrain_all(sorted(set(zip(scored_chunks().cy, scored_chunks().cx)) | set(start_chunks())))
    elif a.step == "history":
        cells = cells_of(set(zip(scored_chunks().cy, scored_chunks().cx)) | set(start_chunks()))
        for n, (i, j) in enumerate(cells):
            history_cell(i, j)
            if n % 20 == 0 or n == len(cells) - 1:
                print(f"history cell {n + 1}/{len(cells)}", flush=True)
    elif a.step == "check-need":
        check_need()
    elif a.step == "check":
        check()
    elif a.step == "score-starts":
        score_starts()
    else:
        score()


if __name__ == "__main__":
    main()
