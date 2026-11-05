# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "pillow", "shapely", "pyogrio", "rasterio", "scikit-learn", "scipy", "psutil", "h5py", "obstore"]
# ///
"""Follow each GOES event over its first day: the fuel under every pixel seen so far, in the ring around it, and
just ahead of the newest pixels, read at fixed times after the first scan; with and without terrain parallax.

goes_fuel.py reads fuel at the first box only. Here every scan of the event's first HOURS counts. The 10 m fuel
grids are the same (goes_fuel.py grids, one folder per embedding year); only more chunks are needed.

Fuel is summed per GOES cell once: each 10 m pixel goes to the GOES cell its center falls in (rounded fractional
row and column, the same "in" as goes_fuel.py: within half a pixel of a center). Any set of cells then has its
mean fuel from the cell sums, and its ring is the 3 x 3 dilation minus the set (0.5 to 1.5 pixels out).

Terrain: GOES places a hot pixel on the ellipsoid, but the ground can be 1 to 3 km up, and the satellite looks
at California about 50 degrees off vertical, so a ridge shows up 1 to 3 km farther from the satellite than it
is. With terrain on, each 10 m pixel is projected to the GOES grid from its true height (Mapterhorn, zoom 8,
about 230 m), so the fuel under a hot pixel is read where the line of sight meets the ground.

  pixels  every fire pixel of each event in its first HOURS -> data/goes/track_px_<year>.parquet (events rebuilt with
          goes_events.py's own steps; the first boxes are checked against event_px_<year>.parquet)
  plan    the chunks those cells and rings need, with room for the terrain shift -> data/goes/track_chunks_<year>.parquet
          (then: goes_fuel.py grids --year Y --extra-chunks data/goes/track_chunks_Y.parquet)
  cells   per GOES cell, fuel sums from every chunk, flat and terrain -> data/goes/cell_fuel_<year>.parquet
  steps   per event and time after first scan: GOES so far, fuel in / around / ahead -> data/goes/track_fuel_<year>.parquet
  score   checks and AUCs (pooled over --years) -> results/goes_fuel_track_<years>.json

RAM: cells holds one chunk per thread (about 50 MB each) and the cell sums (small); under 2 GB at 8 threads.

    uv run scripts/goes_fuel_track.py pixels --year 2025
    uv run scripts/goes_fuel_track.py plan --year 2025
    uv run scripts/goes_fuel.py grids --year 2025 --workers 32 --max-gb 8 --extra-chunks data/goes/track_chunks_2025.parquet
    uv run scripts/goes_fuel_track.py cells --year 2025
    uv run scripts/goes_fuel_track.py steps --year 2025
    uv run scripts/goes_fuel_track.py score --years 2025
"""
import argparse
import io
import json
import math
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuel_common import CHUNK, MOS_RES, MOS_X0, MOS_Y0  # noqa: E402
from goes_fuel import STEP, grid_params, grids_dir  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
G = ROOT / "data" / "goes"
DEM_DIR = ROOT / "data" / "mapterhorn"
HOURS = 24
HORIZONS = (0, 1, 3, 6, 12, 24)  # hours after the first scan
H_MAX = 4500.0  # m, above anything in California, for chunk planning
READ = ("knn", "head", "lf")


def rss():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


# ---------- terrain ----------

class Dem:
    """Mapterhorn Terrarium tiles at zoom DZ, nearest pixel; tiles cached in data/mapterhorn/ (shared with sample_terrain.py)."""
    DZ, TILE = 8, 512

    def __init__(self):
        self.tiles, self.lock = {}, threading.Lock()
        DEM_DIR.mkdir(parents=True, exist_ok=True)

    def tile(self, x, y):
        k = (x, y)
        with self.lock:
            if k in self.tiles:
                return self.tiles[k]
            klock = self.tiles.setdefault(("lock", k), threading.Lock())
        with klock:  # one thread fetches and decodes a tile; the others wait for it
            if k in self.tiles:
                return self.tiles[k]
            from PIL import Image
            p = DEM_DIR / f"{self.DZ}_{x}_{y}.webp"
            if not p.exists():
                for attempt in range(4):
                    try:
                        req = urllib.request.Request(f"https://tiles.mapterhorn.com/{self.DZ}/{x}/{y}.webp",
                                                     headers={"User-Agent": "aef-hls-s1-fire-knn"})
                        tmp = p.with_suffix(f".tmp{threading.get_ident()}")
                        tmp.write_bytes(urllib.request.urlopen(req, timeout=60).read())
                        os.replace(tmp, p)
                        break
                    except Exception:
                        if attempt == 3:
                            raise
                        time.sleep(2 + 3 * attempt)
            rgb = np.asarray(Image.open(io.BytesIO(p.read_bytes())).convert("RGB")).astype(np.float32)
            z = rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768
            with self.lock:
                self.tiles[k] = z
            return z

    def height(self, lon, lat):
        n = 2 ** self.DZ * self.TILE
        px = ((np.asarray(lon) + 180) / 360 * n).astype(np.int64)
        py = ((1 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2 * n).astype(np.int64)
        tx, ty = px // self.TILE, py // self.TILE
        out = np.zeros(px.shape, np.float32)
        for x, y in set(zip(tx.ravel().tolist(), ty.ravel().tolist())):
            m = (tx == x) & (ty == y)
            out[m] = self.tile(x, y)[py[m] % self.TILE, px[m] % self.TILE]
        return np.maximum(out, 0)  # the sea and below-sea basins: the ellipsoid is close enough


def lonlat_h_to_rc(lon, lat, h, g):
    """GOES fractional row and column of ground points at height h (m above the ellipsoid; geodetic lat)."""
    req, rpol = g["semi_major_axis"], g["semi_minor_axis"]
    H = g["perspective_point_height"] + req
    e2 = 1 - rpol**2 / req**2
    phi, lam = np.radians(lat), np.radians(lon) - math.radians(g["longitude_of_projection_origin"])
    N = req / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    X = (N + h) * np.cos(phi) * np.cos(lam)
    Y = (N + h) * np.cos(phi) * np.sin(lam)
    Z = (N * (1 - e2) + h) * np.sin(phi)
    sx, sy, sz = H - X, -Y, Z
    x = np.arcsin(-sy / np.sqrt(sx * sx + sy * sy + sz * sz))
    y = np.arctan(sz / sx)
    return (y - g["y0"]) / g["dy"], (x - g["x0"]) / g["dx"]


def rc_h_to_lonlat(r, c, h, g):
    """Where a GOES line of sight meets the ellipsoid raised by h (plan margins only)."""
    from goes_ca_detections import xy_to_lonlat
    gg = dict(g)
    H = g["perspective_point_height"] + g["semi_major_axis"]
    gg["semi_major_axis"], gg["semi_minor_axis"] = g["semi_major_axis"] + h, g["semi_minor_axis"] + h
    gg["perspective_point_height"] = H - gg["semi_major_axis"]
    return xy_to_lonlat(g["x0"] + g["dx"] * np.asarray(c, float), g["y0"] + g["dy"] * np.asarray(r, float), gg)


# ---------- stages ----------

def pixels(a):
    import shapely
    from goes_events import STRENGTH, events
    from pick_fuel_windows import ca_polygon
    det = pd.read_parquet(G / f"det_{a.year}.parquet")
    det = det[shapely.contains_xy(ca_polygon(), det.lon.to_numpy(), det.lat.to_numpy())].reset_index(drop=True)
    det["strength"] = (det["mask"] % 20).map(STRENGTH).fillna(0).astype(int)
    det = events(det)
    t0 = det.groupby("event").t.min()
    det["dt_h"] = (det.t - det.event.map(t0)).dt.total_seconds() / 3600
    # the rebuilt events must be goes_events.py's: same first boxes
    fb = det[det.dt_h == 0][["event", "row", "col"]].sort_values(["event", "row", "col"]).reset_index(drop=True)
    ref = pd.read_parquet(G / f"event_px_{a.year}.parquet")[["event", "row", "col"]].sort_values(["event", "row", "col"]).reset_index(drop=True)
    if not fb.equals(ref):
        sys.exit("rebuilt first boxes differ from event_px; rerun goes_events.py first")
    out = det[det.dt_h <= HOURS][["event", "t", "dt_h", "row", "col", "power", "strength", "lon", "lat"]]
    out.to_parquet(G / f"track_px_{a.year}.parquet", index=False)
    n = out.groupby("event").size()
    print(f"{len(out)} px over {len(n)} events in their first {HOURS} h (first boxes match); "
          f"{int((out.groupby('event').t.nunique() > 1).sum())} events with 2+ scans")


def cells_needed(px):
    """Distinct cells of the event pixels and their 3 x 3 rings."""
    c = px[["row", "col"]].drop_duplicates()
    d = [c.assign(row=c.row + dr, col=c.col + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    return pd.concat(d).drop_duplicates().reset_index(drop=True)


def plan(a):
    g = grid_params(a.year)
    px = pd.read_parquet(G / f"track_px_{a.year}.parquet")
    c = cells_needed(px)
    rows = []
    for h in (0.0, H_MAX):  # the ground under a cell lies between its flat place and its place at H_MAX, toward the satellite
        for dr in (-0.5, 0.5):
            for dc in (-0.5, 0.5):
                lon, lat = rc_h_to_lonlat(c.row + dr, c.col + dc, h, g)
                rows.append(np.c_[lon, lat])
    ll = np.stack(rows)  # (8, n, 2)
    lon0, lon1 = np.nanmin(ll[..., 0], 0), np.nanmax(ll[..., 0], 0)
    lat0, lat1 = np.nanmin(ll[..., 1], 0), np.nanmax(ll[..., 1], 0)
    cy0, cy1 = ((MOS_Y0 - lat1) / MOS_RES).astype(int) // CHUNK, ((MOS_Y0 - lat0) / MOS_RES).astype(int) // CHUNK
    cx0, cx1 = ((lon0 - MOS_X0) / MOS_RES).astype(int) // CHUNK, ((lon1 - MOS_X0) / MOS_RES).astype(int) // CHUNK
    s = set()
    for a0, a1, b0, b1 in zip(cy0, cy1, cx0, cx1):
        s.update((cy, cx) for cy in range(a0, a1 + 1) for cx in range(b0, b1 + 1))
    ch = pd.DataFrame(sorted(s), columns=["cy", "cx"])
    ch.to_parquet(G / f"track_chunks_{a.year}.parquet", index=False)
    first = pd.read_parquet(G / f"chunks_{a.year}.parquet")[["cy", "cx"]].drop_duplicates()
    gd = grids_dir(a.year - 1)
    done = sum((gd / f"{cy}_{cx}.npz").exists() for cy, cx in zip(ch.cy, ch.cx))
    new = len(ch.merge(first, how="left", indicator=True).query("_merge == 'left_only'"))
    print(f"{len(c)} cells with rings -> {len(ch)} chunks ({new} beyond the first-box plan, {done} already made)")


def chunk_cells(cy, cx, g, dem, step):
    """Rounded GOES cell (flat, terrain) of every step-th pixel center of a chunk."""
    i = np.arange(0, CHUNK, step) + 0.5
    lon, lat = np.meshgrid(MOS_X0 + (cx * CHUNK + i) * MOS_RES, MOS_Y0 - (cy * CHUNK + i) * MOS_RES)
    out = {}
    for name, h in (("flat", 0.0), ("terrain", dem.height(lon, lat))):
        r, c = lonlat_h_to_rc(lon, lat, h, g)
        out[name] = (np.round(r).astype(np.int32).ravel(), np.round(c).astype(np.int32).ravel())
    return out


def cells(a):
    g = grid_params(a.year)
    gd = grids_dir(a.year - 1)
    ch = pd.concat([pd.read_parquet(G / f"chunks_{a.year}.parquet")[["cy", "cx"]],
                    pd.read_parquet(G / f"track_chunks_{a.year}.parquet")]).drop_duplicates()
    need = cells_needed(pd.read_parquet(G / f"track_px_{a.year}.parquet"))
    keep = set(zip(need.row, need.col))
    dem = Dem()
    acc, lock = {}, threading.Lock()
    missing = [0]

    def one(k):
        cy, cx = k
        f = gd / f"{cy}_{cx}.npz"
        if not f.exists():
            missing[0] += 1
            return
        z = np.load(f)
        vals = {"knn": z["knn"], "head": z["head"], "lf": z["lf"]}
        fuel = {}
        for key, v in vals.items():
            if key == "lf":
                fuel[key] = np.where(v > 0, (~((v >= 91) & (v <= 99))).astype(np.float32), np.nan)
            else:
                fuel[key] = np.where(v == 255, np.nan, v / 254.0)
        rc1, rc2 = chunk_cells(cy, cx, g, dem, 1), chunk_cells(cy, cx, g, dem, STEP)
        local = {}
        for terr in ("flat", "terrain"):
            for key in READ:
                r, c = (rc2 if key == "knn" else rc1)[terr]
                v = fuel[key].ravel()
                ok = np.isfinite(v)
                df = pd.DataFrame(dict(row=r[ok], col=c[ok], v=v[ok]))
                s = df.groupby(["row", "col"]).v.agg(["sum", "count"])
                for (rr, cc), (sm, n) in zip(s.index, s.to_numpy()):
                    if (rr, cc) in keep:
                        local[(terr, key, rr, cc)] = local.get((terr, key, rr, cc), (0.0, 0)) + np.array([sm, n])
        with lock:
            for k2, v2 in local.items():
                acc[k2] = acc.get(k2, 0) + v2

    todo = list(zip(ch.cy, ch.cx))
    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        for n, _ in enumerate(ex.map(one, todo)):
            if n % 2000 == 0:
                print(f"{n}/{len(todo)} chunks, {time.time() - t0:.0f} s, rss {rss():.2f} GB", flush=True)
    rows = [(t, k, r, c, float(v[0]), int(v[1])) for (t, k, r, c), v in acc.items()]
    df = pd.DataFrame(rows, columns=["terrain", "reading", "row", "col", "s", "n"])
    df.to_parquet(G / f"cell_fuel_{a.year}.parquet", index=False)
    print(f"{len(todo)} chunks ({missing[0]} missing grids) -> {len(df)} cell sums, {time.time() - t0:.0f} s, peak rss {rss():.2f} GB")


def steps(a):
    g = grid_params(a.year)
    px = pd.read_parquet(G / f"track_px_{a.year}.parquet")
    cf = pd.read_parquet(G / f"cell_fuel_{a.year}.parquet")
    tab = {}
    for (t, k), d in cf.groupby(["terrain", "reading"]):
        tab[(t, k)] = {(r, c): (s, n) for r, c, s, n in zip(d.row, d.col, d.s, d.n)}

    def mean(cellset, t, k):
        s = n = 0.0
        tb = tab[(t, k)]
        for x in cellset:
            v = tb.get(x)
            if v is not None:
                s += v[0]
                n += v[1]
        return s / n if n else np.nan

    def ring(cellset):
        out = set()
        for r, c in cellset:
            out.update((r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
        return out - cellset

    dem = Dem()
    out = []
    for e, p in px.groupby("event"):
        p = p.sort_values("t")
        cell = list(zip(p.row, p.col))
        first_seen = {}
        for x, dt in zip(cell, p.dt_h):
            first_seen.setdefault(x, dt)
        fb = p[p.dt_h == 0]
        lon, lat = fb.lon.mean(), fb.lat.mean()
        hgt = float(dem.height(np.array([lon]), np.array([lat]))[0])
        # how far the terrain moves this place on the GOES grid, in km (one GOES pixel ~ 2 km at nadir, more here)
        r0, c0 = lonlat_h_to_rc(lon, lat, 0.0, g)
        r1, c1 = lonlat_h_to_rc(lon, lat, hgt, g)
        for hz in HORIZONS:
            q = p[p.dt_h <= hz]
            S = set(zip(q.row, q.col))
            new = {x for x, dt in first_seen.items() if hz - 1 < dt <= hz} if hz > 0 else S
            ahead = ring(new) - S if new else set()
            row = dict(event=e, h=hz, elev_m=hgt, shift_px=float(np.hypot(r1 - r0, c1 - c0)),
                       scans=int(q.t.nunique()), px=int(len(q)), cells=len(S), new_cells=len(new),
                       power_max=float(q.power.max()) if q.power.notna().any() else 0.0, strength_max=int(q.strength.max()),
                       power_last=float(q[q.t == q.t.max()].power.max()) if q.power.notna().any() else 0.0)
            for t in ("flat", "terrain"):
                for k in READ:
                    row[f"{k}_in_{t}"] = mean(S, t, k)
                    row[f"{k}_around_{t}"] = mean(ring(S), t, k)
                    row[f"{k}_ahead_{t}"] = mean(ahead, t, k) if ahead else np.nan
            out.append(row)
    df = pd.DataFrame(out)
    ev = pd.read_parquet(G / f"events_{a.year}.parquet")[["event", "label", "fire", "fire_acres", "lon", "lat", "recur", "t0",
                                                          "scans", "first_power", "first_strength", "first_px"]]
    df = df.merge(ev.rename(columns={"scans": "scans_total"}), on="event")
    df.to_parquet(G / f"track_fuel_{a.year}.parquet", index=False)
    print(f"{df.event.nunique()} events x {len(HORIZONS)} horizons -> data/goes/track_fuel_{a.year}.parquet")
    print("median terrain shift (GOES px):", round(float(df[df.h == 0].shift_px.median()), 2),
          "; 90th pct:", round(float(df[df.h == 0].shift_px.quantile(0.9)), 2))


# ---------- scoring ----------

def score(a):
    from goes_fuel_check import paired
    from goes_fuel_score import oof
    from scipy.stats import spearmanr
    fs = []
    for y in a.years:
        d = pd.read_parquet(G / f"track_fuel_{y}.parquet").assign(year=y)
        fs.append(d)
    f = pd.concat(fs, ignore_index=True)
    f["place"] = np.floor(f.lon / 0.1).astype(int) * 10000 + np.floor(f.lat / 0.1).astype(int)
    f["y"] = (f.label != "unmatched").astype(int)
    res = {}

    # 1. the cell sums reproduce goes_fuel.py's first-box shares (flat)
    chk = []
    for y in a.years:
        ef = pd.read_parquet(G / f"event_fuel_{y}.parquet")[["event", "knn_in", "knn_around", "head_in", "lf_in"]]
        ef = ef.rename(columns={c: f"{c}_ref" for c in ef.columns if c != "event"})
        m = f[(f.year == y) & (f.h == 0)].merge(ef, on="event")
        chk.append(m)
    m = pd.concat(chk)
    res["check vs goes_fuel shares"] = {k: dict(max_abs_diff=round(float((m[f"{k}_flat"] - m[f"{k}_ref"]).abs().max()), 4),
                                                corr=round(float(m[[f"{k}_flat", f"{k}_ref"]].corr().iloc[0, 1]), 4))
                                        for k in ("knn_in", "knn_around", "head_in", "lf_in")}
    print("check:", res["check vs goes_fuel shares"])

    f0 = f[(f.h == 0) & (f.knn_in_flat.notna())].reset_index(drop=True)
    res["terrain shift (GOES px) at first boxes"] = dict(median=round(float(f0.shift_px.median()), 2),
                                                         p90=round(float(f0.shift_px.quantile(0.9)), 2),
                                                         share_over_half_px=round(float((f0.shift_px > 0.5).mean()), 3))

    def sig(d):
        return np.c_[np.log1p(d.power_max.fillna(0)), d.strength_max, np.log(d.px), np.log(d.scans), np.log1p(d.power_last.fillna(0))]

    def fx(d, cols):
        return np.c_[[d[c].fillna(d[c].mean()) for c in cols]].T

    # 2. terrain on vs off, first box
    y, gp = f0.y.to_numpy(), f0.place.to_numpy()
    base = sig(f0)
    sc = {"GOES first scan": oof(base, y, gp)}
    for t in ("flat", "terrain"):
        sc[f"GOES + knn {t}"] = oof(np.c_[base, fx(f0, [f"knn_in_{t}", f"knn_around_{t}"])], y, gp)
    for t in ("flat", "terrain"):
        sc[f"knn in {t} alone"] = f0[f"knn_in_{t}"].fillna(f0[f"knn_in_{t}"].mean()).to_numpy()
    r = paired(y, sc, gp)
    res["first box, terrain vs flat"] = r
    for sub, msk in (("2+ scans", f0.scans_total >= 2), ("first box elevation over 1000 m", f0.elev_m > 1000)):
        d = f0[msk].reset_index(drop=True)
        yy, gg = d.y.to_numpy(), d.place.to_numpy()
        if min(yy.sum(), (1 - yy).sum()) < 15:
            continue
        s2 = {"knn in flat": d.knn_in_flat.fillna(d.knn_in_flat.mean()).to_numpy(),
              "knn in terrain": d.knn_in_terrain.fillna(d.knn_in_terrain.mean()).to_numpy()}
        res[f"first box {sub}, terrain vs flat"] = paired(yy, s2, gg) | dict(n=int(len(d)), matched=int(yy.sum()))

    # 3. a dashboard's view: at each time after the first scan, GOES so far alone vs with fuel so far
    terr = a.terrain
    res["by time after first scan"] = {}
    for hz in HORIZONS:
        d = f[(f.h == hz) & f[f"knn_in_{terr}"].notna()].reset_index(drop=True)
        for sub, msk in (("all events", np.ones(len(d), bool)), ("events still burning past this time", d.scans_total > d.scans)):
            dd = d[msk].reset_index(drop=True)
            yy, gg = dd.y.to_numpy(), dd.place.to_numpy()
            if min(yy.sum(), (1 - yy).sum()) < 15:
                continue
            b = sig(dd)
            s3 = {"GOES so far": oof(b, yy, gg),
                  "GOES + fuel in, around": oof(np.c_[b, fx(dd, [f"knn_in_{terr}", f"knn_around_{terr}"])], yy, gg),
                  "GOES + fuel in, around, ahead": oof(np.c_[b, fx(dd, [f"knn_in_{terr}", f"knn_around_{terr}", f"knn_ahead_{terr}"])], yy, gg)}
            r3 = paired(yy, s3, gg) | dict(n=int(len(dd)), matched=int(yy.sum()))
            res["by time after first scan"][f"{hz} h, {sub}"] = r3
            print(f"{hz:>2} h {sub:38s} n {len(dd):5d} ({yy.sum():4d} matched): GOES {r3['GOES so far'][0]:.3f}, "
                  f"+ fuel {r3['GOES + fuel in, around'][0]:.3f} ({r3['gain GOES + fuel in, around over GOES so far']['gain']:+.3f} "
                  f"{r3['gain GOES + fuel in, around over GOES so far']['ci']}), + ahead {r3['GOES + fuel in, around, ahead'][0]:.3f}", flush=True)

    # 4. is it moving into fuel: among matched events seen past 1 h, fuel just ahead vs final acres (one event per fire)
    res["moving into fuel (matched, Spearman with final acres)"] = {}
    for hz in (1, 3, 6, 12):
        d = f[(f.h == hz) & (f.y == 1) & f[f"knn_ahead_{terr}"].notna() & f.fire_acres.notna()]
        d = d.sort_values("t0").drop_duplicates(["year", "fire"])
        if len(d) < 20:
            continue
        rho = spearmanr(d[f"knn_ahead_{terr}"], d.fire_acres)
        rho_px = spearmanr(d.cells, d.fire_acres)
        res["moving into fuel (matched, Spearman with final acres)"][f"{hz} h"] = dict(
            n_fires=int(len(d)), fuel_ahead=round(float(rho.statistic), 3), p=round(float(rho.pvalue), 4),
            cells_so_far_for_reference=round(float(rho_px.statistic), 3))
    print(json.dumps(res["moving into fuel (matched, Spearman with final acres)"], indent=1))
    tag = "_".join(map(str, a.years))
    json.dump(res, open(ROOT / "results" / f"goes_fuel_track_{tag}.json", "w"), indent=1, default=str)
    print(f"wrote results/goes_fuel_track_{tag}.json")
    for k, v in r.items():
        print(" ", k, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pixels", "plan", "cells", "steps", "score"])
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--years", type=int, nargs="+", default=[2025])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--terrain", choices=["flat", "terrain"], default="terrain", help="score: which placement the time view uses")
    a = ap.parse_args()
    {"pixels": pixels, "plan": plan, "cells": cells, "steps": steps, "score": score}[a.stage](a)


if __name__ == "__main__":
    main()
