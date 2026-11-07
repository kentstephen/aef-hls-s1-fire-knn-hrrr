# /// script
# requires-python = ">=3.12"
# dependencies = ["zarr>=3", "obstore", "numpy", "pandas", "pyarrow", "shapely", "psutil", "pillow", "icechunk", "xarray", "scikit-learn", "scipy", "h5py", "rasterio"]
# ///
"""Is there signal in where one fire spread next? GOES-West cells of one fire, hour by hour.

The fire's ground is the GOES-18 fixed grid (about 2 x 3 km a cell over northern California). Each cell's arrival is
its first fire-mask detection (data/goes/det_<year>.parquet, one scan per 15 min). Every hour from discovery on, the
frontier is the unburned cells next to a burned one (8 neighbors); a frontier cell is a case when GOES first sees it
within the next H hours, a control otherwise. Cases and controls of an hour share the hour (same time, nearby place),
so the hour's weather level is common to both and only the ground and the wind's direction relative to the fire
differ.

Per frontier cell and hour:
  geometry  burned neighbors (8 and 4), hours since its newest burned neighbor first burned
  wind      HRRR analysis 10 m wind at the burned neighbors that hour, earth-relative (HRRR's u / v are along its
            Lambert grid, turned here by sin(38.5) x (lon + 97.5) degrees): the component along the line from the burned
            neighbors' center to the cell (m/s), and its cosine
  slope     the cell's mean height minus the burned neighbors', per km between them (Mapterhorn, zoom 8)
  fuel      from the year before's embeddings (EMB_YEAR): the volume head's cover x height (meters, its log scale and
            canopy cover), the fuel head's P(fuel), the kNN vote, LANDFIRE LF2024's burnable share
  weather   FFWI at the cell that hour (a check: shared across an hour's frontier, so near 0.5)

AUC per hour pooled (each hour's case-control pairs summed), 95 % from resampling hours. Then logistic models out of
fold (GroupKFold on 4 x 4 cell blocks): geometry + wind + slope, with and without fuel.

  cells   the fire's cells and arrivals -> data/spread/<fire>_cells.parquet
  fuel    fuel per cell from the mosaic (one year read per chunk) and the goes_fuel.py grids -> <fire>_fuel.parquet
  wx      HRRR per cell and hour -> <fire>_wx.parquet
  score   -> results/spread_signal_<fire>.json

RAM: fuel holds one chunk per thread (4 MB int8, about 30 MB as float); under 2 GB at 8 threads, stops past --max-gb.
wx reads whole HRRR chunks (45 x 45 x 2160 float32, 17.5 MB) for 4 variables and keeps only the cells; under 1 GB.

    uv run scripts/spread_signal.py cells
    uv run scripts/spread_signal.py fuel [--workers 8]
    uv run scripts/spread_signal.py wx
    uv run scripts/spread_signal.py score
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuel_common import CHUNK, MOS_RES, MOS_X0, MOS_Y0, YEAR0  # noqa: E402
from goes_ca_detections import FIRE_CODES, xy_to_lonlat  # noqa: E402
from goes_fuel import chunk_rc, grid_params, grids_dir  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "spread"
FIRES = {
    # name in data/fire_test/perimeters.parquet, its year, the embeddings year before it, days of GOES to follow
    "park": dict(name="PARK", year=2024, emb_year=2023, days=21),
}
BUF_DEG = 0.07  # the domain: cell centers within about 7 km of the final outline, so ground it never reached is in
VOLUME = json.loads((ROOT / "models" / "fuel_volume.json").read_text())


def rss():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def perimeter(f):
    import shapely
    p = pd.read_parquet(ROOT / "data" / "fire_test" / "perimeters.parquet")
    r = p[(p.name == f["name"]) & (p.year == f["year"])].sort_values("acres").iloc[-1]
    return r, shapely.from_wkb(r.geom)


def rc_lonlat(r, c, g):
    return xy_to_lonlat(g["x0"] + g["dx"] * np.asarray(c, float), g["y0"] + g["dy"] * np.asarray(r, float), g)


# ---------- cells ----------

def cells(a):
    import shapely
    f = FIRES[a.fire]
    g = grid_params(f["year"])
    rec, geom = perimeter(f)
    t0 = pd.Timestamp(rec.discovered)
    zone = geom.buffer(BUF_DEG)
    w, s, e, n = zone.bounds
    # every grid cell whose center falls in the zone
    from goes_fuel import goes_rc
    lo, la = np.meshgrid(np.linspace(w, e, 200), np.linspace(s, n, 200))
    rr, cc = goes_rc(lo, la, g)
    R = np.arange(int(np.floor(rr.min())) - 1, int(np.ceil(rr.max())) + 2)
    C = np.arange(int(np.floor(cc.min())) - 1, int(np.ceil(cc.max())) + 2)
    r2, c2 = [v.ravel() for v in np.meshgrid(R, C, indexing="ij")]
    lon, lat = rc_lonlat(r2, c2, g)
    keep = shapely.contains_xy(zone, lon, lat)
    df = pd.DataFrame(dict(row=r2[keep], col=c2[keep], lon=lon[keep], lat=lat[keep]))
    df["inside"] = shapely.contains_xy(geom, df.lon, df.lat)
    det = pd.read_parquet(ROOT / "data" / "goes" / f"det_{f['year']}.parquet")
    det = det[det["mask"].isin(FIRE_CODES) & (det.t >= t0 - pd.Timedelta(hours=1)) & (det.t < t0 + pd.Timedelta(days=f["days"]))]
    first = det.groupby(["row", "col"]).agg(arrival=("t", "min"), scans=("t", "size"), power=("power", "max")).reset_index()
    df = df.merge(first, on=["row", "col"], how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"{a.fire}_cells.parquet")
    seen = df.arrival.notna()
    print(f"{len(df)} cells in the zone ({df.inside.sum()} centers inside the outline); {seen.sum()} seen burning "
          f"({(seen & df.inside).sum()} inside); discovered {t0}, start {rec.start_lon:.4f}, {rec.start_lat:.4f}")
    print(df[seen].arrival.dt.tz_convert("America/Los_Angeles").dt.floor("D").value_counts().sort_index().to_string())


# ---------- fuel ----------

def volume_forward(q):
    """The volume head on int8 embeddings [n, 64] -> scaled log volume (0..1), canopy cover, tree lifeform (bool)."""
    v = VOLUME
    f = q.astype(np.float32) / 127.5
    x = (np.sign(f) * f * f - np.asarray(v["mean"], np.float32)) / np.asarray(v["scale"], np.float32)
    w1 = np.asarray(v["w1"], np.float32).reshape(v["hidden"], v["n_in"])
    w2 = np.asarray(v["w2"], np.float32).reshape(v["n_out"], v["hidden"])
    h = np.maximum(x @ w1.T + np.asarray(v["b1"], np.float32), 0)
    z = h @ w2.T + np.asarray(v["b2"], np.float32)
    nr, nk = len(v["reg"]), len(v["kinds"])
    life = z[:, nr + nk:nr + nk + 4].argmax(1)
    return np.clip(z[:, 0], 0, 1), np.clip(z[:, 3], 0, 1), life == 1, z


def check_volume():
    v = VOLUME
    q = np.asarray(v["probe_q"], np.int8).reshape(-1, v["n_in"])
    *_, z = volume_forward(q)
    gap = np.abs(z - np.asarray(v["probe_out"], np.float32).reshape(len(q), -1)).max()
    print(f"volume head vs the export's probes: largest gap {gap:.2e}")
    assert gap < 1e-3


def vol_m(s):
    return np.maximum(np.exp(s * (VOLUME["vol_hi"] - VOLUME["vol_lo"]) + VOLUME["vol_lo"]) - 0.02, 0)


def fuel(a):
    from fire_test_grids import open_aef
    check_volume()
    f = FIRES[a.fire]
    g = grid_params(f["year"])
    cl = pd.read_parquet(OUT / f"{a.fire}_cells.parquet")
    key = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(cl.row, cl.col))}
    lon0, lon1 = cl.lon.min() - 0.03, cl.lon.max() + 0.03
    lat0, lat1 = cl.lat.min() - 0.03, cl.lat.max() + 0.03
    cy0, cy1 = int((MOS_Y0 - lat1) / MOS_RES // CHUNK), int((MOS_Y0 - lat0) / MOS_RES // CHUNK)
    cx0, cx1 = int((lon0 - MOS_X0) / MOS_RES // CHUNK), int((lon1 - MOS_X0) / MOS_RES // CHUNK)
    todo = [(cy, cx) for cy in range(cy0, cy1 + 1) for cx in range(cx0, cx1 + 1)]
    GR = grids_dir(f["emb_year"])
    print(f"{len(todo)} chunks; goes_fuel.py grids on disk for {sum((GR / f'{cy}_{cx}.npz').exists() for cy, cx in todo)}")
    arr = open_aef()
    yi = f["emb_year"] - YEAR0
    # per cell sums: [n_vol, s_vol, m_vol, cc, tree, n_head, head, n_knn, knn, n_lf, lf_fuel]
    S = np.zeros((len(cl), 11))
    import threading
    lock = threading.Lock()
    peak, t0 = 0.0, time.time()

    def add(idx, col, vals):
        m = idx >= 0
        np.add.at(S[:, col], idx[m], vals[m])

    def one(c):
        cy, cx = c
        for attempt in range(4):
            try:
                b = np.asarray(arr[yi, :, cy * CHUNK:(cy + 1) * CHUNK, cx * CHUNK:(cx + 1) * CHUNK])
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 + 3 * attempt)
        r, cc = chunk_rc(cy, cx, g)
        ri, ci = np.rint(r).astype(int).ravel(), np.rint(cc).astype(int).ravel()
        idx = np.array([key.get((x, y), -1) for x, y in zip(ri, ci)])
        if (idx < 0).all():
            return c
        q = b.reshape(64, -1).T
        ok = ~(q == -128).all(axis=1) & (idx >= 0)
        s, ccov, tree, _ = volume_forward(q[ok])
        out = []
        ii = idx[ok]
        out.append((ii, 0, np.ones(len(ii))))
        out.append((ii, 1, s))
        out.append((ii, 2, vol_m(s)))
        out.append((ii, 3, ccov))
        out.append((ii, 4, tree.astype(float)))
        p = GR / f"{cy}_{cx}.npz"
        if p.exists():
            z = np.load(p)
            hd = z["head"].ravel()
            hv = hd != 255
            out.append((np.where(hv, idx, -1), 5, np.ones(len(idx))))
            out.append((np.where(hv, idx, -1), 6, hd / 254))
            lf = z["lf"].ravel()
            lv = lf > 0
            out.append((np.where(lv, idx, -1), 9, np.ones(len(idx))))
            out.append((np.where(lv, idx, -1), 10, ((lf < 91) | (lf > 99)).astype(float)))
            kn = z["knn"].ravel()
            r2, c2 = chunk_rc(cy, cx, g, step=2)
            i2 = np.array([key.get((x, y), -1) for x, y in zip(np.rint(r2).astype(int).ravel(), np.rint(c2).astype(int).ravel())])
            kv = kn != 255
            out.append((np.where(kv, i2, -1), 7, np.ones(len(i2))))
            out.append((np.where(kv, i2, -1), 8, kn / 254))
        with lock:
            for ii, col, vals in out:
                add(ii, col, vals)
        return c

    with ThreadPoolExecutor(a.workers) as ex:
        for k, _ in enumerate(ex.map(one, todo)):
            gb = rss()
            peak = max(peak, gb)
            if gb > a.max_gb:
                print(f"STOP: {gb:.1f} GB > --max-gb {a.max_gb}", flush=True)
                os._exit(2)
            if k % 50 == 0 or k == len(todo) - 1:
                el = time.time() - t0
                print(f"{k + 1}/{len(todo)} {el / (k + 1):.2f} s/chunk, eta {el / (k + 1) * (len(todo) - k - 1) / 60:.1f} min, "
                      f"rss {gb:.2f} GB (peak {peak:.2f})", flush=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = cl[["row", "col"]].copy()
        out["px"] = S[:, 0]
        out["vol_s"] = S[:, 1] / S[:, 0]
        out["vol_m"] = S[:, 2] / S[:, 0]
        out["canopy"] = S[:, 3] / S[:, 0]
        out["tree"] = S[:, 4] / S[:, 0]
        out["p_fuel"] = np.where(S[:, 5] > 0.5 * S[:, 0], S[:, 6] / S[:, 5], np.nan)
        out["knn"] = np.where(S[:, 7] * 4 > 0.5 * S[:, 0], S[:, 8] / S[:, 7], np.nan)
        out["lf_fuel"] = np.where(S[:, 9] > 0.5 * S[:, 0], S[:, 10] / S[:, 9], np.nan)
    out.to_parquet(OUT / f"{a.fire}_fuel.parquet")
    print(out.describe().T.to_string())
    print(f"done in {(time.time() - t0) / 60:.1f} min, peak rss {peak:.2f} GB")


# ---------- weather ----------

def ffwi(t_c, rh, wind_ms):
    t_f = t_c * 1.8 + 32
    h = np.clip(rh, 0, 100)
    m = np.where(h < 10, 0.03229 + 0.281073 * h - 0.000578 * h * t_f,
                 np.where(h < 50, 2.22749 + 0.160107 * h - 0.014784 * t_f, 21.0606 + 0.005565 * h * h - 0.00035 * h * t_f - 0.483199 * h))
    mr = np.clip(m, 0, 30) / 30
    eta = 1 - 2 * mr + 1.5 * mr**2 - 0.5 * mr**3
    w = wind_ms * 2.2369363
    return eta * np.sqrt(1 + w * w) / 0.3002


def wx(a):
    import icechunk
    import xarray as xr
    f = FIRES[a.fire]
    cl = pd.read_parquet(OUT / f"{a.fire}_cells.parquet")
    rec, _ = perimeter(f)
    t0 = pd.Timestamp(rec.discovered).tz_convert(None).floor("h") - pd.Timedelta(hours=2)
    t1 = t0 + pd.Timedelta(days=f["days"])
    st = icechunk.s3_storage(bucket="dynamical-noaa-hrrr", prefix="noaa-hrrr-analysis/v0.2.0.icechunk", region="us-west-2", anonymous=True)
    ds = xr.open_dataset(icechunk.Repository.open(st).readonly_session("main").store, engine="zarr", consolidated=False)
    # the nearest HRRR cell to each GOES cell center, searched in a window around the fire
    la, lo = ds.latitude.values, ds.longitude.values
    lo = np.where(lo > 180, lo - 360, lo)
    near = (np.abs(la - cl.lat.mean()) < 1.5) & (np.abs(lo - cl.lon.mean()) < 1.5)
    yy, xx = np.nonzero(near)
    d = (la[yy, xx][None] - cl.lat.values[:, None]) ** 2 + ((lo[yy, xx][None] - cl.lon.values[:, None]) * math.cos(math.radians(cl.lat.mean()))) ** 2
    k = d.argmin(1)
    hy, hx = yy[k], xx[k]
    print(f"{len(cl)} cells -> {len(set(zip(hy, hx)))} HRRR cells; {t0} to {t1}")
    sub = ds[["temperature_2m", "relative_humidity_2m", "wind_u_10m", "wind_v_10m"]].sel(time=slice(t0, t1))
    sub = sub.isel(y=slice(hy.min(), hy.max() + 1), x=slice(hx.min(), hx.max() + 1)).load()
    print(f"read: rss {rss():.2f} GB")
    yi, xi = hy - hy.min(), hx - hx.min()
    tt = pd.DatetimeIndex(sub.time.values).tz_localize("UTC")
    T = sub.temperature_2m.values[:, yi, xi]
    RH = sub.relative_humidity_2m.values[:, yi, xi]
    ug, vg = sub.wind_u_10m.values[:, yi, xi], sub.wind_v_10m.values[:, yi, xi]
    # grid-relative to earth-relative (NCEP's rotation for this Lambert grid: tangent latitude 38.5, central lon -97.5)
    ang = np.radians(math.sin(math.radians(38.5)) * (cl.lon.values + 97.5))[None]
    ue = np.cos(ang) * ug + np.sin(ang) * vg
    ve = -np.sin(ang) * ug + np.cos(ang) * vg
    nt, nc = T.shape
    out = pd.DataFrame(dict(t=np.repeat(tt, nc), cell=np.tile(np.arange(nc), nt), t_c=T.ravel(), rh=RH.ravel(), u=ue.ravel(), v=ve.ravel()))
    out["ffwi"] = ffwi(out.t_c, out.rh, np.hypot(out.u, out.v))
    out.to_parquet(OUT / f"{a.fire}_wx.parquet")
    print(out.describe().T.to_string())


# ---------- score ----------

def elevation(cl, g):
    from goes_fuel_track import Dem
    dem = Dem()
    o = np.linspace(-0.4, 0.4, 5)
    dr, dc = [v.ravel() for v in np.meshgrid(o, o)]
    lon, lat = rc_lonlat(cl.row.values[:, None] + dr[None], cl.col.values[:, None] + dc[None], g)
    return dem.height(lon.ravel(), lat.ravel()).reshape(len(cl), -1).mean(1)


def pooled_auc(y, x, grp):
    """Mann-Whitney AUC within each group (hour), summed over groups; NaN rows dropped."""
    from scipy.stats import rankdata
    num = den = 0.0
    per = []
    for k, ix in pd.Series(np.arange(len(y))).groupby(grp):
        ix = ix.values
        yy, xx = y[ix], x[ix]
        m = ~np.isnan(xx)
        yy, xx = yy[m], xx[m]
        p, n = yy.sum(), len(yy) - yy.sum()
        if p == 0 or n == 0:
            continue
        r = rankdata(xx)
        u = r[yy == 1].sum() - p * (p + 1) / 2
        per.append((k, u, p * n))
        num += u
        den += p * n
    return (num / den if den else np.nan), per


def auc_ci(per, reps=2000, seed=0):
    if not per:
        return [np.nan, np.nan]
    u = np.array([x[1] for x in per])
    d = np.array([x[2] for x in per])
    rng = np.random.default_rng(seed)
    b = rng.integers(0, len(per), (reps, len(per)))
    v = u[b].sum(1) / d[b].sum(1)
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def frontier_rows(cl, wxd, H, start, step_h=1):
    """One row per frontier cell per hour: its features and whether it burns within H hours.

    Two direction baselines sit beside the hour's wind, so the wind's own part can be told from the fire's heading:
    outward_cos, the spread line against the line from the ignition point (start) to the cell, and prevail_cos, the
    spread line against the fire's mean wind over every hour and cell (one fixed bearing).
    """
    n = len(cl)
    rc = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(cl.row, cl.col))}
    nb8 = [[rc.get((r + dr, c + dc), -1) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc] for r, c in zip(cl.row, cl.col)]
    nb4 = [[rc.get((r + dr, c + dc), -1) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))] for r, c in zip(cl.row, cl.col)]
    arr = cl.arrival.values.astype("datetime64[ns]")
    has = ~np.isnat(arr)
    t_first = arr[has].min()
    t_last = arr[has].max()
    hours = pd.date_range(pd.Timestamp(t_first).ceil("h"), pd.Timestamp(t_last), freq=f"{step_h}h")
    kx = np.cos(np.radians(cl.lat.mean())) * 111.32
    W = wxd.set_index(["t", "cell"])
    wt = {t: g for t, g in wxd.groupby("t")}
    mu, mv = wxd.u.mean(), wxd.v.mean()
    ms = math.hypot(mu, mv)
    sx, sy = start
    rows = []
    for t in hours:
        tn = np.datetime64(t.tz_convert(None) if t.tzinfo else t, "ns")
        burned = has & (arr <= tn)
        if not burned.any():
            continue
        th = pd.Timestamp(t).floor("h")
        w = wt.get(th if th.tzinfo else th.tz_localize("UTC"))
        if w is None:
            continue
        wu, wv, wf = w.u.values, w.v.values, w.ffwi.values
        for i in range(n):
            if burned[i]:
                continue
            nb = [j for j in nb8[i] if j >= 0 and burned[j]]
            if not nb:
                continue
            nb_a = np.array(nb)
            bx, by = cl.lon.values[nb_a].mean(), cl.lat.values[nb_a].mean()
            ex, ny_ = (cl.lon.values[i] - bx) * kx, (cl.lat.values[i] - by) * 110.57
            dist = math.hypot(ex, ny_) or 1e-6
            uu, vv = wu[nb_a].mean(), wv[nb_a].mean()
            along = (uu * ex + vv * ny_) / dist
            spd = math.hypot(uu, vv)
            newest = (tn - arr[nb_a].max()) / np.timedelta64(1, "h")
            label = has[i] and arr[i] <= tn + np.timedelta64(H, "h")
            ox, oy = (cl.lon.values[i] - sx) * kx, (cl.lat.values[i] - sy) * 110.57
            od = math.hypot(ox, oy) or 1e-6
            rows.append(dict(hour=t, cell=i, y=int(label), nb8=len(nb), nb4=sum(1 for j in nb4[i] if j >= 0 and burned[j]),
                             newest_h=newest, downwind=along, wind_cos=along / spd if spd > 0 else 0.0, wind=spd,
                             upslope=(cl.elev.values[i] - cl.elev.values[nb_a].mean()) / dist, ffwi=wf[i], dist=dist,
                             outward_cos=(ex * ox + ny_ * oy) / (dist * od), prevail_cos=(ex * mu + ny_ * mv) / (dist * ms)))
    return pd.DataFrame(rows)


def score(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    f = FIRES[a.fire]
    g = grid_params(f["year"])
    cl = pd.read_parquet(OUT / f"{a.fire}_cells.parquet")
    fu = pd.read_parquet(OUT / f"{a.fire}_fuel.parquet")
    cl = cl.merge(fu, on=["row", "col"], how="left")
    cl["elev"] = elevation(cl, g)
    wxd = pd.read_parquet(OUT / f"{a.fire}_wx.parquet")
    FUEL = ["vol_s", "vol_m", "canopy", "tree", "p_fuel", "knn", "lf_fuel"]
    BASE = ["nb8", "nb4", "newest_h", "outward_cos", "prevail_cos", "downwind", "wind_cos", "upslope", "ffwi"]
    rec, _ = perimeter(f)
    res = {"fire": a.fire, "emb_year": f["emb_year"], "cells": int(len(cl)), "seen": int(cl.arrival.notna().sum()), "runs": {}}
    for H in a.hours:
        fr = frontier_rows(cl, wxd, H, (rec.start_lon, rec.start_lat))
        fr = fr.merge(cl[FUEL].reset_index().rename(columns={"index": "cell"}), on="cell", how="left")
        hrs = pd.DatetimeIndex(fr.hour)
        t0 = hrs.min()
        for span, m in (("first 72 h", hrs < t0 + pd.Timedelta(hours=72)), ("all", np.ones(len(fr), bool))):
            d = fr[m].reset_index(drop=True)
            y = d.y.values
            out = {"rows": int(len(d)), "cases": int(y.sum()), "hours": int(d.hour.nunique()), "auc": {}}
            for col in BASE + FUEL:
                v, per = pooled_auc(y, d[col].values.astype(float), d.hour.values)
                out["auc"][col] = {"auc": round(float(v), 3), "ci": [round(x, 3) for x in auc_ci(per)]}
            # fuel within a fixed contact: does fuel order cells that touch the fire the same way?
            out["fuel_by_contact"] = {}
            for col in ("vol_s", "p_fuel"):
                for lo, hi in ((1, 2), (3, 8)):
                    mm = (d.nb8 >= lo) & (d.nb8 <= hi)
                    v, per = pooled_auc(y[mm], d[col].values[mm].astype(float), d.hour.values[mm])
                    out["fuel_by_contact"][f"{col} nb8 {lo}-{hi}"] = {"auc": round(float(v), 3), "ci": [round(x, 3) for x in auc_ci(per)]}
            # models out of fold, spatial blocks
            grp = (cl.row.values[d.cell] // 4) * 10000 + cl.col.values[d.cell] // 4
            geo = ["nb8", "nb4", "newest_h"]
            head = geo + ["outward_cos", "prevail_cos"]
            models = {"geometry": geo, "geometry + heading": head, "geometry + heading + hour's wind": head + ["downwind", "wind_cos"],
                      "geometry + wind + slope": head + ["downwind", "wind_cos", "upslope"]}
            for fc in ("vol_s", "p_fuel", "lf_fuel"):
                models[f"geometry + wind + slope + {fc}"] = models["geometry + wind + slope"] + [fc]
            models["geometry + wind + slope + vol_s + canopy + p_fuel"] = models["geometry + wind + slope"] + ["vol_s", "canopy", "p_fuel"]
            models["geometry + heading + fuel (vol_s, canopy, p_fuel), no wind"] = head + ["upslope", "vol_s", "canopy", "p_fuel"]
            out["models"] = {}
            # every model on the same rows: those with every reading (P(fuel) and LANDFIRE are missing where the grids are)
            common = d[sorted({c for cols in models.values() for c in cols})].notna().all(axis=1).values
            for name, cols in models.items():
                X = d[cols].astype(float)
                ok = common
                X, yy, gg, hh = X.values[ok], y[ok], grp[ok], d.hour.values[ok]
                p = np.full(len(yy), np.nan)
                for tr, te in GroupKFold(5).split(X, yy, gg):
                    sc = StandardScaler().fit(X[tr])
                    lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X[tr]), yy[tr])
                    p[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
                v, per = pooled_auc(yy, p, hh)
                out["models"][name] = {"auc": round(float(v), 3), "ci": [round(x, 3) for x in auc_ci(per)], "rows": int(ok.sum())}
            res["runs"][f"H={H} h, {span}"] = out
            print(f"\n== H = {H} h, {span}: {out['rows']} frontier rows, {out['cases']} cases, {out['hours']} hours")
            for k, v in out["auc"].items():
                print(f"  {k:10s} {v['auc']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]")
            for k, v in out["fuel_by_contact"].items():
                print(f"  {k:18s} {v['auc']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]")
            for k, v in out["models"].items():
                print(f"  model {k:50s} {v['auc']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]  ({v['rows']} rows)")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"spread_signal_{a.fire}.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cells", "fuel", "wx", "score"])
    ap.add_argument("--fire", default="park", choices=list(FIRES))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-gb", type=float, default=4.0)
    ap.add_argument("--hours", type=int, nargs="+", default=[1, 3, 6])
    a = ap.parse_args()
    {"cells": cells, "fuel": fuel, "wx": wx, "score": score}[a.stage](a)


if __name__ == "__main__":
    main()
