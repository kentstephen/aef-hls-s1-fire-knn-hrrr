# /// script
# requires-python = ">=3.12"
# dependencies = ["icechunk", "xarray", "zarr>=3", "numpy", "pandas", "pyarrow", "pyproj", "scikit-learn", "scipy"]
# ///
"""Among GOES events that are WFIGS wildfires, does the fuel in and around the first box say how big the fire gets?

From "is it a fire" to "will it grow". Events: matched rows (perimeter or start) of data/goes/event_fuel_<year>.parquet
(goes_events.py, goes_fuel.py). Several events can match one fire (a fire seen, lost, seen again): only the first
event per fire is kept (fire = year, name, acres), so each fire counts once and is read at its first sighting.
Target: the fire's final acres (log10), as WFIGS / NIFC give it.

Predictors, all known at the first scan (or, for weather, from a forecast at that time):
  GOES     first-scan power (log), confidence class, pixels (log)
  fuel     kNN, head or LANDFIRE fuel share in the first box and in the ring around it
  weather  HRRR analysis (dynamical.org Icechunk, 3 km, hourly) at the first box's center: at the first scan hour
           (10 m wind, gust, 2 m RH and temperature, Fosberg FFWI), the next 48 h (max gust, min RH, max FFWI; the
           analysis stands in for a forecast, so it is a little better than a dashboard would have), and the 7
           days before (rain total)
  remoteness  km to the nearest road, major road and building (Overture; goes_fuel_check.py), used when
           data/goes/ext/event_dist_<year>.parquet exists: does fuel still add once remoteness is held fixed?
Whole-event fields (scans, pixels, peak power, distinct pixels) grow with the fire and leak the answer; they are
scored alone only as a ceiling, never used as predictors.

Scores: Spearman of out-of-fold predictions (ridge on log acres) with log acres, and AUC for fires past 100 /
1,000 / 10,000 acres (logistic), with folds and 95 % bootstrap intervals over 0.1 degree places. Gains (GOES + fuel
minus GOES; GOES + fuel + weather minus GOES + fuel) are paired: the same place resamples score both. Perimeter vs
start is also scored, though it is mostly a size split (NIFC perimeters are fires of 100 acres and up).

Weather is cached per year in data/goes/hrrr_events/<year>.parquet (one row per event; only missing events are read).
RAM: one store chunk group per worker (6 variables x 2160 h x 45 x 45 float32, about 105 MB); under 1.5 GB at 4.

    uv run scripts/goes_growth.py [--years 2025] [--workers 4] [--no-weather]
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
G = ROOT / "data" / "goes"
WX = G / "hrrr_events"
LCC = "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs"
HX0, HY0, HDX, HDY = -2699020.142521929, 1588193.847443335, 3000, -3000  # grid edge and step (make_samples.py)
CHUNK_T, CHUNK_XY = 2160, 45
VARS = ["temperature_2m", "relative_humidity_2m", "wind_u_10m", "wind_v_10m", "wind_gust_surface", "precipitation_surface"]
BEFORE_H, AFTER_H = 7 * 24, 48
CUTS = (100, 1000, 10000)
DIST_COLS = ["road", "major_road", "building"]
WX_COLS = ["wind_t0", "gust_t0", "rh_t0", "temp_t0", "ffwi_t0", "gust_max_48h", "rh_min_48h", "ffwi_max_48h", "rain_7d"]


def ffwi(t_c, rh, wind_ms):
    """Fosberg Fire Weather Index from 2 m temperature (C), RH (%) and 10 m sustained wind (m/s)."""
    t_f = t_c * 1.8 + 32
    h = np.clip(rh, 0, 100)
    m = np.where(h < 10, 0.03229 + 0.281073 * h - 0.000578 * h * t_f,
                 np.where(h < 50, 2.22749 + 0.160107 * h - 0.014784 * t_f, 21.0606 + 0.005565 * h * h - 0.00035 * h * t_f - 0.483199 * h))
    mr = np.clip(m, 0, 30) / 30
    eta = 1 - 2 * mr + 1.5 * mr**2 - 0.5 * mr**3
    w = wind_ms * 2.2369363
    return eta * np.sqrt(1 + w * w) / 0.3002


# ---------------------------------------------------------------- events

def load_events(years):
    out = []
    for y in years:
        f = pd.read_parquet(G / f"event_fuel_{y}.parquet")
        f = f[(f.label != "unmatched") & (f.n_knn_in > 0)].copy()
        px = pd.read_parquet(G / f"event_px_{y}.parquet")
        c = px.groupby("event")[["lon", "lat"]].mean().rename(columns=dict(lon="lon_first", lat="lat_first"))
        f = f.join(c, on="event")
        f["year"] = y
        d = G / "ext" / f"event_dist_{y}.parquet"  # goes_fuel_check.py: km to the nearest road, major road, building
        if d.exists():
            f = f.merge(pd.read_parquet(d).drop(columns="year"), on="event", how="left")
        out.append(f)
    f = pd.concat(out, ignore_index=True)
    n_ev = len(f)
    f = f.dropna(subset=["fire_acres"])
    f = f[f.fire_acres > 0]
    n_acres = len(f)
    f["fire_key"] = f.year.astype(str) + "|" + f.fire.astype(str) + "|" + f.fire_acres.round(1).astype(str)
    f = f.sort_values("t0").drop_duplicates("fire_key").reset_index(drop=True)
    print(f"{n_ev} matched events, {n_acres} with acres, {len(f)} fires (first event per fire)", flush=True)
    return f, dict(matched_events=n_ev, with_acres=n_acres, fires=len(f))


# ---------------------------------------------------------------- weather

DS = None


def open_hrrr():
    global DS
    import icechunk
    import xarray as xr
    st = icechunk.s3_storage(bucket="dynamical-noaa-hrrr", prefix="noaa-hrrr-analysis/v0.2.0.icechunk", region="us-west-2", anonymous=True)
    DS = xr.open_dataset(icechunk.Repository.open(st).readonly_session("main").store, engine="zarr", consolidated=False)[VARS]


def read_group(job):
    """Hourly series of the 6 variables at the given pixels over one store chunk group."""
    cx, cy, ct, hx, hy = job
    for attempt in range(4):
        try:
            sub = DS.isel(x=slice(cx * CHUNK_XY, (cx + 1) * CHUNK_XY), y=slice(cy * CHUNK_XY, (cy + 1) * CHUNK_XY),
                          time=slice(ct * CHUNK_T, (ct + 1) * CHUNK_T))
            lx, ly = hx - cx * CHUNK_XY, hy - cy * CHUNK_XY
            arr = {v: sub[v].values[:, ly, lx] for v in VARS}  # (hours, pixels)
            return job, pd.DatetimeIndex(sub.time.values), arr
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 + 5 * attempt)


def weather(ev, workers):
    """Per event: HRRR at the first scan hour, over the next 48 h and the 7 days before. Cached per year."""
    WX.mkdir(parents=True, exist_ok=True)
    have = []
    for y in ev.year.unique():
        p = WX / f"{y}.parquet"
        if p.exists():
            have.append(pd.read_parquet(p).assign(year=y))
    have = pd.concat(have, ignore_index=True) if have else pd.DataFrame(columns=["year", "event"])
    need = ev.merge(have[["year", "event"]], on=["year", "event"], how="left", indicator=True)
    need = need[need._merge == "left_only"].drop(columns="_merge").reset_index(drop=True)
    if len(need):
        open_hrrr()
        times = pd.DatetimeIndex(DS.time.values)
        x, y = Transformer.from_crs("EPSG:4326", LCC, always_xy=True).transform(need.lon_first.values, need.lat_first.values)
        need["hx"] = np.floor((x - HX0) / HDX).astype(int)
        need["hy"] = np.floor((y - HY0) / HDY).astype(int)
        t0 = need.t0.dt.tz_convert(None).dt.floor("h").to_numpy()
        i0 = np.searchsorted(times.values, t0)
        need["i_lo"], need["i_hi"] = i0 - BEFORE_H, i0 + AFTER_H
        need["i0"] = i0
        jobs = {}
        for r in need.itertuples():
            for ct in range(r.i_lo // CHUNK_T, r.i_hi // CHUNK_T + 1):
                jobs.setdefault((r.hx // CHUNK_XY, r.hy // CHUNK_XY, ct), set()).add((r.hx, r.hy))
        jl = [(cx, cy, ct, np.array([p[0] for p in s]), np.array([p[1] for p in s])) for (cx, cy, ct), s in jobs.items()]
        print(f"HRRR: {len(need)} events, {len(jl)} chunk groups to read", flush=True)
        series = {}  # (hx, hy) -> list of DataFrames
        t_start = time.time()
        with ThreadPoolExecutor(workers) as ex:
            for n, (job, tt, arr) in enumerate(ex.map(read_group, jl)):
                cx, cy, ct, hx, hy = job
                for k in range(len(hx)):
                    series.setdefault((int(hx[k]), int(hy[k])), []).append(
                        pd.DataFrame({v: arr[v][:, k] for v in VARS}, index=np.arange(ct * CHUNK_T, ct * CHUNK_T + len(tt))))
                if n % 10 == 0 or n == len(jl) - 1:
                    print(f"  {n + 1}/{len(jl)} groups, {(time.time() - t_start) / (n + 1):.1f} s/group", flush=True)
        rows = []
        for r in need.itertuples():
            s = pd.concat(series[(r.hx, r.hy)]).sort_index()
            s = s[~s.index.duplicated()]
            w = np.hypot(s.wind_u_10m, s.wind_v_10m)
            fw = pd.Series(ffwi(s.temperature_2m.to_numpy(), s.relative_humidity_2m.to_numpy(), w.to_numpy()), index=s.index)
            at = s.index.get_indexer([r.i0], method="pad")[0]
            a = s.index[at]
            nxt = slice(a, a + AFTER_H)
            rows.append(dict(year=r.year, event=r.event, hrrr_time=times[a], wind_t0=float(w.loc[a]), gust_t0=float(s.wind_gust_surface.loc[a]),
                             rh_t0=float(s.relative_humidity_2m.loc[a]), temp_t0=float(s.temperature_2m.loc[a]), ffwi_t0=float(fw.loc[a]),
                             gust_max_48h=float(s.wind_gust_surface.loc[nxt].max()), rh_min_48h=float(s.relative_humidity_2m.loc[nxt].min()),
                             ffwi_max_48h=float(fw.loc[nxt].max()),
                             rain_7d=float(s.precipitation_surface.loc[a - BEFORE_H:a - 1].sum() * 3600)))  # kg m-2 s-1 to mm per hour
        new = pd.DataFrame(rows)
        have = pd.concat([have, new], ignore_index=True)
        for y, g in have.groupby("year"):
            g.drop(columns="year").to_parquet(WX / f"{y}.parquet", index=False)
    return ev.merge(have, on=["year", "event"], how="left")


# ---------------------------------------------------------------- scoring

def places(f):
    return (np.floor(f.lon_first / 0.1).astype(int) * 10000 + np.floor(f.lat_first / 0.1).astype(int)).to_numpy()


def oof(X, y, groups, binary):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, groups=groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        if binary:
            m = LogisticRegression(max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
            p[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
        else:
            m = Ridge(alpha=1.0).fit((X[tr] - mu) / sd, y[tr])
            p[te] = m.predict((X[te] - mu) / sd)
    return p


def metric(y, s, binary):
    if binary:
        return roc_auc_score(y, s) if y.min() != y.max() else np.nan
    return spearmanr(y, s).statistic


def boot(y, preds, groups, binary, n=2000, seed=0):
    """Point and 95 % interval of each prediction's metric, plus paired differences, over the same place resamples."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.nonzero(groups == g)[0] for g in ug}
    draws = {k: [] for k in preds}
    for _ in range(n):
        i = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if binary and y[i].min() == y[i].max():
            continue
        for k, s in preds.items():
            draws[k].append(metric(y[i], s[i], binary))
    out = {k: [round(float(metric(y, s, binary)), 3), *[round(float(q), 3) for q in np.nanpercentile(draws[k], [2.5, 97.5])]]
           for k, s in preds.items()}
    return out, {k: np.array(v) for k, v in draws.items()}


def paired(draws, a, b, point):
    d = draws[a] - draws[b]
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return dict(gain=round(float(point[a][0] - point[b][0]), 3), ci=[round(float(lo), 3), round(float(hi), 3)],
                share_above_0=round(float(np.nanmean(d > 0)), 3))


def score(f, use_wx):
    y_log = np.log10(f.fire_acres.to_numpy())
    groups = places(f)
    goes = np.c_[np.log1p(f.first_power.fillna(0)), f.first_strength, np.log(f.first_px)]
    fuel = {r: np.c_[f[f"{r}_in"].fillna(f[f"{r}_in"].mean()), f[f"{r}_around"].fillna(f[f"{r}_around"].mean())]
            for r in ("knn", "head", "lf")}
    sets = {"GOES alone": goes, **{f"GOES + {r} fuel": np.c_[goes, fuel[r]] for r in fuel}, "kNN fuel alone": fuel["knn"]}
    remote = all(c in f for c in DIST_COLS) and f[DIST_COLS].notna().all(axis=None)
    if remote:  # fuel can stand for remoteness (fires near roads and homes are put out fast): hold remoteness fixed
        dist = np.log1p(f[DIST_COLS].to_numpy(float))
        sets["GOES + remoteness"] = np.c_[goes, dist]
        sets["GOES + remoteness + kNN fuel"] = np.c_[goes, dist, fuel["knn"]]
    if use_wx:
        wx = f[WX_COLS].to_numpy(float)
        wx = np.where(np.isfinite(wx), wx, np.nanmean(wx, 0))
        sets["GOES + weather"] = np.c_[goes, wx]
        sets["GOES + kNN fuel + weather"] = np.c_[goes, fuel["knn"], wx]
        sets["weather alone"] = wx
    res = dict(n_fires=int(len(f)), places=int(len(np.unique(groups))),
               acres_quartiles=[round(float(q), 1) for q in np.percentile(f.fire_acres, [25, 50, 75])])

    def pairs(draws, point):
        out = {"kNN fuel over GOES": paired(draws, "GOES + knn fuel", "GOES alone", point),
               "head fuel over GOES": paired(draws, "GOES + head fuel", "GOES alone", point),
               "LANDFIRE over GOES": paired(draws, "GOES + lf fuel", "GOES alone", point)}
        if remote:
            out["kNN fuel over GOES + remoteness"] = paired(draws, "GOES + remoteness + kNN fuel", "GOES + remoteness", point)
            out["remoteness over GOES"] = paired(draws, "GOES + remoteness", "GOES alone", point)
        if use_wx:
            out["weather over GOES + kNN fuel"] = paired(draws, "GOES + kNN fuel + weather", "GOES + knn fuel", point)
            out["weather over GOES"] = paired(draws, "GOES + weather", "GOES alone", point)
        return out

    # log acres: single readings (no fit), then out-of-fold sets
    single = {f"{r} {p}": f[f"{r}_{p}"].fillna(f[f"{r}_{p}"].mean()).to_numpy() for r in ("knn", "head", "lf") for p in ("in", "around")}
    single |= {"first power (log)": goes[:, 0], "first class": goes[:, 1], "first pixels (log)": goes[:, 2]}
    if use_wx:
        single |= {c: np.nan_to_num(f[c].to_numpy(float), nan=np.nanmean(f[c])) for c in WX_COLS}
    leak = {f"LEAK {c}": np.log1p(np.nan_to_num(f[c].to_numpy(float))) for c in ("scans", "px", "power_max", "distinct_px")}
    pt, _ = boot(y_log, single | leak, groups, False)
    res["Spearman with log acres, one reading"] = pt
    preds = {k: oof(X, y_log, groups, False) for k, X in sets.items()}
    pt, dr = boot(y_log, preds, groups, False)
    res["Spearman with log acres, out of fold"] = pt
    res["paired gains, Spearman"] = pairs(dr, pt)

    for cut in CUTS:
        yb = (f.fire_acres.to_numpy() >= cut).astype(int)
        key = f"AUC past {cut:,} acres"
        if min(yb.sum(), (1 - yb).sum()) < 10:
            res[key] = dict(note=f"too few: {int(yb.sum())} past the cut")
            continue
        preds = {k: oof(X, yb, groups, True) for k, X in sets.items()}
        pt, dr = boot(yb, preds, groups, True)
        res[key] = dict(n_past=int(yb.sum()), **pt, **{"paired gains": pairs(dr, pt)})

    # perimeter vs start (mostly a size split: perimeters are NIFC fires of 100 acres and up)
    yp = (f.label == "perimeter").to_numpy().astype(int)
    if min(yp.sum(), (1 - yp).sum()) < 10:
        return res
    preds = {k: oof(X, yp, groups, True) for k, X in sets.items()}
    pt, dr = boot(yp, preds, groups, True)
    res["AUC perimeter vs start"] = dict(n_perimeter=int(yp.sum()), **pt, **{"paired gains": pairs(dr, pt)})
    return res


def show(res):
    print(f"\n{res['n_fires']} fires, {res['places']} places, acres quartiles {res['acres_quartiles']}")
    for k, v in res.items():
        if not isinstance(v, dict):
            continue
        print(f"\n== {k}" + (f" ({v['n_past']} past)" if "n_past" in v else "") + (f" ({v['n_perimeter']} perimeter)" if "n_perimeter" in v else ""))
        for kk, vv in v.items():
            if isinstance(vv, list):
                print(f"  {kk:34s} {vv[0]:+.3f} [{vv[1]:+.3f}, {vv[2]:+.3f}]")
            elif kk.startswith("paired"):
                for pk, pv in vv.items():
                    print(f"  gain {pk:29s} {pv['gain']:+.3f} [{pv['ci'][0]:+.3f}, {pv['ci'][1]:+.3f}]  above 0 in {pv['share_above_0']:.0%}")
            elif isinstance(vv, dict) and "gain" in vv:
                print(f"  gain {kk:29s} {vv['gain']:+.3f} [{vv['ci'][0]:+.3f}, {vv['ci'][1]:+.3f}]  above 0 in {vv['share_above_0']:.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2025", help="comma separated, e.g. 2023,2024,2025,2026")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-weather", action="store_true")
    a = ap.parse_args()
    years = [int(y) for y in a.years.split(",")]
    f, counts = load_events(years)
    use_wx = not a.no_weather
    if use_wx:
        f = weather(f, a.workers)
        print(f"weather for {int(f.wind_t0.notna().sum())} of {len(f)} fires", flush=True)
    res = dict(years=years, **counts, dedupe="first event per fire (year, name, acres)", all=score(f, use_wx))
    if f.label.nunique() > 1:
        res["starts only (no perimeter)"] = score(f[f.label == "start"].reset_index(drop=True), use_wx)
    show(res["all"])
    (ROOT / "results").mkdir(exist_ok=True)
    tag = "" if years == [2025] else "_" + "_".join(map(str, years))
    out = ROOT / "results" / f"goes_growth{tag}.json"
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
