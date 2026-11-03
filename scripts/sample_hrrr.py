# /// script
# requires-python = ">=3.12"
# dependencies = ["icechunk", "xarray", "zarr>=3", "numpy", "pandas", "pyarrow"]
# ///
"""Daily Fosberg Fire Weather Index at every site, September 2019 to now, from the HRRR analysis
(dynamical.org Icechunk on AWS Open Data, 3 km, hourly). The hourly FFWI is computed at the site's
HRRR pixel from 2 m temperature, 2 m relative humidity and 10 m sustained wind, then summarized per
day (a fixed UTC-8 day, Pacific standard): max and mean FFWI, plus the day's max temperature, min RH,
mean wind and total precipitation, kept for reference.

One store chunk group at a time per worker (45 x 45 pixels, 2160 hours, 5 variables, about 87 MB of
float32); WORKERS at once, so 8 workers peak near 2 GB. Parts per group in data/hrrr_parts/, resumable.

    uv run scripts/sample_hrrr.py [--workers 8]
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
PARTS = ROOT / "data" / "hrrr_parts"
OUT = ROOT / "data" / "hrrr_daily.parquet"
CHUNK_T, CHUNK_XY = 2160, 45
VARS = ["temperature_2m", "relative_humidity_2m", "wind_u_10m", "wind_v_10m", "precipitation_surface"]
START = "2019-09-01"


def ffwi(t_c, rh, wind_ms):
    t_f = t_c * 1.8 + 32
    h = np.clip(rh, 0, 100)
    m = np.where(h < 10, 0.03229 + 0.281073 * h - 0.000578 * h * t_f,
                 np.where(h < 50, 2.22749 + 0.160107 * h - 0.014784 * t_f, 21.0606 + 0.005565 * h * h - 0.00035 * h * t_f - 0.483199 * h))
    mr = np.clip(m, 0, 30) / 30
    eta = 1 - 2 * mr + 1.5 * mr**2 - 0.5 * mr**3
    w = wind_ms * 2.2369363
    return eta * np.sqrt(1 + w * w) / 0.3002


DS = None


def _open():
    global DS
    storage = icechunk.s3_storage(bucket="dynamical-noaa-hrrr", prefix="noaa-hrrr-analysis/v0.2.0.icechunk", region="us-west-2", anonymous=True)
    DS = xr.open_dataset(icechunk.Repository.open(storage).readonly_session("main").store, engine="zarr", consolidated=False)[VARS]


def _group(job):
    key, cx, cy, ct, sites = job
    for attempt in range(4):
        try:
            sub = DS.isel(x=slice(cx * CHUNK_XY, (cx + 1) * CHUNK_XY), y=slice(cy * CHUNK_XY, (cy + 1) * CHUNK_XY),
                          time=slice(ct * CHUNK_T, (ct + 1) * CHUNK_T))
            times = pd.DatetimeIndex(sub.time.values)
            lx = (sites.hx.values - cx * CHUNK_XY).astype(int)
            ly = (sites.hy.values - cy * CHUNK_XY).astype(int)
            arr = {v: sub[v].values[:, ly, lx] for v in VARS}  # (hours, sites)
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 + 5 * attempt)
    t, rh = arr["temperature_2m"], arr["relative_humidity_2m"]
    w = np.hypot(arr["wind_u_10m"], arr["wind_v_10m"])
    f = ffwi(t, rh, w)
    day = (times - pd.Timedelta(hours=8)).floor("D")
    frames = []
    for k, sid in enumerate(sites.site_id.values):
        h = pd.DataFrame(dict(day=day, ffwi=f[:, k], t=t[:, k], rh=rh[:, k], wind=w[:, k], precip=arr["precipitation_surface"][:, k]))
        g = h.groupby("day").agg(ffwi_max=("ffwi", "max"), ffwi_mean=("ffwi", "mean"), t_max=("t", "max"), rh_min=("rh", "min"),
                                 wind_mean=("wind", "mean"), precip_mm=("precip", "sum"), hours=("ffwi", "count"))
        g["site_id"] = int(sid)
        frames.append(g.reset_index())
    pd.concat(frames).to_parquet(PARTS / f"{key}.parquet", index=False)
    return key, len(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    _open()
    times = pd.DatetimeIndex(DS.time.values)
    t0i = int(np.searchsorted(times, np.datetime64(START)))
    ct0, ct1 = t0i // CHUNK_T, (len(times) - 1) // CHUNK_T
    sites = pd.read_parquet(ROOT / "data" / "sites.parquet")
    sites["cx"], sites["cy"] = sites.hx // CHUNK_XY, sites.hy // CHUNK_XY
    PARTS.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in PARTS.glob("*.parquet")}
    jobs = []
    for (cx, cy), s in sites.groupby(["cx", "cy"]):
        for ct in range(ct0, ct1 + 1):
            key = f"{cx}_{cy}_{ct}"
            # the newest group is still filling: always reread it
            if key not in done or ct == ct1:
                jobs.append((key, int(cx), int(cy), ct, s[["site_id", "hx", "hy"]]))
    print(f"store {times[0]} to {times[-1]}; {len(jobs)} groups to read ({len(done)} done)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_open) as ex:
        for n, (key, nh) in enumerate(ex.map(_group, jobs)):
            el = time.time() - t0
            print(f"{n + 1}/{len(jobs)} group {key} {nh} h, {el / (n + 1):.2f} s/group, eta {el / (n + 1) * (len(jobs) - n - 1) / 60:.0f} min", flush=True)
    allp = pd.concat([pd.read_parquet(p) for p in PARTS.glob("*.parquet")], ignore_index=True)
    allp = allp[allp.day >= START].sort_values(["site_id", "day"])
    # a day split across two time chunks shows up twice: keep the fuller half
    allp = allp.sort_values("hours").drop_duplicates(["site_id", "day"], keep="last").sort_values(["site_id", "day"])
    allp.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(allp)} site-days, {allp.site_id.nunique()} sites, {allp.day.min():%Y-%m-%d} to {allp.day.max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
