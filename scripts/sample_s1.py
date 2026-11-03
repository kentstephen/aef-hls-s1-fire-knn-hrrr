# /// script
# requires-python = ">=3.12"
# dependencies = ["pystac-client", "planetary-computer", "rasterio", "numpy", "pandas", "pyarrow", "pyproj"]
# ///
"""Sentinel-1 backscatter at every site-date of data/samples.parquet, from Microsoft Planetary
Computer's sentinel-1-rtc (radiometrically terrain corrected, gamma0, 10 m, COGs on Azure blob).

Per site: one STAC search for every scene over the site. Per sample date: the scenes in the 12 days up
to and including the date, newest first; the first with valid VV and VH over the 5 x 5 pixels (50 m)
at the site gives the mean linear gamma0 of each, kept with the orbit direction and track. Up to 3
scenes are tried. Past only, as a live map would have.

Parts per site in data/s1_parts/, resumable. Under 1 GB of RAM.

    uv run scripts/sample_s1.py [--sites 16] [--limit N]
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent.parent
PARTS = ROOT / "data" / "s1_parts"
OUT = ROOT / "data" / "s1.parquet"
WINDOW_DAYS = 12
MAX_TRIES = 3
ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
           GDAL_INGESTED_BYTES_AT_OPEN="65536", GDAL_CACHEMAX="64", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2",
           CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff", VSI_CACHE="FALSE")
os.environ.update(ENV)


def read5x5(href, lon, lat):
    with rasterio.open(pc.sign(href)) as r:
        x, y = Transformer.from_crs("EPSG:4326", r.crs, always_xy=True).transform(lon, lat)
        row, col = r.index(x, y)
        if not (2 <= row < r.height - 2 and 2 <= col < r.width - 2):
            return None
        a = r.read(1, window=Window(col - 2, row - 2, 5, 5)).astype(np.float64)
        bad = (a == r.nodata) | ~np.isfinite(a) | (a <= 0)
        return None if bad.mean() > 0.5 else float(a[~bad].mean())


def scenes(cat, lon, lat):
    rows = []
    for it in cat.search(collections=["sentinel-1-rtc"], intersects={"type": "Point", "coordinates": [lon, lat]},
                         datetime="2017-12-01/2026-12-31").items():
        rows.append(dict(id=it.id, time=pd.Timestamp(it.datetime).tz_convert(None), orbit=it.properties["sat:orbit_state"],
                         track=int(it.properties["sat:relative_orbit"]), vv=it.assets["vv"].href, vh=it.assets["vh"].href))
    return pd.DataFrame(rows).sort_values("time", ascending=False).reset_index(drop=True)


def site_job(args):
    site, dates, cat = args
    lon, lat = float(site.longitude), float(site.latitude)
    for attempt in range(3):
        try:
            sc = scenes(cat, lon, lat)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5)
    cache, out = {}, []
    for d in dates:
        end = d + pd.Timedelta(days=1)
        cand = sc[(sc.time < end) & (sc.time >= end - pd.Timedelta(days=WINDOW_DAYS))]
        rec = dict(site_id=int(site.site_id), date=d)
        for c in cand.head(MAX_TRIES).itertuples():
            if c.id not in cache:
                try:
                    cache[c.id] = (read5x5(c.vv, lon, lat), read5x5(c.vh, lon, lat))
                except Exception:
                    cache[c.id] = (None, None)
            vv, vh = cache[c.id]
            if vv is None or vh is None:
                continue
            rec.update(vv=vv, vh=vh, s1_scene=c.id, s1_orbit=c.orbit, s1_track=c.track, s1_age_days=float((d - c.time.floor("D")).days))
            break
        out.append(rec)
    df = pd.DataFrame(out)
    df.to_parquet(PARTS / f"{int(site.site_id)}.parquet", index=False)
    return int(site.site_id), len(dates), int(df["vv"].notna().sum()) if "vv" in df else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sites = pd.read_parquet(ROOT / "data" / "sites.parquet").set_index("site_id", drop=False)
    smp = pd.read_parquet(ROOT / "data" / "samples.parquet")
    sd = smp[smp.date >= "2020-01-01"][["site_id", "date"]].drop_duplicates()
    PARTS.mkdir(parents=True, exist_ok=True)
    done = {int(p.stem) for p in PARTS.glob("*.parquet")}
    todo = [s for s in sd.site_id.unique() if s not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(todo)} sites to read ({len(done)} done), {len(sd)} site-dates", flush=True)
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    jobs = [(sites.loc[s], sorted(sd[sd.site_id == s].date), cat) for s in todo]
    t0, ndates, nok = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=a.sites) as ex:
        for n, (sid, nd, nc) in enumerate(ex.map(site_job, jobs)):
            ndates += nd
            nok += nc
            el = time.time() - t0
            print(f"{n + 1}/{len(todo)} site {sid}: {nc}/{nd} dates; all {nok}/{ndates}, "
                  f"{el / (n + 1):.1f} s/site, eta {el / (n + 1) * (len(todo) - n - 1) / 60:.0f} min", flush=True)
    allp = pd.concat([pd.read_parquet(p) for p in PARTS.glob("*.parquet")], ignore_index=True)
    allp.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(allp)} site-dates, valid {int(allp.vv.notna().sum())}")


if __name__ == "__main__":
    main()
