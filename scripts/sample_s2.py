# /// script
# requires-python = ">=3.12"
# dependencies = ["pystac-client", "rasterio", "numpy", "pandas", "pyarrow", "pyproj"]
# ///
"""Sentinel-2 L2A surface reflectance at every site-date of data/samples.parquet, from Element 84 Earth Search
(sentinel-2-c1-l2a, COGs on public S3, no token).

Same rule as sample_hls.py: the scenes in the 20 days up to and including the sample date, newest first; the first
whose SCL is clear (vegetation, bare soil or water) over the 5 x 5 SCL pixels (100 m) at the site gives the bands.
A pass that lands in two MGRS tiles counts once; the other tile is tried if the first has no data there.

Three footprints, to see what 10 m adds:
  p   the 10 m pixel at the site (20 m bands: the 20 m pixel at the site)
  w3  3 x 3 mean, 30 m like HLS (20 m bands: the 20 m pixel at the site)
  w9  9 x 9 mean, 90 m like the HLS window used so far (20 m bands: 5 x 5 mean, 100 m)
10 m bands: blue, green, red, nir (B08). 20 m: re1, re2, re3 (B05 to B07), nir08 (B8A), swir1, swir2.

Parts per site in data/s2_parts/, resumable. Expect about 2.5 GB RSS with 16 sites (GDAL and Python per thread).

    uv run scripts/sample_s2.py [--sites 16] [--limit N]
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pystac_client
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent.parent
PARTS = ROOT / "data" / "s2_parts"
OUT = ROOT / "data" / "s2.parquet"
WINDOW_DAYS = 20
MAX_TRIES = 6
B10 = {"blue": "blue", "green": "green", "red": "red", "nir": "nir"}
B20 = {"re1": "rededge1", "re2": "rededge2", "re3": "rededge3", "nir08": "nir08", "swir1": "swir16", "swir2": "swir22"}
CLEAR_SCL = (4, 5, 6)
ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
           GDAL_INGESTED_BYTES_AT_OPEN="32768", GDAL_CACHEMAX="64", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2",
           CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", VSI_CACHE="FALSE", AWS_NO_SIGN_REQUEST="YES")
os.environ.update(ENV)  # process-wide, so every reader thread sees it


def read_win(href, lon, lat, n):
    """n x n window centered on the pixel holding the site, or None if it runs off the tile."""
    h = n // 2
    with rasterio.open(href) as r:
        x, y = Transformer.from_crs("EPSG:4326", r.crs, always_xy=True).transform(lon, lat)
        row, col = r.index(x, y)
        if not (h <= row < r.height - h and h <= col < r.width - h):
            return None
        return r.read(1, window=Window(col - h, row - h, n, n))


def scenes(cat, lon, lat):
    rows = []
    for it in cat.search(collections=["sentinel-2-c1-l2a"], intersects={"type": "Point", "coordinates": [lon, lat]},
                         datetime="2017-12-01/2026-12-31").items():
        if it.properties.get("eo:cloud_cover", 0) > 95:
            continue
        rows.append(dict(id=it.id, time=pd.Timestamp(it.datetime).tz_convert(None),
                         hrefs={k: it.assets[k].href for k in list(B10.values()) + list(B20.values()) + ["scl"]}))
    df = pd.DataFrame(rows).sort_values("time", ascending=False).reset_index(drop=True)
    df["pass"] = df.time.dt.round("min")  # the same pass in two MGRS tiles
    return df


def try_scene(c, lon, lat):
    """Band values for one tile of a pass, or a reason it is not usable."""
    scl = read_win(c.hrefs["scl"], lon, lat, 5)
    if scl is None or (scl == 0).any():
        return "nodata"
    if not np.isin(scl, CLEAR_SCL).all():
        return "cloud"
    rec = {}
    for name, key in B10.items():
        v = read_win(c.hrefs[key], lon, lat, 9)
        if v is None or (v == 0).any():
            return "nodata"
        v = v.astype(np.float64) * 1e-4 - 0.1
        rec[f"{name}_p"], rec[f"{name}_w3"], rec[f"{name}_w9"] = v[4, 4], v[3:6, 3:6].mean(), v.mean()
    for name, key in B20.items():
        v = read_win(c.hrefs[key], lon, lat, 5)
        if v is None or (v == 0).any():
            return "nodata"
        v = v.astype(np.float64) * 1e-4 - 0.1
        rec[f"{name}_p"] = rec[f"{name}_w3"] = v[2, 2]
        rec[f"{name}_w9"] = v.mean()
    return rec


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
    seen, out = {}, []
    for d in dates:
        end = d + pd.Timedelta(days=1)
        cand = sc[(sc.time < end) & (sc.time >= end - pd.Timedelta(days=WINDOW_DAYS))]
        rec = dict(site_id=int(site.site_id), date=d, tried=0)
        for p, tiles in list(cand.groupby("pass", sort=False))[:MAX_TRIES]:
            rec["tried"] += 1
            got = None
            for c in tiles.itertuples():
                if c.id not in seen:
                    try:
                        seen[c.id] = try_scene(c, lon, lat)
                    except Exception:
                        seen[c.id] = "error"
                got = seen[c.id]
                if isinstance(got, dict) or got == "cloud":
                    break  # cloud in one tile is cloud in the other: same pass
            if isinstance(got, dict):
                rec.update(got)
                rec.update(scene=c.id, age_days=float((d - c.time.floor("D")).days))
                break
        out.append(rec)
    df = pd.DataFrame(out)
    df.to_parquet(PARTS / f"{int(site.site_id)}.parquet", index=False)
    return int(site.site_id), len(dates), int(df.get("nir_p", pd.Series(dtype=float)).notna().sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sites = pd.read_parquet(ROOT / "data" / "sites.parquet").set_index("site_id", drop=False)
    smp = pd.read_parquet(ROOT / "data" / "samples.parquet")
    sd = smp[["site_id", "date"]].drop_duplicates()
    PARTS.mkdir(parents=True, exist_ok=True)
    done = {int(p.stem) for p in PARTS.glob("*.parquet")}
    todo = [s for s in sd.site_id.unique() if s not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(todo)} sites to read ({len(done)} done), {len(sd)} site-dates", flush=True)
    cat = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
    jobs = [(sites.loc[s], sorted(sd[sd.site_id == s].date), cat) for s in todo]
    t0, ndates, nclear = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=a.sites) as ex:
        for n, (sid, nd, nc) in enumerate(ex.map(site_job, jobs)):
            ndates += nd
            nclear += nc
            el = time.time() - t0
            print(f"{n + 1}/{len(todo)} site {sid}: {nc}/{nd} dates clear; all {nclear}/{ndates}, "
                  f"{el / (n + 1):.1f} s/site, eta {el / (n + 1) * (len(todo) - n - 1) / 60:.0f} min", flush=True)
    allp = pd.concat([pd.read_parquet(p) for p in PARTS.glob("*.parquet")], ignore_index=True)
    allp.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(allp)} site-dates, clear {int(allp.nir_p.notna().sum())}")


if __name__ == "__main__":
    main()
