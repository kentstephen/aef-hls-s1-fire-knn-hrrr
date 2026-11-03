# /// script
# requires-python = ">=3.12"
# dependencies = ["pystac-client", "planetary-computer", "rasterio", "numpy", "pandas", "pyarrow", "pyproj"]
# ///
"""HLS v2.0 surface reflectance at every site-date of data/samples.parquet, from Microsoft Planetary
Computer (hls2-s30 and hls2-l30, COGs on Azure blob, free SAS token).

Per site: one STAC search for every S30 and L30 scene over the site, 2017-12 to now. Per sample date:
the scenes in the 20 days up to and including the date, newest first; the first one whose Fmask is
clear (no cloud, adjacent cloud, shadow or snow) over the 3 x 3 pixels (90 m) at the site gives the
six bands (blue, green, red, NIR narrow, SWIR 1, SWIR 2), averaged over those pixels. Only the past is
used: a live map has nothing after today. Up to 6 scenes are tried per date.

Parts per site in data/hls_parts/, resumable. SITES sites at once, each a few small block reads at a
time. Measured with 16 sites: 2.2 GB RSS (GDAL and Python per thread, not the data).

    uv run scripts/sample_hls.py [--sites 16] [--limit N]
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
PARTS = ROOT / "data" / "hls_parts"
OUT = ROOT / "data" / "hls.parquet"
WINDOW_DAYS = 20
MAX_TRIES = 6
BANDS = {"hls2-s30": ["B02", "B03", "B04", "B8A", "B11", "B12"], "hls2-l30": ["B02", "B03", "B04", "B05", "B06", "B07"]}
NAMES = ["blue", "green", "red", "nir", "swir1", "swir2"]
ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
           GDAL_INGESTED_BYTES_AT_OPEN="32768", GDAL_CACHEMAX="64", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2",
           CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", VSI_CACHE="FALSE")
os.environ.update(ENV)  # process-wide, so every reader thread sees it


def read3x3(href, lon, lat):
    with rasterio.open(pc.sign(href)) as r:
        x, y = Transformer.from_crs("EPSG:4326", r.crs, always_xy=True).transform(lon, lat)
        row, col = r.index(x, y)
        if not (1 <= row < r.height - 1 and 1 <= col < r.width - 1):
            return None
        return r.read(1, window=Window(col - 1, row - 1, 3, 3))


def clear(fm):
    # Fmask bits: 1 cloud, 2 adjacent to cloud, 3 shadow, 4 snow/ice; 255 fill
    return fm is not None and not (fm == 255).any() and not (fm & 0b11110).any()


def scenes(cat, lon, lat):
    rows = []
    for c in BANDS:
        for it in cat.search(collections=[c], intersects={"type": "Point", "coordinates": [lon, lat]},
                             datetime="2017-12-01/2026-12-31").items():
            if it.properties.get("eo:cloud_cover", 0) > 95:
                continue
            rows.append(dict(collection=c, id=it.id, time=pd.Timestamp(it.datetime).tz_convert(None),
                             hrefs={k: it.assets[k].href for k in BANDS[c] + ["Fmask"]}))
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
    fmask_cache, out = {}, []
    for d in dates:
        end = d + pd.Timedelta(days=1)
        cand = sc[(sc.time < end) & (sc.time >= end - pd.Timedelta(days=WINDOW_DAYS))]
        rec = dict(site_id=int(site.site_id), date=d, tried=0)
        for c in cand.head(MAX_TRIES).itertuples():
            rec["tried"] += 1
            if c.id not in fmask_cache:
                try:
                    fmask_cache[c.id] = read3x3(c.hrefs["Fmask"], lon, lat)
                except Exception:
                    fmask_cache[c.id] = None
            if not clear(fmask_cache[c.id]):
                continue
            try:
                vals = [read3x3(c.hrefs[b], lon, lat) for b in BANDS[c.collection]]
            except Exception:
                continue
            if any(v is None or (v == -9999).any() for v in vals):
                continue
            rec.update({n: float(v.mean()) * 1e-4 for n, v in zip(NAMES, vals)})
            rec.update(scene=c.id, sensor=c.collection[-3:], age_days=float((d - c.time.floor("D")).days))
            break
        out.append(rec)
    df = pd.DataFrame(out)
    df.to_parquet(PARTS / f"{int(site.site_id)}.parquet", index=False)
    return int(site.site_id), len(dates), int(df.get("nir", pd.Series(dtype=float)).notna().sum())


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
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
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
    print(f"wrote {OUT}: {len(allp)} site-dates, clear {int(allp.nir.notna().sum())}")


if __name__ == "__main__":
    main()
