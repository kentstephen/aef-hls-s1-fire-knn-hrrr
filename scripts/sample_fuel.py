# /// script
# requires-python = ">=3.12"
# dependencies = ["zarr>=3", "obstore", "numpy", "pandas", "pyarrow", "pyproj", "rasterio", "pystac-client"]
# ///
"""Pixels for the fuel / not-fuel test, within 150 km of Redding.

Windows are AlphaEarth mosaic chunks (256 x 256 pixels at 10 m, about 2 x 2.5 km): 70 at random in the
circle plus 10 on towns, so built-up ground and roads show up often enough to score. In each window,
1,000 random pixels that AEF covers. Per pixel:
  labels   LANDFIRE LF2024 FBFM40 (fuel model; 91 to 99 are non-burnable) and EVT (7299 = Developed-Roads),
           exported from LANDFIRE's image services onto the AEF chunk grid (nearest neighbor from 30 m).
  aef      AlphaEarth 2024, 64 int8 (dequantized in the fit)
  tessera  TESSERA v1.1 2024, 128 float32;  olmo  OlmoEarth v1.3 2024, 128 float32 (nearest 10 m pixel)
  s2       Sentinel-2 L2A 2024, three seasons (Apr-May, Jul-Aug, Oct-Nov), 10 bands each: the
           no-embedding rival. Per season the least cloudy scene whose SCL is mostly clear over the window.

One part per window in data/fuel_parts/, resumable. Downloads: AEF 4 MB and OlmoEarth 4 to 16 MB per
window, S2 about 30 small window reads. Under 1.5 GB RAM with 6 windows in flight.

    uv run scripts/sample_fuel.py [--workers 6] [--limit N]
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
import zarr
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.windows import Window

from make_samples import MOS_RES, MOS_X0, MOS_Y0, REDDING, haversine_km
from sample_aef import CHUNK, YEAR0
from sample_embed import T0 as EMB_T0, open_store

ROOT = Path(__file__).resolve().parent.parent
PARTS = ROOT / "data" / "fuel_parts"
OUT = ROOT / "data" / "fuel.parquet"
YEAR = 2024
N_RANDOM, PER_WIN = 70, 1000
TOWNS = {"Redding": (-122.39, 40.59), "Anderson": (-122.30, 40.45), "Red Bluff": (-122.24, 40.18), "Chico": (-121.84, 39.73),
         "Oroville": (-121.56, 39.51), "Corning": (-122.18, 39.93), "Weaverville": (-122.94, 40.73), "Mount Shasta": (-122.31, 41.31),
         "Burney": (-121.66, 40.88), "Susanville": (-120.65, 40.42)}
SEASONS = {"spring": ("2024-04-01", "2024-05-31"), "summer": ("2024-07-01", "2024-08-31"), "fall": ("2024-10-01", "2024-11-30")}
S2B = ["blue", "green", "red", "nir", "rededge1", "rededge2", "rededge3", "nir08", "swir16", "swir22"]
LF = "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2024/LF2024_{}_CONUS/ImageServer/exportImage"
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                  GDAL_CACHEMAX="64", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                  VSI_CACHE="FALSE", AWS_NO_SIGN_REQUEST="YES")


def pick_windows(seed=0):
    """Chunk indices (cy, cx): towns first, then random points in the 150 km circle, no repeats."""
    rng = np.random.default_rng(seed)
    out, names = [], []
    def add(lon, lat, name):
        k = (int((MOS_Y0 - lat) / MOS_RES) // CHUNK, int((lon - MOS_X0) / MOS_RES) // CHUNK)
        if k not in out:
            out.append(k)
            names.append(name)
    for name, (lon, lat) in TOWNS.items():
        add(lon, lat, name)
    while len(out) < len(TOWNS) + N_RANDOM:
        lon, lat = REDDING[0] + rng.uniform(-1.8, 1.8), REDDING[1] + rng.uniform(-1.4, 1.4)
        if haversine_km(REDDING[0], REDDING[1], lon, lat) < 150:
            add(lon, lat, "random")
    return list(zip(out, names))


def get(url, params, tries=4):
    import urllib.parse, urllib.request
    q = url + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            with urllib.request.urlopen(q, timeout=120) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(3 + 3 * i)


def landfire(layer, lon0, lat0):
    """The LF2024 layer on the chunk's own 256 x 256 grid (EPSG:4326, nearest neighbor)."""
    lon1, lat1 = lon0 + CHUNK * MOS_RES, lat0 - CHUNK * MOS_RES
    b = get(LF.format(layer), dict(bbox=f"{lon0},{lat1},{lon1},{lat0}", bboxSR=4326, imageSR=4326, size=f"{CHUNK},{CHUNK}",
                                   format="tiff", pixelType="S16", interpolation="RSP_NearestNeighbor", f="image"))
    with MemoryFile(b) as m, m.open() as r:
        return r.read(1)


def sample_raster(href, lon, lat):
    """Values of one band at the points (nearest pixel), 0 where off the tile."""
    with rasterio.open(href) as r:
        xs, ys = Transformer.from_crs(4326, r.crs, always_xy=True).transform(lon, lat)
        rows, cols = rasterio.transform.rowcol(r.transform, xs, ys)
        rows, cols = np.asarray(rows), np.asarray(cols)
        r0, c0 = rows.min(), cols.min()
        a = r.read(1, window=Window(c0, r0, cols.max() - c0 + 1, rows.max() - r0 + 1), boundless=True, fill_value=0)
        return a[rows - r0, cols - c0]


def s2_season(cat, lon, lat, start, end):
    """10 bands at the points from the least cloudy mostly clear scene of the season; NaN where SCL is not clear."""
    bbox = [float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())]
    items = sorted(cat.search(collections=["sentinel-2-c1-l2a"], bbox=bbox, datetime=f"{start}/{end}",
                              query={"eo:cloud_cover": {"lt": 40}}).items(), key=lambda it: it.properties["eo:cloud_cover"])
    for it in items[:6]:
        scl = sample_raster(it.assets["scl"].href, lon, lat)
        ok = np.isin(scl, (4, 5, 6))
        if ok.mean() < 0.8:
            continue
        out = np.full((len(lon), len(S2B)), np.nan, np.float32)
        for j, b in enumerate(S2B):
            v = sample_raster(it.assets[b].href, lon, lat).astype(np.float32)
            out[:, j] = np.where(ok & (v > 0), v * 1e-4 - 0.1, np.nan)
        return out, it.id
    return np.full((len(lon), len(S2B)), np.nan, np.float32), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)
    zarr.config.set({"async.concurrency": 32})
    from obstore.store import S3Store
    from zarr.storage import ObjectStore
    aef = zarr.open_array(ObjectStore(S3Store("us-west-2.opendata.source.coop", prefix="tge-labs/aef-mosaic", region="us-west-2",
                                              skip_signature=True, client_options={"timeout": "60s"}), read_only=True), path="embeddings", mode="r")
    emb = {s: zarr.open_group(open_store(s), mode="r", path="utm10") for s in ("tessera", "olmo")}
    utm = Transformer.from_crs(4326, emb["olmo"].attrs["proj:code"], always_xy=True)
    cat = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
    wins = pick_windows()
    todo = [(i, k, n) for i, (k, n) in enumerate(wins) if not (PARTS / f"{i}.parquet").exists()]
    print(f"{len(todo)} windows to read ({len(wins) - len(todo)} done)", flush=True)
    if a.limit:
        todo = todo[: a.limit]

    def job(arg):
        i, (cy, cx), name = arg
        rng = np.random.default_rng(1000 + i)
        for attempt in range(4):
            try:
                block = aef[YEAR - YEAR0, :, cy * CHUNK:(cy + 1) * CHUNK, cx * CHUNK:(cx + 1) * CHUNK]
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(3 + 3 * attempt)
        valid = np.argwhere(~(block == -128).all(axis=0))
        if len(valid) < 100:
            pd.DataFrame().to_parquet(PARTS / f"{i}.parquet")  # ocean or no AEF: an empty part marks it done
            return i, name, 0
        pick = valid[rng.choice(len(valid), size=min(PER_WIN, len(valid)), replace=False)]
        r, c = pick[:, 0], pick[:, 1]
        lon0, lat0 = MOS_X0 + cx * CHUNK * MOS_RES, MOS_Y0 - cy * CHUNK * MOS_RES
        lon, lat = lon0 + (c + 0.5) * MOS_RES, lat0 - (r + 0.5) * MOS_RES
        df = pd.DataFrame(dict(win=i, win_name=name, lon=lon, lat=lat,
                               fbfm=landfire("FBFM40", lon0, lat0)[r, c], evt=landfire("EVT", lon0, lat0)[r, c]))
        df["aef"] = [block[:, y, x].tobytes() for y, x in zip(r, c)]
        x, y = utm.transform(lon, lat)
        for s, g in emb.items():
            ta, _, x0, _, te, y0 = g.attrs["spatial:transform"]
            col, row = ((x - x0) // ta).astype(int), ((y - y0) // te).astype(int)
            sl = (YEAR - EMB_T0, slice(None), slice(row.min(), row.max() + 1), slice(col.min(), col.max() + 1))
            q = g["embeddings"][sl]
            if s == "tessera":
                v = q.astype(np.float32) * g["scales"][sl[0], sl[2], sl[3]][None]
            else:
                f = np.where(q == -128, np.nan, q).astype(np.float32) / 127.5
                v = np.sign(f) * f * f
            vv = v[:, row - row.min(), col - col.min()].T
            df[s] = [p.astype(np.float32).tobytes() for p in vv]
        for season, (s0, s1) in SEASONS.items():
            vals, sid = s2_season(cat, lon, lat, s0, s1)
            df[f"s2_{season}"] = [p.tobytes() for p in vals]
            df[f"s2_{season}_scene"] = sid
        df.to_parquet(PARTS / f"{i}.parquet", index=False)
        return i, name, len(df)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for n, (i, name, m) in enumerate(ex.map(job, todo)):
            el = time.time() - t0
            print(f"{n + 1}/{len(todo)} window {i} ({name}): {m} pixels, {el / (n + 1):.1f} s/window, "
                  f"eta {el / (n + 1) * (len(todo) - n - 1) / 60:.0f} min", flush=True)
    allp = pd.concat([pd.read_parquet(p) for p in PARTS.glob("*.parquet")], ignore_index=True)
    allp.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(allp)} pixels in {allp.win.nunique()} windows, non-burnable {(allp.fbfm.between(91, 99)).mean():.1%}, "
          f"roads {(allp.evt == 7299).sum()}")


if __name__ == "__main__":
    main()
