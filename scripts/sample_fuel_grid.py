# /// script
# requires-python = ">=3.12"
# dependencies = ["zarr>=3", "obstore", "numpy", "pandas", "pyarrow", "rasterio"]
# ///
"""Full 256 x 256 grids for the fuel / not-fuel lab, one file per window from data/fuel_ca_windows.parquet.

Per window (the AlphaEarth mosaic chunk grid, EPSG:4326, about 10 m):
  aef   AlphaEarth 2023, 2024, 2025: int8 (3, 64, 256, 256); all bands -128 = no data
  fbfm  LANDFIRE LF2024 FBFM40 (30 m, nearest neighbor onto the grid); 91 to 99 non-burnable
  evt   LANDFIRE LF2024 EVT (7299 Developed-Roads)
  wc    ESA WorldCover 2021 v200 (10 m, nearest pixel): 10 trees, 20 shrub, 30 grass, 40 crop, 50 built-up,
        60 bare / sparse, 70 snow, 80 water, 90 wetland, 95 mangrove, 100 moss / lichen; 0 = none

Full grids, not samples, so the lab can build neighbor means, drop pixels on mixed 30 m label edges and compare
years. Files data/fuel_grid/<cy>_<cx>.npz (about 9 MB each, 5 GB for 580 windows). Resumable.
RAM: about 40 MB per window in flight; under 1 GB with 6 workers.

    uv run scripts/sample_fuel_grid.py [--workers 6] [--limit N]
"""
import argparse
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import zarr
from rasterio.io import MemoryFile

from fuel_common import CHUNK, LF, MOS_RES, MOS_X0, MOS_Y0, YEAR0, get

ROOT = Path(__file__).resolve().parent.parent
GRID = ROOT / "data" / "fuel_grid"
YEARS = (2023, 2024, 2025)
WC = "/vsis3/esa-worldcover/v200/2021/map/ESA_WorldCover_10m_2021_v200_{}_Map.tif"
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_CACHEMAX="64", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2",
                  CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", VSI_CACHE="FALSE", AWS_NO_SIGN_REQUEST="YES", AWS_REGION="eu-central-1")


def landfire(layer, lon0, lat0):
    lon1, lat1 = lon0 + CHUNK * MOS_RES, lat0 - CHUNK * MOS_RES
    b = get(LF.format(layer), dict(bbox=f"{lon0},{lat1},{lon1},{lat0}", bboxSR=4326, imageSR=4326, size=f"{CHUNK},{CHUNK}",
                                   format="tiff", pixelType="S16", interpolation="RSP_NearestNeighbor", f="image"))
    with MemoryFile(b) as m, m.open() as r:
        return r.read(1).astype(np.int16)


def worldcover(lon0, lat0):
    """WorldCover class at each grid pixel center; tiles are 3 x 3 degrees named by their lower-left corner."""
    lon = lon0 + (np.arange(CHUNK) + 0.5) * MOS_RES
    lat = lat0 - (np.arange(CHUNK) + 0.5) * MOS_RES
    LON, LAT = np.meshgrid(lon, lat)
    out = np.zeros((CHUNK, CHUNK), np.uint8)
    tx, ty = (np.floor(LON / 3) * 3).astype(int), (np.floor(LAT / 3) * 3).astype(int)
    for x, y in set(zip(tx.ravel().tolist(), ty.ravel().tolist())):
        name = f"{'N' if y >= 0 else 'S'}{abs(y):02d}{'E' if x >= 0 else 'W'}{abs(x):03d}"
        m = (tx == x) & (ty == y)
        for attempt in range(4):
            try:
                with rasterio.open(WC.format(name)) as r:
                    rows, cols = rasterio.transform.rowcol(r.transform, LON[m], LAT[m])
                    rows, cols = np.asarray(rows), np.asarray(cols)
                    r0, c0 = rows.min(), cols.min()
                    a = r.read(1, window=((r0, rows.max() + 1), (c0, cols.max() + 1)))
                    out[m] = a[rows - r0, cols - c0]
                break
            except rasterio.errors.RasterioIOError:
                if attempt == 3:
                    raise
                time.sleep(3 + 3 * attempt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    GRID.mkdir(parents=True, exist_ok=True)
    zarr.config.set({"async.concurrency": 32})
    from obstore.store import S3Store
    from zarr.storage import ObjectStore
    aef = zarr.open_array(ObjectStore(S3Store("us-west-2.opendata.source.coop", prefix="tge-labs/aef-mosaic", region="us-west-2",
                                              skip_signature=True, client_options={"timeout": "120s"}), read_only=True),
                          path="embeddings", mode="r")
    wins = pd.read_parquet(ROOT / "data" / "fuel_ca_windows.parquet")
    todo = [(int(w.cy), int(w.cx)) for w in wins.itertuples() if not (GRID / f"{w.cy}_{w.cx}.npz").exists()]
    print(f"{len(todo)} windows to read ({len(wins) - len(todo)} done)", flush=True)
    if a.limit:
        todo = todo[: a.limit]
    y0, y1 = YEARS[0] - YEAR0, YEARS[-1] - YEAR0 + 1

    def job(k):
        cy, cx = k
        for attempt in range(5):
            try:
                block = np.asarray(aef[y0:y1, :, cy * CHUNK:(cy + 1) * CHUNK, cx * CHUNK:(cx + 1) * CHUNK])
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(5 + 5 * attempt)
        lon0, lat0 = MOS_X0 + cx * CHUNK * MOS_RES, MOS_Y0 - cy * CHUNK * MOS_RES
        valid = ~(block[1] == -128).all(axis=0)
        if valid.sum() < 1000:
            np.savez_compressed(GRID / f"{cy}_{cx}.npz", empty=np.array(1))
            return k, 0
        tmp = GRID / f"{cy}_{cx}.tmp.npz"
        np.savez_compressed(tmp, aef=block, fbfm=landfire("FBFM40", lon0, lat0), evt=landfire("EVT", lon0, lat0),
                            wc=worldcover(lon0, lat0), lon0=np.float64(lon0), lat0=np.float64(lat0), years=np.array(YEARS))
        tmp.rename(GRID / f"{cy}_{cx}.npz")
        return k, int(valid.sum())

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for n, (k, m) in enumerate(ex.map(job, todo)):
            el = time.time() - t0
            print(f"{n + 1}/{len(todo)} window {k}: {m} valid px, {el / (n + 1):.1f} s/window, "
                  f"eta {el / (n + 1) * (len(todo) - n - 1) / 60:.0f} min", flush=True)
    print(f"done: {len(list(GRID.glob('*.npz')))} window files in {GRID}")


if __name__ == "__main__":
    main()
