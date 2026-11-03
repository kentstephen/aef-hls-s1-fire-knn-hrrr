# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "pillow"]
# ///
"""Elevation, slope and aspect at every site from Mapterhorn (tiles.mapterhorn.com: Terrarium-encoded
512 px WebP tiles, 10 m sources country-wide in the US), the terrain the map will use too.

Zoom 13 (about 7 m per pixel at 40 N). Slope and aspect from a plane fit over the 21 x 21 pixels
(about 150 m) around the site, so one rough pixel does not set them. Aspect is kept as northness (cos)
and eastness (sin). Tiles are cached in data/mapterhorn/ (gitignored, about 300 KB each).

    uv run scripts/sample_terrain.py
"""
import io
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "mapterhorn"
Z, TILE, HALF = 13, 512, 10


def tile(x, y):
    p = CACHE / f"{Z}_{x}_{y}.webp"
    if not p.exists():
        for attempt in range(4):
            try:
                req = urllib.request.Request(f"https://tiles.mapterhorn.com/{Z}/{x}/{y}.webp", headers={"User-Agent": "aef-hls-s1-fire-knn"})
                p.write_bytes(urllib.request.urlopen(req, timeout=60).read())
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 + 3 * attempt)
    rgb = np.asarray(Image.open(io.BytesIO(p.read_bytes())).convert("RGB")).astype(np.float64)
    return rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768


def window(lon, lat):
    """The (2 HALF + 1) square of elevations around a point, across tile edges, and its pixel size in m."""
    n = 2 ** Z * TILE
    px = (lon + 180) / 360 * n
    py = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    cx, cy = int(px), int(py)
    tiles = {}
    out = np.empty((2 * HALF + 1, 2 * HALF + 1))
    for j, gy in enumerate(range(cy - HALF, cy + HALF + 1)):
        for i, gx in enumerate(range(cx - HALF, cx + HALF + 1)):
            k = (gx // TILE, gy // TILE)
            if k not in tiles:
                tiles[k] = tile(*k)
            out[j, i] = tiles[k][gy % TILE, gx % TILE]
    m = 40075016.686 * math.cos(math.radians(lat)) / n
    return out, m


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    sites = pd.read_parquet(ROOT / "data" / "sites.parquet")
    out = []
    for s in sites.itertuples():
        z, m = window(s.longitude, s.latitude)
        yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]]
        A = np.c_[xx.ravel() * m, -yy.ravel() * m, np.ones(z.size)]
        (gx, gy, _), *_ = np.linalg.lstsq(A, z.ravel(), rcond=None)  # dz/dx east, dz/dy north
        aspect = (math.degrees(math.atan2(-gx, -gy)) + 360) % 360  # the direction the slope faces
        out.append(dict(site_id=s.site_id, elev_m=float(z[HALF, HALF]), slope_deg=math.degrees(math.atan(math.hypot(gx, gy))),
                        northness=math.cos(math.radians(aspect)), eastness=math.sin(math.radians(aspect))))
    df = pd.DataFrame(out)
    df.to_parquet(ROOT / "data" / "terrain.parquet", index=False)
    chk = df.merge(sites[["site_id", "elevation"]], on="site_id")
    print(df.describe().T[["mean", "min", "max"]])
    print("median |Mapterhorn - site's own elevation (ft -> m)|:", float((chk.elev_m - chk.elevation * 0.3048).abs().median()), "m")


if __name__ == "__main__":
    main()
