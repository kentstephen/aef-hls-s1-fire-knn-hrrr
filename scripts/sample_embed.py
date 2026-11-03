# /// script
# requires-python = ">=3.12"
# dependencies = ["zarr>=3", "obstore", "numpy", "pandas", "pyarrow", "pyproj"]
# ///
"""TESSERA v1.1 or OlmoEarth v1.3 embeddings at every site, for the years before the sample years (2019 to 2025),
as sample_aef.py does for AlphaEarth: the 10 m pixel and the means of the 5 x 5 (50 m) and 15 x 15 (150 m)
windows around it, 128 float32 each. Both stores are Zarr v3 on source.coop, one group per UTM zone, 10 m.

  tessera  int8 times a per-pixel float32 scale (NaN scale = not embedded). Inner chunks 32 x 32: small reads.
  olmo     int8, -128 = nodata, sign(q) (q / 127.5)^2 like AlphaEarth. Inner chunks 64 x 256 x 256 (4 MB), so
           each site-year pulls 2 to 8 of them: about 50 MB per site, 15 to 20 GB for all sites.

One part per site in data/<source>_parts/, resumable. Peak RAM about 100 MB per site in flight (under 1 GB at 8).

    uv run scripts/sample_embed.py --source tessera|olmo [--workers 8]
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from obstore.store import HTTPStore, S3Store
from pyproj import Transformer
from zarr.storage import ObjectStore

ROOT = Path(__file__).resolve().parent.parent
YEARS = list(range(2019, 2026))  # embedding of the year before a 2020 to 2026 sample
T0 = 2017  # first year in both stores


def open_store(source):
    if source == "tessera":
        return ObjectStore(HTTPStore.from_url("https://data.source.coop/tessera/tessera/zarr/v1.1-dclimate"), read_only=True)
    return ObjectStore(S3Store.from_url("s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr", endpoint="https://data.source.coop",
                                        skip_signature=True, client_options={"timeout": "120s", "connect_timeout": "10s"}), read_only=True)


def window(v, h):
    """Mean of the (2h+1)^2 window at the center of v (years, 128, 15, 15), over embedded pixels."""
    c = v.shape[-1] // 2
    w = v[:, :, c - h:c + h + 1, c - h:c + h + 1].reshape(v.shape[0], v.shape[1], -1)
    return np.nanmean(w, axis=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["tessera", "olmo"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    parts = ROOT / "data" / f"{a.source}_parts"
    parts.mkdir(parents=True, exist_ok=True)
    zarr.config.set({"async.concurrency": 64})
    store = open_store(a.source)
    sites = pd.read_parquet(ROOT / "data" / "sites.parquet")
    todo = [r for r in sites.itertuples() if not (parts / f"{r.site_id}.parquet").exists()]
    print(f"{len(todo)} sites to read ({len(sites) - len(todo)} done)", flush=True)
    zones = {}

    def zone(lon):
        n = int((lon + 180) // 6) + 1
        if n not in zones:
            g = zarr.open_group(store, mode="r", path=f"utm{n:02d}")
            zones[n] = (g, Transformer.from_crs(4326, g.attrs["proj:code"], always_xy=True))
        return zones[n]

    for r in sites.head(1).itertuples():
        zone(r.longitude)  # open one group before the threads start

    def fetch(r):
        g, tr = zone(r.longitude)
        a_, _, x0, _, e_, y0 = g.attrs["spatial:transform"]
        x, y = tr.transform(r.longitude, r.latitude)
        col, row = int((x - x0) // a_), int((y - y0) // e_)
        sl = (slice(YEARS[0] - T0, YEARS[-1] - T0 + 1), slice(None), slice(row - 7, row + 8), slice(col - 7, col + 8))
        for attempt in range(4):
            try:
                q = g["embeddings"][sl]
                if a.source == "tessera":
                    s = g["scales"][sl[0], sl[2], sl[3]]
                    v = q.astype(np.float32) * s[:, None]  # NaN scale -> NaN
                else:
                    f = np.where(q == -128, np.nan, q).astype(np.float32) / 127.5
                    v = np.sign(f) * f * f
                return r, v
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 + attempt * 3)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for n, (r, v) in enumerate(ex.map(fetch, todo)):
            pix = v[:, :, 7, 7]
            w5, w15 = window(v, 2), window(v, 7)
            out = [dict(site_id=int(r.site_id), year=yr, nodata=bool(np.isnan(w15[i]).all()),
                        pix=pix[i].astype(np.float32).tobytes(), w5=w5[i].astype(np.float32).tobytes(), w15=w15[i].astype(np.float32).tobytes())
                   for i, yr in enumerate(YEARS)]
            pd.DataFrame(out).to_parquet(parts / f"{r.site_id}.parquet", index=False)
            el = time.time() - t0
            print(f"{n + 1}/{len(todo)} {el / (n + 1):.2f} s/site, eta {el / (n + 1) * (len(todo) - n - 1) / 60:.0f} min", flush=True)
    allp = pd.concat([pd.read_parquet(p) for p in parts.glob("*.parquet")], ignore_index=True)
    out = ROOT / "data" / f"{a.source}.parquet"
    allp.to_parquet(out, index=False)
    print(f"wrote {out}: {len(allp)} site-years, nodata {int(allp.nodata.sum())}")


if __name__ == "__main__":
    main()
