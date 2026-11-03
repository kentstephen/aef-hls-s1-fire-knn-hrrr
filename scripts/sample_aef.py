# /// script
# requires-python = ">=3.12"
# dependencies = ["zarr>=3", "obstore", "numpy", "pandas", "pyarrow"]
# ///
"""The AlphaEarth embedding at every site, for every year before a sample year (2017 to 2025), from
the 10 m Zarr mosaic (tge-labs/aef-mosaic on source.coop, read from the S3 host). One inner chunk per
(mosaic chunk, year): 64 x 256 x 256 int8, about 4 MB. Per site: the 10 m pixel and the
unit-normalized means of the 5 x 5 (50 m) and 15 x 15 (150 m) windows around it. Parts per chunk in
data/aef_parts/, resumable. 8 reads in flight: about 40 MB, peak RAM well under 1 GB.

    uv run scripts/sample_aef.py [--workers 8]
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from obstore.store import S3Store
from zarr.storage import ObjectStore

ROOT = Path(__file__).resolve().parent.parent
PARTS = ROOT / "data" / "aef_parts"
OUT = ROOT / "data" / "aef.parquet"
CHUNK = 256
YEAR0 = 2017


def dequant(q):
    f = q.astype(np.float32) / 127.5
    return np.sign(f) * f * f


def win_mean(block, x, y, h):
    y0, y1, x0, x1 = max(0, y - h), min(CHUNK, y + h + 1), max(0, x - h), min(CHUNK, x + h + 1)
    w = block[:, y0:y1, x0:x1].reshape(64, -1)
    ok = ~(w == -128).all(axis=0)
    if not ok.any():
        return np.full(64, np.nan, np.float32)
    m = dequant(w[:, ok]).mean(axis=1)
    n = np.linalg.norm(m)
    return (m / n if n > 0 else m).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    sites = pd.read_parquet(ROOT / "data" / "sites.parquet")
    smp = pd.read_parquet(ROOT / "data" / "samples.parquet")
    need = smp.assign(year=smp.date.dt.year - 1)[["site_id", "year"]].drop_duplicates()
    need = need[need.year >= YEAR0].merge(sites[["site_id", "mx", "my"]], on="site_id")
    need["key"] = (need.my // CHUNK).astype(str) + "_" + (need.mx // CHUNK).astype(str) + "_" + need.year.astype(str)
    PARTS.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in PARTS.glob("*.parquet")}
    keys = [k for k in need.key.unique() if k not in done]
    print(f"{len(keys)} chunks to read ({len(done)} done), {len(need)} site-years", flush=True)
    store = ObjectStore(S3Store("us-west-2.opendata.source.coop", prefix="tge-labs/aef-mosaic", region="us-west-2", skip_signature=True,
                                client_options={"timeout": "60s", "connect_timeout": "10s"}), read_only=True)
    arr = zarr.open_array(store, path="embeddings", mode="r")

    def fetch(k):
        rows = need[need.key == k]
        cy0 = int(rows.my.iloc[0] // CHUNK) * CHUNK
        cx0 = int(rows.mx.iloc[0] // CHUNK) * CHUNK
        for attempt in range(4):
            try:
                block = arr[int(rows.year.iloc[0]) - YEAR0, :, cy0:cy0 + CHUNK, cx0:cx0 + CHUNK]
                return k, rows, cx0, cy0, block
            except Exception as e:  # a dropped connection: try again
                if attempt == 3:
                    raise
                time.sleep(2 + attempt * 3)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for n, (k, rows, cx0, cy0, block) in enumerate(ex.map(fetch, keys)):
            out = []
            for r in rows.itertuples():
                x, y = int(r.mx - cx0), int(r.my - cy0)
                pix = block[:, y, x]
                out.append(dict(site_id=int(r.site_id), year=int(r.year), nodata=bool((pix == -128).all()),
                                pix=pix.tobytes(), w5=win_mean(block, x, y, 2).tobytes(), w15=win_mean(block, x, y, 7).tobytes()))
            pd.DataFrame(out).to_parquet(PARTS / f"{k}.parquet", index=False)
            el = time.time() - t0
            print(f"{n + 1}/{len(keys)} {el / (n + 1):.2f} s/chunk, eta {el / (n + 1) * (len(keys) - n - 1) / 60:.0f} min", flush=True)
    allp = pd.concat([pd.read_parquet(p) for p in PARTS.glob("*.parquet")], ignore_index=True)
    allp.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(allp)} site-years, nodata {int(allp.nodata.sum())}")


if __name__ == "__main__":
    main()
