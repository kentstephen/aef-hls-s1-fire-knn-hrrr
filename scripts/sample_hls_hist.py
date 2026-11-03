# /// script
# requires-python = ">=3.12"
# dependencies = ["pystac-client", "planetary-computer", "rasterio", "numpy", "pandas", "pyarrow", "pyproj"]
# ///
"""HLS history before every sample: one clear scene per 8-day step over the 120 days before each
sample date (2020-02 on), so a model can see the greening and drying curve, not only the latest scene.

Per site: one STAC search (sample_hls.scenes). The steps are fixed 8-day bins from 2019-01-01; only bins
that fall inside some sample's 120-day window are read. Per bin: scenes newest first, up to 3 tried;
the first with a clear Fmask over the 3 x 3 pixels (90 m) gives red, NIR, SWIR 1 and SWIR 2 (enough for
NDVI, NDMI and NBR). One row per clear bin: site_id, time, sensor, the four bands.

Parts per site in data/hls_hist_parts/, resumable. Measured earlier with 16 sites: 2.2 GB RSS; expect
about 3 GB with 24.

    uv run scripts/sample_hls_hist.py [--sites 24] [--limit N]
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pystac_client

from sample_hls import BANDS, NAMES, clear, read3x3, scenes  # also sets the GDAL env

ROOT = Path(__file__).resolve().parent.parent
PARTS = ROOT / "data" / "hls_hist_parts"
OUT = ROOT / "data" / "hls_hist.parquet"
LOOKBACK = 120
BIN = 8
T0 = pd.Timestamp("2019-01-01")
MAX_TRIES = 3
KEEP = ["red", "nir", "swir1", "swir2"]


def needed_bins(dates):
    bins = set()
    for d in dates:
        a = (d - pd.Timedelta(days=LOOKBACK) - T0).days // BIN
        b = (d - T0).days // BIN
        bins.update(range(a, b + 1))
    return bins


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
    sc["bin"] = (sc.time - T0).dt.days // BIN
    sc = sc[sc.bin.isin(needed_bins(dates))]
    out = []
    for b, cand in sc.groupby("bin"):
        for c in cand.sort_values("time", ascending=False).head(MAX_TRIES).itertuples():
            try:
                if not clear(read3x3(c.hrefs["Fmask"], lon, lat)):
                    continue
                keep = [b_ for b_, n in zip(BANDS[c.collection], NAMES) if n in KEEP]
                vals = [read3x3(c.hrefs[k], lon, lat) for k in keep]
            except Exception:
                continue
            if any(v is None or (v == -9999).any() for v in vals):
                continue
            out.append(dict(site_id=int(site.site_id), time=c.time, sensor=c.collection[-3:],
                            **{n: float(v.mean()) * 1e-4 for n, v in zip(KEEP, vals)}))
            break
    df = pd.DataFrame(out, columns=["site_id", "time", "sensor"] + KEEP)
    df.to_parquet(PARTS / f"{int(site.site_id)}.parquet", index=False)
    return int(site.site_id), len(set(sc.bin)), len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sites = pd.read_parquet(ROOT / "data" / "sites.parquet").set_index("site_id", drop=False)
    smp = pd.read_parquet(ROOT / "data" / "samples.parquet")
    smp = smp[~smp.new_growth & (smp.date >= "2020-02-01")]
    sd = smp[["site_id", "date"]].drop_duplicates()
    PARTS.mkdir(parents=True, exist_ok=True)
    done = {int(p.stem) for p in PARTS.glob("*.parquet")}
    todo = [s for s in sd.site_id.unique() if s not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(todo)} sites to read ({len(done)} done)", flush=True)
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    jobs = [(sites.loc[s], sorted(sd[sd.site_id == s].date), cat) for s in todo]
    t0, nb, nc = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=a.sites) as ex:
        for n, (sid, bins, got) in enumerate(ex.map(site_job, jobs)):
            nb += bins
            nc += got
            el = time.time() - t0
            print(f"{n + 1}/{len(todo)} site {sid}: {got}/{bins} bins clear; all {nc}/{nb}, "
                  f"{el / (n + 1):.1f} s/site, eta {el / (n + 1) * (len(todo) - n - 1) / 60:.0f} min", flush=True)
    allp = pd.concat([pd.read_parquet(p) for p in PARTS.glob("*.parquet")], ignore_index=True)
    allp.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(allp)} clear scenes at {allp.site_id.nunique()} sites")


if __name__ == "__main__":
    main()
