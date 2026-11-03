# /// script
# requires-python = ">=3.12"
# dependencies = ["pystac-client", "planetary-computer", "rasterio", "numpy", "pandas", "pyarrow", "pyproj"]
# ///
"""Sentinel-1 history before every sample: one RTC scene per 12-day step over the 120 days before each
sample date (2020-02 on), so a model can see the backscatter curve, not only the latest scene.

Per site: one STAC search (sample_s1.scenes). Fixed 12-day bins from 2019-01-01 (about one repeat cycle);
only bins inside some sample's 120-day window are read. Per bin: the newest scene with valid VV and VH
over the 5 x 5 pixels (50 m), up to 2 tried. Orbit and track are kept, since the two tracks over a site
see it from different angles.

Parts per site in data/s1_hist_parts/, resumable. Under 1 GB of RAM with 16 sites.

    uv run scripts/sample_s1_hist.py [--sites 16] [--limit N]
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pystac_client

from sample_s1 import read5x5, scenes  # also sets the GDAL env

ROOT = Path(__file__).resolve().parent.parent
PARTS = ROOT / "data" / "s1_hist_parts"
OUT = ROOT / "data" / "s1_hist.parquet"
LOOKBACK = 120
BIN = 12
T0 = pd.Timestamp("2019-01-01")
MAX_TRIES = 2


def needed_bins(dates):
    bins = set()
    for d in dates:
        bins.update(range((d - pd.Timedelta(days=LOOKBACK) - T0).days // BIN, (d - T0).days // BIN + 1))
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
                vv, vh = read5x5(c.vv, lon, lat), read5x5(c.vh, lon, lat)
            except Exception:
                continue
            if vv is None or vh is None:
                continue
            out.append(dict(site_id=int(site.site_id), time=c.time, orbit=c.orbit, track=c.track, vv=vv, vh=vh))
            break
    df = pd.DataFrame(out, columns=["site_id", "time", "orbit", "track", "vv", "vh"])
    df.to_parquet(PARTS / f"{int(site.site_id)}.parquet", index=False)
    return int(site.site_id), len(set(sc.bin)), len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=16)
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
            print(f"{n + 1}/{len(todo)} site {sid}: {got}/{bins} bins; all {nc}/{nb}, "
                  f"{el / (n + 1):.1f} s/site, eta {el / (n + 1) * (len(todo) - n - 1) / 60:.0f} min", flush=True)
    allp = pd.concat([pd.read_parquet(p) for p in PARTS.glob("*.parquet")], ignore_index=True)
    allp.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(allp)} scenes at {allp.site_id.nunique()} sites")


if __name__ == "__main__":
    main()
