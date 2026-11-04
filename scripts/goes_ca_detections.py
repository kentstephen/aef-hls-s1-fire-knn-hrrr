# /// script
# requires-python = ">=3.12"
# dependencies = ["obstore", "h5py", "numpy", "pandas", "pyarrow", "psutil"]
# ///
"""Every GOES-West fire pixel over California in one year, from NOAA's bucket, one scan every --every minutes.

The product is ABI-L2-FDCC (Fire/Hot Spot Characterization, CONUS sector, a scan every 5 minutes) in
s3://noaa-goes18. Each file is about 270 KB, so it is fetched whole and opened with h5py from memory; only the
rows and columns over California are read. Kept: every pixel whose Mask is a fire category, low probability
included (10 to 15, and 30 to 35 for the temporally filtered ones), with its Power (MW) and the scan time.

Mask categories: 10 processed fire, 11 saturated, 12 cloud contaminated, 13 high, 14 medium, 15 low probability;
30 to 35 the same, temporally filtered.

Writes data/goes/parts/<year>-<doy>.parquet (one per day, resumable) and data/goes/det_<year>.parquet (all of
them, plus each pixel's lon / lat). Under 1 GB of RAM at 16 threads; stops on its own past --max-gb.

    uv run scripts/goes_ca_detections.py --year 2025 [--every 15] [--threads 16] [--days 1-365]
"""
import argparse
import io
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import obstore
import pandas as pd
import psutil
from obstore.store import S3Store

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "goes"
PARTS = OUT / "parts"
BUCKET, PRODUCT = "noaa-goes18", "ABI-L2-FDCC"
FIRE_CODES = (10, 11, 12, 13, 14, 15, 30, 31, 32, 33, 34, 35)
CA = (-124.6, 32.4, -113.9, 42.1)  # west, south, east, north

s3 = S3Store(BUCKET, region="us-east-1", skip_signature=True, client_options={"timeout": "60s", "connect_timeout": "10s"})


def scan_start(key):
    return datetime.strptime(key.split("_s")[1][:13], "%Y%j%H%M%S").replace(tzinfo=timezone.utc)


def list_day(year, doy):
    keys = []
    for b in obstore.list(s3, prefix=f"{PRODUCT}/{year}/{doy:03d}/"):
        keys += [o["path"] for o in b]
    # a scan written twice at a site handover carries start == end in its name; skip those
    return sorted((k for k in keys if k.endswith(".nc") and k.split("_s")[1][:13] != k.split("_e")[1][:13]), key=scan_start)


def pick(keys, every):
    """One scan per `every` minutes: the first in each slot."""
    out, seen = [], set()
    for k in keys:
        t = scan_start(k)
        slot = (t.hour * 60 + t.minute) // every
        if slot not in seen:
            seen.add(slot)
            out.append(k)
    return out


def proj_of(f):
    p = f["goes_imager_projection"].attrs
    return {k: float(np.asarray(p[k]).ravel()[0]) for k in ("semi_major_axis", "semi_minor_axis", "perspective_point_height",
                                                             "longitude_of_projection_origin")}


def lonlat_to_xy(lon, lat, p):
    """GOES fixed grid scan angles (radians) of lon / lat arrays (GOES-R PUG)."""
    req, rpol, H = p["semi_major_axis"], p["semi_minor_axis"], p["perspective_point_height"] + p["semi_major_axis"]
    lon0 = math.radians(p["longitude_of_projection_origin"])
    e2 = 1 - rpol**2 / req**2
    lat, lon = np.radians(lat), np.radians(lon)
    latc = np.arctan((rpol**2 / req**2) * np.tan(lat))
    rc = rpol / np.sqrt(1 - e2 * np.cos(latc) ** 2)
    sx = H - rc * np.cos(latc) * np.cos(lon - lon0)
    sy = -rc * np.cos(latc) * np.sin(lon - lon0)
    sz = rc * np.sin(latc)
    return np.arcsin(-sy / np.sqrt(sx * sx + sy * sy + sz * sz)), np.arctan(sz / sx)


def xy_to_lonlat(x, y, p):
    """Lon / lat of scan angle arrays (GOES-R PUG navigation); NaN off the earth."""
    req, rpol, H = p["semi_major_axis"], p["semi_minor_axis"], p["perspective_point_height"] + p["semi_major_axis"]
    lon0 = math.radians(p["longitude_of_projection_origin"])
    a = np.sin(x) ** 2 + np.cos(x) ** 2 * (np.cos(y) ** 2 + (req**2 / rpol**2) * np.sin(y) ** 2)
    b = -2 * H * np.cos(x) * np.cos(y)
    c = H**2 - req**2
    with np.errstate(invalid="ignore"):
        rs = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
    sx, sy, sz = rs * np.cos(x) * np.cos(y), -rs * np.sin(x), rs * np.cos(x) * np.sin(y)
    lat = np.degrees(np.arctan((req**2 / rpol**2) * sz / np.sqrt((H - sx) ** 2 + sy**2)))
    lon = np.degrees(lon0 - np.arctan(sy / (H - sx)))
    return lon, lat


def grid_of(f):
    xa, ya = f["x"].attrs, f["y"].attrs
    g = lambda a, k: float(np.asarray(a[k]).ravel()[0])  # noqa: E731
    return dict(x0=g(xa, "add_offset"), dx=g(xa, "scale_factor"), y0=g(ya, "add_offset"), dy=g(ya, "scale_factor"),
                nx=f["x"].shape[0], ny=f["y"].shape[0])


def ca_window(p, g):
    """Rows and columns of the GOES grid that cover the California box."""
    lon, lat = np.meshgrid(np.linspace(CA[0], CA[2], 60), np.linspace(CA[1], CA[3], 60))
    x, y = lonlat_to_xy(lon, lat, p)
    cols, rows = (x - g["x0"]) / g["dx"], (y - g["y0"]) / g["dy"]
    return (max(0, int(np.floor(rows.min())) - 2), min(g["ny"], int(np.ceil(rows.max())) + 3),
            max(0, int(np.floor(cols.min())) - 2), min(g["nx"], int(np.ceil(cols.max())) + 3))


def fetch(key):
    for attempt in range(5):
        try:
            return bytes(obstore.get(s3, key).bytes())
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 + 3 * attempt)


def read_scan(key):
    """Fire pixels over California in one scan, and the file's projection and grid (to check they hold)."""
    with h5py.File(io.BytesIO(fetch(key)), "r") as f:
        p, g = proj_of(f), grid_of(f)
        r0, r1, c0, c1 = ca_window(p, g)
        m = f["Mask"][r0:r1, c0:c1]
        rr, cc = np.nonzero(np.isin(m, FIRE_CODES))
        pw = f["Power"]
        pv = pw[r0:r1, c0:c1][rr, cc].astype(np.float64)
        sf, ao = pw.attrs.get("scale_factor"), pw.attrs.get("add_offset")
        fill = pw.attrs.get("_FillValue")
        if fill is not None:
            pv[pv == float(np.asarray(fill).ravel()[0])] = np.nan
        if sf is not None:
            pv = pv * float(np.asarray(sf).ravel()[0]) + (float(np.asarray(ao).ravel()[0]) if ao is not None else 0.0)
        return dict(t=scan_start(key), row=(rr + r0).astype(np.int16), col=(cc + c0).astype(np.int16),
                    mask=m[rr, cc].astype(np.int16), power=pv.astype(np.float32), proj=p, grid=g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--every", type=int, default=15, help="minutes between scans kept")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--days", default=None, help="day-of-year range, e.g. 180-190")
    ap.add_argument("--max-gb", type=float, default=2.0)
    a = ap.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)
    ndays = (datetime(a.year + 1, 1, 1) - datetime(a.year, 1, 1)).days
    d0, d1 = (map(int, a.days.split("-")) if a.days else (1, ndays))
    days = [d for d in range(d0, d1 + 1) if not (PARTS / f"{a.year}-{d:03d}.parquet").exists()]
    print(f"{len(days)} days to read ({d1 - d0 + 1 - len(days)} done), a scan every {a.every} min", flush=True)
    proc = psutil.Process(os.getpid())
    peak, t0, nfiles, ref = 0.0, time.time(), 0, None
    with ThreadPoolExecutor(a.threads) as ex:
        for i, doy in enumerate(days):
            keys = pick(list_day(a.year, doy), a.every)
            rows, bad = [], 0

            def safe(k):
                try:
                    return read_scan(k)
                except Exception as e:  # a broken file: say so and go on
                    print(f"skip {k}: {e}", flush=True)
                    return None

            for s in ex.map(safe, keys):
                if s is None:
                    bad += 1
                    continue
                form = (tuple(sorted(s["proj"].items())), tuple(sorted(s["grid"].items())))
                if ref is None:
                    ref = form
                    (OUT / f"grid_{a.year}.json").write_text(pd.Series(dict(**s["proj"], **s["grid"])).to_json(indent=1))
                elif form != ref:
                    sys.exit(f"projection or grid changed at {s['t']}: the lon / lat step assumes one grid; split the year")
                n = len(s["row"])
                rows.append(pd.DataFrame(dict(t=[s["t"]] * n, row=s["row"], col=s["col"], mask=s["mask"], power=s["power"])))
                rows.append(pd.DataFrame(dict(t=[s["t"]], row=np.int16(-1), col=np.int16(-1), mask=np.int16(-1), power=np.float32(np.nan))))
            df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["t", "row", "col", "mask", "power"])
            df.to_parquet(PARTS / f"{a.year}-{doy:03d}.parquet", index=False)  # row -1 marks a scan read with no fire
            nfiles += len(keys)
            gb = proc.memory_info().rss / 1e9
            peak = max(peak, gb)
            if gb > a.max_gb:
                print(f"STOP: {gb:.1f} GB > --max-gb {a.max_gb}; rerun to resume", flush=True)
                os._exit(2)
            el = time.time() - t0
            print(f"day {doy:03d}: {len(keys)} scans ({bad} skipped), {int((df.row >= 0).sum())} fire px; {i + 1}/{len(days)} days, "
                  f"{el / nfiles:.2f} s/scan, eta {el / (i + 1) * (len(days) - i - 1) / 60:.0f} min, rss {gb:.2f} GB (peak {peak:.2f})",
                  flush=True)

    parts = sorted(PARTS.glob(f"{a.year}-*.parquet"))
    det = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    scans = det.t.nunique()
    det = det[det.row >= 0].reset_index(drop=True)
    g = pd.read_json(OUT / f"grid_{a.year}.json", typ="series")
    x = g.x0 + g.dx * det.col.to_numpy(np.float64)
    y = g.y0 + g.dy * det.row.to_numpy(np.float64)
    det["lon"], det["lat"] = xy_to_lonlat(x, y, g)
    det = det[(det.lon >= CA[0]) & (det.lon <= CA[2]) & (det.lat >= CA[1]) & (det.lat <= CA[3])]
    det.to_parquet(OUT / f"det_{a.year}.parquet", index=False)
    print(f"{len(parts)} days, {scans} scans, {len(det)} fire px in the box, {det.groupby(['row', 'col']).ngroups} distinct GOES pixels")


if __name__ == "__main__":
    main()
