# /// script
# requires-python = ">=3.12"
# dependencies = ["zarr>=3", "obstore", "numpy", "pandas", "pyarrow", "rasterio", "psutil", "shapely"]
# ///
"""Pixels for a fuel head that can see change: every embedding year, and LANDFIRE labels from four editions.

The first head learned one label edition (LF2024) and never saw a pixel's label change. Here each pixel carries
its AlphaEarth embedding for every year 2017 to 2025 and LANDFIRE's canopy cover (CC, %), canopy height (CH,
m x 10) and fuel model (FBFM40) from the 2016, 2022, 2024 and 2025 editions. An edition is paired with the
embedding of its own year (LF2016 with 2017, the first embedding year).

Windows (mosaic chunks, 256 x 256 at 10 m):
  random  the fuel lab's 580 windows across California (data/fuel_ca_windows.parquet)
  burn    up to --burn chunks at least 80 % inside 2020 to 2023 perimeters (4 per fire, largest fires first),
          so burned ground is common enough to learn
  test    the chunks scripts/fire_change.py read, inside and outside Caldor, Dixie, Creek, Park and Bobcat
Train windows (random, burn) within 10 km of a test fire are dropped, so the test fires are never seen.

Per window --px random pixels with an embedding in every year. Writes data/change/<kind>_<cy>_<cx>.npz
(q: int8 [n, 9, 64]; rows, cols; cc_<ed>, ch_<ed>, fb_<ed>: int16 [n]) and data/change/windows.parquet.
Resumable. RAM: about 60 MB per window in flight, under 1.5 GB at the defaults; stops past --max-gb.

    uv run scripts/sample_change.py [--burn 200] [--px 1500] [--workers 6] [--max-gb 4]
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import shapely
from rasterio.io import MemoryFile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fire_change import FIRES, pixel_xy  # noqa: E402
from fire_test_grids import open_aef  # noqa: E402
from fuel_common import CHUNK, MOS_RES, MOS_X0, MOS_Y0, YEAR0, get  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "change"
YEARS = list(range(2017, 2026))
EDITIONS = [2016, 2022, 2024, 2025]
LAYERS = {"cc": "CC", "ch": "CH", "fb": "FBFM40"}
LF_URL = "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF{v}/LF{v}_{lyr}_CONUS/ImageServer/exportImage"


def chunk_center(cy, cx):
    return MOS_X0 + (cx + 0.5) * CHUNK * MOS_RES, MOS_Y0 - (cy + 0.5) * CHUNK * MOS_RES


def landfire(cy, cx, ed, lyr):
    lon0, lat0 = MOS_X0 + cx * CHUNK * MOS_RES, MOS_Y0 - cy * CHUNK * MOS_RES
    lon1, lat1 = lon0 + CHUNK * MOS_RES, lat0 - CHUNK * MOS_RES
    b = get(LF_URL.format(v=ed, lyr=lyr), dict(bbox=f"{lon0},{lat1},{lon1},{lat0}", bboxSR=4326, imageSR=4326,
                                               size=f"{CHUNK},{CHUNK}", format="tiff", pixelType="S16",
                                               interpolation="RSP_NearestNeighbor", f="image"))
    with MemoryFile(b) as m, m.open() as r:
        return r.read(1).astype(np.int16)


def burn_windows(per, test_ids, n, rng):
    """Chunks at least 80 % inside 2020 to 2023 perimeters, 4 per fire, largest fires first."""
    out = []
    for s in per[(per.year <= 2023) & ~per.fire_id.isin(test_ids)].sort_values("acres", ascending=False).itertuples():
        g = shapely.make_valid(s.g)
        shapely.prepare(g)
        x0, y0, x1, y1 = g.bounds
        cys = range(int((MOS_Y0 - y1) / MOS_RES) // CHUNK, int((MOS_Y0 - y0) / MOS_RES) // CHUNK + 1)
        cxs = range(int((x0 - MOS_X0) / MOS_RES) // CHUNK, int((x1 - MOS_X0) / MOS_RES) // CHUNK + 1)
        cand = []
        for cy in cys:
            for cx in cxs:
                lon, lat = pixel_xy(cy, cx, 32)
                if shapely.contains_xy(g, lon, lat).mean() >= 0.8:
                    cand.append((cy, cx))
        for i in rng.permutation(len(cand))[:4]:
            out.append((*cand[i], s.name, int(s.year)))
        if len(out) >= n:
            break
    return out[:n]


def one(arr, w, px, rng_seed):
    path = OUT / f"{w.kind}_{w.cy}_{w.cx}.npz"
    if path.exists():
        return 0
    q = np.stack([arr[y - YEAR0, :, w.cy * CHUNK:(w.cy + 1) * CHUNK, w.cx * CHUNK:(w.cx + 1) * CHUNK] for y in YEARS])
    q = q.reshape(len(YEARS), 64, -1)
    ok = np.flatnonzero(~(q == -128).all(axis=1).any(axis=0))
    if len(ok) < 200:
        np.savez_compressed(path, q=np.zeros((0, len(YEARS), 64), np.int8))
        return 0
    pick = np.sort(np.random.default_rng(rng_seed).choice(ok, min(px, len(ok)), replace=False))
    out = dict(q=np.ascontiguousarray(q[:, :, pick].transpose(2, 0, 1)), rows=(pick // CHUNK).astype(np.int16),
               cols=(pick % CHUNK).astype(np.int16))
    del q
    jobs = [(ed, k, lyr) for ed in EDITIONS for k, lyr in LAYERS.items()]
    with ThreadPoolExecutor(max_workers=4) as ex:
        for (ed, k, _), g in zip(jobs, ex.map(lambda j: landfire(w.cy, w.cx, j[0], j[2]), jobs)):
            out[f"{k}_{ed}"] = g.ravel()[pick]
    np.savez_compressed(path, **out)
    return len(pick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burn", type=int, default=200)
    ap.add_argument("--px", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-gb", type=float, default=4.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    per = pd.read_parquet(ROOT / "data" / "fire_test" / "perimeters.parquet")
    per["g"] = shapely.from_wkb(per.geom)
    tests = [per[(per.name == n) & (per.year == y)].sort_values("acres").iloc[-1] for n, y, _ in FIRES]
    test_ids = [t.fire_id for t in tests]
    # 10 km around the test fires, in degrees (0.09 deg lat ~ 10 km; lon a little wider at 35 to 41 N)
    keep_out = shapely.union_all([shapely.buffer(shapely.make_valid(t.g), 0.12) for t in tests])
    shapely.prepare(keep_out)

    rnd = pd.read_parquet(ROOT / "data" / "fuel_ca_windows.parquet")
    rows = [dict(kind="random", cy=int(r.cy), cx=int(r.cx), fire="", fire_year=0) for r in rnd.itertuples()]
    rows += [dict(kind="burn", cy=cy, cx=cx, fire=f, fire_year=y) for cy, cx, f, y in burn_windows(per, test_ids, a.burn, rng)]
    w = pd.DataFrame(rows).drop_duplicates(["cy", "cx"])
    c = np.array([chunk_center(r.cy, r.cx) for r in w.itertuples()])
    near = shapely.contains_xy(keep_out, c[:, 0], c[:, 1])
    print(f"train windows: {len(w)}, dropped near test fires: {near.sum()}", flush=True)
    w = w[~near]
    fc = pd.read_parquet(ROOT / "data" / "fire_change" / "rows.parquet")[["fire", "fire_year", "side", "cy", "cx"]].drop_duplicates()
    t = pd.DataFrame(dict(kind="test_" + fc.side, cy=fc.cy, cx=fc.cx, fire=fc.fire, fire_year=fc.fire_year))
    w = pd.concat([w, t], ignore_index=True)
    w.to_parquet(OUT / "windows.parquet")
    print(w.kind.value_counts().to_string(), flush=True)

    arr = open_aef()
    proc = psutil.Process(os.getpid())
    peak, t0, todo = 0.0, time.time(), list(w.itertuples())
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, arr, r, a.px, i) for i, r in enumerate(todo)]
        for i, f in enumerate(futs):
            try:
                f.result()
            except Exception as e:
                print(f"window {todo[i].kind} {todo[i].cy}_{todo[i].cx}: {e}", flush=True)
            gb = proc.memory_info().rss / 1e9
            peak = max(peak, gb)
            if gb > a.max_gb:
                print(f"STOP: {gb:.1f} GB > --max-gb {a.max_gb}; rerun to resume", flush=True)
                os._exit(2)
            if i % 25 == 0 or i == len(futs) - 1:
                el = time.time() - t0
                print(f"{i + 1}/{len(futs)} {el / (i + 1):.1f} s/window, eta {el / (i + 1) * (len(futs) - i - 1) / 60:.1f} min, "
                      f"rss {gb:.2f} GB (peak {peak:.2f})", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min, peak rss {peak:.2f} GB")


if __name__ == "__main__":
    main()
