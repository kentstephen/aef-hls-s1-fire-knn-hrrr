# /// script
# requires-python = ">=3.12"
# dependencies = ["zarr>=3", "obstore", "numpy", "pandas", "pyarrow", "rasterio", "psutil", "shapely"]
# ///
"""Do the embeddings see a fire, and what grows back? Year by year, inside a burn and just outside it.

For a few large fires (timber, grass and oak, chaparral), a sample of mosaic chunks mostly inside the perimeter
and a sample 1 to 6 km outside it. Every embedding year 2017 to 2025 per chunk. Per year and side:

  sim_before  cosine of each pixel's embedding with its own embedding the year before the fire (1 = unchanged)
  sim_prev    cosine with the pixel's embedding the year before (year-to-year step)
  kinds       the fuel head's 11 kind probabilities, averaged, and the share of pixels where each kind leads
              (years 2019 on: the head reads each year with the two before)

Pixels inside any other perimeter on file (data/fire_test/perimeters.parquet, 100 acres and up, 2020 to 2026) are
left out of both sides. Fires before 2020 are not on file, so older scars are not removed.

Writes results/fire_change.json and data/fire_change/rows.parquet. Four reads at a time, ~400 MB each in flight:
peak RAM about 2 GB; stops on its own past --max-gb.

    uv run scripts/fire_change.py [--chunks 8] [--workers 4] [--max-gb 4]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fire_test_grids import Head, dequant, open_aef  # noqa: E402
from fuel_common import CHUNK, MOS_RES, MOS_X0, MOS_Y0, YEAR0  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "fire_change"
YEARS = list(range(2017, 2026))
FIRES = [("Dixie", 2021, "timber"), ("Caldor", 2021, "timber"), ("Creek", 2020, "timber"),
         ("PARK", 2024, "grass and oak"), ("Bobcat", 2020, "chaparral")]
KINDS = json.loads((ROOT / "models" / "fuel_head.json").read_text())["kinds"]


def kinds_of(head, x):
    """x (n, 128) raw [e, sd] -> kind probabilities (n, 11), averaged over the nets."""
    xs = (x - head.mu) / head.sc
    out = np.zeros((len(x), len(KINDS)), np.float32)
    for w1, b1, w2, b2 in head.nets:
        z = np.maximum(xs @ w1 + b1, 0) @ w2 + b2
        z = np.exp(z - z.max(1, keepdims=True))
        out += z / z.sum(1, keepdims=True) / len(head.nets)
    return out


def chunk_poly(cy, cx):
    lon0, lat0 = MOS_X0 + cx * CHUNK * MOS_RES, MOS_Y0 - cy * CHUNK * MOS_RES
    return shapely.box(lon0, lat0 - CHUNK * MOS_RES, lon0 + CHUNK * MOS_RES, lat0)


def pixel_xy(cy, cx, step=1):
    i = np.arange(0, CHUNK, step) + step / 2
    lon = MOS_X0 + (cx * CHUNK + i) * MOS_RES
    lat = MOS_Y0 - (cy * CHUNK + i) * MOS_RES
    return np.meshgrid(lon, lat)


def pick_chunks(geom, rng, n):
    """n chunks at least 80 % inside geom, n chunks fully outside with centers 1 to 6 km from it."""
    lat0 = (geom.bounds[1] + geom.bounds[3]) / 2
    km_lat, km_lon = 111.32, 111.32 * np.cos(np.radians(lat0))
    pad = 7 / km_lat * 1.5
    x0, y0, x1, y1 = geom.bounds
    cys = range(int((MOS_Y0 - (y1 + pad)) / MOS_RES) // CHUNK, int((MOS_Y0 - (y0 - pad)) / MOS_RES) // CHUNK + 1)
    cxs = range(int((x0 - pad - MOS_X0) / MOS_RES) // CHUNK, int((x1 + pad - MOS_X0) / MOS_RES) // CHUNK + 1)
    shapely.prepare(geom)
    km = shapely.transform(geom, lambda c: np.c_[(c[:, 0] - x0) * km_lon, (c[:, 1] - y0) * km_lat])
    shapely.prepare(km)
    inside, outside = [], []
    for cy in cys:
        for cx in cxs:
            box = chunk_poly(cy, cx)
            if not geom.intersects(box):
                c = box.centroid
                d = km.distance(shapely.Point((c.x - x0) * km_lon, (c.y - y0) * km_lat))
                if 1 <= d <= 6:
                    outside.append((cy, cx))
                continue
            lon, lat = pixel_xy(cy, cx, 16)
            if shapely.contains_xy(geom, lon, lat).mean() >= 0.8:
                inside.append((cy, cx))
    pick = lambda xs: [xs[i] for i in rng.choice(len(xs), min(n, len(xs)), replace=False)]  # noqa: E731
    return pick(inside), pick(outside), len(inside), len(outside)


def one_chunk(arr, head, fire, side, cy, cx, geom, others):
    blocks = []
    for y in YEARS:
        for attempt in range(4):
            try:
                blocks.append(arr[y - YEAR0, :, cy * CHUNK:(cy + 1) * CHUNK, cx * CHUNK:(cx + 1) * CHUNK])
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 + 3 * attempt)
    q = np.stack(blocks).reshape(len(YEARS), 64, -1)  # (9, 64, n)
    del blocks
    ok = ~(q == -128).all(axis=1).any(axis=0)
    lon, lat = pixel_xy(cy, cx)
    lon, lat = lon.ravel(), lat.ravel()
    ins = shapely.contains_xy(geom, lon, lat)
    other = shapely.contains_xy(others, lon, lat) if others is not None else np.zeros_like(ins)
    keep = ok & ~other & (ins if side == "inside" else ~ins)
    if keep.sum() < 500:
        return []
    v = dequant(q[:, :, keep])  # (9, 64, m)
    del q
    u = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
    b = YEARS.index(fire["year"] - 1)
    rows = []
    for yi, y in enumerate(YEARS):
        r = dict(fire=fire["name"], kind=fire["kind"], fire_year=fire["year"], side=side, cy=cy, cx=cx, year=y,
                 n=int(keep.sum()), sim_before=float((u[yi] * u[b]).sum(0).mean()),
                 sim_prev=float((u[yi] * u[yi - 1]).sum(0).mean()) if yi else np.nan)
        if yi >= 2:
            w = v[yi - 2:yi + 1]
            x = np.concatenate([w[-1], w.std(axis=0)]).T  # (m, 128)
            k = kinds_of(head, x)
            lead = np.bincount(k.argmax(1), minlength=len(KINDS)) / len(k)
            for j, name in enumerate(KINDS):
                r[f"p_{name}"] = float(k[:, j].mean())
                r[f"lead_{name}"] = float(lead[j])
            r["p_fuel"] = float(k[:, 5:].sum(1).mean())
        rows.append(r)
    return rows


def summarize(df):
    out = {}
    for (name, side), g in df.groupby(["fire", "side"]):
        w = g.n / g.groupby("year").n.transform("sum")
        cols = [c for c in g.columns if c.startswith(("sim_", "p_", "lead_"))]
        per = (g[cols].multiply(w, axis=0)).groupby(g.year).sum(min_count=1)
        out.setdefault(name, {"fire_year": int(g.fire_year.iloc[0]), "kind": g.kind.iloc[0]})[side] = {
            "chunks": int(g[["cy", "cx"]].drop_duplicates().shape[0]), "pixels": int(g[g.year == YEARS[0]].n.sum()),
            "by_year": {int(y): {c: (None if pd.isna(v) else round(float(v), 4)) for c, v in row.items()} for y, row in per.iterrows()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=8, help="per side per fire")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-gb", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    per = pd.read_parquet(ROOT / "data" / "fire_test" / "perimeters.parquet")
    per["g"] = shapely.from_wkb(per.geom)
    rng = np.random.default_rng(a.seed)
    head, arr = Head(), open_aef()
    jobs = []
    for name, year, kind in FIRES:
        s = per[(per.name == name) & (per.year == year)].sort_values("acres").iloc[-1]
        geom = shapely.make_valid(s.g)
        rest = per[per.fire_id != s.fire_id].g
        rest = rest[shapely.intersects(rest.to_numpy(), shapely.box(*shapely.buffer(geom, 0.2).bounds))]
        others = shapely.union_all(shapely.make_valid(rest.to_numpy())) if len(rest) else None
        if others is not None:
            shapely.prepare(others)
        shapely.prepare(geom)
        ins, outs, ni, no = pick_chunks(geom, rng, a.chunks)
        print(f"{name} {year}: {len(ins)} of {ni} inside chunks, {len(outs)} of {no} outside", flush=True)
        f = dict(name=name, year=year, kind=kind)
        jobs += [(f, "inside", cy, cx, geom, others) for cy, cx in ins] + [(f, "outside", cy, cx, geom, others) for cy, cx in outs]
    proc = psutil.Process(os.getpid())
    peak, rows, t0 = 0.0, [], time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(lambda j: one_chunk(arr, head, *j), jobs)):
            rows += r
            gb = proc.memory_info().rss / 1e9
            peak = max(peak, gb)
            if gb > a.max_gb:
                print(f"STOP: {gb:.1f} GB > --max-gb {a.max_gb}", flush=True)
                os._exit(2)
            el = time.time() - t0
            print(f"{i + 1}/{len(jobs)} {el / (i + 1):.1f} s/chunk, eta {el / (i + 1) * (len(jobs) - i - 1) / 60:.1f} min, "
                  f"rss {gb:.2f} GB (peak {peak:.2f})", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "rows.parquet")
    (ROOT / "results" / "fire_change.json").write_text(json.dumps(summarize(df), indent=1))
    print(f"done in {(time.time() - t0) / 60:.1f} min, peak rss {peak:.2f} GB")


if __name__ == "__main__":
    main()
