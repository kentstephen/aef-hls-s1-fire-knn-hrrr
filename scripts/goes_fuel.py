# /// script
# requires-python = ">=3.12"
# dependencies = ["zarr>=3", "obstore", "numpy", "pandas", "pyarrow", "rasterio", "psutil", "faiss-cpu", "h5py"]
# ///
"""The fuel in and around each GOES event's first box, from the 10 m pixels of the year before.

Each mosaic pixel is placed on the GOES grid (its fractional row and column, from the same navigation GOES uses)
and measured by its distance, in GOES pixels, to the nearest first-box pixel center (the larger of the row and
column gaps): "in" is up to 0.5 (inside a hot pixel), "around" is 0.5 to 1.5 (the ring of pixels next to it).

Three readings of each 10 m pixel's fuel, embeddings of --emb-year (the year before the fires):
  knn   kNN vote (k = K, distance weighted) of the pixel's AlphaEarth embedding among labeled California
        reference pixels (data/fuel_lab: LANDFIRE LF2024 FBFM40 fuel / not fuel over 580 windows, drawn by their
        weights so the reference mix matches the windows' own). Run on every second pixel each way (20 m).
        All 580 windows are references, the same ones the head was fit on, so the two stand on equal footing
        (a few hundred of the ~36,000 chunks hold a window). Search: HNSW (M 32, efSearch 64), 99.5 % of the
        exact neighbors at a tenth of the time.
  head  the fuel head (models/fuel_head.json): P(fuel) from the newest embedding and its 3-year spread
  lf    LANDFIRE LF2024 FBFM40 itself (fuel = not 91 to 99), the map the other two learned from

  plan    the chunks each event needs -> data/goes/chunks_<year>.parquet (with --max-events per label)
  grids   the three readings per chunk -> data/goes/grids/<emb_year>/<cy>_<cx>.npz (resumable); LANDFIRE in 8 x 8 chunk
          blocks -> data/goes/lf/
  shares  per event: fuel share in and around -> data/goes/event_fuel_<year>.parquet

RAM: the reference set (REFS x 64 float32 and its HNSW graph, about 100 MB) plus about 150 MB per grid in
flight; 1.3 GB measured at 8 workers. Stops on its own past --max-gb.

    uv run scripts/goes_fuel.py plan --year 2025 [--max-events 800]
    uv run scripts/goes_fuel.py grids --year 2025 [--workers 8]
    uv run scripts/goes_fuel.py shares --year 2025
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuel_common import CHUNK, MOS_RES, MOS_X0, MOS_Y0  # noqa: E402
from goes_ca_detections import lonlat_to_xy, xy_to_lonlat  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
G = ROOT / "data" / "goes"
GRIDS_ROOT = G / "grids"
LAB = ROOT / "data" / "fuel_lab"
LFB = G / "lf"
LF_BLOCK = 8  # chunks a side per LANDFIRE request (2048 px)
RING = 1.5  # GOES pixels
K, REFS, STEP = 15, 200_000, 2


def grids_dir(emb_year):
    """One folder per embedding year: the same chunk reads different embeddings for fires of different years."""
    return GRIDS_ROOT / str(emb_year)


def grid_params(year):
    return json.loads((G / f"grid_{year}.json").read_text())


def goes_rc(lon, lat, g):
    x, y = lonlat_to_xy(lon, lat, g)
    return (y - g["y0"]) / g["dy"], (x - g["x0"]) / g["dx"]


def plan(a):
    g = grid_params(a.year)
    ev = pd.read_parquet(G / f"events_{a.year}.parquet")
    px = pd.read_parquet(G / f"event_px_{a.year}.parquet")
    rng = np.random.default_rng(a.seed)
    keep = []
    for lab, e in ev.groupby("label"):
        keep.append(e if len(e) <= a.max_events else e.iloc[np.sort(rng.choice(len(e), a.max_events, replace=False))])
    ev = pd.concat(keep).sort_values("event")
    px = px[px.event.isin(ev.event)]
    rows = []
    for e, p in px.groupby("event"):
        r0, r1, c0, c1 = p.row.min() - RING, p.row.max() + RING, p.col.min() - RING, p.col.max() + RING
        rr, cc = np.meshgrid(np.linspace(r0, r1, 9), np.linspace(c0, c1, 9))
        lon, lat = xy_to_lonlat(g["x0"] + g["dx"] * cc, g["y0"] + g["dy"] * rr, g)
        cy0, cy1 = int((MOS_Y0 - np.nanmax(lat)) / MOS_RES) // CHUNK, int((MOS_Y0 - np.nanmin(lat)) / MOS_RES) // CHUNK
        cx0, cx1 = int((np.nanmin(lon) - MOS_X0) / MOS_RES) // CHUNK, int((np.nanmax(lon) - MOS_X0) / MOS_RES) // CHUNK
        rows += [(e, cy, cx) for cy in range(cy0, cy1 + 1) for cx in range(cx0, cx1 + 1)]
    ch = pd.DataFrame(rows, columns=["event", "cy", "cx"])
    ch.to_parquet(G / f"chunks_{a.year}.parquet", index=False)
    ev[["event"]].to_parquet(G / f"events_kept_{a.year}.parquet", index=False)
    u = ch[["cy", "cx"]].drop_duplicates()
    done = sum((grids_dir(a.emb_year) / f"{cy}_{cx}.npz").exists() for cy, cx in zip(u.cy, u.cx))
    print(f"{len(ev)} events kept ({ev.label.value_counts().to_dict()}), {len(ch)} event-chunks, {len(u)} distinct chunks ({done} done)")


class Knn:
    def __init__(self, seed=0):
        import faiss
        px = pd.read_parquet(LAB / "pixels.parquet", columns=["win", "w", "nf"])
        w = px.w.to_numpy()
        rng = np.random.default_rng(seed)
        pick = np.sort(rng.choice(len(px), size=min(REFS, len(px)), replace=False, p=w / w.sum()))
        e = np.load(LAB / "e24.npy", mmap_mode="r")
        x = np.ascontiguousarray(e[pick], np.float32)
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        self.y = 1 - px.nf.to_numpy()[pick].astype(np.float32)  # 1 = fuel
        self.index = faiss.IndexHNSWFlat(64, 32)
        self.index.hnsw.efConstruction = 80
        self.index.add((x - self.mu) / self.sd)
        self.index.hnsw.efSearch = 64
        print(f"kNN references: {len(pick)} pixels from {px.win[pick].nunique()} windows, "
              f"fuel share {self.y.mean():.2f}", flush=True)

    def p_fuel(self, e):
        """e (n, 64) dequantized -> distance-weighted share of fuel among the K nearest references."""
        d, i = self.index.search((e - self.mu) / self.sd, K)
        wt = 1 / (np.sqrt(np.maximum(d, 0)) + 1e-6)
        return (wt * self.y[i]).sum(1) / wt.sum(1)


def save(path, **arrays):
    """Write to a temporary name, then rename: a run stopped mid-write leaves no broken file behind."""
    tmp = path.with_name(path.stem + f".tmp{threading.get_ident()}.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


_locks, _locks_guard = {}, threading.Lock()


def lf_block(by, bx):
    """LANDFIRE LF2024 FBFM40 over a block of LF_BLOCK x LF_BLOCK mosaic chunks, nearest onto the mosaic's own pixels
    (one request for many chunks; chunks cluster around fires). Cached in data/goes/lf/. -1 where it failed."""
    from fuel_common import LF, get
    from rasterio.io import MemoryFile
    f = LFB / f"{by}_{bx}.npz"
    with _locks_guard:
        lock = _locks.setdefault((by, bx), threading.Lock())
    with lock:
        if f.exists():
            return np.load(f)["fbfm"]
        n = LF_BLOCK * CHUNK
        lon0, lat0 = MOS_X0 + bx * n * MOS_RES, MOS_Y0 - by * n * MOS_RES
        lon1, lat1 = lon0 + n * MOS_RES, lat0 - n * MOS_RES
        try:
            b = get(LF.format("FBFM40"), dict(bbox=f"{lon0},{lat1},{lon1},{lat0}", bboxSR=4326, imageSR=4326, size=f"{n},{n}",
                                              format="tiff", pixelType="S16", interpolation="RSP_NearestNeighbor", f="image"))
            with MemoryFile(b) as m, m.open() as r:
                v = r.read(1).astype(np.int16)
            assert v.shape == (n, n), v.shape
        except Exception as ex:  # LANDFIRE's service can fail; the baseline is then missing for these chunks
            print(f"landfire block {by}_{bx}: {ex}", flush=True)
            return np.full((n, n), -1, np.int16)
        LFB.mkdir(parents=True, exist_ok=True)
        save(f, fbfm=v)
        return v


def lf_chunk(cy, cx):
    v = lf_block(cy // LF_BLOCK, cx // LF_BLOCK)
    i, j = (cy % LF_BLOCK) * CHUNK, (cx % LF_BLOCK) * CHUNK
    return v[i:i + CHUNK, j:j + CHUNK]


def grids(a):
    from fire_test_grids import Head, dequant, open_aef, p_fuel_grid, read_blocks
    import faiss
    faiss.omp_set_num_threads(4)
    GRIDS = grids_dir(a.emb_year)
    GRIDS.mkdir(parents=True, exist_ok=True)
    ch = pd.concat([pd.read_parquet(G / f"chunks_{a.year}.parquet")] + [pd.read_parquet(p) for p in a.extra_chunks])
    u = ch[["cy", "cx"]].drop_duplicates()
    todo = sorted((int(cy), int(cx)) for cy, cx in zip(u.cy, u.cx) if not (GRIDS / f"{cy}_{cx}.npz").exists())
    todo.sort(key=lambda c: (c[0] // LF_BLOCK, c[1] // LF_BLOCK, c))  # block by block, so each LANDFIRE block is fetched once and reused
    print(f"{len(todo)} chunks to make ({len(u) - len(todo)} done)", flush=True)
    if a.limit:
        todo = todo[:a.limit]
    head, knn, arr = Head(), Knn(), open_aef()
    proc = psutil.Process(os.getpid())
    peak, t0 = 0.0, time.time()

    def one(c):
        cy, cx = c
        blocks = read_blocks(arr, cy, cx, a.emb_year)
        ph = p_fuel_grid(head, blocks)
        b = blocks[-1][:, ::STEP, ::STEP]
        nod = (b == -128).all(0)
        e = dequant(b).reshape(64, -1).T
        pk = knn.p_fuel(np.ascontiguousarray(e, np.float32)).reshape(b.shape[1:])
        pk[nod] = np.nan
        lf = lf_chunk(cy, cx)
        q = lambda p: np.where(np.isnan(p), 255, np.round(np.nan_to_num(p) * 254)).astype(np.uint8)  # noqa: E731
        save(GRIDS / f"{cy}_{cx}.npz", head=q(ph), knn=q(pk), lf=lf)
        return c

    with ThreadPoolExecutor(a.workers) as ex:
        for n, _ in enumerate(ex.map(one, todo)):
            gb = proc.memory_info().rss / 1e9
            peak = max(peak, gb)
            if gb > a.max_gb:
                print(f"STOP: {gb:.1f} GB > --max-gb {a.max_gb}; rerun to resume", flush=True)
                os._exit(2)
            if n % 50 == 0 or n == len(todo) - 1:
                el = time.time() - t0
                print(f"{n + 1}/{len(todo)} {el / (n + 1):.2f} s/chunk, eta {el / (n + 1) * (len(todo) - n - 1) / 60:.1f} min, "
                      f"rss {gb:.2f} GB (peak {peak:.2f})", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min, peak rss {peak:.2f} GB")


def chunk_rc(cy, cx, g, step=1):
    """GOES fractional row and column of each mosaic pixel center of a chunk (every `step`-th pixel)."""
    i = np.arange(0, CHUNK, step) + 0.5  # the sampled pixel's own center
    lon = MOS_X0 + (cx * CHUNK + i) * MOS_RES
    lat = MOS_Y0 - (cy * CHUNK + i) * MOS_RES
    lon, lat = np.meshgrid(lon, lat)
    return goes_rc(lon, lat, g)


def shares(a):
    g = grid_params(a.year)
    ev = pd.read_parquet(G / f"events_{a.year}.parquet")
    kept = pd.read_parquet(G / f"events_kept_{a.year}.parquet").event
    px = pd.read_parquet(G / f"event_px_{a.year}.parquet")
    ch = pd.read_parquet(G / f"chunks_{a.year}.parquet")
    GRIDS = grids_dir(a.emb_year)
    cache = {}

    def load(cy, cx):
        if (cy, cx) not in cache:
            f = GRIDS / f"{cy}_{cx}.npz"
            if not f.exists():
                cache[(cy, cx)] = None
            else:
                z = np.load(f)
                r1, c1 = chunk_rc(cy, cx, g, 1)
                r2, c2 = chunk_rc(cy, cx, g, STEP)
                cache[(cy, cx)] = dict(head=z["head"], knn=z["knn"], lf=z["lf"], r1=r1, c1=c1, r2=r2, c2=c2)
            if len(cache) > 3000:
                cache.pop(next(iter(cache)))
        return cache[(cy, cx)]

    out = []
    for e in kept:
        p = px[px.event == e]
        cs = ch[ch.event == e]
        acc = {k: [[], []] for k in ("knn", "head", "lf")}
        missing = 0
        for cy, cx in zip(cs.cy, cs.cx):
            z = load(cy, cx)
            if z is None:
                missing += 1
                continue
            for key, (rr, cc) in (("head", ("r1", "c1")), ("lf", ("r1", "c1")), ("knn", ("r2", "c2"))):
                r, c = z[rr], z[cc]
                d = np.full(r.shape, np.inf)
                for pr, pc in zip(p.row, p.col):
                    d = np.minimum(d, np.maximum(np.abs(r - pr), np.abs(c - pc)))
                v = z[key]
                if key == "lf":
                    fuel = np.where(v > 0, (~((v >= 91) & (v <= 99))).astype(np.float32), np.nan)
                else:
                    fuel = np.where(v == 255, np.nan, v / 254.0)
                for j, m in enumerate((d <= 0.5, (d > 0.5) & (d <= RING))):
                    acc[key][j].append(fuel[m])
        row = dict(event=e, missing_chunks=missing)
        for key, (ins, rng) in acc.items():
            for name, parts in (("in", ins), ("around", rng)):
                v = np.concatenate(parts) if parts else np.array([])
                v = v[np.isfinite(v)]
                row[f"{key}_{name}"] = float(v.mean()) if len(v) else np.nan  # mean P(fuel) = expected fuel share
                if key != "lf":
                    row[f"{key}_{name}_frac"] = float((v >= 0.5).mean()) if len(v) else np.nan  # share called fuel
                row[f"n_{key}_{name}"] = int(len(v))
        out.append(row)
    f = pd.DataFrame(out).merge(ev, on="event")
    f.to_parquet(G / f"event_fuel_{a.year}.parquet", index=False)
    print(f"{len(f)} events; chunks missing in {int((f.missing_chunks > 0).sum())}")
    print(f.groupby("label")[["knn_in", "knn_around", "head_in", "head_around", "lf_in", "lf_around"]].median().round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["plan", "grids", "shares"])
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--emb-year", type=int, default=None, help="default: --year minus 1")
    ap.add_argument("--max-events", type=int, default=800, help="per label, random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-gb", type=float, default=4.0)
    ap.add_argument("--extra-chunks", nargs="*", default=[], help="grids: more chunk lists (parquet with cy, cx) to make too, "
                    "e.g. data/goes/track_chunks_<year>.parquet from goes_fuel_track.py")
    ap.add_argument("--limit", type=int, default=0, help="grids: only the first n chunks (a timing trial)")
    a = ap.parse_args()
    a.emb_year = a.emb_year or a.year - 1
    {"plan": plan, "grids": grids, "shares": shares}[a.stage](a)


if __name__ == "__main__":
    main()
