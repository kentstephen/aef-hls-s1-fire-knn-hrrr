# /// script
# requires-python = ">=3.12"
# dependencies = ["zarr>=3", "obstore", "numpy", "pandas", "pyarrow", "rasterio", "psutil"]
# ///
"""P(fuel) over every chunk the fire test needs (data/fire_test/grids_needed.parquet), from the year before the fire.

The head (models/fuel_head.json) takes the newest year's AlphaEarth embedding (dequantized, not unit-normalized)
and the spread of the pixel's embedding over that year and the two before it. Here the newest year is the
embedding year (the fire year minus 1), so a 2026 fire gets exactly the head's own 2023 to 2025 input.

Per grid: data/fire_test/grids/<cy>_<cx>_<emb_year>.npz with p (uint8, P(fuel) * 254 rounded, 255 = no data).
Per chunk: data/fire_test/lf/<cy>_<cx>.npz with fbfm (LANDFIRE LF2024 FBFM40, 30 m, nearest onto the grid).
Resumable. Three 4 MB reads per grid; RAM about 150 MB per worker in flight, under 2 GB at 8. Stops on its own
if the process passes --max-gb.

    uv run scripts/fire_test_grids.py --check     # the numpy head against the export's own probe pixels
    uv run scripts/fire_test_grids.py [--workers 8] [--max-gb 4]
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
import zarr
from obstore.store import S3Store
from rasterio.io import MemoryFile
from zarr.storage import ObjectStore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuel_common import CHUNK, LF, MOS_RES, MOS_X0, MOS_Y0, YEAR0, get  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "fire_test"
GRIDS, LFD = D / "grids", D / "lf"


class Head:
    def __init__(self, path=ROOT / "models" / "fuel_head.json"):
        h = json.loads(Path(path).read_text())
        self.mu, self.sc = np.array(h["mean"], np.float32), np.array(h["scale"], np.float32)
        no = h["n_out"]
        self.nets = [(np.array(n["w1"], np.float32).reshape(64, -1).T, np.array(n["b1"], np.float32),
                      np.array(n["w2"], np.float32).reshape(no, 64).T, np.array(n["b2"], np.float32)) for n in h["nets"]]
        self.nf = h["nf_cols"]

    def p_notfuel(self, x):
        """x (n, 128) raw [e, sd] -> P(not fuel) (n,)."""
        xs = (x - self.mu) / self.sc
        out = np.zeros(len(x), np.float32)
        for w1, b1, w2, b2 in self.nets:
            z = np.maximum(xs @ w1 + b1, 0) @ w2 + b2
            z = np.exp(z - z.max(1, keepdims=True))
            out += z[:, self.nf].sum(1) / z.sum(1) / len(self.nets)
        return out


def dequant(q):
    f = q.astype(np.float32) / 127.5
    return np.sign(f) * f * f


def p_fuel_grid(head, blocks):
    """blocks: 3 int8 (64, 256, 256), oldest first -> P(fuel) (256, 256) float32, NaN where any year has no data."""
    nod = np.zeros((CHUNK, CHUNK), bool)
    for b in blocks:
        nod |= (b == -128).all(axis=0)
    v = np.stack([dequant(b) for b in blocks])  # (3, 64, 256, 256)
    x = np.concatenate([v[-1], v.std(axis=0)]).reshape(128, -1).T  # (65536, 128)
    del v
    p = 1 - head.p_notfuel(x).reshape(CHUNK, CHUNK)
    p[nod] = np.nan
    return p


def open_aef():
    store = ObjectStore(S3Store("us-west-2.opendata.source.coop", prefix="tge-labs/aef-mosaic", region="us-west-2", skip_signature=True,
                                client_options={"timeout": "60s", "connect_timeout": "10s"}), read_only=True)
    return zarr.open_array(store, path="embeddings", mode="r")


def read_blocks(arr, cy, cx, emb_year):
    out = []
    for y in (emb_year - 2, emb_year - 1, emb_year):
        for attempt in range(4):
            try:
                out.append(arr[y - YEAR0, :, cy * CHUNK:(cy + 1) * CHUNK, cx * CHUNK:(cx + 1) * CHUNK])
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 + 3 * attempt)
    return out


def landfire(cy, cx):
    lon0, lat0 = MOS_X0 + cx * CHUNK * MOS_RES, MOS_Y0 - cy * CHUNK * MOS_RES
    lon1, lat1 = lon0 + CHUNK * MOS_RES, lat0 - CHUNK * MOS_RES
    b = get(LF.format("FBFM40"), dict(bbox=f"{lon0},{lat1},{lon1},{lat0}", bboxSR=4326, imageSR=4326, size=f"{CHUNK},{CHUNK}",
                                      format="tiff", pixelType="S16", interpolation="RSP_NearestNeighbor", f="image"))
    with MemoryFile(b) as m, m.open() as r:
        return r.read(1).astype(np.int16)


def check(head):
    """The export's probe pixels (2025 embedding, spread over 2023 to 2025) through this numpy head."""
    chk = json.loads((ROOT / "models" / "fuel_head_check.json").read_text())
    wins = pd.read_parquet(ROOT / "data" / "fuel_ca_windows.parquet")
    c = pd.DataFrame(chk)
    arr = open_aef()
    worst = 0.0
    for wi in c.win.unique()[:4]:
        cy, cx = int(wins.cy.iloc[wi]), int(wins.cx.iloc[wi])
        p = p_fuel_grid(head, read_blocks(arr, cy, cx, 2025))
        s = c[c.win == wi]
        d = np.abs((1 - p[s.row, s.col]) - s.p_notfuel.to_numpy())
        worst = max(worst, float(np.nanmax(d)))
        print(f"window {wi} ({cy}, {cx}): {len(s)} probe px, max abs diff {np.nanmax(d):.2e}", flush=True)
    print("CHECK OK" if worst < 1e-3 else f"CHECK FAILED: {worst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-gb", type=float, default=4.0)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    head = Head()
    if a.check:
        return check(head)
    GRIDS.mkdir(parents=True, exist_ok=True)
    LFD.mkdir(parents=True, exist_ok=True)
    need = pd.read_parquet(D / "grids_needed.parquet")
    todo = [r for r in need.itertuples() if not (GRIDS / f"{r.cy}_{r.cx}_{r.emb_year}.npz").exists()]
    print(f"{len(todo)} grids to make ({len(need) - len(todo)} done)", flush=True)
    arr = open_aef()
    proc = psutil.Process(os.getpid())
    peak = 0.0

    def one(r):
        p = p_fuel_grid(head, read_blocks(arr, r.cy, r.cx, r.emb_year))
        q = np.where(np.isnan(p), 255, np.round(np.nan_to_num(p) * 254)).astype(np.uint8)
        np.savez_compressed(GRIDS / f"{r.cy}_{r.cx}_{r.emb_year}.npz", p=q)
        lf = LFD / f"{r.cy}_{r.cx}.npz"
        if not lf.exists():
            try:
                np.savez_compressed(lf, fbfm=landfire(r.cy, r.cx))
            except Exception as e:  # LANDFIRE's service can fail; the baseline is then missing for this chunk
                print(f"landfire {r.cy}_{r.cx}: {e}", flush=True)
        return r

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for n, _ in enumerate(ex.map(one, todo)):
            gb = proc.memory_info().rss / 1e9
            peak = max(peak, gb)
            if gb > a.max_gb:
                print(f"STOP: {gb:.1f} GB > --max-gb {a.max_gb}; rerun to resume", flush=True)
                os._exit(2)
            if n % 20 == 0 or n == len(todo) - 1:
                el = time.time() - t0
                print(f"{n + 1}/{len(todo)} {el / (n + 1):.2f} s/grid, eta {el / (n + 1) * (len(todo) - n - 1) / 60:.1f} min, "
                      f"rss {gb:.2f} GB (peak {peak:.2f})", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min, peak rss {peak:.2f} GB")


if __name__ == "__main__":
    main()
