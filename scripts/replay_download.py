# /// script
# requires-python = ">=3.12"
# dependencies = ["zarr>=3", "obstore", "numpy", "pandas", "pyarrow", "shapely", "psutil", "rasterio"]
# ///
"""A fire's AlphaEarth embeddings kept on disk for developing the replay (web/src/replay.ts), so the page reads them
from localhost instead of Source Cooperative.

Every 256 x 256 mosaic chunk the fire's final outline touches, for each year asked, as the raw int8 block the browser
reads (band-major, 64 x 256 x 256, 4 MiB): data/replay/<fire>/<year>/<cy>_<cx>.bin. The dev server serves them at
/local-aef/ (web/vite.config.ts) and the chunk worker tries there first (web/src/aef/aef.worker.ts). Resumable.

The Park Fire, 2023 and 2025: 422 chunks a year, 3.3 GiB. RAM: one block per thread, under 1 GB at 16 threads.

    uv run scripts/replay_download.py [--fire park] [--years 2023 2025] [--threads 16]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fire_test_grids import open_aef  # noqa: E402
from fuel_common import CHUNK, MOS_RES, MOS_X0, MOS_Y0, YEAR0  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIRES = {"park": ("PARK", 2024)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", default="park", choices=list(FIRES))
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2025])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-gb", type=float, default=3.0)
    a = ap.parse_args()
    name, year = FIRES[a.fire]
    p = pd.read_parquet(ROOT / "data" / "fire_test" / "perimeters.parquet")
    r = p[(p.name == name) & (p.year == year)].sort_values("acres").iloc[-1]
    g = shapely.from_wkb(r.geom)
    w, s, e, n = g.bounds
    cy0, cy1 = int((MOS_Y0 - n) / MOS_RES // CHUNK), int((MOS_Y0 - s) / MOS_RES // CHUNK)
    cx0, cx1 = int((w - MOS_X0) / MOS_RES // CHUNK), int((e - MOS_X0) / MOS_RES // CHUNK)
    chunks = [(cy, cx) for cy in range(cy0, cy1 + 1) for cx in range(cx0, cx1 + 1)
              if g.intersects(shapely.box(MOS_X0 + cx * CHUNK * MOS_RES, MOS_Y0 - (cy + 1) * CHUNK * MOS_RES,
                                          MOS_X0 + (cx + 1) * CHUNK * MOS_RES, MOS_Y0 - cy * CHUNK * MOS_RES))]
    out = ROOT / "data" / "replay" / a.fire
    todo = [(y, cy, cx) for y in a.years for cy, cx in chunks if not (out / str(y) / f"{cy}_{cx}.bin").exists()]
    print(f"{len(chunks)} chunks a year touch the outline; {len(todo)} chunk-years to fetch ({len(chunks) * len(a.years) - len(todo)} on disk)", flush=True)
    for y in a.years:
        (out / str(y)).mkdir(parents=True, exist_ok=True)
    arr = open_aef()
    proc = psutil.Process(os.getpid())
    t0, peak = time.time(), 0.0

    def one(job):
        y, cy, cx = job
        for attempt in range(4):
            try:
                b = np.ascontiguousarray(arr[y - YEAR0, :, cy * CHUNK:(cy + 1) * CHUNK, cx * CHUNK:(cx + 1) * CHUNK], dtype=np.int8)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 + 3 * attempt)
        f = out / str(y) / f"{cy}_{cx}.bin"
        tmp = f.with_suffix(".tmp")
        tmp.write_bytes(b.tobytes())
        os.replace(tmp, f)

    with ThreadPoolExecutor(a.threads) as ex:
        for k, _ in enumerate(ex.map(one, todo)):
            gb = proc.memory_info().rss / 1e9
            peak = max(peak, gb)
            if gb > a.max_gb:
                print(f"STOP: {gb:.1f} GB > --max-gb {a.max_gb}; rerun to resume", flush=True)
                os._exit(2)
            if k % 50 == 0 or k == len(todo) - 1:
                el = time.time() - t0
                print(f"{k + 1}/{len(todo)} {el / (k + 1):.2f} s each, eta {el / (k + 1) * (len(todo) - k - 1) / 60:.1f} min, rss {gb:.2f} GB", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min, peak rss {peak:.2f} GB")


if __name__ == "__main__":
    main()
