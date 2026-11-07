# /// script
# requires-python = ">=3.12"
# dependencies = ["zarr>=3", "obstore", "numpy", "pandas", "pyarrow", "rasterio", "psutil", "shapely", "torch"]
# ///
"""The change head on every 2024 fire on file: does predicted canopy drop inside the burn and not beside it?

For each 2024 perimeter (100 acres and up), up to 3 mosaic chunks its outer line crosses, so burned and unburned
ground sit side by side. Chunks within 3 km of a training window (data/change/windows.parquet, random and burn)
are left out: a random window can sit on a 2024 burn and was trained on LF2024 / LF2025 labels that hold it.
Burned = inside the perimeter and at least 150 m from its line; unburned = outside and at least 150 m away. Pixels
inside any other perimeter on file are left out of both.

Per chunk, the year before (2023) and the year after (2025): the head's CC and CH, the cosine of each pixel's
embedding between the two years, and LANDFIRE's CC from LF2023 and LF2025 on the same pixels as a reference.
The score per fire is the difference in differences: (inside 2025 - 2023) - (outside 2025 - 2023).

Writes results/change_test.json and data/change/test2024.parquet. Six chunks in flight, about 1 GB of RAM.

    uv run scripts/change_test.py [--head data/change/head_e_all.pt] [--per-fire 3] [--workers 6]
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fire_change import pixel_xy  # noqa: E402
from fire_test_grids import open_aef  # noqa: E402
from fit_change import Net, deq, feats  # noqa: E402
from fuel_common import CHUNK, MOS_RES, MOS_X0, MOS_Y0, YEAR0  # noqa: E402
from sample_change import chunk_center, landfire  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "change"
BEFORE, AFTER = 2023, 2025


def edge_chunks(g, n, rng):
    """Chunks the outer line of g crosses, at least 20 % burned and 20 % unburned."""
    x0, y0, x1, y1 = g.bounds
    cys = range(int((MOS_Y0 - y1) / MOS_RES) // CHUNK, int((MOS_Y0 - y0) / MOS_RES) // CHUNK + 1)
    cxs = range(int((x0 - MOS_X0) / MOS_RES) // CHUNK, int((x1 - MOS_X0) / MOS_RES) // CHUNK + 1)
    out = []
    for cy in cys:
        for cx in cxs:
            lon, lat = pixel_xy(cy, cx, 32)
            f = shapely.contains_xy(g, lon, lat).mean()
            if 0.2 <= f <= 0.8:
                out.append((cy, cx))
    return [out[i] for i in rng.permutation(len(out))[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, default=D / "head_e_all.pt")
    ap.add_argument("--per-fire", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-gb", type=float, default=4.0)
    a = ap.parse_args()
    h = torch.load(a.head, weights_only=False)
    s, mu, sd = h["set"], h["mu"], h["sd"]
    if s not in ("e", "e_all"):
        raise SystemExit("this test reads two years only; use a head on the year's embedding alone")
    net = Net(len(mu))
    net.load_state_dict(h["state"])
    net.eval()

    per = pd.read_parquet(ROOT / "data" / "fire_test" / "perimeters.parquet")
    per["g"] = [shapely.make_valid(g) for g in shapely.from_wkb(per.geom)]
    tw = pd.read_parquet(D / "windows.parquet")
    tw = tw[tw.kind.isin(["random", "burn"])]
    tc = np.array([chunk_center(r.cy, r.cx) for r in tw.itertuples()])
    rng = np.random.default_rng(0)
    jobs = []
    for f in per[per.year == 2024].sort_values("acres", ascending=False).itertuples():
        for cy, cx in edge_chunks(f.g, a.per_fire, rng):
            lon, lat = chunk_center(cy, cx)
            d_km = np.hypot((tc[:, 0] - lon) * 111.32 * np.cos(np.radians(lat)), (tc[:, 1] - lat) * 111.32).min()
            if d_km >= 3:
                jobs.append((f.fire_id, f.name, f.acres, cy, cx))
    print(f"{len(jobs)} chunks over {len({j[0] for j in jobs})} fires of 2024", flush=True)
    arr = open_aef()
    bufd = 150 / 111320  # 150 m in degrees (lat); lon a little wider, fine for a margin

    def one(j):
        fid, name, acres, cy, cx = j
        g = per.loc[per.fire_id == fid, "g"].iloc[0]
        others = per[(per.fire_id != fid)].g
        others = shapely.union_all(others[shapely.intersects(others.to_numpy(), shapely.buffer(g, 0.1))].to_numpy())
        q = np.stack([arr[y - YEAR0, :, cy * CHUNK:(cy + 1) * CHUNK, cx * CHUNK:(cx + 1) * CHUNK] for y in (BEFORE, AFTER)])
        q = q.reshape(2, 64, -1).transpose(2, 0, 1)  # (n, 2, 64)
        ok = ~(q == -128).all(axis=2).any(axis=1)
        lon, lat = pixel_xy(cy, cx)
        lon, lat = lon.ravel(), lat.ravel()
        inner, outer = shapely.buffer(g, -bufd), shapely.buffer(g, bufd)
        ins = shapely.contains_xy(inner, lon, lat)
        outs = ~shapely.contains_xy(outer, lon, lat)
        oth = shapely.contains_xy(others, lon, lat) if not others.is_empty else np.zeros_like(ins)
        side = np.where(ins, 1, np.where(outs, 0, -1))
        side[~ok | oth] = -1
        lf = {y: landfire(cy, cx, y, "CC").ravel() for y in (BEFORE, AFTER)}
        rows = []
        keep = side >= 0
        if keep.sum() < 1000:
            return rows
        qq = q[keep]
        pr = {}
        with torch.no_grad():
            for k, y in enumerate((BEFORE, AFTER)):
                x = torch.from_numpy((feats(qq, k, s) - mu) / sd)
                o = net(x).numpy()
                pr[y] = (o[:, 0].clip(0, 1) * 100, o[:, 1].clip(0, None) * 50)
        e0, e1 = deq(qq[:, 0]), deq(qq[:, 1])
        cos = (e0 * e1).sum(1) / np.maximum(np.linalg.norm(e0, axis=1) * np.linalg.norm(e1, axis=1), 1e-6)
        sd_ = side[keep]
        for sv, sname in ((1, "inside"), (0, "outside")):
            m = sd_ == sv
            if m.sum() < 300:
                continue
            l0, l1 = lf[BEFORE][keep][m].astype(float), lf[AFTER][keep][m].astype(float)
            lv = (l0 >= 0) & (l0 <= 100) & (l1 >= 0) & (l1 <= 100)
            rows.append(dict(fire_id=fid, fire=name, acres=acres, cy=cy, cx=cx, side=sname, n=int(m.sum()),
                             cc0=float(pr[BEFORE][0][m].mean()), cc1=float(pr[AFTER][0][m].mean()),
                             ch0=float(pr[BEFORE][1][m].mean()), ch1=float(pr[AFTER][1][m].mean()),
                             cos=float(cos[m].mean()), lf0=float(l0[lv].mean()) if lv.any() else np.nan,
                             lf1=float(l1[lv].mean()) if lv.any() else np.nan))
        return rows

    proc = psutil.Process(os.getpid())
    rows, peak, t0 = [], 0.0, time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(one, jobs)):
            rows += r
            gb = proc.memory_info().rss / 1e9
            peak = max(peak, gb)
            if gb > a.max_gb:
                print(f"STOP: {gb:.1f} GB > --max-gb {a.max_gb}", flush=True)
                os._exit(2)
            if i % 25 == 0 or i == len(jobs) - 1:
                el = time.time() - t0
                print(f"{i + 1}/{len(jobs)} eta {el / (i + 1) * (len(jobs) - i - 1) / 60:.1f} min, rss {gb:.2f} GB", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(D / "test2024.parquet")

    # per chunk with both sides, then per fire (pixel-weighted within a side)
    w = df.pivot_table(index=["fire_id", "fire", "cy", "cx"], columns="side", values=["n", "cc0", "cc1", "ch0", "ch1", "cos", "lf0", "lf1"])
    w = w.dropna(subset=[("n", "inside"), ("n", "outside")])

    def did(c0, c1):
        return (w[(c1, "inside")] - w[(c0, "inside")]) - (w[(c1, "outside")] - w[(c0, "outside")])

    ch = pd.DataFrame(dict(cc=did("cc0", "cc1"), ch=did("ch0", "ch1"), lf=did("lf0", "lf1"),
                           cos=w[("cos", "inside")] - w[("cos", "outside")], cc_before=w[("cc0", "inside")],
                           lf_before=w[("lf0", "inside")]))
    fires = ch.groupby(level=["fire_id", "fire"]).mean()
    ok = fires.dropna(subset=["lf"])
    res = dict(head=str(a.head.name), chunks=int(len(ch)), fires=int(len(fires)),
               cc_did_median=float(fires.cc.median()), cc_did_mean=float(fires.cc.mean()),
               share_fires_cc_down=float((fires.cc < 0).mean()), share_fires_cc_down_2=float((fires.cc < -2).mean()),
               lf_did_median=float(ok.lf.median()), corr_cc_vs_lf=float(np.corrcoef(ok.cc, ok.lf)[0, 1]),
               corr_cc_vs_cos=float(np.corrcoef(fires.cc, fires.cos)[0, 1]), ch_did_median=float(fires.ch.median()),
               cos_gap_median=float(fires.cos.median()))
    for lo, hi in ((0, 10), (10, 25), (25, 101)):
        m = (fires.cc_before >= lo) & (fires.cc_before < hi)
        mo = m & fires.lf.notna()
        res[f"canopy_before_{lo}_{hi}"] = dict(fires=int(m.sum()), cc_did_median=float(fires.cc[m].median()) if m.any() else None,
                                               lf_did_median=float(fires.lf[mo].median()) if mo.any() else None,
                                               share_cc_down=float((fires.cc[m] < 0).mean()) if m.any() else None)
    res["by_fire"] = {f"{n}": dict(cc=round(r.cc, 1), lf=None if pd.isna(r.lf) else round(r.lf, 1), cos=round(r.cos, 3),
                                   cc_before=round(r.cc_before, 1)) for (fid, n), r in fires.sort_values("cc").iterrows()}
    (ROOT / "results" / "change_test.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "by_fire"}, indent=1))
    print(f"done in {(time.time() - t0) / 60:.1f} min, peak rss {peak:.2f} GB")


if __name__ == "__main__":
    main()
