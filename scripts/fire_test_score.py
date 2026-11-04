# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "rasterio", "scipy", "shapely", "tabulate"]
# ///
"""Score the fuel head against fires: do starts and burns land on ground it calls burnable, more than nearby
ground with no fire? Reads data/fire_test/ (fire_test_build.py, fire_test_grids.py); the map is always the
year before the fire.

starts  P(fuel) at each WFIGS point of origin against controls in the same year:
          near   random pixels 300 m to ~2 km away, in the start's own 2.3 km chunk
          5-10 km, 20-30 km   random pixels in a chunk that far away (half the starts each)
        Read at the pixel and as the most burnable ground within 30 m and 100 m (a box), since points of origin
        are rough and many sit on a road. AUC case vs control (0.5 = no better than chance), 95 % interval from a
        bootstrap over starts.
edges   each perimeter chunk split by signed distance to the fire line: inside vs outside, and the line itself
        (within 20 m) vs 100 to 300 m to either side.
LANDFIRE LF2024 FBFM40 (burnable = not 91 to 99) is scored the same way, but only on fires of 2025 and 2026:
LF2024 can already show older burn scars.

Writes results/fire_test.json and data/fire_test/start_scores.parquet. Small: under 1 GB.

    uv run scripts/fire_test_score.py [--year 2025]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuel_common import CHUNK, MOS_RES, MOS_X0, MOS_Y0  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "fire_test"
N_CTRL = 20
REACH_M = (0, 30, 100)
BOOT = 500
rng = np.random.default_rng(0)


def grid(cy, cx, emb_year):
    f = D / "grids" / f"{cy}_{cx}_{emb_year}.npz"
    if not f.exists():
        return None
    q = np.load(f)["p"]
    return np.where(q == 255, np.nan, q / 254.0).astype(np.float32)


def lf_burnable(cy, cx):
    f = D / "lf" / f"{cy}_{cx}.npz"
    if not f.exists():
        return None
    b = np.load(f)["fbfm"].astype(np.float32)
    out = ((b < 91) | (b > 99)).astype(np.float32)
    out[b <= 0] = np.nan
    return out


def rowcol(cy, cx, lon, lat):
    return int((MOS_Y0 - lat) / MOS_RES) - cy * CHUNK, int((lon - MOS_X0) / MOS_RES) - cx * CHUNK


def px_m(lat):
    """Pixel size in meters (rows, cols) at a latitude: the mosaic is in degrees."""
    dy = MOS_RES * 111_320
    return dy, dy * np.cos(np.radians(lat))


def reach(g, r, c, m, lat):
    """Most burnable value within m meters of (r, c) (a box), NaN if all no data."""
    if g is None:
        return np.nan
    dy, dx = px_m(lat)
    hy, hx = int(round(m / dy)), int(round(m / dx))
    w = g[max(0, r - hy):r + hy + 1, max(0, c - hx):c + hx + 1]
    return float(np.nanmax(w)) if np.isfinite(w).any() else np.nan


def auc(pos, neg):
    """Mann-Whitney AUC, ties counted half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def auc_ci(df, col, ctrl_kind):
    """AUC of starts vs one control kind, with a bootstrap over starts (each start keeps its own controls)."""
    case = df[df.kind == "start"].set_index("sid")[col]
    ctrl = df[df.kind == ctrl_kind]
    sids = ctrl.sid.unique()
    if not len(sids):
        return np.nan, np.nan, np.nan, 0
    by = {s: g[col].to_numpy() for s, g in ctrl.groupby("sid")}
    a = auc(case.loc[sids], np.concatenate([by[s] for s in sids]))
    boots = []
    for _ in range(BOOT):
        pick = rng.choice(sids, len(sids))
        boots.append(auc(case.loc[pick], np.concatenate([by[s] for s in pick])))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return a, lo, hi, len(sids)


def score_starts(year=None):
    s = pd.read_parquet(D / "starts.parquet")
    if year:
        s = s[s.year == year]
    rows = []
    for sid, r in enumerate(s.itertuples()):
        ey = r.year - 1
        g, lf = grid(r.cy, r.cx, ey), lf_burnable(r.cy, r.cx)
        if g is None:
            continue
        rr, cc = rowcol(r.cy, r.cx, r.lon, r.lat)
        base = dict(sid=sid, year=r.year, size=r.size, cause=r.cause, lat=r.lat)

        def add(kind, gg, ll, i, j):
            d = dict(base, kind=kind, row=i, col=j)
            for m in REACH_M:
                d[f"p{m}"] = reach(gg, i, j, m, r.lat)
                d[f"lf{m}"] = reach(ll, i, j, m, r.lat) if r.year >= 2025 else np.nan
            rows.append(d)

        add("start", g, lf, rr, cc)
        # near controls: valid pixels at least 300 m from the start in its own chunk
        dy, dx = px_m(r.lat)
        ii, jj = np.mgrid[0:CHUNK, 0:CHUNK]
        far = (np.hypot((ii - rr) * dy, (jj - cc) * dx) >= 300) & np.isfinite(g)
        cand = np.flatnonzero(far)
        for k in rng.choice(cand, min(N_CTRL, len(cand)), replace=False):
            add("near", g, lf, *divmod(int(k), CHUNK))
        gc = grid(r.ctrl_cy, r.ctrl_cx, ey)
        if gc is not None:
            lfc = lf_burnable(r.ctrl_cy, r.ctrl_cx)
            cand = np.flatnonzero(np.isfinite(gc))
            for k in rng.choice(cand, min(N_CTRL, len(cand)), replace=False):
                add(r.ring, gc, lfc, *divmod(int(k), CHUNK))
    return pd.DataFrame(rows)


def start_tables(df):
    out, tab = {}, []
    groups = ([("all", df)] + [(f"size {k}", df[df["size"] == k]) for k in df["size"].unique()]
              + [(f"cause {k}", df[df.cause == k]) for k in ["Natural", "Human"]])
    for gname, gdf in groups:
        for kind in ["near", "5-10 km", "20-30 km"]:
            for col in [f"p{m}" for m in REACH_M] + [f"lf{m}" for m in REACH_M]:
                a, lo, hi, n = auc_ci(gdf, col, kind)
                if not np.isfinite(a):
                    continue
                out[f"{gname}|{kind}|{col}"] = dict(auc=a, lo=lo, hi=hi, n=n)
                tab.append([gname, kind, col, n, f"{a:.3f}", f"[{lo:.3f}, {hi:.3f}]"])
    print(tabulate(tab, headers=["starts", "control", "map (reach m)", "n starts", "AUC", "95 %"]))
    st = df[df.kind == "start"]
    print("\nshare of starts on P(fuel) >= 0.5:", {k: round(float((g.p0 >= 0.5).mean()), 3) for k, g in st.groupby("size")},
          " near controls:", round(float((df[df.kind == "near"].p0 >= 0.5).mean()), 3))
    return out


def score_edges(year=None):
    per = pd.read_parquet(D / "perimeters.parquet").set_index("fire_id")
    e = pd.read_parquet(D / "edges.parquet")
    if year:
        e = e[e.year == year]
    bands = {"inside > 300 m": (-1e9, -300), "inside 0-100 m": (-100, 0), "outside 0-100 m": (0, 100), "outside > 300 m": (300, 1e9),
             "line (20 m)": (-20, 20), "inside 100-300 m": (-300, -100), "outside 100-300 m": (100, 300)}
    rows = []
    for r in e.itertuples():
        g = grid(r.cy, r.cx, r.year - 1)
        if g is None:
            continue
        geom = shapely.from_wkb(per.loc[r.fire_id, "geom"])
        lon0, lat0 = MOS_X0 + r.cx * CHUNK * MOS_RES, MOS_Y0 - r.cy * CHUNK * MOS_RES
        inside = rasterize([(geom, 1)], out_shape=(CHUNK, CHUNK), transform=from_origin(lon0, lat0, MOS_RES, MOS_RES), fill=0).astype(bool)
        if inside.all() or not inside.any():
            continue
        sam = px_m(lat0 - CHUNK * MOS_RES / 2)
        sd = np.where(inside, -distance_transform_edt(inside, sampling=sam), distance_transform_edt(~inside, sampling=sam))
        lf = lf_burnable(r.cy, r.cx) if r.year >= 2025 else None
        d = dict(fire_id=r.fire_id, year=r.year, cy=r.cy, cx=r.cx, acres=per.loc[r.fire_id, "acres"])
        ok = np.isfinite(g)
        d["auc_in_out"] = auc(g[inside & ok], g[~inside & ok])
        if lf is not None:
            okl = np.isfinite(lf)
            d["lf_auc_in_out"] = auc(lf[inside & okl], lf[~inside & okl])
        for name, (lo, hi) in bands.items():
            m = (sd > lo) & (sd <= hi) & ok
            d[f"mean {name}"] = float(g[m].mean()) if m.any() else np.nan
            d[f"lit {name}"] = float((g[m] >= 0.5).mean()) if m.any() else np.nan
            if lf is not None:
                ml = (sd > lo) & (sd <= hi) & np.isfinite(lf)
                d[f"lf {name}"] = float(lf[ml].mean()) if ml.any() else np.nan
        rows.append(d)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no edge chunks scored")
        return {}, df
    per_fire = df.groupby("fire_id").mean(numeric_only=True)
    out = {"fires": int(len(per_fire)), "chunks": int(len(df))}
    tab = []
    for name in bands:
        v = per_fire[f"mean {name}"]
        l = per_fire[f"lit {name}"]
        lfv = per_fire[f"lf {name}"] if f"lf {name}" in per_fire else pd.Series(dtype=float)
        out[name] = dict(mean_p=float(v.mean()), share_lit=float(l.mean()), lf_burnable=float(lfv.mean()) if len(lfv.dropna()) else None)
        tab.append([name, f"{v.mean():.3f}", f"{l.mean():.3f}", f"{lfv.mean():.3f}" if len(lfv.dropna()) else ""])
    print(tabulate(tab, headers=["band (per fire, mean)", "mean P(fuel)", "share P >= 0.5", "LANDFIRE burnable (2025+)"]))
    a = per_fire.auc_in_out
    boots = [rng.choice(a.dropna().to_numpy(), a.notna().sum()).mean() for _ in range(BOOT)]
    out["auc_in_out"] = dict(mean=float(a.mean()), median=float(a.median()), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)))
    print(f"\ninside vs outside AUC per fire: mean {a.mean():.3f} [{out['auc_in_out']['lo']:.3f}, {out['auc_in_out']['hi']:.3f}], "
          f"median {a.median():.3f}, {len(per_fire)} fires, {len(df)} chunks")
    if "lf_auc_in_out" in per_fire:
        la = per_fire.lf_auc_in_out.dropna()
        ha = per_fire.loc[la.index, "auc_in_out"]
        out["lf_auc_in_out"] = dict(mean=float(la.mean()), head_same_fires=float(ha.mean()), n=int(len(la)))
        print(f"2025+ fires ({len(la)}): head {ha.mean():.3f} vs LANDFIRE {la.mean():.3f}")
    return out, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=None, help="only fires of this year")
    a = ap.parse_args()
    tag = f"_{a.year}" if a.year else ""
    print(f"== starts{(' ' + str(a.year)) if a.year else ''}")
    df = score_starts(a.year)
    df.to_parquet(D / f"start_scores{tag}.parquet", index=False)
    st = start_tables(df)
    print(f"\n== edges{(' ' + str(a.year)) if a.year else ''}")
    ed, edf = score_edges(a.year)
    edf.to_parquet(D / f"edge_scores{tag}.parquet", index=False)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"fire_test{tag}.json").write_text(json.dumps({"starts": st, "edges": ed}, indent=1, default=float))


if __name__ == "__main__":
    main()
