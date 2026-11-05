# /// script
# requires-python = ">=3.12"
# dependencies = ["duckdb", "numpy", "pandas", "pyarrow", "scikit-learn", "shapely", "psutil"]
# ///
"""Checks on the GOES + fuel result (goes_fuel_score.py): is the gain real, where does it come from, and why do
single-scan events carry nothing.

  gain     paired bootstrap of AUC(GOES + fuel) minus AUC(GOES alone): both out-of-fold scores are fixed once,
           then the same resampled places (0.1 degree cells) score both, so the interval is of the difference
  labels   perimeter vs unmatched and start vs unmatched, scored apart (perimeter fires are 100+ acres)
  single   what the matched single-scan events are: acres, cause, distance to the nearest road and building
           (Overture), next to matched 2+ scan events and unmatched single-scan events. The distances are kept per
           event in data/goes/ext/event_dist_<year>.parquet (km to the nearest road, major road, building center)

Several years pool into one table (--years 2023 2024 2025); places stay the resampling unit across years, since
one place can host fires in many years.

  overture   cache California's roads (geometry, class) and building centers (bbox middles) from Overture
             -> data/goes/ext/overture_roads_ca.parquet, overture_bld_ca.parquet. Peak about 5 GB.
  run        the three checks -> results/goes_fuel_check_<years>.json

    uv run scripts/goes_fuel_check.py overture
    uv run scripts/goes_fuel_check.py run --years 2025
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from goes_fuel_score import oof  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
G = ROOT / "data" / "goes"
EXT = G / "ext"
WFIGS = ROOT.parent / "aef-landcover-fosberg-cog" / "data" / "ignition" / "wfigs.parquet"
REL = "s3://overturemaps-us-west-2/release/2026-10-21.0"
CA = (-124.6, 32.4, -113.9, 42.1)
MAJOR = ("motorway", "trunk", "primary", "secondary", "tertiary")
READINGS = ("knn", "head", "lf")
DIST = ["road", "major_road", "building"]


def rss():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def overture(a):
    import duckdb
    EXT.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect()
    c.execute("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2'; "
              "SET enable_progress_bar=false; SET memory_limit='4GB';")
    box = f"bbox.xmin < {CA[2]} and bbox.xmax > {CA[0]} and bbox.ymin < {CA[3]} and bbox.ymax > {CA[1]}"
    f = EXT / "overture_roads_ca.parquet"
    if not f.exists():
        c.execute(f"copy (select class, ST_AsWKB(geometry) as g from read_parquet('{REL}/theme=transportation/type=segment/*') "
                  f"where subtype = 'road' and {box}) to '{f}' (format parquet)")
        print(f"roads: {c.execute(f'select count(*) from read_parquet({chr(39)}{f}{chr(39)})').fetchone()[0]} segments, rss {rss():.1f} GB", flush=True)
    f = EXT / "overture_bld_ca.parquet"
    if not f.exists():
        c.execute(f"copy (select ((bbox.xmin + bbox.xmax) / 2)::float as lon, ((bbox.ymin + bbox.ymax) / 2)::float as lat "
                  f"from read_parquet('{REL}/theme=buildings/type=building/*') where {box}) to '{f}' (format parquet)")
        print(f"buildings: {c.execute(f'select count(*) from read_parquet({chr(39)}{f}{chr(39)})').fetchone()[0]}, rss {rss():.1f} GB", flush=True)


def load(years):
    fs = []
    for y in years:
        f = pd.read_parquet(G / f"event_fuel_{y}.parquet").assign(year=y)
        fs.append(f[f.n_knn_in > 0])
    f = pd.concat(fs, ignore_index=True)
    f["place"] = np.floor(f.lon / 0.1).astype(int) * 10000 + np.floor(f.lat / 0.1).astype(int)
    return f


def goes_sig(f):
    return np.c_[np.log1p(f.first_power.fillna(0)), f.first_strength, np.log(f.first_px)]


def fuel_x(f, r):
    return np.c_[goes_sig(f), f[f"{r}_in"].fillna(f[f"{r}_in"].mean()), f[f"{r}_around"].fillna(f[f"{r}_around"].mean())]


def paired(y, scores, groups, n=2000, seed=0):
    """scores: dict name -> out-of-fold score. Returns each AUC with its interval and each one's gain over the first,
    all from the same resampled places."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.nonzero(groups == g)[0] for g in ug}
    names = list(scores)
    full = {k: roc_auc_score(y, s) for k, s in scores.items()}
    boot = {k: [] for k in names}
    for _ in range(n):
        i = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if y[i].min() == y[i].max():
            continue
        for k in names:
            boot[k].append(roc_auc_score(y[i], scores[k][i]))
    b = {k: np.array(v) for k, v in boot.items()}
    base = names[0]
    out = {}
    for k in names:
        out[k] = [round(full[k], 3), round(float(np.percentile(b[k], 2.5)), 3), round(float(np.percentile(b[k], 97.5)), 3)]
        if k != base:
            d = b[k] - b[base]
            out[f"gain {k} over {base}"] = dict(gain=round(full[k] - full[base], 3), ci=[round(float(np.percentile(d, 2.5)), 3),
                                                round(float(np.percentile(d, 97.5)), 3)], share_above_0=round(float((d > 0).mean()), 3))
    return out


def gain_block(f, name, pos=("perimeter", "start")):
    f = f[f.label.isin(pos) | (f.label == "unmatched")].reset_index(drop=True)
    y = (f.label != "unmatched").to_numpy().astype(int)
    g = f.place.to_numpy()
    res = dict(n=int(len(f)), matched=int(y.sum()), places=int(len(np.unique(g))))
    if min(y.sum(), (1 - y).sum()) < 15:
        return res | dict(note="too few in one group")
    scores = {"GOES alone": oof(goes_sig(f), y, g)}
    for r in READINGS:
        scores[f"GOES + {r}"] = oof(fuel_x(f, r), y, g)
    for r in READINGS:
        scores[f"{r} in alone"] = f[f"{r}_in"].fillna(f[f"{r}_in"].mean()).to_numpy()
    res |= paired(y, scores, g)
    if all(c in f for c in DIST):  # hold remoteness fixed: WFIGS records fires near people, and fuel thins near people
        dist = np.log1p(f[DIST].to_numpy(float))
        s2 = {"GOES + remoteness": oof(np.c_[goes_sig(f), dist], y, g),
              "GOES + remoteness + knn": oof(np.c_[fuel_x(f, "knn"), dist], y, g)}
        res |= paired(y, s2, g)
    print(f"\n== {name}: {res['matched']} matched of {res['n']}, {res['places']} places")
    for k, v in res.items():
        if isinstance(v, list):
            print(f"  {k:36s} AUC {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]")
        elif isinstance(v, dict) and "gain" in v:
            print(f"  {k:36s} {v['gain']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}], above 0 in {v['share_above_0']:.1%}")
    return res


def distances(f):
    """km from each event's center to the nearest road (any, and major) and the nearest building center."""
    import shapely
    from scipy.spatial import cKDTree  # scikit-learn brings scipy
    kx = np.cos(np.radians(37.0)) * 111.32  # km per degree of lon, at California's middle
    ky = 111.32
    b = pd.read_parquet(EXT / "overture_bld_ca.parquet")
    t = cKDTree(np.c_[b.lon.to_numpy() * kx, b.lat.to_numpy() * ky])
    d_b, _ = t.query(np.c_[f.lon.to_numpy() * kx, f.lat.to_numpy() * ky])
    del b, t
    r = pd.read_parquet(EXT / "overture_roads_ca.parquet")
    geoms = shapely.from_wkb(r.g.to_numpy())
    major = r["class"].isin(MAJOR).to_numpy()
    del r
    pts = shapely.points(f.lon.to_numpy(), f.lat.to_numpy())
    out = {}
    for name, m in (("road", np.ones(len(geoms), bool)), ("major_road", major)):
        tree = shapely.STRtree(geoms[m])
        i = tree.nearest(pts)
        near = geoms[m][i]
        # distance in km on a local flat frame around each point (lon scaled by its latitude)
        p = shapely.get_coordinates(pts)
        q = shapely.get_coordinates(shapely.shortest_line(pts, near))[1::2]
        out[name] = np.hypot((q[:, 0] - p[:, 0]) * np.cos(np.radians(p[:, 1])) * 111.32, (q[:, 1] - p[:, 1]) * 111.32)
    out["building"] = d_b
    print(f"distances done, rss {rss():.1f} GB", flush=True)
    return pd.DataFrame(out, index=f.index)


def cause_of(f):
    """WFIGS cause (Human / Natural / Undetermined) for matched events, by incident name and year."""
    w = pd.read_parquet(WFIGS, columns=["IncidentName", "FireCause", "FireDiscoveryDateTime", "POOState", "IncidentTypeCategory"])
    w = w[(w.POOState == "US-CA") & (w.IncidentTypeCategory == "WF")]
    w = w.assign(year=w.FireDiscoveryDateTime.dt.year, key=w.IncidentName.str.upper().str.strip())
    c = w.groupby(["year", "key"]).FireCause.first()
    k = list(zip(f.year, f.fire.fillna("").str.upper().str.strip()))
    return pd.Series([c.get(x, np.nan) for x in k], index=f.index)


def event_distances(f):
    """distances() for every event, cached per year in data/goes/ext/event_dist_<year>.parquet (roads load once)."""
    todo = [y for y in f.year.unique() if not (EXT / f"event_dist_{y}.parquet").exists()]
    if todo:
        g = f[f.year.isin(todo)]
        d = distances(g).assign(event=g.event.to_numpy(), year=g.year.to_numpy())
        for y, dy in d.groupby("year"):
            dy.to_parquet(EXT / f"event_dist_{y}.parquet", index=False)
    d = pd.concat([pd.read_parquet(EXT / f"event_dist_{y}.parquet") for y in f.year.unique()])
    return f[["event", "year"]].reset_index().merge(d, on=["event", "year"]).set_index("index")[["road", "major_road", "building"]]


def single(f):
    if "road" not in f:
        f = f.join(event_distances(f))
    f["cause"] = cause_of(f).where(f.label != "unmatched")
    f["group"] = np.select([(f.label != "unmatched") & (f.scans == 1), (f.label != "unmatched") & (f.scans >= 2),
                            (f.label == "unmatched") & (f.scans == 1)], ["matched single scan", "matched 2+ scans", "unmatched single scan"], "other")
    out = {}
    for gname, g in f[f.group != "other"].groupby("group"):
        o = dict(n=int(len(g)))
        if gname.startswith("matched"):
            o["label"] = g.label.value_counts().to_dict()
            ac = g.fire_acres.dropna()
            o["acres quartiles"] = [round(float(x), 1) for x in ac.quantile([0.25, 0.5, 0.75])]
            o["share under 10 acres"] = round(float((ac < 10).mean()), 3)
            o["share 100+ acres"] = round(float((ac >= 100).mean()), 3)
            o["cause"] = g.cause.fillna("unknown").value_counts().to_dict()
        for k in ("road", "major_road", "building"):
            o[f"{k} km median"] = round(float(g[k].median()), 2)
            o[f"share within 0.5 km of {k}"] = round(float((g[k] <= 0.5).mean()), 3)
        o["first_power median MW"] = round(float(g.first_power.median()), 1)
        o["first_strength"] = g.first_strength.value_counts().sort_index().to_dict()
        o["knn in median"] = round(float(g.knn_in.median()), 3)
        o["knn around median"] = round(float(g.knn_around.median()), 3)
        out[gname] = o
        print(f"\n-- {gname}: {json.dumps(o, default=str)}")
    # within matched single-scan events: does fuel separate them from unmatched single-scan ones at all
    # once the roadside ones (within 0.5 km of a road) are set aside?
    s = f[f.scans == 1]
    for name, m in (("single scan, over 0.5 km from any road", s.road > 0.5), ("single scan, within 0.5 km of a road", s.road <= 0.5)):
        g = s[m]
        y = (g.label != "unmatched").to_numpy().astype(int)
        if min(y.sum(), (1 - y).sum()) >= 15:
            out[name] = dict(n=int(len(g)), matched=int(y.sum()), knn_in_auc=round(float(roc_auc_score(y, g.knn_in.fillna(g.knn_in.mean()))), 3))
            print(f"  {name}: {out[name]}")
    return out


def run(a):
    f = load(a.years)
    print(f"{len(f)} events over {a.years}: {f.label.value_counts().to_dict()}, rss {rss():.1f} GB")
    if all((EXT / f"event_dist_{y}.parquet").exists() for y in a.years) or a.distances:
        f = f.join(event_distances(f))
    res = {"gain": {}, "labels": {}}
    res["gain"]["all events"] = gain_block(f, "all events")
    res["gain"]["events seen in 2+ scans"] = gain_block(f[f.scans >= 2], "2+ scans")
    res["gain"]["single-scan events"] = gain_block(f[f.scans == 1], "single scan")
    res["gain"]["first-scan class 6"] = gain_block(f[f.first_strength == 6], "class 6")
    if len(a.years) > 1:
        for y in a.years:
            res["gain"][f"all events, {y}"] = gain_block(f[f.year == y], f"all, {y}")
    for lab in ("perimeter", "start"):
        res["labels"][f"{lab} vs unmatched"] = gain_block(f, f"{lab} vs unmatched", pos=(lab,))
        res["labels"][f"{lab} vs unmatched, 2+ scans"] = gain_block(f[f.scans >= 2], f"{lab} vs unmatched, 2+ scans", pos=(lab,))
    res["median fuel by label"] = f.groupby("label")[["knn_in", "knn_around", "head_in", "lf_in"]].median().round(3).to_dict("index")
    if not a.no_single:
        res["single"] = single(f)
    tag = "_".join(map(str, a.years))
    json.dump(res, open(ROOT / "results" / f"goes_fuel_check_{tag}.json", "w"), indent=1, default=str)
    print(f"\nwrote results/goes_fuel_check_{tag}.json, peak rss {rss():.1f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["overture", "run"])
    ap.add_argument("--years", type=int, nargs="+", default=[2025])
    ap.add_argument("--distances", action="store_true", help="compute missing road / building distances first (peak ~6 GB)")
    ap.add_argument("--no-single", action="store_true", help="skip the road / building distances")
    a = ap.parse_args()
    {"overture": overture, "run": run}[a.stage](a)


if __name__ == "__main__":
    main()
