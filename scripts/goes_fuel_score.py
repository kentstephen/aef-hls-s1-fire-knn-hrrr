# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn"]
# ///
"""Does the fuel in and around a GOES event's first box tell WFIGS wildfires from the rest?

Matched = perimeter or start (goes_events.py), unmatched = everything else. AUC > 0.5 means matched events sit
on more fuel. Intervals: 95 %, resampling places (0.1 degree cells), since one place can host many events.
Also: the same with GOES's own signal (first-scan power, confidence class) alone and together with fuel
(a logistic fit, scored out of fold by place), and the share of each group with almost no fuel around.

Writes results/goes_fuel_<year>.json.

    uv run scripts/goes_fuel_score.py --year 2025
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
G = ROOT / "data" / "goes"
READINGS = ["knn", "head", "lf"]


def boot_auc(y, s, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.nonzero(groups == g)[0] for g in ug}
    out = []
    for _ in range(n):
        i = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if y[i].min() != y[i].max():
            out.append(roc_auc_score(y[i], s[i]))
    return [round(float(roc_auc_score(y, s)), 3), round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]


def oof(X, y, groups):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, groups=groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = LogisticRegression(max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
        p[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
    return p


def block(f, name):
    y = (f.label != "unmatched").to_numpy().astype(int)
    groups = (np.floor(f.lon / 0.1).astype(int) * 10000 + np.floor(f.lat / 0.1).astype(int)).to_numpy()
    res = dict(n=int(len(f)), matched=int(y.sum()), unmatched=int((1 - y).sum()), places=int(len(np.unique(groups))))
    if y.min() == y.max() or min(y.sum(), (1 - y).sum()) < 15:
        return res | dict(note="too few in one group")
    for r in READINGS:
        for part in ("in", "around"):
            s = f[f"{r}_{part}"].to_numpy()
            ok = np.isfinite(s)
            res[f"{r} {part}"] = boot_auc(y[ok], s[ok], groups[ok])
    sig = np.c_[np.log1p(f.first_power.fillna(0)), f.first_strength, np.log(f.first_px)]
    res["GOES alone (power, class, pixels)"] = boot_auc(y, oof(sig, y, groups), groups)
    for r in ("knn", "head", "lf"):
        X = np.c_[sig, f[f"{r}_in"].fillna(f[f"{r}_in"].mean()), f[f"{r}_around"].fillna(f[f"{r}_around"].mean())]
        res[f"GOES + {r} fuel"] = boot_auc(y, oof(X, y, groups), groups)
    for r in ("knn", "lf"):
        lo = (f[f"{r}_in"] + f[f"{r}_around"]) / 2 < 0.1
        res[f"share with < 10 % fuel in and around ({r})"] = dict(matched=round(float(lo[y == 1].mean()), 3),
                                                                  unmatched=round(float(lo[y == 0].mean()), 3))
    res["median fuel share"] = f.groupby("label")[[f"{r}_{p}" for r in READINGS for p in ("in", "around")]].median().round(3).to_dict("index")
    print(f"\n== {name}: {res['matched']} matched, {res['unmatched']} unmatched, {res['places']} places")
    for k, v in res.items():
        if isinstance(v, list):
            print(f"  {k:40s} AUC {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]")
        elif k.startswith("share"):
            print(f"  {k:40s} matched {v['matched']:.1%}, unmatched {v['unmatched']:.1%}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    a = ap.parse_args()
    f = pd.read_parquet(G / f"event_fuel_{a.year}.parquet")
    f = f[f.n_knn_in > 0].reset_index(drop=True)
    out = {"all events": block(f, "all events"),
           "not at a recurring pixel (under 5 events there in the year)": block(f[f.recur < 5], "not recurring"),
           "single-scan events": block(f[f.scans == 1], "single scan"),
           "events seen in 2+ scans": block(f[f.scans >= 2], "2+ scans")}
    for s, g in f.groupby("first_strength"):
        out[f"first-scan class {s}"] = block(g, f"class {s}")
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "results" / f"goes_fuel_{a.year}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
