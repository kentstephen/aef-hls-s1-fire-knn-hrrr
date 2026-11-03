# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn"]
# ///
"""Can a pixel's embedding tell fuel from not-fuel? Labels are LANDFIRE LF2024 FBFM40: 91 to 99 are
non-burnable (91 urban and roads, 93 farmland, 98 water, 99 bare), everything else burnable.

Inputs, one at a time and in pairs: AlphaEarth (64), TESSERA (128), OlmoEarth (128), Sentinel-2 three
seasons of 2024 (30 bands plus a seasonal NDVI each: the rival without embeddings).
Models: kNN (k tuned in inner folds) and logistic regression (a linear head, the cheapest thing a shader can
run). Holdout: 5 folds by window (about 2 x 2.5 km each), so a test pixel never has a neighbor from its own
window in training.

Scores: balanced accuracy, and for the non-burnable class its recall (share of non-fuel found) and
precision (share of "not fuel" calls that are right). Also the share of LANDFIRE road pixels (EVT 7299)
called not-fuel, and the same for each non-burnable kind.

    uv run scripts/fit_fuel.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
KS = [5, 15, 45]
KINDS = {91: "urban and roads", 93: "farmland", 98: "water", 99: "bare"}


def load():
    d = pd.read_parquet(ROOT / "data" / "fuel.parquet")
    d = d[d.fbfm > 0].reset_index(drop=True)  # LANDFIRE fill
    q = np.stack([np.frombuffer(b, np.int8) for b in d.aef]).astype(np.float32) / 127.5
    X = {"AEF": np.sign(q) * q * q}
    for s, name in [("tessera", "TESSERA"), ("olmo", "OlmoEarth")]:
        X[name] = np.stack([np.frombuffer(b, np.float32) for b in d[s]])
    s2 = []
    for season in ("spring", "summer", "fall"):
        v = np.stack([np.frombuffer(b, np.float32) for b in d[f"s2_{season}"]])
        ndvi = (v[:, 3] - v[:, 2]) / (v[:, 3] + v[:, 2])
        s2.append(np.c_[v, ndvi])
    X["S2 3 seasons"] = np.hstack(s2)
    ok = np.all([np.isfinite(x).all(1) for x in X.values()], axis=0)
    print(f"{len(d)} pixels, {ok.sum()} with every input; dropped: " +
          ", ".join(f"{k} {(~np.isfinite(x).all(1)).sum()}" for k, x in X.items()), flush=True)
    d = d[ok].reset_index(drop=True)
    X = {k: x[ok] for k, x in X.items()}
    return d, X


def fit_predict(model, X, y, groups, tr, te):
    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X)
    if model == "logistic":
        m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xs[tr], y[tr])
        return m.predict(Xs[te])
    best, best_s = None, -1
    for k in KS:
        s = []
        for itr, ite in GroupKFold(n_splits=3).split(tr, groups=groups[tr]):
            m = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(Xs[tr[itr]], y[tr[itr]])
            s.append(balanced_accuracy_score(y[tr[ite]], m.predict(Xs[tr[ite]])))
        if np.mean(s) > best_s:
            best, best_s = k, np.mean(s)
    m = KNeighborsClassifier(n_neighbors=best, weights="distance").fit(Xs[tr], y[tr])
    return m.predict(Xs[te])


def scores(d, y, p):
    out = dict(balanced_accuracy=balanced_accuracy_score(y, p), nonfuel_recall=recall_score(y, p), nonfuel_precision=precision_score(y, p),
               roads_called_nonfuel=float(p[d.evt.values == 7299].mean()))
    for code, kind in KINDS.items():
        m = d.fbfm.values == code
        if m.sum() >= 20:
            out[f"{kind} called non-fuel"] = float(p[m].mean())
    out["burnable called non-fuel"] = float(p[~y.astype(bool)].mean())
    return {k: round(float(v), 3) for k, v in out.items()}


def win_ci(y, pa, pb, groups, n=2000, seed=0):
    """95% interval of balanced accuracy (b) - (a), resampling windows."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.nonzero(groups == g)[0] for g in ug}
    out = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ug, size=len(ug))])
        if y[s].min() == y[s].max():
            continue
        out.append(balanced_accuracy_score(y[s], pb[s]) - balanced_accuracy_score(y[s], pa[s]))
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]


def main():
    d, X = load()
    y = d.fbfm.between(91, 99).to_numpy().astype(int)
    groups = d.win.to_numpy()
    print(f"non-burnable {y.mean():.1%} ({y.sum()}), roads {int((d.evt == 7299).sum())}, windows {d.win.nunique()}; by kind: " +
          ", ".join(f"{k} {int((d.fbfm == c).sum())}" for c, k in KINDS.items()), flush=True)
    sets = {k: [k] for k in X} | {"AEF + S2 3 seasons": ["AEF", "S2 3 seasons"]}
    preds, res = {}, {}
    for model in ("logistic", "kNN"):
        for name, parts in sets.items():
            Xm = np.hstack([X[p] for p in parts])
            p = np.zeros(len(y), int)
            for tr, te in GroupKFold(n_splits=5).split(Xm, groups=groups):
                p[te] = fit_predict(model, Xm, y, groups, tr, te)
            key = f"{model}: {name}"
            preds[key] = p
            res[key] = scores(d, y, p)
            print(f"  {key:32s} " + "  ".join(f"{k} {v:.2f}" for k, v in res[key].items()), flush=True)
    res["intervals (balanced accuracy b - a, 95%, by window)"] = {
        f"{m}: {b} vs {a}": win_ci(y, preds[f"{m}: {a}"], preds[f"{m}: {b}"], groups)
        for m in ("logistic", "kNN") for a, b in [("S2 3 seasons", "AEF"), ("S2 3 seasons", "AEF + S2 3 seasons"),
                                                    ("AEF", "TESSERA"), ("AEF", "OlmoEarth")]}
    for k, v in res["intervals (balanced accuracy b - a, 95%, by window)"].items():
        print(f"  {k}: [{v[0]:+.3f}, {v[1]:+.3f}]")
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(res, open(ROOT / "results" / "fuel.json", "w"), indent=1)
    out = d[["win", "win_name", "lon", "lat", "fbfm", "evt"]].copy()
    for k, p in preds.items():
        out[k] = p
    out.to_parquet(ROOT / "data" / "predictions_fuel.parquet", index=False)


if __name__ == "__main__":
    main()
