# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn"]
# ///
"""Does the AEF embedding help predict live fuel moisture, on top of HLS, Sentinel-1, HRRR Fosberg,
terrain and the date? kNN with and without it, on field samples held out by site.

Rows: live fuel moisture samples (no new growth), 2020-02 onward, that have every input: a clear HLS
scene in the 20 days before, a Sentinel-1 scene in the 12 days before, 90 days of HRRR, terrain, and
the AEF embedding of the year before.

Feature blocks (each standardized; a block's weight is spread over its columns, so a 64-column block
does not outvote a 2-column one by size alone):
  season   sin and cos of day of year
  terrain  elevation, slope, northness, eastness (Mapterhorn)
  fosberg  mean daily max FFWI over 7, 30, 90 days; mean FFWI over 30 days (HRRR analysis)
  hls      six bands, NDVI, NDMI, NBR
  s1       VV and VH (dB), VH - VV, ascending or not
  aef      64 bands of the year before (50 m window mean; --aef pix or w15 for the others)
  species  one-hot (only in the "species known" runs; a map does not know the species of a pixel)

Holdouts:
  space  5 folds by site group (sites within 10 km share a group). Scored over California and over
         the sites within 150 km of Redding.
  time   train before 2025, test 2025 on at the held-out site groups of each fold.
k and the AEF block weight are tuned inside each training fold (3 inner folds by group), never on the
held-out sites. A gradient boosting model with and without AEF is the check that the answer is not
kNN's alone. The AEF difference gets a 95% interval from resampling site groups.

    uv run scripts/fit_knn.py [--aef w5] [--partial]   # --partial: read the *_parts dirs while pulls run
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
KS = [5, 10, 20, 40]
AEF_W = [0.5, 1, 2, 4]


def load(name, partial):
    p = D / f"{name}.parquet"
    if partial or not p.exists():
        return pd.concat([pd.read_parquet(f) for f in (D / f"{name}_parts").glob("*.parquet")], ignore_index=True)
    return pd.read_parquet(p)


def fosberg_features(h):
    h = h.sort_values(["site_id", "day"]).copy()
    g = h.groupby("site_id")
    # the window ends the day before the sample: the sample day's own afternoon has not happened at 05Z
    for n in (7, 30, 90):
        h[f"ffwi_max_{n}d"] = g.ffwi_max.transform(lambda s: s.shift(1).rolling(n, min_periods=int(n * 0.8)).mean())
    h["ffwi_mean_30d"] = g.ffwi_mean.transform(lambda s: s.shift(1).rolling(30, min_periods=24).mean())
    return h.rename(columns={"day": "date"})[["site_id", "date", "ffwi_max_7d", "ffwi_max_30d", "ffwi_max_90d", "ffwi_mean_30d"]]


def build(aef_col, partial):
    smp = pd.read_parquet(D / "samples.parquet")
    smp = smp[~smp.new_growth & (smp.date >= "2020-02-01")].copy()
    hls = load("hls", partial).dropna(subset=["nir"])
    s1 = load("s1", partial).dropna(subset=["vv"])
    if partial:  # a day split across two store chunks is in two parts: keep the fuller half
        h = load("hrrr", True).sort_values("hours").drop_duplicates(["site_id", "day"], keep="last")
    else:
        h = pd.read_parquet(D / "hrrr_daily.parquet")
    wx = fosberg_features(h)
    ter = pd.read_parquet(D / "terrain.parquet")
    aef = pd.read_parquet(D / "aef.parquet")
    aef = aef[~aef.nodata]
    emb = np.stack([np.frombuffer(b, np.int8 if aef_col == "pix" else np.float32) for b in aef[aef_col]])
    if aef_col == "pix":
        f = emb.astype(np.float32) / 127.5
        emb = np.sign(f) * f * f
    A = pd.DataFrame(emb, columns=[f"a{i:02d}" for i in range(64)])
    A["site_id"], A["year"] = aef.site_id.values, aef.year.values + 1  # the embedding of the year before
    df = (smp.assign(year=smp.date.dt.year)
          .merge(hls, on=["site_id", "date"]).merge(s1, on=["site_id", "date"]).merge(wx, on=["site_id", "date"])
          .merge(ter, on="site_id").merge(A, on=["site_id", "year"]))
    df = df.dropna(subset=["ffwi_max_90d", "ffwi_mean_30d", "a00"])
    doy = df.date.dt.dayofyear / 365.25 * 2 * np.pi
    df["doy_sin"], df["doy_cos"] = np.sin(doy), np.cos(doy)
    df["ndvi"] = (df.nir - df.red) / (df.nir + df.red)
    df["ndmi"] = (df.nir - df.swir1) / (df.nir + df.swir1)
    df["nbr"] = (df.nir - df.swir2) / (df.nir + df.swir2)
    df["vv_db"], df["vh_db"] = 10 * np.log10(df.vv), 10 * np.log10(df.vh)
    df["cross_db"] = df.vh_db - df.vv_db
    df["asc"] = (df.s1_orbit == "ascending").astype(float)
    return df.reset_index(drop=True)


def blocks(df):
    sp = pd.get_dummies(df.species).astype(float)
    sp.columns = [f"sp_{c}" for c in sp.columns]
    return {
        "season": df[["doy_sin", "doy_cos"]],
        "terrain": df[["elev_m", "slope_deg", "northness", "eastness"]],
        "fosberg": df[["ffwi_max_7d", "ffwi_max_30d", "ffwi_max_90d", "ffwi_mean_30d"]],
        "hls": df[["blue", "green", "red", "nir", "swir1", "swir2", "ndvi", "ndmi", "nbr"]],
        "s1": df[["vv_db", "vh_db", "cross_db", "asc"]],
        "aef": df[[f"a{i:02d}" for i in range(64)]],
        "species": sp,
    }


SETS = {
    "weather+season+terrain": ["season", "terrain", "fosberg"],
    "+ HLS": ["season", "terrain", "fosberg", "hls"],
    "+ HLS + S1": ["season", "terrain", "fosberg", "hls", "s1"],
    "+ HLS + S1 + AEF": ["season", "terrain", "fosberg", "hls", "s1", "aef"],
    "AEF, no HLS or S1": ["season", "terrain", "fosberg", "aef"],
}


def design(B, names, train_idx, aef_w):
    """Standardize on the training rows; scale each block to weight / sqrt(columns)."""
    cols = []
    for n in names:
        X = B[n].to_numpy(np.float64)
        mu, sd = X[train_idx].mean(0), X[train_idx].std(0)
        sd[sd == 0] = 1
        w = (aef_w if n == "aef" else 1.0) / np.sqrt(X.shape[1])
        cols.append((X - mu) / sd * w)
    return np.hstack(cols)


def knn_fit_predict(B, names, y, groups, tr, te, inner=3):
    has_aef = "aef" in names
    grid = [(k, w) for k in KS for w in (AEF_W if has_aef else [1.0])]
    best, best_err = None, np.inf
    gkf = GroupKFold(n_splits=inner)
    for k, w in grid:
        err = []
        for itr, ite in gkf.split(tr, groups=groups[tr]):
            a, b = tr[itr], tr[ite]
            X = design(B, names, a, w)
            m = KNeighborsRegressor(n_neighbors=k, weights="distance").fit(X[a], y[a])
            err.append(np.mean((m.predict(X[b]) - y[b]) ** 2))
        if np.mean(err) < best_err:
            best, best_err = (k, w), np.mean(err)
    X = design(B, names, tr, best[1])
    m = KNeighborsRegressor(n_neighbors=best[0], weights="distance").fit(X[tr], y[tr])
    return m.predict(X[te]), best


def gbm_fit_predict(B, names, y, tr, te):
    X = np.hstack([B[n].to_numpy(np.float64) for n in names])
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    return m.fit(X[tr], y[tr]).predict(X[te])


def score(y, p):
    e = p - y
    return dict(rmse=float(np.sqrt(np.mean(e**2))), mae=float(np.mean(np.abs(e))), r2=float(1 - np.sum(e**2) / np.sum((y - y.mean()) ** 2)),
                bias=float(e.mean()), n=int(len(y)))


def group_ci(y, p_a, p_b, groups, n=2000, seed=0):
    """95% interval of RMSE(b) - RMSE(a), resampling site groups."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.nonzero(groups == g)[0] for g in ug}
    out = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ug, size=len(ug))])
        out.append(np.sqrt(np.mean((p_b[s] - y[s]) ** 2)) - np.sqrt(np.mean((p_a[s] - y[s]) ** 2)))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def run(df, species_known, split):
    B = blocks(df)
    y = df.value.to_numpy(np.float64)
    groups = df.group.to_numpy()
    preds = {name: np.full(len(df), np.nan) for name in SETS}
    preds.update({"gbm + HLS + S1": np.full(len(df), np.nan), "gbm + HLS + S1 + AEF": np.full(len(df), np.nan)})
    chosen = {}
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(df, groups=groups)):
        if split == "time":
            tr = tr[(df.date.values[tr] < np.datetime64("2025-01-01"))]
            te = te[(df.date.values[te] >= np.datetime64("2025-01-01"))]
        for name, names in SETS.items():
            names = names + (["species"] if species_known else [])
            preds[name][te], best = knn_fit_predict(B, names, y, groups, tr, te)
            chosen.setdefault(name, []).append(best)
        base = SETS["+ HLS + S1"] + (["species"] if species_known else [])
        preds["gbm + HLS + S1"][te] = gbm_fit_predict(B, base, y, tr, te)
        preds["gbm + HLS + S1 + AEF"][te] = gbm_fit_predict(B, base + ["aef"], y, tr, te)
    ok = ~np.isnan(preds["+ HLS + S1"])
    res = {}
    for area, mask in [("California", ok), ("Redding 150 km", ok & (df.redding_km.values < 150))]:
        if mask.sum() < 30:
            continue
        r = {name: score(y[mask], p[mask]) for name, p in preds.items()}
        r["AEF effect, kNN (RMSE with - without, 95%)"] = group_ci(y[mask], preds["+ HLS + S1"][mask], preds["+ HLS + S1 + AEF"][mask], groups[mask])
        r["AEF effect, gbm (RMSE with - without, 95%)"] = group_ci(y[mask], preds["gbm + HLS + S1"][mask], preds["gbm + HLS + S1 + AEF"][mask], groups[mask])
        res[area] = r
    res["chosen (k, aef weight) per fold"] = {k: [list(map(float, b)) for b in v] for k, v in chosen.items()}
    return res, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aef", default="w5", choices=["pix", "w5", "w15"])
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    df = build(a.aef, a.partial)
    print(f"{len(df)} samples at {df.site_id.nunique()} sites in {df.group.nunique()} groups, "
          f"{(df.redding_km < 150).sum()} within 150 km of Redding; {df.date.min():%Y-%m-%d} to {df.date.max():%Y-%m-%d}", flush=True)
    allres = {}
    for split in ["space", "time"]:
        for sk in [False, True]:
            tag = f"{split}, species {'known' if sk else 'unknown'}"
            res, preds = run(df, sk, split)
            allres[tag] = res
            print(f"\n== {tag}")
            for area in ["California", "Redding 150 km"]:
                if area not in res:
                    continue
                print(f"  {area}")
                for name, r in res[area].items():
                    if isinstance(r, dict):
                        print(f"    {name:26s} RMSE {r['rmse']:6.1f}  MAE {r['mae']:5.1f}  R2 {r['r2']:5.2f}  bias {r['bias']:+5.1f}  n {r['n']}")
                    else:
                        print(f"    {name}: [{r[0]:+.1f}, {r[1]:+.1f}]")
            if split == "space" and not sk:
                out = df[["sample_id", "site_id", "group", "date", "species", "value", "redding_km"]].copy()
                for k, p in preds.items():
                    out[f"pred {k}"] = p
                out.to_parquet(D / f"predictions_{a.aef}.parquet", index=False)
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(allres, open(ROOT / "results" / f"knn_aef_{a.aef}{'_partial' if a.partial else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
