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

    uv run scripts/fit_knn.py [--emb aef|tessera|olmo] [--aef w5] [--common] [--mlp] [--wxhist] [--partial]
--mlp adds a small network (32 -> 16) next to kNN and the GBM; --wxhist adds rain and drying history (see WXH).
--s2 keeps only rows with a clear Sentinel-2 scene too and adds S2 rows (10 m pixel, 30 m and 90 m means; see S2 sets).
--sathist adds HLS and Sentinel-1 history over the 120 days before each sample (blocks hhist, shist; see sat_history_features).
--wide extends the kNN k grid to 160 and the MLP penalty grid to 30 (the old grids' top values were picked most folds).
--partial reads the *_parts dirs while pulls run.
--emb swaps the embedding block (AlphaEarth 64, TESSERA 128, OlmoEarth 128); the block keeps the name "aef" inside.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

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


WXH = ["rain_7d", "rain_30d", "rain_90d", "rain_wy", "dry_days", "vpd_7d", "vpd_30d", "vpd_90d", "rhmin_30d", "tmax_30d"]


def wx_history_features(h):
    """Rain and drying history that moisture follows over weeks (Fosberg is mostly today's wind and humidity).
    The HRRR precip column is a relative amount (its yearly sums look like meters); it is log-scaled and
    standardized, so its units do not matter. Every window ends the day before the sample."""
    h = h.sort_values(["site_id", "day"]).copy()
    es = 0.6108 * np.exp(17.27 * h.t_max / (h.t_max + 237.3))
    h["vpd"] = es * (1 - h.rh_min / 100)  # kPa at the afternoon peak
    g = h.groupby("site_id")
    prev = lambda col: g[col].shift(1)
    h["p1"], h["vpd1"], h["rh1"], h["t1"] = prev("precip_mm"), prev("vpd"), prev("rh_min"), prev("t_max")
    g = h.groupby("site_id")
    for n in (7, 30, 90):
        h[f"rain_{n}d"] = np.log1p(1000 * g.p1.transform(lambda s: s.rolling(n, min_periods=int(n * 0.8)).sum()))
        h[f"vpd_{n}d"] = g.vpd1.transform(lambda s: s.rolling(n, min_periods=int(n * 0.8)).mean())
    h["rhmin_30d"] = g.rh1.transform(lambda s: s.rolling(30, min_periods=24).mean())
    h["tmax_30d"] = g.t1.transform(lambda s: s.rolling(30, min_periods=24).mean())
    wy = h.day.dt.year + (h.day.dt.month >= 10)  # water year starts Oct 1
    h["rain_wy"] = np.log1p(1000 * h.groupby(["site_id", wy]).p1.cumsum())
    h.loc[(h.day < "2019-10-02"), "rain_wy"] = np.nan  # the HRRR pull starts 2019-09-01, so WY2020 is the first whole one
    wet = h.p1 > 0.0005  # about the top 10% of days
    last = h.day.where(wet).groupby(h.site_id).ffill()
    h["dry_days"] = np.log1p((h.day - last).dt.days.clip(upper=365))
    return h.rename(columns={"day": "date"})[["site_id", "date"] + WXH]


S2_BANDS = ["blue", "green", "red", "nir", "re1", "re2", "re3", "nir08", "swir1", "swir2"]
S2_FOOT = {"p": "10 m", "w3": "30 m", "w9": "90 m"}


def s2_features(partial):
    """Ten bands per footprint plus NDVI (B08), NDMI (B8A, B11), NBR (B8A, B12) and a red edge index (B8A, B05).
    Columns are renamed s2<foot>_<band> so they do not collide with the HLS bands."""
    s2 = load("s2", partial).dropna(subset=["nir_p"])
    out = s2[["site_id", "date"]].copy()
    nd = lambda a, b: (a - b) / (a + b)
    for f in S2_FOOT:
        b = {k: s2[f"{k}_{f}"] for k in S2_BANDS}
        for k, v in b.items():
            out[f"s2{f}_{k}"] = v
        out[f"s2{f}_ndvi"] = nd(b["nir"], b["red"])
        out[f"s2{f}_ndmi"] = nd(b["nir08"], b["swir1"])
        out[f"s2{f}_nbr"] = nd(b["nir08"], b["swir2"])
        out[f"s2{f}_ndre"] = nd(b["nir08"], b["re1"])
    return out


HSPANS = [(0, 30), (30, 60), (60, 121)]


def _spans(t, v, d, lookback=121):
    """Means of v over the day spans before d, the slope per 30 days, and the latest value minus the max."""
    age = (d - t) / np.timedelta64(1, "D")
    m = (age >= 0) & (age < lookback)
    out = [np.nanmean(v[m & (age >= a) & (age < b)], 0) if (m & (age >= a) & (age < b)).any() else np.full(v.shape[1], np.nan)
           for a, b in HSPANS]
    if m.sum() >= 3:
        A = np.c_[-age[m] / 30, np.ones(m.sum())]
        slope = np.linalg.lstsq(A, v[m], rcond=None)[0][0]
        latest = v[m][np.argmin(age[m])]
        drop = latest - np.nanmax(v[m], 0)
    else:
        slope = drop = np.full(v.shape[1], np.nan)
    return np.concatenate(out + [slope, drop])


def sat_history_features(sd, partial):
    """Per site-date, from data/hls_hist and data/s1_hist: HLS NDVI, NDMI, NBR and S1 VV, VH (dB) and VH - VV,
    each as means over 0-30, 30-60 and 60-120 days back, slope per 30 days and latest minus the 120-day max.
    Empty spans are filled with the 120-day mean of the same index, then the overall median."""
    h = load("hls_hist", partial)
    h["ndvi"], h["ndmi"], h["nbr"] = [(h.nir - h[b]) / (h.nir + h[b]) for b in ("red", "swir1", "swir2")]
    s = load("s1_hist", partial)
    s["vv_db"], s["vh_db"] = 10 * np.log10(s.vv), 10 * np.log10(s.vh)
    s["xr_db"] = s.vh_db - s.vv_db
    rows = []
    for (src, cols), pref in [((h, ["ndvi", "ndmi", "nbr"]), "hh"), ((s, ["vv_db", "vh_db", "xr_db"]), "sh")]:
        g = {k: (v.time.values, v[cols].to_numpy(np.float64)) for k, v in src.groupby("site_id")}
        names = [f"{pref}_{c}_{a}_{b}" for a, b in HSPANS for c in cols] + [f"{pref}_{c}_slope" for c in cols] + [f"{pref}_{c}_drop" for c in cols]
        vals = np.full((len(sd), len(names)), np.nan)
        for i, (sid, d) in enumerate(zip(sd.site_id.values, sd.date.values)):
            if sid in g:
                vals[i] = _spans(*g[sid], d + np.timedelta64(1, "D"))
        f = pd.DataFrame(vals, columns=names)
        for c in cols:  # an empty span takes the mean of the others
            span_cols = [f"{pref}_{c}_{a}_{b}" for a, b in HSPANS]
            f[span_cols] = f[span_cols].T.fillna(f[span_cols].mean(axis=1)).T
        rows.append(f)
    out = pd.concat([sd.reset_index(drop=True)] + rows, axis=1)
    feat = [c for c in out.columns if c.startswith(("hh_", "sh_"))]
    out[feat] = out[feat].fillna(out[feat].median())
    print(f"  sat history: {np.isfinite(np.vstack([r.to_numpy() for r in rows])).all(1).mean():.0%} of site-dates have every history value before filling", flush=True)
    return out


def build(aef_col, partial, emb="aef", s2=False, sathist=False):
    smp = pd.read_parquet(D / "samples.parquet")
    smp = smp[~smp.new_growth & (smp.date >= "2020-02-01")].copy()
    hls = load("hls", partial).dropna(subset=["nir"])
    s1 = load("s1", partial).dropna(subset=["vv"])
    if partial:  # a day split across two store chunks is in two parts: keep the fuller half
        h = load("hrrr", True).sort_values("hours").drop_duplicates(["site_id", "day"], keep="last")
    else:
        h = pd.read_parquet(D / "hrrr_daily.parquet")
    wx = fosberg_features(h).merge(wx_history_features(h), on=["site_id", "date"])
    ter = pd.read_parquet(D / "terrain.parquet")
    aef = pd.read_parquet(D / f"{emb}.parquet")
    aef = aef[~aef.nodata]
    pix_int8 = emb == "aef" and aef_col == "pix"
    vec = np.stack([np.frombuffer(b, np.int8 if pix_int8 else np.float32) for b in aef[aef_col]])
    if pix_int8:
        f = vec.astype(np.float32) / 127.5
        vec = np.sign(f) * f * f
    A = pd.DataFrame(vec, columns=[f"a{i:03d}" for i in range(vec.shape[1])])
    A["site_id"], A["year"] = aef.site_id.values, aef.year.values + 1  # the embedding of the year before
    df = (smp.assign(year=smp.date.dt.year)
          .merge(hls, on=["site_id", "date"]).merge(s1, on=["site_id", "date"]).merge(wx, on=["site_id", "date"])
          .merge(ter, on="site_id").merge(A, on=["site_id", "year"]))
    if s2:
        df = df.merge(s2_features(partial), on=["site_id", "date"])
    df = df.dropna(subset=["ffwi_max_90d", "ffwi_mean_30d", "a000"])
    if sathist:
        df = df.merge(sat_history_features(df[["site_id", "date"]].drop_duplicates(), partial), on=["site_id", "date"])
    df[WXH] = df[WXH].fillna(df[WXH].median())  # dry_days before the first wet day of the record
    doy = df.date.dt.day_of_year / 365.25 * 2 * np.pi
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
        "wxhist": df[WXH],
        "hhist": df[[c for c in df.columns if c.startswith("hh_")]],
        "shist": df[[c for c in df.columns if c.startswith("sh_")]],
        "aef": df[[c for c in df.columns if c.startswith("a") and c[1:].isdigit()]],
        "species": sp,
    } | {f"s2_{f}": df[[c for c in df.columns if c.startswith(f"s2{f}_")]] for f in S2_FOOT if f"s2{f}_nir" in df}


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


MLP_ALPHA = [1e-2, 0.3, 3.0]


def widen():
    """Grids past the old top values (k = 40, alpha = 3), which were picked in most folds."""
    KS[:] = [10, 20, 40, 80, 160]
    MLP_ALPHA[:] = [0.3, 3.0, 10.0, 30.0]


def _mlp(X, y, tr, te, alpha, seeds):
    ym, ys = y[tr].mean(), y[tr].std()
    out = []
    for s in seeds:
        m = MLPRegressor(hidden_layer_sizes=(32, 16), alpha=alpha, learning_rate_init=1e-3, batch_size=256, max_iter=300,
                         early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, random_state=s)
        out.append(m.fit(X[tr], (y[tr] - ym) / ys).predict(X[te]) * ys + ym)
    return np.mean(out, 0)


def mlp_fit_predict(B, names, y, groups, tr, te, inner=3):
    """A small network (inputs -> 32 -> 16 -> 1, ReLU): a few KB of weights, the size a shader can run per pixel.
    Inputs and target standardized on the training rows; early stopping on 15% of them. The L2 penalty is
    picked by 3 inner folds by site group (one seed), then the final fit is the mean of two seeds."""
    X = np.hstack([B[n].to_numpy(np.float64) for n in names])
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1
    X = (X - mu) / sd
    errs = {}
    for alpha in MLP_ALPHA:
        e = []
        for itr, ite in GroupKFold(n_splits=inner).split(tr, groups=groups[tr]):
            e.append(np.mean((_mlp(X, y, tr[itr], tr[ite], alpha, (0,)) - y[tr[ite]]) ** 2))
        errs[alpha] = np.mean(e)
    alpha = min(errs, key=errs.get)
    return _mlp(X, y, tr, te, alpha, (0, 1)), alpha


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


S2_SETS = {
    "+ S2 10 m": ["season", "terrain", "fosberg", "s2_p"],
    "+ S2 30 m": ["season", "terrain", "fosberg", "s2_w3"],
    "+ S2 90 m": ["season", "terrain", "fosberg", "s2_w9"],
    "+ S2 10 m + S1 + AEF": ["season", "terrain", "fosberg", "s2_p", "s1", "aef"],
    "+ S2 90 m + S1 + AEF": ["season", "terrain", "fosberg", "s2_w9", "s1", "aef"],
    "+ HLS + S2 10 m + S1 + AEF": ["season", "terrain", "fosberg", "hls", "s2_p", "s1", "aef"],
}
S2_GBM = {
    "gbm + S2 10 m + S1 + AEF": S2_SETS["+ S2 10 m + S1 + AEF"],
    "gbm + S2 90 m + S1 + AEF": S2_SETS["+ S2 90 m + S1 + AEF"],
    "gbm + HLS + S2 10 m + S1 + AEF": S2_SETS["+ HLS + S2 10 m + S1 + AEF"],
}
# (label, a, b): interval of RMSE(b) - RMSE(a); below 0 means b is better
S2_CI = [
    ("S2 10 m vs HLS, kNN (RMSE S2 - HLS, 95%)", "+ HLS", "+ S2 10 m"),
    ("S2 30 m vs HLS, kNN", "+ HLS", "+ S2 30 m"),
    ("S2 90 m vs HLS, kNN", "+ HLS", "+ S2 90 m"),
    ("S2 10 m vs S2 90 m, kNN (RMSE 10 m - 90 m)", "+ S2 90 m", "+ S2 10 m"),
    ("S2 10 m in place of HLS, kNN with S1 + AEF", "+ HLS + S1 + AEF", "+ S2 10 m + S1 + AEF"),
    ("S2 10 m added to HLS, kNN with S1 + AEF", "+ HLS + S1 + AEF", "+ HLS + S2 10 m + S1 + AEF"),
    ("S2 10 m in place of HLS, gbm with S1 + AEF", "gbm + HLS + S1 + AEF", "gbm + S2 10 m + S1 + AEF"),
    ("S2 10 m vs S2 90 m, gbm with S1 + AEF", "gbm + S2 90 m + S1 + AEF", "gbm + S2 10 m + S1 + AEF"),
    ("S2 10 m added to HLS, gbm with S1 + AEF", "gbm + HLS + S1 + AEF", "gbm + HLS + S2 10 m + S1 + AEF"),
]


def run(df, species_known, split, mlp=False, wxhist=False, s2=False, sathist=False):
    B = blocks(df)
    y = df.value.to_numpy(np.float64)
    groups = df.group.to_numpy()
    base = SETS["+ HLS + S1"]
    knn_sets = dict(SETS) | (S2_SETS if s2 else {})
    other = {"gbm + HLS + S1": base, "gbm + HLS + S1 + AEF": base + ["aef"]} | (S2_GBM if s2 else {})
    if wxhist:
        knn_sets["+ HLS + S1 + wx history"] = base + ["wxhist"]
        knn_sets["+ HLS + S1 + AEF + wx history"] = base + ["aef", "wxhist"]
        other["gbm + HLS + S1 + wx history"] = base + ["wxhist"]
        other["gbm + HLS + S1 + AEF + wx history"] = base + ["aef", "wxhist"]
    if sathist:
        sh = ["hhist", "shist"]
        wx = ["wxhist"] if wxhist else []
        tag = " + wx" if wxhist else ""
        for aef_ in ([], ["aef"]):
            label = f"+ HLS + S1{' + AEF' if aef_ else ''}{tag} + sat history"
            knn_sets[label] = base + aef_ + wx + sh
            other["gbm " + label] = base + aef_ + wx + sh
    if mlp:
        for k in list(other):
            other["mlp" + k[3:]] = other[k]
    preds = {name: np.full(len(df), np.nan) for name in list(knn_sets) + list(other)}
    chosen = {}
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(df, groups=groups)):
        if split == "time":
            tr = tr[(df.date.values[tr] < np.datetime64("2025-01-01"))]
            te = te[(df.date.values[te] >= np.datetime64("2025-01-01"))]
        sp = ["species"] if species_known else []
        for name, names in knn_sets.items():
            preds[name][te], best = knn_fit_predict(B, names + sp, y, groups, tr, te)
            chosen.setdefault(name, []).append(best)
        for name, names in other.items():
            if name.startswith("gbm"):
                preds[name][te] = gbm_fit_predict(B, names + sp, y, tr, te)
            else:
                preds[name][te], alpha = mlp_fit_predict(B, names + sp, y, groups, tr, te)
                chosen.setdefault(name, []).append([alpha])
        print(f"   fold {fold} done", flush=True)
    ok = ~np.isnan(preds["+ HLS + S1"])
    res = {}
    for area, mask in [("California", ok), ("Redding 150 km", ok & (df.redding_km.values < 150))]:
        if mask.sum() < 30:
            continue
        r = {name: score(y[mask], p[mask]) for name, p in preds.items()}
        ci = lambda a, b: group_ci(y[mask], preds[a][mask], preds[b][mask], groups[mask])
        r["AEF effect, kNN (RMSE with - without, 95%)"] = ci("+ HLS + S1", "+ HLS + S1 + AEF")
        r["AEF effect, gbm (RMSE with - without, 95%)"] = ci("gbm + HLS + S1", "gbm + HLS + S1 + AEF")
        if mlp:
            r["AEF effect, mlp (RMSE with - without, 95%)"] = ci("mlp + HLS + S1", "mlp + HLS + S1 + AEF")
            r["mlp vs kNN, with AEF (RMSE mlp - kNN, 95%)"] = ci("+ HLS + S1 + AEF", "mlp + HLS + S1 + AEF")
            r["mlp vs gbm, with AEF (RMSE mlp - gbm, 95%)"] = ci("gbm + HLS + S1 + AEF", "mlp + HLS + S1 + AEF")
        if s2:
            for label, a_, b_ in S2_CI:
                r[label] = ci(a_, b_)
        if sathist:
            tag = " + wx" if wxhist else ""
            no = "+ HLS + S1 + wx history" if wxhist else "+ HLS + S1"
            for m_ in ["", "gbm ", "mlp "] if mlp else ["", "gbm "]:
                r[f"sat history effect, {m_ or 'kNN '}no AEF (95%)"] = ci(m_ + no if m_ else no, f"{m_}+ HLS + S1{tag} + sat history")
                r[f"AEF effect on top of sat history, {m_ or 'kNN '}(95%)"] = ci(f"{m_}+ HLS + S1{tag} + sat history", f"{m_}+ HLS + S1 + AEF{tag} + sat history")
            r["sat history in place of AEF, gbm (RMSE history - AEF, 95%)"] = ci(
                "gbm + HLS + S1 + AEF" + (" + wx history" if wxhist else ""), f"gbm + HLS + S1{tag} + sat history")
        if wxhist:
            r["wx history effect, kNN with AEF (95%)"] = ci("+ HLS + S1 + AEF", "+ HLS + S1 + AEF + wx history")
            r["wx history effect, gbm with AEF (95%)"] = ci("gbm + HLS + S1 + AEF", "gbm + HLS + S1 + AEF + wx history")
            if mlp:
                r["wx history effect, mlp with AEF (95%)"] = ci("mlp + HLS + S1 + AEF", "mlp + HLS + S1 + AEF + wx history")
        res[area] = r
    res["chosen per fold (kNN: k, aef weight; mlp: alpha)"] = {k: [list(map(float, b)) for b in v] for k, v in chosen.items()}
    return res, preds


def noise_floor(df):
    """Error no model can beat. With species unknown every input is shared by all samples at a site-date,
    so the best any model can do there is the site-date mean; with species known, the site-date-species
    mean. RMSE against those means (in-sample, so a lower bound)."""
    out = {}
    for area, m in [("California", np.ones(len(df), bool)), ("Redding 150 km", df.redding_km.values < 150)]:
        d = df[m]
        for name, keys in [("species unknown", ["site_id", "date"]), ("species known", ["site_id", "date", "species"])]:
            e = d.value - d.groupby(keys).value.transform("mean")
            rmse = float(np.sqrt(np.mean(e**2)))
            out[f"{area}, {name}"] = dict(rmse=round(rmse, 1), r2_ceiling=round(float(1 - rmse**2 / d.value.var(ddof=0)), 2))
    return out


def suffix(a):
    flags = [("common", a.common), ("mlp", a.mlp), ("wxh", a.wxhist), ("s2", a.s2), ("wide", a.wide), ("sathist", a.sathist)]
    return f"{a.emb}_{a.aef}" + "".join(f"_{n}" for n, on in flags if on)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aef", default="w5", choices=["pix", "w5", "w15"])
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--emb", default="aef", choices=["aef", "tessera", "olmo"])
    ap.add_argument("--common", action="store_true", help="only site-years that AEF, TESSERA and OlmoEarth all cover")
    ap.add_argument("--mlp", action="store_true", help="add small-network rows")
    ap.add_argument("--wxhist", action="store_true", help="add rain and drying history rows")
    ap.add_argument("--s2", action="store_true", help="only rows with clear S2 too; add S2 rows")
    ap.add_argument("--wide", action="store_true", help="wider k and MLP penalty grids")
    ap.add_argument("--sathist", action="store_true", help="add HLS and S1 history over the 120 days before")
    a = ap.parse_args()
    if a.wide:
        widen()
    df = build(a.aef, a.partial, a.emb, a.s2, a.sathist)
    df = df.dropna(subset=[c for c in df.columns if c.startswith("a") and c[1:].isdigit()])
    if a.common:
        ok = None
        for e in ["aef", "tessera", "olmo"]:
            t = pd.read_parquet(D / f"{e}.parquet", columns=["site_id", "year", "nodata"])
            t = set(zip(t.site_id[~t.nodata], t.year[~t.nodata] + 1))
            ok = t if ok is None else ok & t
        df = df[[k in ok for k in zip(df.site_id, df.year)]].reset_index(drop=True)
    print(f"{len(df)} samples at {df.site_id.nunique()} sites in {df.group.nunique()} groups, "
          f"{(df.redding_km < 150).sum()} within 150 km of Redding; {df.date.min():%Y-%m-%d} to {df.date.max():%Y-%m-%d}", flush=True)
    allres = {"noise floor": noise_floor(df)}
    for k, v in allres["noise floor"].items():
        print(f"  floor {k}: {v}")
    for split in ["space", "time"]:
        for sk in [False, True]:
            tag = f"{split}, species {'known' if sk else 'unknown'}"
            res, preds = run(df, sk, split, a.mlp, a.wxhist, a.s2, a.sathist)
            allres[tag] = res
            print(f"\n== {tag}")
            for area in ["California", "Redding 150 km"]:
                if area not in res:
                    continue
                print(f"  {area}")
                for name, r in res[area].items():
                    if isinstance(r, dict):
                        print(f"    {name:34s} RMSE {r['rmse']:6.1f}  MAE {r['mae']:5.1f}  R2 {r['r2']:5.2f}  bias {r['bias']:+5.1f}  n {r['n']}")
                    else:
                        print(f"    {name}: [{r[0]:+.1f}, {r[1]:+.1f}]")
            if split == "space" and not sk:
                out = df[["sample_id", "site_id", "group", "date", "species", "value", "redding_km"]].copy()
                for k, p in preds.items():
                    out[f"pred {k}"] = p
                out.to_parquet(D / f"predictions_{suffix(a)}{'_partial' if a.partial else ''}.parquet", index=False)
    (ROOT / "results").mkdir(exist_ok=True)
    tag = suffix(a) + ("_partial" if a.partial else "")
    json.dump(allres, open(ROOT / "results" / f"knn_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
