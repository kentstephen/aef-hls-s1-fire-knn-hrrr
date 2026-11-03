# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "torch", "lightgbm", "psutil", "tabulate"]
# ///
"""Fuel / not-fuel lab: a pixel table from the full window grids, then a queue of scenarios scored the same way.

The point of the head is to never hide real fuel. So the main score is: with the cutoff set so that at most
0.5 / 1 / 2 / 5 % of fuel pixels are called not-fuel, what share of non-fuel is caught? The cutoff for each test
fold comes from the other folds' out-of-fold predictions (honest), and the fuel actually hidden is reported next
to it. Folds are by 0.5 degree block (about 50 km), so no test pixel has a training neighbor nearby.

Truth is LANDFIRE LF2024 FBFM40 (91 to 99 not burnable). Views of the same predictions:
  all      every window, each pixel weighted back to its window's true fuel / non-fuel mix
  random   only windows placed at random (closest to "a pixel of California")
  clean    only pixels whose 7 x 7 neighborhood (70 m) has one label status (no 30 m label edges)
  agree    only pixels where LANDFIRE and ESA WorldCover 2021 agree on fuel / not-fuel
  strict   separate truth: not-fuel only where both maps say so (LANDFIRE "urban" yards and street trees are fuel)
Also by non-fuel kind, fuel hidden by fuel family and by distance to the nearest non-fuel pixel, calibration
(after a one-number logit rescale fit on the other folds, foldable into a shader), a north / south travel test and
a 2024 -> 2025 year test (labels are 2024; the map will show the newest year).

Every model is something a fragment shader can run: a linear head or a small MLP on per-pixel features
(AEF, neighbor means, year spread). LightGBM runs only as a ceiling: if it barely beats them, the limit is
labels or inputs, not the model.

    uv run scripts/fuel_lab.py build            # pixel table from data/fuel_grid (once)
    uv run scripts/fuel_lab.py run [--loop]     # every scenario in fuel_scenarios.py not yet in results/fuel_lab
    uv run scripts/fuel_lab.py board            # rewrite results/fuel_lab/LEADERBOARD.md

RAM: build about 3 GB, a run about 6 to 10 GB at peak; a watchdog stops the process above --cap GB (default 20).
"""
import argparse
import importlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuel_common import FUEL_FAMILIES, NONFUEL_KINDS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GRID = ROOT / "data" / "fuel_grid"
LAB = ROOT / "data" / "fuel_lab"
RES = ROOT / "results" / "fuel_lab"
PER_CLASS = 1500
BORDER = 4  # keep the 9 x 9 mean inside the window
BUDGETS = (0.005, 0.01, 0.02, 0.05)
WC_NONFUEL = (40, 50, 60, 70, 80)  # crop, built-up, bare, snow, water


# ---------------------------------------------------------------- build

def deq(q):
    f = q.astype(np.float32) / 127.5
    return np.sign(f) * f * f


def box_mean(v, valid, size):
    """Mean of the size x size neighborhood over valid pixels; v (64, H, W), valid (H, W)."""
    from scipy.ndimage import uniform_filter
    num = uniform_filter(np.where(valid[None], v, 0), size=(1, size, size), mode="nearest")
    den = uniform_filter(valid.astype(np.float32), size=size, mode="nearest")
    return num / np.maximum(den, 1e-6)[None]


def build_window(args):
    path, wi = args
    from scipy.ndimage import distance_transform_edt, uniform_filter
    z = np.load(path)
    if "empty" in z:
        return None
    q, fb, evt, wc = z["aef"], z["fbfm"], z["evt"], z["wc"]
    valid = ~(q == -128).all(axis=1).any(axis=0)  # AEF present in every year
    ok = valid & (fb > 0)
    inner = np.zeros_like(ok)
    inner[BORDER:-BORDER, BORDER:-BORDER] = True
    nf = (fb >= 91) & (fb <= 99)
    rng = np.random.default_rng(wi)
    rows, cols, wts = [], [], []
    for cls in (True, False):
        idx = np.argwhere(ok & inner & (nf == cls))
        if len(idx) == 0:
            continue
        pick = idx[rng.choice(len(idx), size=min(PER_CLASS, len(idx)), replace=False)]
        rows.append(pick[:, 0])
        cols.append(pick[:, 1])
        wts.append(np.full(len(pick), len(idx) / len(pick), np.float32))
    if not rows:
        return None
    r, c, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
    share = uniform_filter(nf.astype(np.float32), size=7, mode="nearest")[r, c]
    d_nf = distance_transform_edt(~nf)[r, c] if nf.any() else np.full(len(r), 999.0)
    feats = {}
    years = list(z["years"])
    for yr in (2024, 2025):
        v = deq(q[years.index(yr)])
        vm = ~(q[years.index(yr)] == -128).all(axis=0)
        tag = str(yr)[2:]
        feats[f"e{tag}"] = v[:, r, c].T
        feats[f"m3_{tag}"] = box_mean(v, vm, 3)[:, r, c].T
        feats[f"m9_{tag}"] = box_mean(v, vm, 9)[:, r, c].T
    allv = np.stack([deq(q[i][:, r, c]) for i in range(len(years))])  # (years, 64, n)
    feats["sd"] = allv.std(axis=0).T
    meta = pd.DataFrame(dict(win=wi, row=r, col=c, w=w, fbfm=fb[r, c], evt=evt[r, c], wc=wc[r, c],
                             nf=nf[r, c].astype(np.int8), clean=(share < 1e-6) | (share > 1 - 1e-6), d_nf=d_nf.astype(np.float32)))
    return meta, {k: v.astype(np.float32) for k, v in feats.items()}


def build():
    LAB.mkdir(parents=True, exist_ok=True)
    wins = pd.read_parquet(ROOT / "data" / "fuel_ca_windows.parquet").reset_index(drop=True)
    jobs = [(GRID / f"{w.cy}_{w.cx}.npz", i) for i, w in wins.iterrows()]
    missing = [p for p, _ in jobs if not p.exists()]
    if missing:
        sys.exit(f"{len(missing)} window grids missing; run sample_fuel_grid.py first")
    metas, feats = [], {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for n, out in enumerate(ex.map(build_window, jobs, chunksize=4)):
            if out is not None:
                metas.append(out[0])
                for k, v in out[1].items():
                    feats.setdefault(k, []).append(v)
            if (n + 1) % 50 == 0:
                print(f"  {n + 1}/{len(jobs)} windows, {time.time() - t0:.0f} s", flush=True)
    meta = pd.concat(metas, ignore_index=True)
    meta = meta.merge(wins[["stratum", "name", "block", "region", "lon", "lat"]].rename_axis("win").reset_index(), on="win")
    meta.to_parquet(LAB / "pixels.parquet", index=False)
    names = []
    for k in feats:
        a = np.concatenate(feats[k])
        np.save(LAB / f"{k}.npy", a)
        names.append(k)
        feats[k] = None
    json.dump(names, open(LAB / "features.json", "w"))
    print(f"wrote {LAB}: {len(meta)} pixels in {meta.win.nunique()} windows, non-fuel {meta.nf.mean():.1%} sampled, "
          f"{np.average(meta.nf, weights=meta.w):.1%} weighted; clean {meta.clean.mean():.0%}")


# ---------------------------------------------------------------- data for a scenario

def load_meta():
    m = pd.read_parquet(LAB / "pixels.parquet")
    m["random_view"] = (m.stratum == "random") | ((m.stratum == "redding") & (m.name == "random"))
    wc_nf = np.isin(m.wc, WC_NONFUEL)
    m["agree"] = (m.wc > 0) & (wc_nf == m.nf.astype(bool))
    kinds = {c: i for i, c in enumerate(NONFUEL_KINDS)}
    fam = np.full(len(m), -1)
    for i, (name, codes) in enumerate(FUEL_FAMILIES.items()):
        fam[np.isin(m.fbfm, list(codes))] = i
    m["family"] = fam
    # multi-kind label: each non-fuel kind, each fuel family, other burnable
    k = np.where(m.nf == 1, m.fbfm.map(kinds).fillna(0).astype(int), len(kinds) + np.where(fam >= 0, fam, len(FUEL_FAMILIES)))
    m["kind"] = k
    ov = LAB / "overture.parquet"  # from Overture buildings and roads drawn on the grid (sample_fuel_overture.py)
    if ov.exists():
        o = pd.read_parquet(ov)
        m["bld"], m["road"] = o.bld.to_numpy(), o.road.to_numpy()
    return m


def features(spec, year):
    tag = str(year)[2:]
    cols = []
    for f in spec:
        name = f if f == "sd" else f"{f}{tag}" if f == "e" else f"{f}_{tag}"
        cols.append(np.load(LAB / f"{name}.npy", mmap_mode="r"))
    return np.hstack([np.asarray(c) for c in cols]).astype(np.float32)


# ---------------------------------------------------------------- models

def train_torch(X, y, sw, Xval, yval, swval, hidden, n_out, wd, seed, max_epochs=40, lr=None, dropout=0.0, stop_on="loss",
                nf_val=None, nf_cols=None, patience=3):
    import torch
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(seed)
    layers, d = [], X.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.ReLU()] + ([torch.nn.Dropout(dropout)] if dropout else [])
        d = h
    layers.append(torch.nn.Linear(d, n_out))
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.AdamW(net.parameters(), lr=lr or (3e-3 if hidden else 1e-2), weight_decay=wd)
    Xt, yt, wt = torch.from_numpy(X), torch.from_numpy(y.astype(np.int64)), torch.from_numpy(sw.astype(np.float32))
    Xv, yv, wv = torch.from_numpy(Xval), torch.from_numpy(yval.astype(np.int64)), torch.from_numpy(swval.astype(np.float32))
    lossf = torch.nn.CrossEntropyLoss(reduction="none")
    best, best_state, bad = np.inf, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(perm), 4096):
            b = perm[i:i + 4096]
            loss = (lossf(net(Xt[b]), yt[b]) * wt[b]).sum() / wt[b].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            out = net(Xv)
            vl = float((lossf(out, yv) * wv).sum() / wv.sum())
            if stop_on == "auc":  # rank quality of P(not fuel) on held-out blocks, higher is better
                pv = torch.softmax(out, dim=1).numpy()[:, nf_cols].sum(axis=1)
                vl = -roc_auc_score(nf_val, pv, sample_weight=swval)
        if vl < best - 1e-5:
            best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, ep + 1


def predict_torch(net, X, nf_cols):
    import torch
    with torch.no_grad():
        p = torch.softmax(net(torch.from_numpy(X)), dim=1).numpy()
    return p[:, nf_cols].sum(axis=1)


def balanced_weights(nf, w, mode):
    if mode == "true":
        return w.astype(np.float64)
    out = np.zeros(len(nf))
    for c in (0, 1):
        m = nf == c
        out[m] = w[m] / w[m].sum()
    return out * len(nf) / 2


def fit_predict(sc, meta, X, tr, te_sets):
    """Fit on tr; return P(not fuel) for each index array in te_sets (dict of name -> (X, idx))."""
    from sklearn.model_selection import GroupShuffleSplit
    nf = meta.nf.to_numpy()
    if sc.get("truth") == "strict":  # not-fuel only where WorldCover also sees no vegetation; the rest trains as fuel
        nf = nf & np.isin(meta.wc.to_numpy(), WC_NONFUEL).astype(nf.dtype)
    multi = sc.get("labels", "binary") == "kinds"
    yall = np.where((meta.nf.to_numpy() == 1) & (nf == 0), meta.kind.max() if multi else 0,
                    meta.kind.to_numpy() if multi else nf) if multi else nf
    n_out = int(meta.kind.max()) + 1 if multi else 2
    nf_cols = list(range(len(NONFUEL_KINDS))) if multi else [1]
    trf = tr
    if sc.get("train_on") == "clean":
        trf = tr[meta.clean.to_numpy()[tr]]
    elif sc.get("train_on") == "agree":
        trf = tr[meta.agree.to_numpy()[tr]]
    sw = balanced_weights(nf[trf], meta.w.to_numpy()[trf], sc.get("weights", "balanced"))
    if sc["model"] == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=sc.get("trees", 400), learning_rate=0.05, num_leaves=63, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=8)
        m.fit(X[trf], nf[trf], sample_weight=sw)
        return {k: m.predict_proba(Xk[idx])[:, 1] for k, (Xk, idx) in te_sets.items()}, {"trees": sc.get("trees", 400)}
    mu, sd = X[trf].mean(0), X[trf].std(0) + 1e-6
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
    itr, ival = next(gss.split(trf, groups=meta.block.to_numpy()[trf]))
    Xs = (X[trf] - mu) / sd
    nets, eps = [], []
    for seed in range(sc.get("ensemble", 1)):
        net, epochs = train_torch(Xs[itr], yall[trf][itr], sw[itr], Xs[ival], yall[trf][ival], sw[ival], sc.get("hidden", []),
                                  n_out, sc.get("wd", 1e-4), sc.get("seed", 0) + seed, lr=sc.get("lr"), dropout=sc.get("dropout", 0.0),
                                  stop_on=sc.get("stop_on", "loss"), nf_val=nf[trf][ival], nf_cols=nf_cols,
                                  patience=sc.get("patience", 3), max_epochs=sc.get("max_epochs", 40))
        nets.append(net)
        eps.append(epochs)
    del Xs
    return {k: np.mean([predict_torch(n, ((Xk[idx] - mu) / sd).astype(np.float32), nf_cols) for n in nets], axis=0)
            for k, (Xk, idx) in te_sets.items()}, {"epochs": eps}


def n_params(sc, d, n_out):
    if sc["model"] == "lgbm":
        return None
    p, prev = 0, d
    for h in sc.get("hidden", []) + [n_out]:
        p += prev * h + h
        prev = h
    return p


# ---------------------------------------------------------------- scoring

def wquantile(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return x[o][np.searchsorted(cw, q * cw[-1])]


def wauc(y, s, w):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s, sample_weight=w))


def score_view(q, nf, w, fold, thr, mask):
    """Honest: each fold's pixels judged with that fold's cutoff (from the other folds)."""
    q, nf, w, fold = q[mask], nf[mask], w[mask], fold[mask]
    out = {"n": int(mask.sum()), "auc": round(wauc(nf, q, w), 4)}
    for b in BUDGETS:
        t = np.array([thr[b][f] for f in fold])
        call = q > t
        fu, nn = nf == 0, nf == 1
        out[f"hidden@{b:g}"] = round(float(np.average(call[fu], weights=w[fu])), 4)
        out[f"caught@{b:g}"] = round(float(np.average(call[nn], weights=w[nn])), 4) if nn.any() else None
    return out


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration(q, nf, w, fold, mask):
    """Rescale the logit (a, b) on the other folds with true-mix weights, then Brier and ECE on this view."""
    from sklearn.linear_model import LogisticRegression
    pc = np.zeros_like(q)
    L = logit(q)[:, None]
    for f in np.unique(fold):
        o = fold != f
        lr = LogisticRegression(C=1e6).fit(L[o], nf[o], sample_weight=w[o])
        pc[fold == f] = lr.predict_proba(L[fold == f])[:, 1]
    pc, y, ww = pc[mask], nf[mask], w[mask]
    bins = np.minimum((pc * 10).astype(int), 9)
    ece = sum(abs(np.average(pc[bins == i], weights=ww[bins == i]) - np.average(y[bins == i], weights=ww[bins == i])) *
              ww[bins == i].sum() for i in range(10) if (bins == i).any()) / ww.sum()
    return {"brier": round(float(np.average((pc - y) ** 2, weights=ww)), 4), "ece": round(float(ece), 4),
            "base_rate": round(float(np.average(y, weights=ww)), 4)}


def score_all(meta, q, fold):
    nf, w = meta.nf.to_numpy(), meta.w.to_numpy()
    thr = {b: {} for b in BUDGETS}
    for f in np.unique(fold):
        o = (fold != f) & (nf == 0)
        for b in BUDGETS:
            thr[b][f] = wquantile(q[o], w[o], 1 - b)
    views = {"all": np.ones(len(q), bool), "random": meta.random_view.to_numpy(), "clean": meta.clean.to_numpy(),
             "agree": meta.agree.to_numpy()}
    res = {"views": {k: score_view(q, nf, w, fold, thr, m) for k, m in views.items()}}
    # strict truth: not-fuel only where LANDFIRE and WorldCover both say so; yard trees and grass count as fuel
    nfs = (nf.astype(bool) & np.isin(meta.wc.to_numpy(), WC_NONFUEL)).astype(int)
    thr_s = {b: {f: wquantile(q[(fold != f) & (nfs == 0)], w[(fold != f) & (nfs == 0)], 1 - b) for f in np.unique(fold)}
             for b in BUDGETS}
    res["strict truth"] = {k: score_view(q, nfs, w, fold, thr_s, m) for k, m in [("all", views["all"]), ("random", views["random"])]}
    t1 = np.array([thr[0.01][f] for f in fold])
    call = q > t1
    res["caught@0.01 by kind"] = {name: round(float(np.average(call[m], weights=w[m])), 4)
                                  for code, name in NONFUEL_KINDS.items() if (m := (meta.fbfm == code).to_numpy()).sum() >= 50}
    res["caught@0.01 roads (EVT 7299)"] = round(float(np.average(call[(m := (meta.evt == 7299).to_numpy())], weights=w[m])), 4)
    fam = meta.family.to_numpy()
    res["hidden@0.01 by fuel family"] = {name: round(float(np.average(call[(fam == i) & (nf == 0)], weights=w[(fam == i) & (nf == 0)])), 4)
                                         for i, name in enumerate(FUEL_FAMILIES)}
    if "bld" in meta:
        for name, sel in [("building", meta.bld.to_numpy() == 1), ("road (motorway to residential)", meta.road.between(1, 3).to_numpy()),
                          ("service road", meta.road.to_numpy() == 4), ("track", meta.road.to_numpy() == 5)]:
            res.setdefault("caught@0.01 Overture mapped", {})[name] = round(float(np.average(call[sel], weights=w[sel])), 4)
    d = meta.d_nf.to_numpy()
    res["hidden@0.01 by distance to non-fuel"] = {
        lab: round(float(np.average(call[m], weights=w[m])), 4)
        for lab, m in [("<=30 m", (nf == 0) & (d <= 3)), ("30-100 m", (nf == 0) & (d > 3) & (d <= 10)), (">100 m", (nf == 0) & (d > 10))]
        if m.sum() > 50}
    res["calibration (random view, after logit rescale)"] = calibration(q, nf, w, fold, views["random"])
    return res, thr


def oracle(q, nf, w, mask):
    q, nf, w = q[mask], nf[mask], w[mask]
    out = {"auc": round(wauc(nf, q, w), 4)}
    for b in (0.01, 0.02):
        t = wquantile(q[nf == 0], w[nf == 0], 1 - b)
        out[f"caught@{b:g} (cutoff set on this test set)"] = round(float(np.average(q[nf == 1] > t, weights=w[nf == 1])), 4)
    return out


# ---------------------------------------------------------------- runner

KNOWN_KEYS = {"name", "features", "model", "hidden", "labels", "weights", "train_on", "train_years", "year_test", "travel", "truth",
              "wd", "seed", "lr", "dropout", "stop_on", "ensemble", "patience", "max_epochs", "trees", "overture"}


def run_one(sc, meta):
    from sklearn.model_selection import GroupKFold
    unknown = set(sc) - KNOWN_KEYS
    if unknown:
        raise ValueError(f"unknown scenario keys {unknown}: this fuel_lab.py does not know them")
    t0 = time.time()
    X = features(sc["features"], 2024)
    multi = sc.get("labels", "binary") == "kinds"
    blocks = meta.block.to_numpy()
    fold = np.zeros(len(meta), int)
    q = np.zeros(len(meta))
    q25 = np.zeros(len(meta)) if sc.get("year_test") else None
    X25 = features(sc["features"], 2025) if sc.get("year_test") or sc.get("train_years") == "both" else None
    info = {}
    for f, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, groups=blocks)):
        fold[te] = f
        Xtr, mtr = X, meta
        if sc.get("train_years") == "both":  # each training pixel twice: 2024 and 2025 embeddings
            Xtr = np.vstack([X[tr], X25[tr]])
            mtr = pd.concat([meta.iloc[tr], meta.iloc[tr]], ignore_index=True)
            tr_idx = np.arange(len(mtr))
        else:
            tr_idx = tr
        sets = {"q": (X, te)}
        if q25 is not None:
            sets["q25"] = (X25, te)
        preds, info = fit_predict(sc, mtr, Xtr, tr_idx, sets)
        q[te] = preds["q"]
        if q25 is not None:
            q25[te] = preds["q25"]
        print(f"    fold {f + 1}/5 {time.time() - t0:.0f} s {info}", flush=True)
        del Xtr
    if sc.get("overture"):  # mapped buildings and roads drawn low directly, whatever the head says
        ov = (meta.bld.to_numpy() == 1) | meta.road.between(1, sc["overture"]).to_numpy()
        q = np.where(ov, 1.0, q)
        if q25 is not None:
            q25 = np.where(ov, 1.0, q25)
    res, thr = score_all(meta, q, fold)
    nf, w = meta.nf.to_numpy(), meta.w.to_numpy()
    if q25 is not None:
        res["year test: 2024-trained, scored on 2025 embeddings"] = score_view(q25, nf, w, fold, thr, np.ones(len(q), bool))
    if sc.get("travel"):
        reg = meta.region.to_numpy()
        res["travel"] = {}
        for a, b in [("north", "south"), ("south", "north")]:
            tr, te = np.nonzero(reg == a)[0], np.nonzero(reg == b)[0]
            preds, _ = fit_predict(sc, meta, X, tr, {"q": (X, te)})
            qq = np.zeros(len(meta))
            qq[te] = preds["q"]
            res["travel"][f"train {a}, test {b}"] = oracle(qq, nf, w, reg == b)
            print(f"    travel {a}->{b} {time.time() - t0:.0f} s", flush=True)
    n_out = int(meta.kind.max()) + 1 if multi else 2
    res["scenario"] = sc
    res["params"] = n_params(sc, X.shape[1], n_out)
    res["aef_reads_per_pixel"] = sum({"e": 1, "m3": 9, "m9": 81, "sd": 3}[f] for f in sc["features"])
    res["seconds"] = round(time.time() - t0)
    np.save(RES / f"{sc['name']}.oof.npy", q.astype(np.float32))
    return res


def board():
    rows = []
    for p in sorted(RES.glob("*.json")):
        r = json.load(open(p))
        if "error" in r:
            rows.append({"scenario": r["scenario"]["name"], "model": "FAILED: " + r["error"][:80]})
            continue
        a, rv = r["views"]["all"], r["views"]["random"]
        k = r.get("caught@0.01 by kind", {})
        rows.append({"scenario": r["scenario"]["name"], "model": r["scenario"]["model"] + (str(r["scenario"].get("hidden", "")) or ""),
                     "params": r["params"], "reads": r["aef_reads_per_pixel"], "auc": a["auc"],
                     "caught@0.5%": a["caught@0.005"], "caught@1%": a["caught@0.01"], "hidden@1%": a["hidden@0.01"],
                     "caught@2%": a["caught@0.02"], "caught@5%": a["caught@0.05"],
                     "random caught@1%": rv["caught@0.01"], "clean caught@1%": r["views"]["clean"]["caught@0.01"],
                     "agree caught@1%": r["views"]["agree"]["caught@0.01"],
                     "strict caught@1%": r["strict truth"]["all"]["caught@0.01"],
                     "strict hidden@1%": r["strict truth"]["all"]["hidden@0.01"],
                     **{f"{n} @1%": k.get(n) for n in ("urban and roads", "farmland", "water", "bare")},
                     "2025 caught@1%": r.get("year test: 2024-trained, scored on 2025 embeddings", {}).get("caught@0.01"),
                     "travel auc N->S / S->N": " / ".join(str(v["auc"]) for v in r.get("travel", {}).values()) or None,
                     "Overture roads @1%": r.get("caught@0.01 Overture mapped", {}).get("road (motorway to residential)"),
                     "brier": r["calibration (random view, after logit rescale)"]["brier"]})
    if not rows:
        return
    d = pd.DataFrame(rows).sort_values("caught@1%", ascending=False, na_position="last")
    txt = ["# Fuel / not-fuel lab leaderboard", "",
           "Share of non-fuel caught (turned to the low end) while at most 0.5 / 1 / 2 / 5 % of fuel is called not-fuel.",
           "Cutoffs set on the other folds; `hidden@1%` is the fuel actually hidden at the 1 % cutoff. Folds by 50 km block.",
           "`strict` = truth where not-fuel needs LANDFIRE and WorldCover to agree (yard trees and grass count as fuel).",
           "`reads` = AEF pixels the shader reads per screen pixel. LightGBM rows are a ceiling only (not shader-ready).", "",
           d.to_markdown(index=False, floatfmt=".3f")]
    (RES / "LEADERBOARD.md").write_text("\n".join(txt) + "\n")


def watchdog(cap_gb):
    import psutil
    p = psutil.Process()
    peak = 0

    def loop():
        nonlocal peak
        while True:
            rss = p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
            peak = max(peak, rss)
            if rss > cap_gb * 2**30:
                print(f"WATCHDOG: RSS {rss / 2**30:.1f} GB over cap {cap_gb} GB, stopping", flush=True)
                os._exit(3)
            time.sleep(2)
    threading.Thread(target=loop, daemon=True).start()
    return lambda: peak / 2**30


def run(loop):
    import torch
    torch.set_num_threads(8)
    RES.mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    print(f"{len(meta)} pixels, {meta.win.nunique()} windows, {meta.block.nunique()} blocks", flush=True)
    while True:
        import fuel_scenarios
        importlib.reload(fuel_scenarios)
        todo = [s for s in fuel_scenarios.SCENARIOS if not (RES / f"{s['name']}.json").exists()]
        if not todo:
            if loop:
                time.sleep(60)
                continue
            break
        sc = todo[0]
        print(f"== {sc['name']} ({len(todo)} left): {sc}", flush=True)
        try:
            res = run_one(sc, meta)
        except Exception as e:  # one broken scenario must not stop the queue
            import traceback
            traceback.print_exc()
            (RES / f"{sc['name']}.json").write_text(json.dumps({"scenario": sc, "error": repr(e)}, indent=1))
            continue
        res["peak_gb"] = round(PEAK(), 2)
        json.dump(res, open(RES / f"{sc['name']}.json", "w"), indent=1)
        a = res["views"]["all"]
        print(f"   auc {a['auc']}  caught@1% {a['caught@0.01']} (hidden {a['hidden@0.01']})  caught@2% {a['caught@0.02']}  "
              f"{res['seconds']} s, peak {res['peak_gb']} GB", flush=True)
        try:
            board()
        except Exception as e:
            print(f"board failed: {e!r}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "board"])
    ap.add_argument("--loop", action="store_true", help="keep watching fuel_scenarios.py for new scenarios")
    ap.add_argument("--cap", type=float, default=20)
    a = ap.parse_args()
    PEAK = watchdog(a.cap)
    {"build": build, "run": lambda: run(a.loop), "board": board}[a.cmd]()
    print(f"peak RSS {PEAK():.2f} GB")
