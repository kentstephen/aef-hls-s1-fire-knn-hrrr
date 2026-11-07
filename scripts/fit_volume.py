# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "torch", "psutil"]
# ///
"""The fuel volume head: how much vegetation stands on a pixel, from that year's embedding alone.

Canopy cover (fit_change.py) only counts trees, so a grass or chaparral burn barely moves it. LANDFIRE's existing
vegetation cover and height (EVC, EVH) name each pixel's leading lifeform (tree, shrub or herb) with its cover %
and height; their product, cover x height in meters, is the vegetation volume: about 11 for a closed forest, 0.5
for chaparral, 0.2 for grass, 0 on roads and water. It is a proxy for how much there is to burn, not a fuel load
in tons.

One network, the year's embedding (64) -> 64 -> 5 values + 11 fuel kinds + 4 lifeforms:
  vol   log(0.02 + cover x height), scaled to about 0..1 (VOL_LO, VOL_HI)
  vc    the leading lifeform's cover / 100;  vh  its height / 40 m
  cc    LANDFIRE canopy cover / 100;  ch  canopy height / 50 m
  kind  the 11 kinds of models/fuel_head.json (FBFM40 families);  life  none, tree, shrub, herb (EVC)
Rows: every LANDFIRE edition sampled (2016, 2022, 2024, 2025), each with the embedding of its own year (LF2016 with
2017), over data/change's random and burn windows (scripts/sample_change.py). Test fires never in training.

Scores: 5 folds by 0.5 degree block (as fit_change.py); the change from LF2016 to LF2022 on the same pixels, out of
fold; the five held-out fires year by year (inside and outside, with LANDFIRE's volume per edition).

Writes results/fit_volume.json and data/change/head_volume.pt. RAM: about 5 GB at peak.

    uv run scripts/fit_volume.py [--epochs 12] [--no-cv]
"""
import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

from fit_change import KINDS, PAIRS, YEARS, blocks, deq, kind_of, load, r2, standardize

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "change"
EDS = [2016, 2022, 2024, 2025]
REG = ["vol", "vc", "vh", "cc", "ch"]
LIFE = ["none", "tree", "shrub", "herb"]
VOL_LO, VOL_HI = np.log(0.02), np.log(50.0)
torch.set_num_threads(8)


def vol_scale(v):
    return (np.log(0.02 + v) - VOL_LO) / (VOL_HI - VOL_LO)


def vol_unscale(s):
    return np.exp(s * (VOL_HI - VOL_LO) + VOL_LO) - 0.02


def lifeform(evc, evh):
    """EVC / EVH codes -> lifeform index, cover %, height m (NaN where LANDFIRE has no value)."""
    evc, evh = evc.astype(np.int32), evh.astype(np.int32)
    life = np.full(evc.shape, -1, np.int64)
    vc = np.full(evc.shape, np.nan, np.float32)
    vh = np.full(evc.shape, np.nan, np.float32)
    non = (evc >= 11) & (evc < 100)
    life[non], vc[non], vh[non] = 0, 0, 0
    for i, lo in ((1, 100), (2, 200), (3, 300)):
        m = (evc > lo) & (evc < lo + 100)
        life[m] = i
        vc[m] = evc[m] - lo
    for lo, scale in ((100, 1.0), (200, 0.1), (300, 0.1)):
        m = (evh > lo) & (evh < lo + 100) & (life > 0)
        vh[m] = (evh[m] - lo) * scale
    vh[(life > 0) & np.isnan(vh)] = np.nan
    return life, vc, vh


def targets(z, ed):
    life, vc, vh = lifeform(z[f"evc_{ed}"], z[f"evh_{ed}"])
    c, h = z[f"cc_{ed}"].astype(np.float32), z[f"ch_{ed}"].astype(np.float32)
    cc = np.where((c >= 0) & (c <= 100), c, np.nan)
    ch = np.where((h >= 0) & (h <= 600), h / 10, np.nan)
    vol = vc / 100 * vh
    y = np.c_[vol_scale(vol), vc / 100, vh / 40, cc / 100, ch / 50].astype(np.float32)
    return y, kind_of(z[f"fb_{ed}"]), life


def table(parts):
    X, Y, K, L, win, px, ed_of = [], [], [], [], [], [], []
    for p in parts:
        if p["kind"] not in ("random", "burn"):
            continue
        z = p["z"]
        for ed in EDS:
            X.append(deq(z["q"][:, YEARS.index(PAIRS[ed])]))
            y, k, life = targets(z, ed)
            Y.append(y)
            K.append(k)
            L.append(life)
            n = len(y)
            win.append(np.full(n, p["win"]))
            px.append(np.arange(n))
            ed_of.append(np.full(n, ed))
    return tuple(np.concatenate(a) for a in (X, Y, K, L, win, px, ed_of))


class Net(torch.nn.Module):
    def __init__(self, d=64, h=64):
        super().__init__()
        self.l1, self.l2 = torch.nn.Linear(d, h), torch.nn.Linear(h, len(REG) + len(KINDS) + len(LIFE))

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))


def fit(Xs, idx, Y, K, L, epochs, seed=0):
    rng = np.random.default_rng(seed)
    Yt = torch.from_numpy(np.nan_to_num(Y, nan=-9.0))
    Kt, Lt = torch.from_numpy(K), torch.from_numpy(L)
    torch.manual_seed(seed)
    net = Net(Xs.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    nr, nk = len(REG), len(KINDS)
    for _ in range(epochs):
        perm = torch.from_numpy(idx[rng.permutation(len(idx))])
        for i in range(0, len(perm), 4096):
            b = perm[i:i + 4096]
            o = net(Xs[b])
            y = Yt[b]
            m = y > -8
            loss = (((o[:, :nr] - y) ** 2) * m).sum() / m.sum().clamp(min=1) * 10
            mk, ml = Kt[b] >= 0, Lt[b] >= 0
            loss = loss + torch.nn.functional.cross_entropy(o[mk, nr:nr + nk], Kt[b][mk])
            loss = loss + torch.nn.functional.cross_entropy(o[ml, nr + nk:], Lt[b][ml])
            opt.zero_grad()
            loss.backward()
            opt.step()
        for g in opt.param_groups:
            g["lr"] *= 0.8
    net.eval()
    return net


def predict(net, Xs):
    out = []
    with torch.no_grad():
        for i in range(0, len(Xs), 65536):
            out.append(net(Xs[i:i + 65536]).numpy())
    o = np.concatenate(out)
    nr, nk = len(REG), len(KINDS)
    return o[:, :nr], o[:, nr:nr + nk].argmax(1), o[:, nr + nk:].argmax(1)


def cv(w, parts, epochs):
    X, Y, K, L, win, px, ed = table(parts)
    Xs, _, _ = standardize(X)
    fold = blocks(w, parts)[win]
    P = np.zeros_like(Y)
    PK, PL = np.zeros(len(X), np.int64), np.zeros(len(X), np.int64)
    for f in range(5):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        net = fit(Xs, tr, Y, K, L, epochs, seed=f)
        P[te], PK[te], PL[te] = predict(net, Xs[torch.from_numpy(te)])
        print(f"  fold {f} done, rss {psutil.Process(os.getpid()).memory_info().rss / 1e9:.1f} GB", flush=True)
    isburn = np.isin(win, [p["win"] for p in parts if p["kind"] == "burn"])
    res = {"rows": int(len(X))}
    for name, sel in (("all", np.ones(len(X), bool)), ("burn windows", isburn)):
        r = {}
        for j, t in enumerate(REG):
            y, p = Y[sel, j], P[sel, j]
            m = np.isfinite(y)
            r[f"{t}_r2"] = r2(y[m], p[m])
        v, pv = vol_unscale(Y[sel, 0]), vol_unscale(P[sel, 0])
        m = np.isfinite(v)
        r["vol_median_abs_log_err"] = float(np.median(np.abs(np.log((0.02 + v[m]) / (0.02 + pv[m].clip(0))))))
        r["cc_mae"] = float(np.nanmean(np.abs(Y[sel, 3] - P[sel, 3])) * 100)
        mk, ml = sel & (K >= 0), sel & (L >= 0)
        r["kind_acc"], r["life_acc"] = float((PK[mk] == K[mk]).mean()), float((PL[ml] == L[ml]).mean())
        res[name] = r
    # the change LF2016 -> LF2022 on the same pixels, in log volume (natural log units)
    a = pd.DataFrame(dict(y=Y[:, 0], p=P[:, 0]), index=pd.MultiIndex.from_arrays([win, px]))
    j = a[ed == 2016].join(a[ed == 2022], lsuffix="16", rsuffix="22", how="inner").dropna()
    s = VOL_HI - VOL_LO
    dt, dp = (j.y22 - j.y16) * s, (j.p22 - j.p16) * s
    big, steady = dt <= -1, dt.abs() < 0.2
    res["vol_change_2016_2022"] = dict(pixels=int(len(j)), corr=float(np.corrcoef(dt, dp)[0, 1]),
                                       lost_e_fold=int(big.sum()), lost_true_mean=float(dt[big].mean()),
                                       lost_pred_mean=float(dp[big].mean()), steady_pred_mean=float(dp[steady].mean()))
    return res


def fires(parts, epochs):
    X, Y, K, L, *_ = table(parts)
    Xs, mu, sd = standardize(X)
    net = fit(Xs, np.arange(len(X)), Y, K, L, epochs, seed=0)
    del X, Xs
    torch.save(dict(state=net.state_dict(), mu=mu, sd=sd, reg=REG, kinds=KINDS, life=LIFE,
                    vol_lo=float(VOL_LO), vol_hi=float(VOL_HI)), D / "head_volume.pt")
    out = {}
    for p in parts:
        if not p["kind"].startswith("test_"):
            continue
        side, z = p["kind"][5:], p["z"]
        for yi, y in enumerate(YEARS):
            r, _, life = predict(net, torch.from_numpy((deq(z["q"][:, yi]) - mu) / sd))
            o = out.setdefault(p["fire"], {}).setdefault(side, {}).setdefault(str(y), dict(n=0, lv=0.0, life=np.zeros(4)))
            o["n"] += len(r)
            o["lv"] += float(r[:, 0].sum())
            o["life"] += np.bincount(life, minlength=4)
        for ed in EDS:
            t, _, _ = targets(z, ed)
            v = t[:, 0][np.isfinite(t[:, 0])]
            o = out[p["fire"]][side].setdefault(f"lf{ed}", dict(n=0, lv=0.0))
            o["n"] += len(v)
            o["lv"] += float(v.sum())
    res = {}
    for f, sides in out.items():
        for side, ys in sides.items():
            for y, o in ys.items():
                r = dict(vol=round(float(vol_unscale(o["lv"] / o["n"])), 2))
                if "life" in o:
                    r["life"] = {LIFE[i]: round(float(c / o["n"] * 100)) for i, c in enumerate(o["life"])}
                res.setdefault(f, {}).setdefault(side, {})[y] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--max-gb", type=float, default=16.0)
    a = ap.parse_args()

    def watch():
        p = psutil.Process(os.getpid())
        while True:
            if p.memory_info().rss / 1e9 > a.max_gb:
                print(f"STOP: rss past --max-gb {a.max_gb}", flush=True)
                os._exit(2)
            time.sleep(1)

    threading.Thread(target=watch, daemon=True).start()
    t0 = time.time()
    w, parts = load()
    path = ROOT / "results" / "fit_volume.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    if not a.no_cv:
        res["cv"] = cv(w, parts, a.epochs)
        print(json.dumps(res["cv"]), flush=True)
    res["fires"] = fires(parts, a.epochs)
    path.write_text(json.dumps(res, indent=1))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
