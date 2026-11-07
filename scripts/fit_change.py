# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "torch", "psutil"]
# ///
"""Fuel heads that can see change, fit on data/change (scripts/sample_change.py) and tested on five held-out fires.

One small network per input set (in -> 64 -> 2 + 11), three outputs at once: LANDFIRE canopy cover (CC, %),
canopy height (CH, m), and the fuel kind (the 11 kinds of models/fuel_head.json). Each training row is a pixel in
one LANDFIRE edition, with the embedding of that edition's year (LF2016 with 2017), so the labels change where the
ground changed.

Input sets (each a browser-ready per-pixel feature):
  e       the year's embedding (64, dequantized)
  e_prev  plus the year before's embedding (128)
  e_d     plus the change from the year before, e minus e_prev (128)
  e_sd    plus the spread over the year and the two before (128; the current map head's input)
Rows are the editions every set can use (2022, 2024, 2025); "e_all" is e with LF2016 / 2017 too.

Scores:
  cv      5 folds by 0.5 degree block over random and burn windows: CC and CH MAE and R^2, kind accuracy,
          and for pixels whose LANDFIRE CC changed 2016 -> 2022 by 20 points or more, how well the predicted change
          (2022 minus 2017, both out of fold) follows it (correlation and mean).
  fires   fit on every train window, then the five test fires year by year (2018 or 2019 to 2025), inside vs
          outside: mean predicted CC and CH, kind shares, next to LANDFIRE's own CC for each edition.

Writes results/fit_change.json and the final fit per set to data/change/head_<set>.pt. RAM: about 5 GB at peak (the table standardized in place, folds trained on row indices).

    uv run scripts/fit_change.py [--sets e,e_prev,e_d,e_sd,e_all] [--epochs 12]
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "change"
YEARS = list(range(2017, 2026))
PAIRS = {2016: 2017, 2022: 2022, 2024: 2024, 2025: 2025}
KINDS = json.loads((ROOT / "models" / "fuel_head.json").read_text())["kinds"]
NONFUEL = {91: 0, 92: 1, 93: 2, 98: 3, 99: 4}
FAMILY = [(101, 109, 5), (121, 124, 6), (141, 149, 7), (161, 165, 8), (181, 189, 9), (201, 204, 10)]
torch.set_num_threads(8)


def kind_of(fb):
    k = np.full(fb.shape, -1, np.int64)
    for c, i in NONFUEL.items():
        k[fb == c] = i
    for lo, hi, i in FAMILY:
        k[(fb >= lo) & (fb <= hi)] = i
    return k


def deq(q):
    f = q.astype(np.float32) / 127.5
    return np.sign(f) * f * f


def load():
    w = pd.read_parquet(D / "windows.parquet")
    parts = []
    for i, r in enumerate(w.itertuples()):
        p = D / f"{r.kind}_{r.cy}_{r.cx}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        if len(z["q"]) == 0:
            continue
        parts.append(dict(win=i, kind=r.kind, fire=r.fire, cy=r.cy, cx=r.cx, z={k: z[k] for k in z.files}))
    print(f"{len(parts)} windows with pixels", flush=True)
    return w, parts


def feats(q, yi, s):
    """q int8 [n, 9, 64]; yi index of the year in YEARS -> features [n, d] float32 for input set s."""
    e = deq(q[:, yi])
    if s in ("e", "e_all"):
        return e
    if s == "e_prev":
        return np.c_[e, deq(q[:, yi - 1])]
    if s == "e_d":
        return np.c_[e, e - deq(q[:, yi - 1])]
    if s == "e_sd":
        return np.c_[e, deq(q[:, yi - 2:yi + 1]).std(axis=1)]
    raise ValueError(s)


def first_year(s):
    return {"e": 0, "e_all": 0, "e_prev": 1, "e_d": 1, "e_sd": 2}[s]


def table(parts, s, eds):
    X, cc, ch, kd, win, px, ed_of = [], [], [], [], [], [], []
    for p in parts:
        if not p["kind"] in ("random", "burn"):
            continue
        z = p["z"]
        for ed in eds:
            yi = YEARS.index(PAIRS[ed])
            if yi < first_year(s):
                continue
            X.append(feats(z["q"], yi, s))
            c, h = z[f"cc_{ed}"].astype(np.float32), z[f"ch_{ed}"].astype(np.float32)
            cc.append(np.where((c >= 0) & (c <= 100), c, np.nan))
            ch.append(np.where((h >= 0) & (h <= 600), h / 10, np.nan))
            kd.append(kind_of(z[f"fb_{ed}"]))
            n = len(c)
            win.append(np.full(n, p["win"]))
            px.append(np.arange(n))
            ed_of.append(np.full(n, ed))
    return (np.concatenate(X), np.concatenate(cc), np.concatenate(ch), np.concatenate(kd), np.concatenate(win),
            np.concatenate(px), np.concatenate(ed_of))


class Net(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.l1, self.l2 = torch.nn.Linear(d, 64), torch.nn.Linear(64, 2 + len(KINDS))

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))


def standardize(X):
    """In place: one float32 copy of the table for every fold (features are scaled on all rows; no labels used)."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X -= mu
    X /= sd
    return torch.from_numpy(X), mu, sd


def fit(Xs, idx, cc, ch, kd, epochs, seed=0):
    """Xs standardized torch table; idx the training rows. Returns the network."""
    rng = np.random.default_rng(seed)
    ycc = torch.from_numpy(np.nan_to_num(cc / 100, nan=-1).astype(np.float32))
    ych = torch.from_numpy(np.nan_to_num(ch / 50, nan=-1).astype(np.float32))
    yk = torch.from_numpy(kd)
    torch.manual_seed(seed)
    net = Net(Xs.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for ep in range(epochs):
        perm = torch.from_numpy(idx[rng.permutation(len(idx))])
        for i in range(0, len(perm), 4096):
            b = perm[i:i + 4096]
            o = net(Xs[b])
            mc, mh, mk = ycc[b] >= 0, ych[b] >= 0, yk[b] >= 0
            loss = (((o[:, 0] - ycc[b]) ** 2)[mc].mean() * 10 + ((o[:, 1] - ych[b]) ** 2)[mh].mean() * 10
                    + torch.nn.functional.cross_entropy(o[mk, 2:], yk[b][mk]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        for g in opt.param_groups:
            g["lr"] *= 0.8
    net.eval()
    return net


def predict(net, Xs):
    """Xs standardized, a torch tensor."""
    out = []
    with torch.no_grad():
        for i in range(0, len(Xs), 65536):
            out.append(net(Xs[i:i + 65536]).numpy())
    o = np.concatenate(out)
    k = np.exp(o[:, 2:] - o[:, 2:].max(1, keepdims=True))
    return o[:, 0] * 100, o[:, 1] * 50, k / k.sum(1, keepdims=True)


def r2(y, p):
    m = np.isfinite(y)
    return float(1 - ((y[m] - p[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum())


def blocks(w, parts):
    lon = -180 + (w.cx.to_numpy() + 0.5) * 256 * 8.983111749910169e-5
    lat = 83.68570533713473 - (w.cy.to_numpy() + 0.5) * 256 * 8.983111749910169e-5
    b = (np.floor(lon / 0.5) * 1000 + np.floor(lat / 0.5)).astype(int)
    ub = np.unique(b[[p["win"] for p in parts if p["kind"] in ("random", "burn")]])
    fold_of_block = dict(zip(np.random.default_rng(0).permutation(ub), np.arange(len(ub)) % 5))
    return np.array([fold_of_block.get(x, -1) for x in b])


def cv(w, parts, s, eds, epochs):
    X, cc, ch, kd, win, px, ed = table(parts, s, eds)
    Xs, _, _ = standardize(X)
    fold = blocks(w, parts)[win]
    pcc, pch, pk = np.zeros(len(X), np.float32), np.zeros(len(X), np.float32), np.zeros(len(X), np.int64)
    for f in range(5):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        net = fit(Xs, tr, cc, ch, kd, epochs, seed=f)
        a, b, k = predict(net, Xs[torch.from_numpy(te)])
        pcc[te], pch[te], pk[te] = a, b, k.argmax(1)
        del net
        print(f"  {s} fold {f} done, rss {psutil.Process(os.getpid()).memory_info().rss / 1e9:.1f} GB", flush=True)
    isburn = np.isin(win, [p["win"] for p in parts if p["kind"] == "burn"])
    res = {"rows": int(len(X))}
    for name, sel in [("all", np.ones(len(X), bool)), ("burn windows", isburn)]:
        mk = sel & (kd >= 0)
        res[name] = dict(cc_mae=float(np.nanmean(np.abs(cc[sel] - pcc[sel]))), cc_r2=r2(cc[sel], pcc[sel]),
                         ch_mae=float(np.nanmean(np.abs(ch[sel] - pch[sel]))), ch_r2=r2(ch[sel], pch[sel]),
                         kind_acc=float((pk[mk] == kd[mk]).mean()))
    if 2016 in eds:  # change 2016 -> 2022 on the same pixels
        key = pd.MultiIndex.from_arrays([win, px])
        a = pd.DataFrame(dict(cc=cc, p=pcc), index=key)
        e16, e22 = a[ed == 2016], a[ed == 2022]
        j = e16.join(e22, lsuffix="16", rsuffix="22", how="inner").dropna()
        dt, dp = j.cc22 - j.cc16, j.p22 - j.p16
        big = dt <= -20
        res["cc_change_2016_2022"] = dict(
            pixels=int(len(j)), corr=float(np.corrcoef(dt, dp)[0, 1]), lost_20_plus=int(big.sum()),
            lost_true_mean=float(dt[big].mean()), lost_pred_mean=float(dp[big].mean()),
            steady_true_mean=float(dt[dt.abs() < 5].mean()), steady_pred_mean=float(dp[dt.abs() < 5].mean()))
    return res


def fires(parts, s, eds, epochs):
    X, cc, ch, kd, *_ = table(parts, s, eds)
    Xs, mu, sd = standardize(X)
    net = fit(Xs, np.arange(len(X)), cc, ch, kd, epochs, seed=0)
    del X, Xs
    torch.save(dict(state=net.state_dict(), mu=mu, sd=sd, set=s, kinds=KINDS), D / f"head_{s}.pt")
    out = {}
    for p in parts:
        if not p["kind"].startswith("test_"):
            continue
        side = p["kind"][5:]
        z = p["z"]
        for yi in range(first_year(s), len(YEARS)):
            a, b, k = predict(net, torch.from_numpy((feats(z["q"], yi, s) - mu) / sd))
            o = out.setdefault(p["fire"], {}).setdefault(side, {}).setdefault(YEARS[yi], dict(n=0, cc=0.0, ch=0.0, lead=np.zeros(len(KINDS))))
            n = len(a)
            o["n"] += n
            o["cc"] += float(a.clip(0, 100).sum())
            o["ch"] += float(b.clip(0, None).sum())
            o["lead"] += np.bincount(k.argmax(1), minlength=len(KINDS))
        for ed in PAIRS:
            c = z[f"cc_{ed}"].astype(float)
            c = c[(c >= 0) & (c <= 100)]
            o = out[p["fire"]][side].setdefault(f"lf{ed}", dict(n=0, cc=0.0))
            o["n"] += len(c)
            o["cc"] += float(c.sum())
    res = {}
    for f, sides in out.items():
        for side, ys in sides.items():
            for y, o in ys.items():
                r = dict(cc=round(o["cc"] / o["n"], 1))
                if "ch" in o:
                    r["ch"] = round(o["ch"] / o["n"], 2)
                    r["lead"] = {KINDS[i]: round(float(v / o["n"] * 100)) for i, v in enumerate(o["lead"]) if v / o["n"] >= 0.02}
                res.setdefault(f, {}).setdefault(side, {})[str(y)] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="e,e_prev,e_d,e_sd,e_all")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--max-gb", type=float, default=16.0)
    ap.add_argument("--redo", action="store_true", help="refit sets already in results/fit_change.json")
    a = ap.parse_args()
    stop = {"x": False}

    def watch():
        p = psutil.Process(os.getpid())
        while not stop["x"]:
            if p.memory_info().rss / 1e9 > a.max_gb:
                print(f"STOP: rss past --max-gb {a.max_gb}", flush=True)
                os._exit(2)
            time.sleep(1)

    import threading
    threading.Thread(target=watch, daemon=True).start()
    w, parts = load()
    path = ROOT / "results" / "fit_change.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for s in a.sets.split(","):
        if s in res and not a.redo:
            print(f"{s}: done already", flush=True)
            continue
        eds = [2016, 2022, 2024, 2025] if s == "e_all" else [2022, 2024, 2025]
        t0 = time.time()
        r = cv(w, parts, s, eds, a.epochs)
        gc.collect()
        r["fires"] = fires(parts, s, eds, a.epochs)
        gc.collect()
        res[s] = r
        path.write_text(json.dumps(res, indent=1))
        print(f"{s}: {json.dumps({k: v for k, v in r.items() if k != 'fires'})} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    stop["x"] = True


if __name__ == "__main__":
    main()
