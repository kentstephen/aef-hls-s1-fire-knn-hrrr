# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn"]
# ///
"""Pick pixels for the NAIP check of the fuel head's errors.

The head is scored against LANDFIRE, so its "errors" mix two things: the head being wrong and LANDFIRE being wrong.
This picks pixels from each kind of disagreement (plus agreed controls) so a look at NAIP aerial photos can split them.

Head: results/fuel_lab/<HEAD>.oof.npy, out-of-fold P(not-fuel) row-aligned with data/fuel_lab/pixels.parquet.
The cutoff is rebuilt exactly as fuel_lab.score_all does it: folds are GroupKFold(5) on block, and each fold's
cutoff is the weighted 99th percentile of the other folds' fuel scores (1 % of fuel hidden).

Within a group, pixels are drawn with probability proportional to their window weight w (so a group's share of
NAIP calls estimates the share over all its pixels), at most PER_WIN per window so one place doesn't dominate.

    uv run scripts/naip_pick.py            # -> data/naip_check/picks.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuel_common import CHUNK, FUEL_FAMILIES, MOS_RES, MOS_X0, MOS_Y0  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HEAD = "mlp64_e_sd_kinds_2yr_ens5"
BUDGET = 0.01
PER_WIN = 2
OUT = ROOT / "data" / "naip_check"

# group: (description, how many)
GROUPS = {
    "hidden_grass_near": ("LANDFIRE grass / grass-shrub, head says not-fuel, <=30 m from non-fuel", 50),
    "hidden_grass_far": ("LANDFIRE grass / grass-shrub, head says not-fuel, >30 m from non-fuel", 50),
    "hidden_woody_near": ("LANDFIRE shrub / timber / other fuel, head says not-fuel, <=30 m from non-fuel", 50),
    "hidden_woody_far": ("LANDFIRE shrub / timber / other fuel, head says not-fuel, >30 m from non-fuel", 50),
    "missed_urban": ("LANDFIRE urban / roads, head colors it as fuel", 50),
    "missed_farm": ("LANDFIRE farmland, head colors it as fuel", 50),
    "missed_bare": ("LANDFIRE bare, head colors it as fuel", 50),
    "missed_water_snow": ("LANDFIRE water / snow, head colors it as fuel", 25),
    "agree_fuel": ("both say fuel (control)", 50),
    "agree_nonfuel": ("both say not-fuel (control)", 50),
}


def wquantile(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return x[o][np.searchsorted(cw, q * cw[-1])]


def main():
    from sklearn.model_selection import GroupKFold
    m = pd.read_parquet(ROOT / "data" / "fuel_lab" / "pixels.parquet")
    q = np.load(ROOT / "results" / "fuel_lab" / f"{HEAD}.oof.npy").astype(np.float64)
    assert len(q) == len(m), (len(q), len(m))
    wins = pd.read_parquet(ROOT / "data" / "fuel_ca_windows.parquet").reset_index(drop=True)
    nf, w = m.nf.to_numpy(), m.w.to_numpy().astype(np.float64)

    fold = np.zeros(len(m), int)
    for f, (_, te) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(m)), groups=m.block.to_numpy())):
        fold[te] = f
    thr = np.zeros(len(m))
    for f in range(5):
        o = (fold != f) & (nf == 0)
        thr[fold == f] = wquantile(q[o], w[o], 1 - BUDGET)
    call = q > thr  # head says not-fuel
    hid = (nf == 0) & call
    print(f"check vs lab: hidden {np.average(call[nf == 0], weights=w[nf == 0]):.4f} (lab 0.0119 on 2025, ~0.01 on 2024), "
          f"caught {np.average(call[nf == 1], weights=w[nf == 1]):.4f} (lab 0.578)")

    fb = m.fbfm.to_numpy()
    grass = np.isin(fb, list(FUEL_FAMILIES["grass"]) + list(FUEL_FAMILIES["grass-shrub"]))
    near = m.d_nf.to_numpy() <= 3
    sel = {
        "hidden_grass_near": hid & grass & near, "hidden_grass_far": hid & grass & ~near,
        "hidden_woody_near": hid & ~grass & near, "hidden_woody_far": hid & ~grass & ~near,
        "missed_urban": (fb == 91) & ~call, "missed_farm": (fb == 93) & ~call, "missed_bare": (fb == 99) & ~call,
        "missed_water_snow": np.isin(fb, [92, 98]) & ~call,
        "agree_fuel": (nf == 0) & ~call, "agree_nonfuel": (nf == 1) & call,
    }

    rng = np.random.default_rng(0)
    picks = []
    for g, (desc, n) in GROUPS.items():
        idx = np.nonzero(sel[g])[0]
        key = rng.random(len(idx)) ** (1 / w[idx])  # weighted draw without replacement (Efraimidis-Spirakis)
        per_win = {}
        took = []
        for i in idx[np.argsort(-key)]:
            wi = m.win.iat[i]
            if per_win.get(wi, 0) < PER_WIN:
                per_win[wi] = per_win.get(wi, 0) + 1
                took.append(i)
                if len(took) == n:
                    break
        print(f"  {g:20s} {len(took):3d} of {len(idx):7d} px in {len(per_win):3d} windows "
              f"(group weight {w[idx].sum():.3g})")
        picks.append(pd.DataFrame(dict(px=took, group=g, group_n=len(idx), group_w=w[idx].sum())))
    p = pd.concat(picks, ignore_index=True)
    row = m.loc[p.px]
    p = pd.concat([p, row[["win", "row", "col", "w", "fbfm", "evt", "wc", "nf", "d_nf", "stratum", "name", "block"]]
                   .reset_index(drop=True)], axis=1)
    p["q"], p["thr"], p["call_nonfuel"] = q[p.px], thr[p.px], call[p.px]
    cy, cx = wins.cy.to_numpy()[p.win], wins.cx.to_numpy()[p.win]
    p["lon"] = MOS_X0 + (cx * CHUNK + p.col + 0.5) * MOS_RES  # AEF pixel center
    p["lat"] = MOS_Y0 - (cy * CHUNK + p.row + 0.5) * MOS_RES
    p = p.sample(frac=1, random_state=1).reset_index(drop=True)  # shuffled, so the chip id says nothing about the group
    p.insert(0, "id", [f"n{i:03d}" for i in range(len(p))])
    OUT.mkdir(parents=True, exist_ok=True)
    p.to_parquet(OUT / "picks.parquet", index=False)
    print(f"wrote {OUT / 'picks.parquet'}: {len(p)} px in {p.win.nunique()} windows")


if __name__ == "__main__":
    main()
