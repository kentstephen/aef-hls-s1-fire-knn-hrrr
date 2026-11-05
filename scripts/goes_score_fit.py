# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn"]
# ///
"""The GOES-only triage score for the web dashboard: one logistic regression on the first scan of each GOES event
(fire power, strength class, pixel count), fit on every year pooled, saved as small JSON the browser applies.

Same features, same events and same labels as "GOES alone" in goes_fuel_check.py (events with fuel read, matched =
perimeter or start). The score is the chance an event is a recorded wildfire, judged from GOES alone; the fuel
terms come later.

    uv run scripts/goes_score_fit.py --years 2023 2024 2025 2026   -> models/goes_score.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "data" / "goes"
FEATURES = ["log1p(first power MW, missing = 0)", "first scan strength class", "log(first scan pixels)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    ap.add_argument("--out", default=str(ROOT / "models" / "goes_score.json"))
    a = ap.parse_args()

    f = pd.concat([pd.read_parquet(G / f"event_fuel_{y}.parquet").assign(year=y) for y in a.years], ignore_index=True)
    f = f[(f.n_knn_in > 0) & f.label.isin(["perimeter", "start", "unmatched"])].reset_index(drop=True)
    X = np.c_[np.log1p(f.first_power.fillna(0)), f.first_strength, np.log(f.first_px)]
    y = (f.label != "unmatched").to_numpy().astype(int)
    groups = (np.floor(f.lon / 0.1).astype(int) * 10000 + np.floor(f.lat / 0.1).astype(int)).to_numpy()

    # out-of-fold AUC by place, to check it matches goes_fuel_check's "GOES alone"
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, groups=groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        p[te] = LogisticRegression(max_iter=2000).fit((X[tr] - mu) / sd, y[tr]).predict_proba((X[te] - mu) / sd)[:, 1]
    auc = roc_auc_score(y, p)

    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(max_iter=2000).fit((X - mu) / sd, y)
    out = dict(
        name="goes_only_first_scan",
        target="event matched to a NIFC / WFIGS perimeter or start (vs unmatched)",
        features=FEATURES,
        strength_classes={"10, 11": 6, "13": 5, "14": 4, "12": 3, "15": 2, "mask 30s": "as their 10s"},
        score="sigmoid(intercept + sum(coef * (x - mean) / scale))",
        mean=[round(float(v), 6) for v in mu],
        scale=[round(float(v), 6) for v in sd],
        coef=[round(float(v), 6) for v in m.coef_[0]],
        intercept=round(float(m.intercept_[0]), 6),
        years=a.years,
        n=int(len(y)),
        matched=int(y.sum()),
        auc_out_of_fold_by_place=round(float(auc), 3),
    )
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
