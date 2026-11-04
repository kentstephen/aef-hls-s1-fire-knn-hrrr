# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "torch", "lightgbm", "psutil", "tabulate"]
# ///
"""Fit the fuel lab's best browser-ready head on every lab pixel and write it for the web map.

The head is mlp64_e_sd_kinds_2yr_ens5 from fuel_scenarios.py: five 128 -> 64 -> 11 networks, averaged.
Input per pixel: the newest year's AEF (64, dequantized sign(q) * (q / 127.5)^2, NOT unit-normalized) and the
spread of the pixel's AEF across 2023, 2024 and 2025 (64, population standard deviation of the dequantized values).
Output: P(not fuel) = the sum of the softmax over the five non-fuel kinds; the map draws P(fuel) = 1 - that.

Trained exactly as the lab's fit_predict (same weights, early stopping on a 15 % block holdout), only on all blocks
instead of four folds of five. Cutoffs for 0.5 / 1 / 2 / 5 % fuel hidden come from the lab's out-of-fold predictions.

    uv run scripts/export_fuel_head.py      # -> models/fuel_head.json, models/fuel_head_check.json

RAM: about 5 GB at peak (both years of features, one standardized copy).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fuel_lab as lab  # noqa: E402
from fuel_common import FUEL_FAMILIES, NONFUEL_KINDS  # noqa: E402

NAME = "mlp64_e_sd_kinds_2yr_ens5"
OUT = lab.ROOT / "models"


def main():
    import pandas as pd
    import torch
    sc = next(s for s in lab.scenarios() if s["name"] == NAME)
    t0 = time.time()
    meta = lab.load_meta()
    X24, X25 = lab.features(sc["features"], 2024), lab.features(sc["features"], 2025)
    X = np.vstack([X24, X25])
    mtr = pd.concat([meta, meta], ignore_index=True)
    del X24
    tr = np.arange(len(mtr))

    # fit_predict standardizes on the training rows; capture the nets and the scaler by fitting the same way
    nets = []
    real_train = lab.train_torch

    def keep(*a, **k):
        net, ep = real_train(*a, **k)
        nets.append(net)
        return net, ep

    lab.train_torch = keep
    probe = np.arange(0, len(meta), 997)  # a fixed sample, scored here and in the browser check
    preds, info = lab.fit_predict(sc, mtr, X, tr, {"q": (X25, probe)})
    lab.train_torch = real_train
    mu, sd = X.mean(0), X.std(0) + 1e-6
    print(f"fit {time.time() - t0:.0f} s, epochs {info['epochs']}", flush=True)

    nf_cols = list(range(len(NONFUEL_KINDS)))
    kinds = list(NONFUEL_KINDS.values()) + list(FUEL_FAMILIES) + ["other burnable"]
    out_nets = []
    for net in nets:
        lin = [m for m in net if isinstance(m, torch.nn.Linear)]
        assert len(lin) == 2
        out_nets.append({
            "w1": lin[0].weight.detach().numpy().astype(np.float32).ravel().tolist(),  # [64, 128] row-major
            "b1": lin[0].bias.detach().numpy().tolist(),
            "w2": lin[1].weight.detach().numpy().astype(np.float32).ravel().tolist(),  # [11, 64] row-major
            "b2": lin[1].bias.detach().numpy().tolist(),
        })

    # cutoffs on P(not fuel) from the lab's out-of-fold predictions: at most this share of fuel above the cutoff
    oof = np.load(lab.RES / f"{NAME}.oof.npy")
    fuel = meta.nf.to_numpy() == 0
    w = meta.w.to_numpy()
    cut = {f"{b:g}": float(lab.wquantile(oof[fuel], w[fuel], 1 - b)) for b in lab.BUDGETS}

    head = {
        "name": NAME,
        "inputs": "e = newest year AEF dequantized sign(q)*(q/127.5)^2 (64); sd = population std of dequantized AEF over "
                  "2023, 2024, 2025 (64); x = ([e, sd] - mean) / scale",
        "years": [2023, 2024, 2025],
        "n_in": int(X.shape[1]),
        "hidden": 64,
        "n_out": len(kinds),
        "kinds": kinds,
        "nf_cols": nf_cols,
        "mean": mu.astype(np.float64).tolist(),
        "scale": sd.astype(np.float64).tolist(),
        "nets": out_nets,
        "p_notfuel_cut": cut,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fuel_head.json").write_text(json.dumps(head, separators=(",", ":")))

    # the same head in numpy from the JSON, against torch, on the probe pixels (2025 embeddings)
    xs = ((X25[probe] - mu) / sd).astype(np.float32)
    pn = np.zeros(len(probe))
    for n in out_nets:
        h = np.maximum(xs @ np.array(n["w1"], np.float32).reshape(64, -1).T + np.array(n["b1"], np.float32), 0)
        z = h @ np.array(n["w2"], np.float32).reshape(len(kinds), 64).T + np.array(n["b2"], np.float32)
        z = np.exp(z - z.max(1, keepdims=True))
        pn += (z[:, nf_cols].sum(1) / z.sum(1)) / len(out_nets)
    print(f"numpy vs torch on {len(probe)} probe px: max abs diff {np.abs(pn - preds['q']).max():.2e}")
    (OUT / "fuel_head_check.json").write_text(json.dumps({
        "win": meta.win.to_numpy()[probe].tolist(), "row": meta.row.to_numpy()[probe].tolist(),
        "col": meta.col.to_numpy()[probe].tolist(), "p_notfuel": pn.round(6).tolist()}))
    print(f"cutoffs on P(not fuel): {cut}")
    print(f"wrote {OUT / 'fuel_head.json'} ({(OUT / 'fuel_head.json').stat().st_size / 1e6:.1f} MB), {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
