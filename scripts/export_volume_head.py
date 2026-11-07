# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch"]
# ///
"""Write the fuel volume head (data/change/head_volume.pt, scripts/fit_volume.py) for the web map.

models/fuel_volume.json: mean and scale of the 64 dequantized inputs, w1 [hidden, 64], b1, w2 [out, hidden], b2,
the output layout (reg, kinds, life) and the volume scale. Plus probes: 64 random embeddings (int8) with the
outputs this numpy forward pass gives, so the browser's math can be checked against them.

    uv run scripts/export_volume_head.py
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def main():
    h = torch.load(ROOT / "data" / "change" / "head_volume.pt", weights_only=False)
    s = h["state"]
    w1, b1 = s["l1.weight"].numpy(), s["l1.bias"].numpy()
    w2, b2 = s["l2.weight"].numpy(), s["l2.bias"].numpy()
    mu, sd = np.asarray(h["mu"], np.float32), np.asarray(h["sd"], np.float32)
    rng = np.random.default_rng(0)
    q = rng.integers(-127, 128, (64, 64)).astype(np.int8)
    f = q.astype(np.float32) / 127.5
    x = (np.sign(f) * f * f - mu) / sd
    o = np.maximum(x @ w1.T + b1, 0) @ w2.T + b2
    r = lambda a: [round(float(v), 6) for v in np.ravel(a)]  # noqa: E731
    out = dict(name="fuel_volume", n_in=64, hidden=int(w1.shape[0]), n_out=int(w2.shape[0]), reg=h["reg"], kinds=h["kinds"],
               life=h["life"], vol_lo=h["vol_lo"], vol_hi=h["vol_hi"], mean=r(mu), scale=r(sd), w1=r(w1), b1=r(b1), w2=r(w2),
               b2=r(b2), probe_q=q.ravel().tolist(), probe_out=r(o))
    (ROOT / "models" / "fuel_volume.json").write_text(json.dumps(out))
    print(f"models/fuel_volume.json: {w1.shape[0]} hidden, {w2.shape[0]} out")


if __name__ == "__main__":
    main()
