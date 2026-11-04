# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "tabulate"]
# ///
"""Read out the NAIP check: how many of the head's errors (scored against LANDFIRE) are LANDFIRE errors.

Inputs: data/naip_check/picks.parquet (naip_pick.py) and data/naip_check/labels/*.csv (rubric in naip_rubric.md).
When an id is labeled more than once, the first file's row is used and the rest measure labeler agreement.

The ten groups split every pixel in the lab table (by LANDFIRE fuel / not-fuel x head call, with the errors cut finer),
and picks within a group were drawn in proportion to window weight. So a group's mean NAIP call times the group's
total weight estimates that group's NAIP-fuel weight, and the head can be re-scored on NAIP truth at the same cutoff.
Burnable: yes 1, mixed 0.5, no 0, unsure dropped. Intervals: 90 % bootstrap within groups.

    uv run scripts/naip_readout.py      # -> results/naip_check.md and printed
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "naip_check"
BURN = {"yes": 1.0, "mixed": 0.5, "no": 0.0}
HIDDEN = ["hidden_grass_near", "hidden_grass_far", "hidden_woody_near", "hidden_woody_far"]
MISSED = ["missed_urban", "missed_farm", "missed_bare", "missed_water_snow"]


def load():
    p = pd.read_parquet(D / "picks.parquet")
    parts = [pd.read_csv(f, dtype=str).assign(src=f.stem) for f in sorted((D / "labels").glob("*.csv"))]
    lab = pd.concat(parts, ignore_index=True)
    lab = lab.apply(lambda s: s.str.strip().str.lower() if s.name not in ("note", "src") else s)
    lab["id"] = lab.id.str.strip()
    first = lab.drop_duplicates("id", keep="first")
    return p.merge(first, on="id", how="left"), lab


def agreement(lab):
    dup = lab[lab.duplicated("id", keep=False)]
    if dup.empty:
        return "no ids labeled twice"
    out = []
    for c in ("c10_burn", "c30_burn"):
        g = dup.groupby("id")[c].agg(lambda s: s.nunique() == 1)
        b = dup.assign(v=dup[c].map(BURN)).groupby("id").v.agg(lambda s: s.notna().all() and abs(s.max() - s.min()) < 1)
        out.append(f"{c}: same call {g.mean():.0%}, never yes-vs-no {b.mean():.0%} (n {len(g)})")
    return "; ".join(out)


def est(df, col, rng=None):
    """Stratified totals: NAIP-fuel weight per group."""
    tot = {}
    for g, s in df.groupby("group"):
        v = s[col].map(BURN).dropna().to_numpy()
        if rng is not None and len(v):
            v = rng.choice(v, len(v))
        tot[g] = (s.group_w.iat[0] * v.mean() if len(v) else np.nan, s.group_w.iat[0])
    return tot


def rescore(tot):
    call_groups = HIDDEN + ["agree_nonfuel"]
    fuel = {g: a for g, (a, _) in tot.items()}
    non = {g: w - a for g, (a, w) in tot.items()}
    hidden = sum(fuel[g] for g in call_groups) / sum(fuel.values())
    caught = sum(non[g] for g in call_groups) / sum(non.values())
    return hidden, caught


def main():
    df, lab = load()
    n_lab = df.c10_burn.notna().sum()
    lines = [f"# NAIP check of the fuel head (mlp64_e_sd_kinds_2yr_ens5, 1 % fuel hidden vs LANDFIRE)", "",
             f"{n_lab} of {len(df)} picks labeled. NAIP 2022 vs LANDFIRE 2024 / AEF 2024. Labeler agreement: {agreement(lab)}.", ""]
    for cell in ("c10", "c30"):
        col = f"{cell}_burn"
        rows = []
        for g in HIDDEN + MISSED + ["agree_fuel", "agree_nonfuel"]:
            s = df[df.group == g][col]
            v = s.map(BURN).dropna()
            rows.append(dict(group=g, n=int(s.notna().sum()), yes=int((s == "yes").sum()), mixed=int((s == "mixed").sum()),
                             no=int((s == "no").sum()), unsure=int((s == "unsure").sum()),
                             naip_burnable=round(v.mean(), 2) if len(v) else None))
        t = pd.DataFrame(rows)
        lines += [f"## {'10 m AEF cell' if cell == 'c10' else '30 m LANDFIRE cell'}", "", t.to_markdown(index=False), ""]

        def wshare(groups, fuel_side):
            num = den = 0.0
            for g in groups:
                v = df[df.group == g][col].map(BURN).dropna()
                if len(v):
                    w = df[df.group == g].group_w.iat[0]
                    num += w * (v.mean() if fuel_side else 1 - v.mean())
                    den += w
            return num / den if den else np.nan

        h, c = rescore(est(df, col))
        rng = np.random.default_rng(0)
        bs = np.array([rescore(est(df, col, rng)) for _ in range(1000)])
        lo, hi = np.nanpercentile(bs, [5, 95], axis=0)
        lines += [
            f"- Head's hidden fuel that NAIP shows is NOT burnable (LANDFIRE wrong): {wshare(HIDDEN, False):.0%}",
            f"- Head's missed non-fuel that NAIP shows IS burnable (LANDFIRE wrong): {wshare(MISSED, True):.0%}",
            f"- Controls: both-say-fuel that NAIP calls not burnable {wshare(['agree_fuel'], False):.0%}; "
            f"both-say-not-fuel that NAIP calls burnable {wshare(['agree_nonfuel'], True):.0%}",
            f"- Re-scored on NAIP truth at the same cutoff: fuel hidden {h:.1%} (90 % {lo[0]:.1%} to {hi[0]:.1%}), "
            f"non-fuel caught {c:.0%} ({lo[1]:.0%} to {hi[1]:.0%}). Against LANDFIRE: 1.1 % and 58 %.", ""]
    out = "\n".join(lines)
    (ROOT / "results" / "naip_check.md").write_text(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
