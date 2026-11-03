"""Queue for fuel_lab.py. The runner reloads this file before each scenario, so new entries can be appended while it
runs; a scenario whose results/fuel_lab/<name>.json exists is skipped (delete the file to rerun it).

Keys:
  name         unique, used for the output file
  features     any of "e" (the pixel's AEF), "m3" / "m9" (mean of the 3 x 3 / 9 x 9 neighborhood), "sd" (spread of
               the pixel's AEF across 2023 to 2025)
  model        "torch" (hidden=[] is a linear head) or "lgbm" (ceiling only, not shader-ready)
  hidden       hidden layer sizes for torch
  labels       "binary" (fuel / not) or "kinds" (each non-fuel kind and fuel family; not-fuel = sum of its kinds)
  weights      "balanced" (both classes count equally) or "true" (each window's real mix)
  train_on     None, "clean" (no 30 m label edge within 70 m) or "agree" (LANDFIRE and WorldCover agree)
  train_years  None (2024) or "both" (each training pixel with its 2024 and its 2025 embedding)
  year_test    also score the fold models on 2025 embeddings
  travel       also train north, test south and the reverse
"""

B = dict(model="torch", hidden=[], labels="binary", weights="balanced", year_test=True)


def S(name, **kw):
    return {**B, "name": name, **kw}


SCENARIOS = [
    # baseline: the head we have, now statewide and scored for hidden fuel
    S("lin_e", features=["e"], travel=True),
    # model shape
    S("lin_e_kinds", features=["e"], labels="kinds"),
    S("mlp32_e", features=["e"], hidden=[32]),
    S("mlp64_e_kinds", features=["e"], hidden=[64], labels="kinds"),
    # neighborhood
    S("lin_e_m3", features=["e", "m3"]),
    S("mlp32_e_m3", features=["e", "m3"], hidden=[32]),
    S("mlp64_e_m3_kinds", features=["e", "m3"], hidden=[64], labels="kinds", travel=True),
    S("mlp64_e_m3_m9_kinds", features=["e", "m3", "m9"], hidden=[64], labels="kinds", travel=True),
    # years
    S("mlp64_e_sd_kinds", features=["e", "sd"], hidden=[64], labels="kinds"),
    S("mlp64_e_m3_kinds_2yr", features=["e", "m3"], hidden=[64], labels="kinds", train_years="both"),
    # labels
    S("mlp64_e_m3_kinds_clean", features=["e", "m3"], hidden=[64], labels="kinds", train_on="clean"),
    S("mlp64_e_m3_kinds_agree", features=["e", "m3"], hidden=[64], labels="kinds", train_on="agree"),
    S("mlp64_e_m3_kinds_true", features=["e", "m3"], hidden=[64], labels="kinds", weights="true"),
    # bigger nets
    S("mlp64x32_e_m3_kinds", features=["e", "m3"], hidden=[64, 32], labels="kinds"),
    S("mlp128_e_m3_m9_sd_kinds", features=["e", "m3", "m9", "sd"], hidden=[128], labels="kinds"),
    # ceilings
    S("lgbm_e", features=["e"], model="lgbm"),
    S("lgbm_e_m3_m9_sd", features=["e", "m3", "m9", "sd"], model="lgbm", travel=True),
]

# Second wave: what counts as not-fuel. LANDFIRE "urban and roads" pixels are 41 % trees and 21 % grass to
# WorldCover (yards, street trees). "strict" trains not-fuel only where WorldCover also sees no vegetation
# (crop, built-up, bare, snow, water); everything else trains as fuel ("kinds": as the other-burnable class).
SCENARIOS += [
    S("lin_e_strict", features=["e"], truth="strict"),
    S("mlp64_e_m3_strict", features=["e", "m3"], hidden=[64], truth="strict"),
    S("mlp64_e_m3_kinds_strict", features=["e", "m3"], hidden=[64], labels="kinds", truth="strict", travel=True),
]
