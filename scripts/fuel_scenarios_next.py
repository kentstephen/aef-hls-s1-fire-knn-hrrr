# Wave 3, to append to fuel_scenarios.py once the runner on the older fuel_lab.py has finished.
# The nets stopped after 1 useful epoch (held-out loss rose from epoch 2): try stopping on held-out AUC,
# more regularization, and averaging several nets. "sd" (year spread) was the best single add in wave 1.
SCENARIOS += [
    S("lin_e_sd_kinds", features=["e", "sd"], labels="kinds"),
    S("mlp64_e_sd_kinds_auc", features=["e", "sd"], hidden=[64], labels="kinds", stop_on="auc", patience=5),
    S("mlp64_e_sd_kinds_reg", features=["e", "sd"], hidden=[64], labels="kinds", stop_on="auc", patience=5,
      dropout=0.3, wd=1e-2, lr=1e-3),
    S("mlp64_e_m3_sd_kinds_reg", features=["e", "m3", "sd"], hidden=[64], labels="kinds", stop_on="auc", patience=5,
      dropout=0.3, wd=1e-2, lr=1e-3),
    S("mlp64_e_sd_kinds_ens5", features=["e", "sd"], hidden=[64], labels="kinds", stop_on="auc", patience=5,
      dropout=0.3, wd=1e-2, lr=1e-3, ensemble=5),
    S("mlp64_e_sd_kinds_2yr_reg", features=["e", "sd"], hidden=[64], labels="kinds", train_years="both", stop_on="auc",
      patience=5, dropout=0.3, wd=1e-2, lr=1e-3),
    S("mlp64_e_sd_kinds_strict_reg", features=["e", "sd"], hidden=[64], labels="kinds", truth="strict", stop_on="auc",
      patience=5, dropout=0.3, wd=1e-2, lr=1e-3, travel=True),
]

# Overture override: mapped buildings and roads up to class N (3 = through residential, 4 = with service roads) are drawn
# low regardless of the head. Shows what the head no longer has to do, and how much fuel that hides (roads under canopy).
SCENARIOS += [
    S("mlp64_e_sd_kinds_reg_ov3", features=["e", "sd"], hidden=[64], labels="kinds", stop_on="auc", patience=5,
      dropout=0.3, wd=1e-2, lr=1e-3, overture=3),
    S("mlp64_e_sd_kinds_reg_ov4", features=["e", "sd"], hidden=[64], labels="kinds", stop_on="auc", patience=5,
      dropout=0.3, wd=1e-2, lr=1e-3, overture=4),
]
