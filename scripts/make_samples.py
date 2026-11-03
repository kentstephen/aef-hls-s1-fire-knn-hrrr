# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "pyarrow", "numpy", "pyproj"]
# ///
"""The table every feature script reads: live fuel moisture samples (Shrub and Tree, not Dead) from
data/nfmd/, 2018 onward (so the year-before AEF embedding exists), joined to site coordinates, with
the grid indices of each site (HRRR pixel, AEF mosaic pixel) and a spatial group for holdout.

- Values outside 20 to 400 % are dropped (the table has typos up to 599,999).
- "New Growth" rows are set aside (flag `new_growth`): fresh leaves are a different quantity.
  "Chamise, Old Growth" is merged into "Chamise".
- `date`: the sample's day. FEMS stores midnight local as 05Z or 00Z; time + 4 h, floored, gives the day.
- `group`: sites within 10 km of each other (single link) share a group, so a held-out site has no
  neighbor a few hundred meters away in training.

Writes data/samples.parquet (one row per sample) and data/sites.parquet (one row per site used).

    uv run scripts/make_samples.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
NFMD = ROOT / "data" / "nfmd"

LCC = "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs"
HX0, HY0, HDX, HDY = -2699020.142521929, 1588193.847443335, 3000, -3000
MOS_RES = 8.983111749910169e-5
MOS_X0, MOS_Y0 = -180.0, 83.68570533713473
REDDING = (-122.39, 40.59)


def haversine_km(lon1, lat1, lon2, lat2):
    p = np.radians
    a = np.sin(p(lat2 - lat1) / 2) ** 2 + np.cos(p(lat1)) * np.cos(p(lat2)) * np.sin(p(lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def groups_within(lon, lat, km):
    n = len(lon)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        d = haversine_km(lon[i], lat[i], lon, lat)
        for j in np.nonzero(d < km)[0]:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri
    roots = [find(i) for i in range(n)]
    _, g = np.unique(roots, return_inverse=True)
    return g


def main():
    s = pd.read_parquet(NFMD / "sites.parquet")
    d = pd.read_parquet(NFMD / "samples_california.parquet")
    d = d[d.category.isin(["Shrub", "Tree"])].copy()
    d = d[(d.value >= 20) & (d.value <= 400)]
    d["date"] = (d.time + pd.Timedelta(hours=4)).dt.tz_convert(None).dt.floor("D")
    d = d[d.date >= "2018-01-01"]
    d["new_growth"] = d.fuel_type.str.contains("New Growth") | (d.sub_category == "New")
    d["species"] = d.fuel_type.str.replace(", Old Growth", "", regex=False)

    used = s[s.siteId.isin(d.site_id.unique())].rename(columns={"siteId": "site_id"})
    used = used[["site_id", "siteName", "longitude", "latitude", "elevation", "groupName"]].reset_index(drop=True)
    hx, hy = Transformer.from_crs("EPSG:4326", LCC, always_xy=True).transform(used.longitude.values, used.latitude.values)
    used["hx"] = np.floor((hx - HX0) / HDX).astype(int)
    used["hy"] = np.floor((hy - HY0) / HDY).astype(int)
    used["mx"] = np.floor((used.longitude - MOS_X0) / MOS_RES).astype(np.int64)
    used["my"] = np.floor((MOS_Y0 - used.latitude) / MOS_RES).astype(np.int64)
    used["group"] = groups_within(used.longitude.values, used.latitude.values, 10.0)
    used["redding_km"] = haversine_km(REDDING[0], REDDING[1], used.longitude.values, used.latitude.values)
    used.to_parquet(ROOT / "data" / "sites.parquet", index=False)

    d = d.merge(used[["site_id", "group", "redding_km"]], on="site_id")
    d = d[["sample_id", "site_id", "group", "date", "species", "category", "new_growth", "value", "redding_km"]]
    d.to_parquet(ROOT / "data" / "samples.parquet", index=False)
    old = d[~d.new_growth]
    print(f"{len(d)} samples ({len(old)} without new growth), {d.site_id.nunique()} sites in {used.group.nunique()} groups")
    print(f"site-dates {old[['site_id', 'date']].drop_duplicates().shape[0]}, within 150 km of Redding: "
          f"{(old.redding_km < 150).sum()} samples at {old[old.redding_km < 150].site_id.nunique()} sites")
    print(old.groupby(old.date.dt.year).size().to_dict())
    print(old.species.value_counts().head(12).to_dict())


if __name__ == "__main__":
    main()
