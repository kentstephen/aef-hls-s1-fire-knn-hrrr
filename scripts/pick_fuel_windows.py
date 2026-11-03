# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "rasterio", "shapely", "pyogrio"]
# ///
"""Choose AlphaEarth chunk windows across California for the fuel / not-fuel lab.

One coarse LANDFIRE LF2024 FBFM40 export of the state (about 700 m pixels, nearest neighbor) tells what kind of
ground a point is on. Windows are drawn per stratum so the rare non-fuel kinds show up often enough to score:
towns and roads (91), farmland (93), water (98), bare (99), each fuel family (grass, grass-shrub, shrub,
timber-understory, timber litter), plus plain random land. The 80 Redding windows of sample_fuel.py are kept.

Each window gets a block id (0.5 degree cells, about 50 km) so cross-validation never trains and tests on
neighbors, and a region (north / south of 37 N) for the travel test.

    uv run scripts/pick_fuel_windows.py
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import pyogrio.raw
import shapely
from rasterio.io import MemoryFile

from fuel_common import CHUNK, LF, MOS_RES, MOS_X0, MOS_Y0, get, redding_windows

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "fuel_ca_windows.parquet"
BBOX = (-124.5, 32.5, -114.1, 42.05)
NX, NY = 1500, 1400
STRATA = {  # name: (FBFM40 codes, windows)
    "urban": ([91], 70), "farmland": ([93], 70), "water": ([98], 40), "bare": ([99], 60),
    "grass": (list(range(101, 110)), 40), "grass-shrub": (list(range(121, 125)), 40), "shrub": (list(range(141, 150)), 40),
    "timber-understory": (list(range(161, 166)), 40), "timber-litter": (list(range(181, 190)), 40),
}
N_RANDOM = 60


def ca_polygon():
    z = ROOT / "data" / "ref" / "cb_2023_us_state_20m.zip"
    with zipfile.ZipFile(z) as f:
        f.extractall(z.parent / "states")
    meta, _, geom, fields = pyogrio.raw.read(z.parent / "states" / "cb_2023_us_state_20m.shp")
    i = list(fields[list(meta["fields"]).index("STUSPS")]).index("CA")
    return shapely.from_wkb(geom[i])  # NAD83 lon/lat, within a meter or two of WGS84


def main():
    ca = ca_polygon()
    x0, y0, x1, y1 = BBOX
    b = get(LF.format("FBFM40"), dict(bbox=f"{x0},{y0},{x1},{y1}", bboxSR=4326, imageSR=4326, size=f"{NX},{NY}", format="tiff",
                                      pixelType="S16", interpolation="RSP_NearestNeighbor", f="image"))
    with MemoryFile(b) as m, m.open() as r:
        fb = r.read(1)
    lon = x0 + (np.arange(NX) + 0.5) * (x1 - x0) / NX
    lat = y1 - (np.arange(NY) + 0.5) * (y1 - y0) / NY
    LON, LAT = np.meshgrid(lon, lat)
    inside = shapely.contains_xy(ca, LON, LAT) & (fb > 0)
    print(f"CA coarse cells {inside.sum()}; share by stratum: " + ", ".join(
        f"{k} {np.isin(fb[inside], c).mean():.1%}" for k, (c, _) in STRATA.items()), flush=True)

    rng = np.random.default_rng(7)
    rows, seen = [], set()

    def add(lo, la, stratum, name=""):
        k = (int((MOS_Y0 - la) / MOS_RES) // CHUNK, int((lo - MOS_X0) / MOS_RES) // CHUNK)
        if k in seen:
            return False
        seen.add(k)
        rows.append(dict(cy=k[0], cx=k[1], stratum=stratum, name=name))
        return True

    for (cy, cx), name in redding_windows():  # the Redding windows of the first test
        seen.add((cy, cx))
        rows.append(dict(cy=cy, cx=cx, stratum="redding", name=name))
    for stratum, (codes, n) in STRATA.items():
        idx = np.argwhere(inside & np.isin(fb, codes))
        got = 0
        for j in rng.permutation(len(idx)):
            r, c = idx[j]
            if add(lon[c], lat[r], stratum):
                got += 1
            if got == n:
                break
    idx = np.argwhere(inside)
    got = 0
    for j in rng.permutation(len(idx)):
        r, c = idx[j]
        got += add(lon[c], lat[r], "random")
        if got == N_RANDOM:
            break
    d = pd.DataFrame(rows)
    d["lon"] = MOS_X0 + (d.cx + 0.5) * CHUNK * MOS_RES
    d["lat"] = MOS_Y0 - (d.cy + 0.5) * CHUNK * MOS_RES
    d["block"] = (np.floor(d.lon / 0.5) * 1000 + np.floor(d.lat / 0.5)).astype(int)
    d["region"] = np.where(d.lat >= 37, "north", "south")
    d.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(d)} windows, {d.block.nunique()} blocks\n{d.stratum.value_counts().to_string()}\n"
          f"{d.region.value_counts().to_string()}")


if __name__ == "__main__":
    main()
