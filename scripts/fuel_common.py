"""Shared pieces of the fuel / not-fuel lab, with no heavy imports (numpy only).

Constants match make_samples.py, sample_aef.py and sample_fuel.py; redding_windows() returns the same 80 windows
as sample_fuel.pick_windows().
"""
import time
import urllib.parse
import urllib.request

import numpy as np

MOS_RES = 8.983111749910169e-5
MOS_X0, MOS_Y0 = -180.0, 83.68570533713473
REDDING = (-122.39, 40.59)
CHUNK = 256
YEAR0 = 2017
LF = "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2024/LF2024_{}_CONUS/ImageServer/exportImage"
TOWNS = {"Redding": (-122.39, 40.59), "Anderson": (-122.30, 40.45), "Red Bluff": (-122.24, 40.18), "Chico": (-121.84, 39.73),
         "Oroville": (-121.56, 39.51), "Corning": (-122.18, 39.93), "Weaverville": (-122.94, 40.73), "Mount Shasta": (-122.31, 41.31),
         "Burney": (-121.66, 40.88), "Susanville": (-120.65, 40.42)}

# FBFM40 codes: 91 urban and roads, 92 snow / ice, 93 farmland, 98 water, 99 bare; everything else burnable.
NONFUEL_KINDS = {91: "urban and roads", 92: "snow and ice", 93: "farmland", 98: "water", 99: "bare"}
FUEL_FAMILIES = {"grass": range(101, 110), "grass-shrub": range(121, 125), "shrub": range(141, 150),
                 "timber-understory": range(161, 166), "timber-litter": range(181, 190)}


def haversine_km(lon1, lat1, lon2, lat2):
    p = np.radians
    a = np.sin(p(lat2 - lat1) / 2) ** 2 + np.cos(p(lat1)) * np.cos(p(lat2)) * np.sin(p(lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def chunk_of(lon, lat):
    return int((MOS_Y0 - lat) / MOS_RES) // CHUNK, int((lon - MOS_X0) / MOS_RES) // CHUNK


def redding_windows(seed=0, n_random=70):
    """Chunk indices (cy, cx) and names: towns first, then random points within 150 km of Redding."""
    rng = np.random.default_rng(seed)
    out, names = [], []

    def add(lon, lat, name):
        k = chunk_of(lon, lat)
        if k not in out:
            out.append(k)
            names.append(name)

    for name, (lon, lat) in TOWNS.items():
        add(lon, lat, name)
    while len(out) < len(TOWNS) + n_random:
        lon, lat = REDDING[0] + rng.uniform(-1.8, 1.8), REDDING[1] + rng.uniform(-1.4, 1.4)
        if haversine_km(REDDING[0], REDDING[1], lon, lat) < 150:
            add(lon, lat, "random")
    return list(zip(out, names))


def get(url, params, tries=4):
    q = url + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            with urllib.request.urlopen(q, timeout=180) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(3 + 3 * i)
