# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "requests", "shapely"]
# ///
"""The fire test's points and grids: does the fuel head call the ground burnable where fires start and burn?

The head never saw a fire (it learned LANDFIRE's fuel map), so fires are an outside check. Each fire is scored
from the embeddings of the year BEFORE it, so the map cannot see the burn scar.

Two sets, California, fires 2020 to 2026 (embedding years 2019 to 2025; the head's spread input needs the two
years before that too, 2017 on):

  starts  WFIGS points of origin, a sample per size class. Each start's own mosaic chunk (2.3 km) holds the
          start and the near controls (random pixels at least 300 m away, read by fire_test_score.py); a control
          chunk at 5 to 10 km or 20 to 30 km (alternating) holds the far controls.
  edges   NIFC WFIGS perimeters of 100 acres and up: up to EDGE_CHUNKS mosaic chunks per fire that the
          perimeter's outer line crosses, so each chunk has burned and unburned ground side by side.

Chunks inside the head's 580 training windows (data/fuel_ca_windows.parquet) are left out.

Writes data/fire_test/{starts,perimeters,edges,grids_needed}.parquet. Small: a few MB, under 1 GB of RAM.

    uv run scripts/fire_test_build.py [--per-size 200] [--edge-fires 200] [--seed 0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import shapely
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuel_common import CHUNK, MOS_RES, MOS_X0, MOS_Y0  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "fire_test"
WFIGS = ROOT.parent / "aef-landcover-fosberg-cog" / "data" / "ignition" / "wfigs.parquet"
PERIM_URL = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters/FeatureServer/0/query"
PERIM_FIELDS = ["poly_IncidentName", "poly_GISAcres", "attr_IncidentSize", "attr_FireDiscoveryDateTime", "attr_FireCause",
                "attr_IrwinID", "attr_InitialLatitude", "attr_InitialLongitude"]
YEARS = range(2020, 2027)
SIZES = {"under 1 ac": (0, 1), "1 to 10 ac": (1, 10), "10 ac and up": (10, np.inf)}
RINGS = {"5-10 km": (5, 10), "20-30 km": (20, 30)}
EDGE_CHUNKS = 3


def chunk_of(lon, lat):
    """Mosaic chunk (cy, cx) of each point, arrays in, arrays out."""
    return ((MOS_Y0 - np.asarray(lat)) / MOS_RES // CHUNK).astype(int), ((np.asarray(lon) - MOS_X0) / MOS_RES // CHUNK).astype(int)


def offset(lon, lat, km, bearing):
    """Points km away at bearing (radians from north), flat-earth (fine at tens of km)."""
    dlat = km * np.cos(bearing) / 111.32
    dlon = km * np.sin(bearing) / (111.32 * np.cos(np.radians(lat)))
    return lon + dlon, lat + dlat


def build_starts(rng, per_size, train):
    w = pd.read_parquet(WFIGS)
    w = w[(w.POOState == "US-CA") & (w.IncidentTypeCategory == "WF") & w.InitialLatitude.notna() & w.InitialLongitude.notna()]
    w = w.assign(year=w.FireDiscoveryDateTime.dt.year)
    w = w[w.year.isin(YEARS)]
    w = w.assign(acres=w[["FinalAcres", "IncidentSize", "DiscoveryAcres"]].max(axis=1).fillna(0),
                 lon=w.InitialLongitude, lat=w.InitialLatitude)
    w = w[(w.lon > -125) & (w.lon < -114) & (w.lat > 32) & (w.lat < 42.1)]
    cy, cx = chunk_of(w.lon, w.lat)
    w = w.assign(cy=cy, cx=cx)
    w = w[~pd.Series(list(zip(w.cy, w.cx)), index=w.index).isin(train)]
    w = w.drop_duplicates(["lon", "lat", "year"])
    parts = []
    for name, (lo, hi) in SIZES.items():
        s = w[(w.acres >= lo) & (w.acres < hi)]
        parts.append(s.sample(min(per_size, len(s)), random_state=int(rng.integers(1 << 31))).assign(size=name))
        print(f"  starts {name}: {len(s)} in pool, {len(parts[-1])} taken", flush=True)
    s = pd.concat(parts, ignore_index=True)
    # one far control chunk per start, the ring alternating, re-drawn when it lands in a training window
    ring_names = list(RINGS)
    s["ring"] = [ring_names[i % 2] for i in range(len(s))]
    ccy, ccx, clon, clat = [], [], [], []
    for r in s.itertuples():
        lo, hi = RINGS[r.ring]
        for _ in range(20):
            lon, lat = offset(r.lon, r.lat, rng.uniform(lo, hi), rng.uniform(0, 2 * np.pi))
            y, x = chunk_of(lon, lat)
            if (int(y), int(x)) not in train:
                break
        ccy.append(int(y)), ccx.append(int(x)), clon.append(lon), clat.append(lat)
    s = s.assign(ctrl_cy=ccy, ctrl_cx=ccx, ctrl_lon=clon, ctrl_lat=clat, cause=s.FireCause.fillna("Unknown"))
    keep = ["IrwinID", "IncidentName", "FireDiscoveryDateTime", "year", "acres", "size", "cause", "lon", "lat", "cy", "cx",
            "ring", "ctrl_cy", "ctrl_cx", "ctrl_lon", "ctrl_lat"]
    return s[keep].rename(columns={"IrwinID": "irwin", "IncidentName": "name", "FireDiscoveryDateTime": "discovered"})


def pull_perimeters():
    feats = []
    for y in YEARS:
        where = (f"attr_FireDiscoveryDateTime >= timestamp '{y}-01-01 00:00:00' AND attr_FireDiscoveryDateTime < timestamp "
                 f"'{y + 1}-01-01 00:00:00' AND attr_IncidentTypeCategory = 'WF' AND attr_POOState = 'US-CA' AND attr_IncidentSize >= 100")
        off = 0
        while True:
            r = requests.get(PERIM_URL, params=dict(where=where, outFields=",".join(PERIM_FIELDS), f="geojson", outSR=4326,
                                                    resultOffset=off, resultRecordCount=200), timeout=180)
            r.raise_for_status()
            fs = r.json()["features"]
            feats += fs
            off += len(fs)
            if len(fs) < 200:
                break
        print(f"  perimeters {y}: {off}", flush=True)
    rows = []
    for f in feats:
        if not f.get("geometry"):
            continue
        p = f["properties"]
        g = shapely.make_valid(shape(f["geometry"]))
        rows.append(dict(name=p["poly_IncidentName"], acres=p["attr_IncidentSize"] or p["poly_GISAcres"], irwin=p["attr_IrwinID"],
                         discovered=pd.to_datetime(p["attr_FireDiscoveryDateTime"], unit="ms", utc=True), cause=p["attr_FireCause"],
                         start_lon=p["attr_InitialLongitude"], start_lat=p["attr_InitialLatitude"], geom=shapely.to_wkb(g)))
    d = pd.DataFrame(rows)
    d["year"] = d.discovered.dt.year
    # one perimeter per fire (the service can hold a fire twice); keep the largest
    d = d.sort_values("acres", ascending=False)
    d = d[d.irwin.isna() | ~d.irwin.duplicated()].reset_index(drop=True)
    d["fire_id"] = np.arange(len(d))
    return d


def build_edges(rng, per, n_fires, train):
    per = per.sample(min(n_fires, len(per)), random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    rows = []
    for r in per.itertuples():
        g = shapely.from_wkb(r.geom)
        line = g.boundary
        if line.is_empty or line.length == 0:
            continue
        # the outer line only: holes (unburned islands) are inside the fire's footprint, not its edge
        outer = shapely.MultiLineString([p.exterior for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon"])
        pts = [outer.interpolate(d) for d in rng.uniform(0, outer.length, 60)]
        cys, cxs = chunk_of([p.x for p in pts], [p.y for p in pts])
        seen = []
        for k in zip(cys.tolist(), cxs.tolist()):
            if k not in seen and k not in train:
                seen.append(k)
            if len(seen) == EDGE_CHUNKS:
                break
        for cy, cx in seen:
            rows.append(dict(fire_id=r.fire_id, year=r.year, cy=cy, cx=cx))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-size", type=int, default=200)
    ap.add_argument("--edge-fires", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    win = pd.read_parquet(ROOT / "data" / "fuel_ca_windows.parquet")
    train = set(zip(win.cy.astype(int), win.cx.astype(int)))

    starts = build_starts(rng, a.per_size, train)
    starts.to_parquet(OUT / "starts.parquet", index=False)
    per = pull_perimeters()
    per.to_parquet(OUT / "perimeters.parquet", index=False)
    edges = build_edges(rng, per, a.edge_fires, train)
    edges.to_parquet(OUT / "edges.parquet", index=False)

    # every (chunk, embedding year) to read; the embedding year is the year before the fire
    need = pd.concat([
        starts[["cy", "cx", "year"]],
        starts[["ctrl_cy", "ctrl_cx", "year"]].rename(columns={"ctrl_cy": "cy", "ctrl_cx": "cx"}),
        edges[["cy", "cx", "year"]],
    ]).assign(emb_year=lambda d: d.year - 1)[["cy", "cx", "emb_year"]].drop_duplicates().reset_index(drop=True)
    need.to_parquet(OUT / "grids_needed.parquet", index=False)
    reads = len(need) * 3
    print(f"starts {len(starts)}, perimeters {len(per)}, edge chunks {len(edges)} from {edges.fire_id.nunique()} fires")
    print(f"grids {len(need)}: {reads} chunk reads, about {reads * 4.2 / 1000:.1f} GB")


if __name__ == "__main__":
    main()
