# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "shapely", "pyogrio", "rasterio"]
# ///
"""GOES-West fire pixels (data/goes/det_<year>.parquet) grouped into events, each matched to WFIGS or not.

Only fire pixels whose center is inside California (Census state outline) are kept: WFIGS here is California
only, so a Nevada or offshore hot spot would count as unmatched for no reason.

An event: fire pixels that touch (same or neighboring GOES pixel) on the same or the next UTC day, chained.
Its first box is the pixels of its first scan: what a dashboard would see first. Each event keeps its first
time, number of scans and pixels, peak fire power, its strongest confidence class, and how many other events
the same pixels hosted that year (a pixel that lights up again and again is likely a fixed hot spot).

Labels (WFIGS, California wildfires only; prescribed and ag burns are not in it, so they fall in "unmatched"):
  perimeter  any event pixel within PERIM_KM of a 2025 burn perimeter (NIFC, 100 acres and up), first time from
             1 day before discovery to PERIM_DAYS after
  start      any event pixel within START_KM of a WFIGS start, first time from 1 day before discovery to
             START_DAYS after
  unmatched  neither

Writes data/goes/events_<year>.parquet (one row per event) and data/goes/event_px_<year>.parquet (the first-box
pixels). Small: under 1 GB of RAM.

    uv run scripts/goes_events.py --year 2025
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuel_common import haversine_km  # noqa: E402
from pick_fuel_windows import ca_polygon  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
G = ROOT / "data" / "goes"
WFIGS = ROOT.parent / "aef-landcover-fosberg-cog" / "data" / "ignition" / "wfigs.parquet"
PERIMS = ROOT / "data" / "fire_test" / "perimeters.parquet"
PERIM_KM, PERIM_DAYS = 3.0, 60
START_KM, START_DAYS = 5.0, 3
STRENGTH = {10: 6, 11: 6, 13: 5, 14: 4, 12: 3, 15: 2}  # higher = stronger; 30s map to their 10s


def events(det):
    """Union-find over (row, col, day) bins: a bin joins its 3 x 3 neighbors on the same and the previous day."""
    det = det.assign(day=det.t.dt.floor("D"))
    bins = det[["row", "col", "day"]].drop_duplicates().reset_index(drop=True)
    idx = {(r, c, d): i for i, (r, c, d) in enumerate(zip(bins.row, bins.col, bins.day))}
    parent = np.arange(len(bins))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    one = pd.Timedelta(days=1)
    for (r, c, d), i in idx.items():
        for dd in (d, d - one):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    j = idx.get((r + dr, c + dc, dd))
                    if j is not None and j != i:
                        a, b = find(i), find(j)
                        if a != b:
                            parent[max(a, b)] = min(a, b)
    bins["event"] = [find(i) for i in range(len(bins))]
    det = det.merge(bins, on=["row", "col", "day"])
    det["event"] = pd.factorize(det.event)[0]
    return det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    a = ap.parse_args()
    det = pd.read_parquet(G / f"det_{a.year}.parquet")
    n0 = len(det)
    det = det[shapely.contains_xy(ca_polygon(), det.lon.to_numpy(), det.lat.to_numpy())].reset_index(drop=True)
    print(f"{len(det)} of {n0} fire px inside California")
    det["strength"] = (det["mask"] % 20).map(STRENGTH).fillna(0).astype(int)
    det = events(det)

    first_t = det.groupby("event").t.min()
    det["first"] = det.t == det.event.map(first_t)
    ev = det.groupby("event").agg(t0=("t", "min"), t1=("t", "max"), scans=("t", "nunique"), px=("row", "size"),
                                  power_max=("power", "max"), strength_max=("strength", "max"),
                                  lon=("lon", "mean"), lat=("lat", "mean"))
    ev["distinct_px"] = det.groupby("event").apply(lambda g: g[["row", "col"]].drop_duplicates().shape[0], include_groups=False)
    fb = det[det["first"]]
    ev["first_px"] = fb.groupby("event").size()
    ev["first_power"] = fb.groupby("event").power.max()
    ev["first_strength"] = fb.groupby("event").strength.max()
    # how many events each pixel hosted in the year; an event's recurrence is the most any of its pixels hosted
    px_events = det[["row", "col", "event"]].drop_duplicates().groupby(["row", "col"]).size().rename("n_events")
    ev["recur"] = det[["row", "col", "event"]].drop_duplicates().join(px_events, on=["row", "col"]).groupby("event").n_events.max()

    # labels
    ev["label"], ev["fire"], ev["fire_acres"] = "unmatched", "", np.nan
    pts = det[["event", "lon", "lat"]].drop_duplicates()
    per = pd.read_parquet(PERIMS)
    per = per[per.year == a.year].reset_index(drop=True)
    geoms = shapely.from_wkb(per.geom)
    tree = shapely.STRtree(geoms)
    deg = PERIM_KM / 111.32
    near = tree.query(shapely.points(pts.lon, pts.lat), predicate="dwithin", distance=deg)  # degrees; lon a bit loose
    hit = pd.DataFrame(dict(event=pts.event.to_numpy()[near[0]], fire=near[1]))
    hit = hit.join(per[["name", "acres", "discovered"]], on="fire").join(ev.t0, on="event")
    ok = (hit.t0 >= hit.discovered - pd.Timedelta(days=1)) & (hit.t0 <= hit.discovered + pd.Timedelta(days=PERIM_DAYS))
    hit = hit[ok].sort_values("acres", ascending=False).drop_duplicates("event")
    ev.loc[hit.event, "label"] = "perimeter"
    ev.loc[hit.event, "fire"] = hit.name.to_numpy()
    ev.loc[hit.event, "fire_acres"] = hit.acres.to_numpy()

    w = pd.read_parquet(WFIGS)
    w = w[(w.POOState == "US-CA") & (w.IncidentTypeCategory == "WF") & (w.FireDiscoveryDateTime.dt.year == a.year)]
    w = w.assign(lon=w.InitialLongitude.fillna(w.geom_lon), lat=w.InitialLatitude.fillna(w.geom_lat),
                 acres=w[["FinalAcres", "IncidentSize", "DiscoveryAcres"]].max(axis=1)).dropna(subset=["lon", "lat"]).reset_index(drop=True)
    left = ev.label == "unmatched"
    cand = pts[pts.event.isin(ev.index[left])]
    rows = []
    for e, g in cand.groupby("event"):
        t0 = ev.at[e, "t0"]
        ww = w[(w.FireDiscoveryDateTime >= t0 - pd.Timedelta(days=START_DAYS)) & (w.FireDiscoveryDateTime <= t0 + pd.Timedelta(days=1))]
        if ww.empty:
            continue
        d = haversine_km(g.lon.to_numpy()[:, None], g.lat.to_numpy()[:, None], ww.lon.to_numpy()[None], ww.lat.to_numpy()[None]).min(0)
        m = d <= START_KM
        if m.any():
            best = ww[m].sort_values("acres", ascending=False).iloc[0]
            rows.append((e, best.IncidentName, best.acres))
    for e, name, acres in rows:
        ev.loc[e, ["label", "fire", "fire_acres"]] = ["start", name, acres]

    ev = ev.reset_index()
    ev.to_parquet(G / f"events_{a.year}.parquet", index=False)
    fb[["event", "row", "col", "lon", "lat", "mask", "power"]].to_parquet(G / f"event_px_{a.year}.parquet", index=False)
    print(f"{len(det)} fire px -> {len(ev)} events; first boxes {len(fb)} px")
    print(ev.label.value_counts().to_string())
    print("\nby strongest class in the first scan:")
    print(pd.crosstab(ev.first_strength, ev.label).to_string())
    print("\nunmatched events at pixels that hosted 5+ events:", int(((ev.label == "unmatched") & (ev.recur >= 5)).sum()))


if __name__ == "__main__":
    main()
