# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "shapely", "pyogrio", "rasterio", "obstore", "h5py", "psutil", "certifi"]
# ///
"""What the GOES events with no WFIGS wildfire record are: other records, other satellites, the ground under them.

"Unmatched" (goes_events.py) only means no NIFC perimeter or WFIGS wildfire start lies near an event. Here each
event (matched ones too, as a check on how often each source sees a known fire) is held against:

  wfigs_near  a WFIGS California wildfire of NEAR_ACRES or more within NEAR_KM of any event pixel, the event
              starting from 1 day before discovery to NEAR_DAYS after: looser than the label (5 km, 3 days), a check
              on whether the label's rules miss lasting fires (not used as a kind: in 2025 it matches at the chance rate)
  calfire     a CAL FIRE incident (incidents.fire.ca.gov, all agencies it lists) within REC_KM, the event starting
              from 1 day before the incident's start to REC_DAYS after
  rx_wfigs    a WFIGS prescribed fire (IncidentTypeCategory RX) or complex within REC_KM, from 1 day before
              discovery to RX_DAYS after (an Rx record's date is its first burn day)
  rx_facts    a USFS FACTS hazardous fuel treatment burn (broadcast, pile, fire use) polygon within FACTS_KM of an
              event pixel, the event from FACTS_BEFORE days before the unit's completion date to 3 days after
  viirs       VIIRS S-NPP 375 m active fire detections (FIRMS yearly country file, no key) within VIIRS_KM of an
              event pixel, from 12 h before the event's first scan to 12 h after its last; type 2 = FIRMS's
              "other static land source" (industry, gas flares)
  ground      LANDFIRE LF2024 FBFM40 sampled 7 x 7 over each first-box pixel (data/goes/lf blocks, cached by
              goes_fuel.py): farmland (93), urban (91), bare (99), water (98), the rest fuel

Every rule is also run with the event times moved by +-45 and +-90 days: the share that still matches is the
chance rate of that rule (a record near a place that burns often, a field burned every year).

Each unmatched event then gets one kind, first that applies: calfire, prescribed (rx_wfigs or
rx_facts), fixed hot spot (5+ events at its pixels in the year, or a VIIRS static source), farmland (half the
box or more), VIIRS heat with no record, nothing else saw it.

The near-zero-fuel group (kNN fuel under 10 % in and around) is looked up in OpenStreetMap (Overpass, within
OSM_KM of the first box's center): power plants (and their source), oil and refineries, landfills, other
industry, farmland, towns.

Downloads are cached in data/goes/ext/. Under 1.5 GB of RAM (FIRMS's US file, 50 LANDFIRE blocks at a time).
Writes results/goes_unmatched_<year>.json and data/goes/unmatched_<year>.parquet (one row per event).

    uv run scripts/goes_unmatched.py --year 2025 [--no-osm]
"""
import argparse
import io
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuel_common import MOS_RES, MOS_X0, MOS_Y0, haversine_km  # noqa: E402
from goes_ca_detections import CA, xy_to_lonlat  # noqa: E402
from goes_events import STRENGTH, events  # noqa: E402
from pick_fuel_windows import ca_polygon  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
G = ROOT / "data" / "goes"
EXT = G / "ext"
LFB = G / "lf"
LF_N = 8 * 256  # mosaic pixels a side per LANDFIRE block (goes_fuel.LF_BLOCK chunks)
WFIGS = ROOT.parent / "aef-landcover-fosberg-cog" / "data" / "ignition" / "wfigs.parquet"
WFIGS_SVC = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Incident_Locations/FeatureServer/0/query"
FACTS_SVC = "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_HazardousFuelsTreatments_01/MapServer/15/query"
CALFIRE = "https://incidents.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=true&year={}"
FIRMS = "https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/{0}/viirs-snpp_{0}_United_States.csv"
OVERPASS = "https://overpass-api.de/api/interpreter"
UA = {"User-Agent": "goes-fuel-research/1.0 (public data study)"}

NEAR_KM, NEAR_DAYS, NEAR_ACRES = 10.0, 30, 10
REC_KM, REC_DAYS, RX_DAYS = 5.0, 7, 14
FACTS_KM, FACTS_BEFORE = 2.0, 60
VIIRS_KM = 3.0
OSM_KM = 2.0
SHIFTS = (-90, -45, 45, 90)
SSL = ssl.create_default_context(cafile=certifi.where())


def ns(x):
    """UTC times (Series or Timestamp) as numpy datetime64[ns] with no zone."""
    if isinstance(x, pd.Timestamp):
        return x.tz_convert(None).to_datetime64() if x.tzinfo else x.to_datetime64()
    x = pd.to_datetime(x, utc=True)
    return x.dt.tz_convert(None).to_numpy("datetime64[ns]")
BURN_TYPES = ("Broadcast Burn", "Machine Pile Burn", "Hand Pile Burn", "Jackpot Burning", "Fire Use", "Underburn")


def http(url, data=None, tries=4, timeout=180):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout, context=SSL) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 + 10 * i)


def arcgis_all(url, params, page=1000):
    """Every feature of an ArcGIS REST query, page by page (GeoJSON)."""
    out, off = [], 0
    while True:
        q = dict(params, f="geojson", outSR=4326, resultOffset=off, resultRecordCount=page)
        d = json.loads(http(url + "?" + urllib.parse.urlencode(q)))
        feats = d.get("features", [])
        out += feats
        if len(feats) < page and not d.get("exceededTransferLimit") and not d.get("properties", {}).get("exceededTransferLimit"):
            break
        off += len(feats)
        if not feats:
            break
    return out


# ---------- sources, cached ----------

def firms(year):
    f = EXT / f"viirs_snpp_ca_{year}.parquet"
    if not f.exists():
        raw = http(FIRMS.format(year), timeout=600)
        v = pd.read_csv(io.BytesIO(raw), dtype={"acq_time": str})
        v = v[(v.longitude >= CA[0]) & (v.longitude <= CA[2]) & (v.latitude >= CA[1]) & (v.latitude <= CA[3])]
        v["t"] = pd.to_datetime(v.acq_date + " " + v.acq_time.str.zfill(4), format="%Y-%m-%d %H%M", utc=True)
        v[["t", "longitude", "latitude", "frp", "confidence", "daynight", "type"]].to_parquet(f, index=False)
    return pd.read_parquet(f)


def calfire(year):
    f = EXT / f"calfire_{year}.json"
    if not f.exists():
        f.write_bytes(http(CALFIRE.format(year)))
    c = pd.DataFrame(json.loads(f.read_text()))
    c = c.dropna(subset=["Longitude", "Latitude"])
    c = c[(c.Longitude != 0) & (c.Latitude != 0)]
    c["start"] = pd.to_datetime(c.Started, utc=True, errors="coerce")
    return c.dropna(subset=["start"]).reset_index(drop=True)


def wfigs_rx(year):
    f = EXT / f"wfigs_ca_not_wf_{year}.parquet"
    if not f.exists():
        where = (f"POOState='US-CA' AND IncidentTypeCategory<>'WF' AND FireDiscoveryDateTime>=date '{year}-01-01' "
                 f"AND FireDiscoveryDateTime<date '{year + 1}-01-01'")
        feats = arcgis_all(WFIGS_SVC, dict(where=where, outFields="IncidentName,IncidentTypeCategory,FireDiscoveryDateTime,"
                                                                   "InitialLatitude,InitialLongitude,IncidentSize,UniqueFireIdentifier"))
        rows = []
        for ft in feats:
            p, g = ft["properties"], ft.get("geometry") or {}
            lon, lat = (g.get("coordinates") or [None, None])[:2]
            rows.append(dict(name=p["IncidentName"], kind=p["IncidentTypeCategory"], acres=p["IncidentSize"],
                             disc=pd.to_datetime(p["FireDiscoveryDateTime"], unit="ms", utc=True),
                             lon=p["InitialLongitude"] if p["InitialLongitude"] else lon,
                             lat=p["InitialLatitude"] if p["InitialLatitude"] else lat))
        pd.DataFrame(rows).to_parquet(f, index=False)
    return pd.read_parquet(f).dropna(subset=["lon", "lat"]).reset_index(drop=True)


def facts_burns(year):
    """USFS burn units completed in the year or up to 120 days after (completion is often recorded late)."""
    f = EXT / f"facts_burns_ca_{year}.parquet"
    if not f.exists():
        types = ",".join(f"'{t}'" for t in BURN_TYPES)
        where = (f"state_abbr='CA' AND treatment_type IN ({types}) AND date_completed>=date '{year}-01-01' "
                 f"AND date_completed<date '{year + 1}-05-01'")
        feats = arcgis_all(FACTS_SVC, dict(where=where, outFields="activity,treatment_type,date_completed,nbr_units_accomplished,"
                                                                   "activity_unit_name", maxAllowableOffset=0.0005), page=500)
        rows = []
        for ft in feats:
            if not ft.get("geometry"):
                continue
            p = ft["properties"]
            rows.append(dict(activity=p["activity"], kind=p["treatment_type"], acres=p["nbr_units_accomplished"],
                             unit=p["activity_unit_name"], done=pd.to_datetime(p["date_completed"], unit="ms", utc=True),
                             geom=shapely.to_wkb(shapely.geometry.shape(ft["geometry"]))))
        pd.DataFrame(rows).to_parquet(f, index=False)
    return pd.read_parquet(f)


def wfigs_wf(year):
    w = pd.read_parquet(WFIGS)
    w = w[(w.POOState == "US-CA") & (w.IncidentTypeCategory == "WF") & (w.FireDiscoveryDateTime.dt.year == year)]
    w = w.assign(lon=w.InitialLongitude.fillna(w.geom_lon), lat=w.InitialLatitude.fillna(w.geom_lat))
    return w.dropna(subset=["lon", "lat"]).rename(columns={"FireDiscoveryDateTime": "disc", "IncidentName": "name"}).reset_index(drop=True)


# ---------- matching ----------

def event_pixels(year):
    """All pixels of every event, the same events goes_events.py made (checked against events_<year>.parquet)."""
    det = pd.read_parquet(G / f"det_{year}.parquet")
    det = det[shapely.contains_xy(ca_polygon(), det.lon.to_numpy(), det.lat.to_numpy())].reset_index(drop=True)
    det["strength"] = (det["mask"] % 20).map(STRENGTH).fillna(0).astype(int)
    det = events(det)
    ev = pd.read_parquet(G / f"events_{year}.parquet").set_index("event")
    chk = det.groupby("event").agg(t0=("t", "min"), px=("row", "size"))
    assert (chk.t0 == ev.t0).all() and (chk.px == ev.px).all(), "events differ from events_<year>.parquet"
    return det[["event", "t", "lon", "lat"]].drop_duplicates(["event", "lon", "lat"])


def near_points(pts, recs, km, ev_t0, lo_days, hi_days, shift_days=0, name_col="name"):
    """event -> name of the closest record within km whose date window holds the event's first time.
    recs: lon, lat, date column 'd'. Window: record date - lo_days <= t0 <= record date + hi_days."""
    out = {}
    if recs.empty:
        return out
    t0 = ev_t0 + pd.Timedelta(days=shift_days)
    rd = ns(recs.d)
    rlon, rlat = recs.lon.to_numpy(), recs.lat.to_numpy()
    deg = km / 111.32 * 1.4
    for e, g in pts.groupby("event", sort=False):
        t = ns(t0[e])
        ok = (rd - np.timedelta64(int(lo_days * 86400), "s") <= t) & (t <= rd + np.timedelta64(int(hi_days * 86400), "s"))
        if not ok.any():
            continue
        lo, la = g.lon.to_numpy(), g.lat.to_numpy()
        box = ok & (rlon > lo.min() - deg) & (rlon < lo.max() + deg) & (rlat > la.min() - deg) & (rlat < la.max() + deg)
        if not box.any():
            continue
        i = np.nonzero(box)[0]
        d = haversine_km(lo[:, None], la[:, None], rlon[i][None], rlat[i][None]).min(0)
        if (d <= km).any():
            out[e] = recs[name_col].iloc[i[np.argmin(d)]]
    return out


def near_polys(pts, polys, km, ev_t0, lo_days, hi_days, shift_days=0):
    out = {}
    if polys.empty:
        return out
    geoms = shapely.from_wkb(polys.geom)
    tree = shapely.STRtree(geoms)
    deg = km / 111.32 / np.cos(np.radians(37))  # degrees of longitude, loose in latitude
    q = tree.query(shapely.points(pts.lon.to_numpy(), pts.lat.to_numpy()), predicate="dwithin", distance=deg)
    h = pd.DataFrame(dict(event=pts.event.to_numpy()[q[0]], poly=q[1]))
    h["t0"] = (ev_t0 + pd.Timedelta(days=shift_days)).reindex(h.event).to_numpy()
    h["done"] = polys.done.to_numpy()[h.poly]
    h = h[(h.t0 >= h.done - pd.Timedelta(days=lo_days)) & (h.t0 <= h.done + pd.Timedelta(days=hi_days))]
    h = h.drop_duplicates("event")
    return dict(zip(h.event, polys.unit.to_numpy()[h.poly] if "unit" in polys else h.poly))


def viirs_match(pts, ev, v, shift_days=0):
    """event -> (VIIRS detections, of them static-source) within VIIRS_KM and [t0 - 12 h, t1 + 12 h]."""
    out = {}
    vt = ns(v.t)
    vlon, vlat, vty = v.longitude.to_numpy(), v.latitude.to_numpy(), v["type"].to_numpy()
    order = np.argsort(vt)
    vt, vlon, vlat, vty = vt[order], vlon[order], vlat[order], vty[order]
    sh = np.timedelta64(int(shift_days * 86400), "s")
    half = np.timedelta64(12 * 3600, "s")
    deg = VIIRS_KM / 111.32 * 1.4
    for e, g in pts.groupby("event", sort=False):
        a = np.searchsorted(vt, ns(ev.at[e, "t0"]) + sh - half)
        b = np.searchsorted(vt, ns(ev.at[e, "t1"]) + sh + half, side="right")
        if b <= a:
            continue
        lo, la = g.lon.to_numpy(), g.lat.to_numpy()
        sl = slice(a, b)
        m = (vlon[sl] > lo.min() - deg) & (vlon[sl] < lo.max() + deg) & (vlat[sl] > la.min() - deg) & (vlat[sl] < la.max() + deg)
        if not m.any():
            continue
        i = np.nonzero(m)[0] + a
        d = haversine_km(lo[:, None], la[:, None], vlon[i][None], vlat[i][None]).min(0)
        k = d <= VIIRS_KM
        if k.any():
            out[e] = (int(k.sum()), int((vty[i][k] == 2).sum()))
    return out


# ---------- ground under the first box ----------

def ground(year, px):
    """Share of LANDFIRE FBFM40 sample points over each event's first box that are farmland, urban, bare, water, fuel."""
    g = json.loads((G / f"grid_{year}.json").read_text())
    u = (np.arange(7) + 0.5) / 7 - 0.5
    dr, dc = np.meshgrid(u, u)
    rows = []
    for p in px.itertuples():
        r, c = p.row + dr.ravel(), p.col + dc.ravel()
        lon, lat = xy_to_lonlat(g["x0"] + g["dx"] * c, g["y0"] + g["dy"] * r, g)
        mr = ((MOS_Y0 - lat) / MOS_RES).astype(np.int64)
        mc = ((lon - MOS_X0) / MOS_RES).astype(np.int64)
        for a, b in zip(mr, mc):
            rows.append((p.event, a // LF_N, b // LF_N, a % LF_N, b % LF_N))
    s = pd.DataFrame(rows, columns=["event", "by", "bx", "i", "j"]).sort_values(["by", "bx"])
    cache = OrderedDict()
    codes = np.full(len(s), -1, np.int16)
    for n, ((by, bx), idx) in enumerate(s.groupby(["by", "bx"]).indices.items()):
        f = LFB / f"{by}_{bx}.npz"
        if (by, bx) not in cache:
            cache[(by, bx)] = np.load(f)["fbfm"] if f.exists() else None
            if len(cache) > 50:
                cache.popitem(last=False)
        v = cache[(by, bx)]
        if v is not None:
            sub = s.iloc[idx]
            codes[idx] = v[sub.i.to_numpy(), sub.j.to_numpy()]
    s["code"] = codes
    s = s[s.code > 0]
    kinds = {91: "urban", 93: "farmland", 98: "water", 99: "bare", 92: "snow"}
    s["kind"] = s.code.map(kinds).fillna("fuel")
    return s.groupby("event").kind.value_counts(normalize=True).unstack(fill_value=0.0)


# ---------- OSM for the near-zero-fuel group ----------

OSM_Q = """[out:json][timeout:120];
nwr(around:{r},{lat},{lon})->.a;
(
 nwr.a["power"~"^(plant|generator)$"];
 nwr.a["landuse"~"^(industrial|landfill|farmland|orchard|vineyard|residential|commercial|retail|quarry|railway|military)$"];
 nwr.a["man_made"~"^(works|petroleum_well|flare|kiln|wastewater_plant|chimney|storage_tank|pumping_station|mineshaft)$"];
 nwr.a["industrial"];
 nwr.a["amenity"~"^(waste_disposal|recycling|crematorium)$"];
 nwr.a["aeroway"="aerodrome"];
);
out tags center;"""


def osm(event, lon, lat):
    f = EXT / "osm" / f"{event}.json"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        q = OSM_Q.format(r=int(OSM_KM * 1000), lat=f"{lat:.5f}", lon=f"{lon:.5f}")
        f.write_bytes(http(OVERPASS, data=urllib.parse.urlencode(dict(data=q)).encode()))
    return json.loads(f.read_text()).get("elements", [])


def osm_kind(els, lon, lat):
    """One kind for the place, most specific first, and the name of the feature that decided it."""
    def dist(e):
        c = e.get("center") or dict(lat=e.get("lat"), lon=e.get("lon"))
        return haversine_km(lon, lat, c["lon"], c["lat"]) if c.get("lat") is not None else 99
    rules = [
        ("geothermal power", lambda t: t.get("power") in ("plant", "generator") and "geothermal" in (t.get("plant:source", "") + t.get("generator:source", ""))),
        ("gas, biomass or other power plant", lambda t: t.get("power") == "plant" and not any(s in t.get("plant:source", "") for s in ("solar", "wind", "hydro"))),
        ("oil field or refinery", lambda t: t.get("man_made") in ("petroleum_well", "flare") or t.get("industrial") in ("refinery", "oil", "well_cluster", "oil_field", "gas") or "refin" in t.get("name", "").lower()),
        ("landfill or waste", lambda t: t.get("landuse") == "landfill" or t.get("amenity") in ("waste_disposal", "recycling") or t.get("industrial") in ("landfill", "waste_handling", "scrap_yard")),
        ("other industry", lambda t: t.get("man_made") in ("works", "kiln", "chimney") or t.get("landuse") in ("industrial", "quarry") or "industrial" in t),
        ("farmland", lambda t: t.get("landuse") in ("farmland", "orchard", "vineyard")),
        ("town", lambda t: t.get("landuse") in ("residential", "commercial", "retail")),
    ]
    for name, rule in rules:
        hits = [e for e in els if rule(e.get("tags", {}))]
        if hits:
            h = min(hits, key=dist)
            t = h.get("tags", {})
            return name, t.get("name") or t.get("operator") or "", round(float(dist(h)), 2)
    return "nothing mapped", "", np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--no-osm", action="store_true")
    a = ap.parse_args()
    EXT.mkdir(parents=True, exist_ok=True)
    f = pd.read_parquet(G / f"event_fuel_{a.year}.parquet").set_index("event")
    px = pd.read_parquet(G / f"event_px_{a.year}.parquet")
    pts = event_pixels(a.year)
    pts = pts[pts.event.isin(f.index)]
    t0 = f.t0

    srcs = {}
    try:
        w = wfigs_wf(a.year).assign(d=lambda x: x.disc)
        w = w[w[["FinalAcres", "IncidentSize", "DiscoveryAcres"]].max(axis=1) >= NEAR_ACRES]
        srcs["wfigs_near"] = (lambda s: near_points(pts, w, NEAR_KM, t0, 1, NEAR_DAYS, s))
    except Exception as ex:
        print(f"WFIGS wildfires: {ex}")
    try:
        c = calfire(a.year).assign(d=lambda x: x.start, lon=lambda x: x.Longitude, lat=lambda x: x.Latitude, name=lambda x: x.Name)
        srcs["calfire"] = (lambda s: near_points(pts, c, REC_KM, t0, 1, REC_DAYS, s))
        print(f"CAL FIRE: {len(c)} incidents")
    except Exception as ex:
        print(f"CAL FIRE failed: {ex}")
    try:
        rx = wfigs_rx(a.year).assign(d=lambda x: x.disc)
        srcs["rx_wfigs"] = (lambda s: near_points(pts, rx, REC_KM, t0, 1, RX_DAYS, s))
        print(f"WFIGS prescribed / complex: {len(rx)} ({rx.kind.value_counts().to_dict()})")
    except Exception as ex:
        print(f"WFIGS RX failed: {ex}")
    try:
        fb = facts_burns(a.year)
        srcs["rx_facts"] = (lambda s: near_polys(pts, fb, FACTS_KM, t0, FACTS_BEFORE, 3, s))
        print(f"USFS FACTS burns: {len(fb)} units ({fb.kind.value_counts().to_dict()})")
    except Exception as ex:
        print(f"FACTS failed: {ex}")
    v = None
    try:
        v = firms(a.year)
        print(f"VIIRS S-NPP in the California box: {len(v)} detections")
    except Exception as ex:
        print(f"FIRMS failed (no yearly file yet?): {ex}")

    obs = {k: fn(0) for k, fn in srcs.items()}
    shifted = {k: [set(fn(s)) for s in SHIFTS] for k, fn in srcs.items()}
    for k, m in obs.items():
        f[k] = f.index.map(m)
    if v is not None:
        vm = viirs_match(pts, f, v)
        f["viirs_n"] = f.index.map(lambda e: vm.get(e, (0, 0))[0])
        f["viirs_static"] = f.index.map(lambda e: vm.get(e, (0, 0))[1])
        shifted["viirs"] = [set(viirs_match(pts, f, v, s)) for s in SHIFTS]
    gr = ground(a.year, px)
    for k in ("farmland", "urban", "bare", "water", "fuel"):
        f[f"lf_{k}"] = gr[k].reindex(f.index) if k in gr else 0.0

    um = f.label == "unmatched"
    has = lambda k: f[k].notna() if k in f else pd.Series(False, index=f.index)  # noqa: E731
    seen = {k: has(k) for k in srcs} | ({"viirs": f.viirs_n > 0} if v is not None else {})

    def tally(m):
        """Per source: events it matches in the subset, and how many it matches by chance (times moved)."""
        idx = set(f.index[m])
        return {k: dict(seen=int(seen[k][m].sum()), by_chance=round(float(np.mean([len(x & idx) for x in shifted[k]])), 1)) for k in seen}

    fixed = (f.recur >= 5) | (f.get("viirs_static", pd.Series(0, index=f.index)) > 0)
    kind = pd.Series("nothing else saw it", index=f.index)
    order = [("VIIRS heat, no record", seen.get("viirs", pd.Series(False, index=f.index))),
             ("farmland (half the box or more)", f.lf_farmland.fillna(0) >= 0.5),
             ("fixed hot spot (5+ events or VIIRS static)", fixed),
             ("prescribed burn record", has("rx_wfigs") | has("rx_facts")),
             ("CAL FIRE incident", has("calfire"))]
    for name, m in order:  # last assignment wins, so the list runs from weakest to strongest evidence
        kind[m] = name
    f["kind"] = kind

    strong = (f.first_strength >= 5) | (f.scans >= 2)
    hi = strong & (f.knn_in >= 0.8)
    subsets = {"all unmatched": um, "unmatched, strong (class 5+ first scan or 2+ scans)": um & strong,
               "unmatched, strong and kNN fuel in >= 0.8": um & hi, "unmatched, single scan and class 2 to 4": um & ~strong,
               "matched (how often each source sees a known fire)": ~um}
    res = dict(year=a.year, n_events=int(len(f)),
               windows=dict(wfigs_near=f"WFIGS wildfires of {NEAR_ACRES}+ acres, {NEAR_KM} km, -1 to +{NEAR_DAYS} d (looser than the label)",
                            calfire=f"{REC_KM} km, -1 to +{REC_DAYS} d", rx_wfigs=f"{REC_KM} km, -1 to +{RX_DAYS} d",
                            rx_facts=f"{FACTS_KM} km, -{FACTS_BEFORE} to +3 d of completion", viirs=f"{VIIRS_KM} km, t0 - 12 h to t1 + 12 h",
                            chance=f"the same rules with event times moved by {SHIFTS} days, mean"))
    for name, m in subsets.items():
        res[name] = dict(n=int(m.sum()), sources=tally(m),
                         farmland_half_or_more=int((f.lf_farmland.fillna(0)[m] >= 0.5).sum()), recurring_5plus=int((f.recur[m] >= 5).sum()))
        if name.startswith("matched"):
            if v is not None:
                res[name]["viirs_share_by_scans"] = {"1 scan": round(float(seen["viirs"][m & (f.scans == 1)].mean()), 3),
                                                     "2+ scans": round(float(seen["viirs"][m & (f.scans >= 2)].mean()), 3)}
        else:
            res[name]["kinds"] = f[m].kind.value_counts().to_dict()
    print(json.dumps({k: res[k] for k in res if k != "windows"}, indent=1, default=str))

    # near-zero-fuel group
    z = ((f.knn_in + f.knn_around) / 2 < 0.1)
    zf = f[z].copy()
    if not a.no_osm and len(zf):
        cen = px.groupby("event")[["lon", "lat"]].mean()
        def look(e):
            lon, lat = cen.at[e, "lon"], cen.at[e, "lat"]
            try:
                return osm_kind(osm(e, lon, lat), lon, lat)
            except Exception as ex:
                print(f"overpass {e}: {ex}", flush=True)
                return ("lookup failed", "", np.nan)

        with ThreadPoolExecutor(2) as ex:  # Overpass gives each address two slots
            out = list(ex.map(look, zf.index))
        zf["osm_kind"], zf["osm_name"], zf["osm_km"] = zip(*out)
        f.loc[zf.index, "osm_kind"] = zf.osm_kind
        f.loc[zf.index, "osm_name"] = zf.osm_name
    zres = dict(n=int(len(zf)), matched=int((zf.label != "unmatched").sum()), unmatched=int((zf.label == "unmatched").sum()),
                lf_majority=zf[["lf_farmland", "lf_urban", "lf_bare", "lf_water", "lf_fuel"]].idxmax(axis=1).str[3:].value_counts().to_dict(),
                viirs_seen=int((zf.viirs_n > 0).sum()) if "viirs_n" in zf else None,
                viirs_static=int((zf.viirs_static > 0).sum()) if "viirs_static" in zf else None,
                recurring_5plus=int((zf.recur >= 5).sum()),
                by_month_and_ground=pd.crosstab(zf.t0.dt.month,
                                                zf[["lf_farmland", "lf_urban", "lf_bare", "lf_water", "lf_fuel"]].idxmax(axis=1).str[3:]).to_dict("index"))
    if "osm_kind" in zf:
        zres["osm_kinds"] = zf.osm_kind.value_counts().to_dict()
        ex = {}
        for k, g in zf.groupby("osm_kind"):
            g = g.sort_values("recur", ascending=False)
            ex[k] = [f"{r.osm_name if isinstance(r.osm_name, str) and r.osm_name else 'unnamed'} ({r.lat:.3f}, {r.lon:.3f}; "
                     f"{r.label}, {int(r.recur)} events there)" for r in g.head(5).itertuples()]
        zres["osm_examples"] = ex
        maj = zf[["lf_farmland", "lf_urban", "lf_bare", "lf_water", "lf_fuel"]].idxmax(axis=1).str[3:]
        zres["osm_kind_by_landfire_majority"] = pd.crosstab(zf.osm_kind, maj).to_dict("index")
    res["near-zero fuel (kNN < 10 % in and around)"] = zres
    print(json.dumps(zres, indent=1, default=str))

    keep = [c for c in f.columns if c in ("label", "fire", "t0", "t1", "scans", "first_strength", "recur", "lon", "lat", "knn_in",
                                         "knn_around", "power_max", "kind", "viirs_n", "viirs_static", "osm_kind", "osm_name")
            or c.startswith("lf_") and c not in ("lf_in", "lf_around") or c in srcs]
    f[keep].reset_index().to_parquet(G / f"unmatched_{a.year}.parquet", index=False)
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(res, open(ROOT / "results" / f"goes_unmatched_{a.year}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
