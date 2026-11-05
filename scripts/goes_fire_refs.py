# /// script
# requires-python = ">=3.12"
# dependencies = ["virtualizarr[hdf]>=2.8", "obstore", "obspec-utils", "xarray", "zarr>=3", "h5netcdf", "numpy", "requests"]
# ///
"""Byte-range references to GOES-West's fire product over one fire, for the web map to read from NOAA's bucket.

The product is ABI-L2-FDCC (Fire/Hot Spot Characterization, CONUS, every 5 minutes, about 2 km at nadir and
3 to 4 km over California) in s3://noaa-goes18. VirtualiZarr opens each netCDF file as references (headers only,
no pixels), and this keeps only the chunks of Mask (fire category) and Power (fire radiative power, MW) that
cover the fire's box. The browser then fetches those byte ranges itself and decodes them.

The files change form over the years (GOES-18: a Shuffle filter from 2023-03-27 17:01; 250 x 250 tiles until
2026-07-09 17:11, full-width strips after, with a variable renamed). Every file's form is fingerprinted (variables,
each needed array's shape, chunks, dtype, fill and codecs, the projection, the grid) and the scans are split into
eras of one form each; the web map reads each era with its own layout.

  scout  one scan an hour over --days from the fire's discovery: fire pixels in the box, to pick the window
  build  every scan from --from to --to: writes web/public/goes/<slug>.json

--region ca in place of --fire: all of California instead of one fire's box (the map's GOES boxes view finds the
fires itself from these scans), written to web/public/goes/ca-<from date>.json.

    uv run scripts/goes_fire_refs.py --fire Green --year 2025 scout
    uv run scripts/goes_fire_refs.py --fire Green --year 2025 build --from 2025-07-02T01 --to 2025-07-08T00
    uv run scripts/goes_fire_refs.py --region ca --year 2024 build --from 2024-07-22T00 --to 2024-08-03T00

Small: headers only, a few hundred KB of output, under 1 GB of RAM.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import obstore
import requests
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import HTTPStore, S3Store
from virtualizarr.parsers import HDFParser

ROOT = Path(__file__).resolve().parent.parent
BUCKET = "noaa-goes18"
ORIGIN = f"https://{BUCKET}.s3.amazonaws.com"
PRODUCT = "ABI-L2-FDCC"
VARS = ("Mask", "Power")
FIRE_MASK = (10, 11, 12, 13, 14, 30, 31, 32, 33, 34)  # processed and temporally filtered fires; low probability left out
MARGIN_KM = 15
# California's outline box (lon / lat), for --region ca
REGIONS = {"ca": [-124.5, 32.5, -114.1, 42.05]}
PERIMS = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters/FeatureServer/0/query"

s3 = S3Store(BUCKET, region="us-east-1", skip_signature=True)
registry = ObjectStoreRegistry({ORIGIN: HTTPStore.from_url(ORIGIN)})


def fire_info(name: str, year: int) -> dict:
    where = (f"poly_IncidentName = '{name}' AND attr_POOState = 'US-CA' AND attr_IncidentTypeCategory = 'WF' AND "
             f"attr_FireDiscoveryDateTime >= timestamp '{year}-01-01 00:00:00' AND attr_FireDiscoveryDateTime < timestamp '{year + 1}-01-01 00:00:00'")
    r = requests.get(PERIMS, params=dict(where=where, outFields="poly_IncidentName,attr_FireDiscoveryDateTime,attr_IncidentSize,attr_FireCause,OBJECTID",
                                         returnGeometry="true", outSR="4326", f="json", returnExtentOnly="false"), timeout=120)
    r.raise_for_status()
    feats = sorted(r.json()["features"], key=lambda f: -(f["attributes"]["attr_IncidentSize"] or 0))
    if not feats:
        sys.exit(f"no perimeter for {name} {year}")
    f = feats[0]
    xs = [p[0] for ring in f["geometry"]["rings"] for p in ring]
    ys = [p[1] for ring in f["geometry"]["rings"] for p in ring]
    a = f["attributes"]
    return dict(name=a["poly_IncidentName"], objectid=a["OBJECTID"], acres=a["attr_IncidentSize"], cause=a["attr_FireCause"],
                discovered=a["attr_FireDiscoveryDateTime"], bbox=[min(xs), min(ys), max(xs), max(ys)])


def scan_start(key: str) -> datetime:
    return datetime.strptime(key.split("_s")[1][:13], "%Y%j%H%M%S").replace(tzinfo=timezone.utc)


def list_hour(t: datetime) -> list[str]:
    keys = []
    for b in obstore.list(s3, prefix=f"{PRODUCT}/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/"):
        keys += [o["path"] for o in b]
    # a scan written twice at a site handover carries start == end in its name; skip those
    return sorted((k for k in keys if k.endswith(".nc") and k.split("_s")[1][:13] != k.split("_e")[1][:13]), key=scan_start)


def h(o) -> str:
    return hashlib.sha1(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()[:10]


def lonlat_to_xy(lon, lat, p):
    """GOES fixed grid scan angles (radians) of a lon/lat (GOES-R PUG, the inverse of the navigation)."""
    req, rpol, H = p["semi_major_axis"], p["semi_minor_axis"], p["perspective_point_height"] + p["semi_major_axis"]
    lon0 = math.radians(p["longitude_of_projection_origin"])
    e2 = 1 - rpol**2 / req**2
    lat, lon = math.radians(lat), math.radians(lon)
    latc = math.atan((rpol**2 / req**2) * math.tan(lat))
    rc = rpol / math.sqrt(1 - e2 * math.cos(latc) ** 2)
    sx = H - rc * math.cos(latc) * math.cos(lon - lon0)
    sy = -rc * math.cos(latc) * math.sin(lon - lon0)
    sz = rc * math.sin(latc)
    return math.asin(-sy / math.sqrt(sx * sx + sy * sy + sz * sz)), math.atan(sz / sx)


class Scan:
    """One file opened as references: its form, its grid, and the chunk refs of the needed arrays."""

    def __init__(self, key: str):
        self.key = key
        self.t = scan_start(key)
        ms = HDFParser()(url=f"{ORIGIN}/{key}", registry=registry)
        self.vds = ms.to_virtual_dataset()
        self.ds = xr.open_zarr(ms, consolidated=False, zarr_format=3, decode_times=False, mask_and_scale=False)
        self.proj = {k: (v.item() if hasattr(v, "item") else v) for k, v in self.ds["goes_imager_projection"].attrs.items()}
        xa, ya = self.ds["x"].attrs, self.ds["y"].attrs
        # x = add_offset + scale_factor * col (stored as int16 with these)
        self.grid = dict(nx=int(self.ds.sizes["x"]), ny=int(self.ds.sizes["y"]),
                         x0=float(xa["add_offset"]), dx=float(xa["scale_factor"]), y0=float(ya["add_offset"]), dy=float(ya["scale_factor"]))
        self.layout = {}
        for v in VARS:
            m = self.vds[v].data.metadata
            a = self.ds[v].attrs
            self.layout[v] = dict(shape=list(m.shape), chunks=list(m.chunks), dtype=str(m.data_type), fill=str(m.fill_value),
                                  codecs=[c.to_dict() for c in m.codecs],
                                  scale=[a.get("scale_factor"), a.get("add_offset"), a.get("_FillValue"), a.get("units")])
        detail = dict(variables=sorted(self.ds.variables), layout=self.layout, projection=self.proj, grid=self.grid)
        self.form = h(json.loads(json.dumps(detail, default=str)))

    def box(self, bbox):
        """Rows and columns of the fire's box plus the margin."""
        w, s, e, n = bbox
        mlat = MARGIN_KM / 111.32
        mlon = mlat / math.cos(math.radians((s + n) / 2))
        cols, rows = [], []
        for lon in (w - mlon, e + mlon):
            for lat in (s - mlat, n + mlat):
                x, y = lonlat_to_xy(lon, lat, self.proj)
                cols.append((x - self.grid["x0"]) / self.grid["dx"])
                rows.append((y - self.grid["y0"]) / self.grid["dy"])
        g = self.grid
        return (max(0, math.floor(min(rows))), min(g["ny"] - 1, math.ceil(max(rows))),
                max(0, math.floor(min(cols))), min(g["nx"] - 1, math.ceil(max(cols))))

    def refs(self, box):
        r0, r1, c0, c1 = box
        out = {}
        for v in VARS:
            ch = self.layout[v]["chunks"]
            man = self.vds[v].data.manifest.dict()
            lst = []
            for cy in range(r0 // ch[0], r1 // ch[0] + 1):
                for cx in range(c0 // ch[1], c1 // ch[1] + 1):
                    e = man.get(f"{cy}.{cx}")
                    if e is None:
                        continue  # a chunk never written: all fill
                    if e["path"] != f"{ORIGIN}/{self.key}":
                        raise ValueError(f"{self.key}: chunk {cy}.{cx} of {v} points at another file")
                    lst.append([cy, cx, int(e["offset"]), int(e["length"])])
            out[v] = lst
        return out

    def fires(self, box):
        r0, r1, c0, c1 = box
        m = self.ds["Mask"][r0:r1 + 1, c0:c1 + 1].values
        return int(np.isin(m, FIRE_MASK).sum())


def open_scan(key):
    for attempt in range(4):
        try:
            return Scan(key)
        except Exception as e:  # a dropped connection: try again; a broken file: say so and skip it
            if attempt == 3:
                print(f"skip {key}: {e}", flush=True)
                return None
            time.sleep(2 + 3 * attempt)


def scout(fire, days):
    t0 = datetime.fromtimestamp(fire["discovered"] / 1000, timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=6)
    hours = [t0 + timedelta(hours=i) for i in range(int(days * 24))]
    keys = []
    with ThreadPoolExecutor(16) as ex:
        for ks in ex.map(list_hour, hours):
            if ks:
                keys.append(ks[0])

    def one(k):
        sc = open_scan(k)
        return (k, None) if sc is None else (k, sc.fires(sc.box(fire["bbox"])))

    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one, keys))
    by_day = {}
    for k, n in res:
        t = scan_start(k)
        by_day.setdefault(t.strftime("%Y-%m-%d"), []).append(n if n is not None else -1)
        if n:
            print(f"{t:%Y-%m-%d %H:%M} UTC  {n} fire px")
    print("\nper day (UTC): hourly scans with fire, max fire px in a scan")
    for d, ns in by_day.items():
        print(f"  {d}: {sum(1 for n in ns if n > 0):2d}/{len(ns)} hours, max {max(ns)}")


def build(fire, t_from, t_to, slug):
    hours = []
    t = t_from.replace(minute=0, second=0, microsecond=0)
    while t <= t_to:
        hours.append(t)
        t += timedelta(hours=1)
    keys = []
    with ThreadPoolExecutor(16) as ex:
        for ks in ex.map(list_hour, hours):
            keys += [k for k in ks if t_from <= scan_start(k) <= t_to]
    print(f"{len(keys)} scans from {t_from:%Y-%m-%d %H:%M} to {t_to:%Y-%m-%d %H:%M} UTC", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(16) as ex:
        scans = [s for s in ex.map(open_scan, keys) if s is not None]
    print(f"opened {len(scans)} in {time.time() - t0:.0f} s", flush=True)
    scans.sort(key=lambda s: s.t)
    eras, out_scans = [], []
    for s in scans:
        if not eras or eras[-1]["form"] != s.form:
            box = s.box(fire["bbox"])
            eras.append(dict(form=s.form, first=s.t.isoformat(), last=s.t.isoformat(), box=list(box), grid=s.grid,
                             projection={k: s.proj[k] for k in ("semi_major_axis", "semi_minor_axis", "perspective_point_height", "longitude_of_projection_origin")},
                             vars=s.layout))
            print(f"era {len(eras) - 1}: form {s.form} from {s.t:%Y-%m-%d %H:%M}", flush=True)
        e = eras[-1]
        e["last"] = s.t.isoformat()
        out_scans.append(dict(t=int(s.t.timestamp() * 1000), era=len(eras) - 1, key=s.key, refs=s.refs(tuple(e["box"]))))
    doc = dict(fire=fire, product=PRODUCT, origin=ORIGIN, fire_mask=list(FIRE_MASK), eras=eras, scans=out_scans)
    out = ROOT / "web" / "public" / "goes" / f"{slug}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {out}: {out.stat().st_size / 1e3:.0f} KB, {len(out_scans)} scans, {len(eras)} era(s)")


def main():
    ap = argparse.ArgumentParser()
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--fire")
    who.add_argument("--region", choices=sorted(REGIONS))
    ap.add_argument("--year", type=int, required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("scout")
    sc.add_argument("--days", type=float, default=21)
    b = sub.add_parser("build")
    b.add_argument("--from", dest="t_from", required=True)
    b.add_argument("--to", dest="t_to", required=True)
    a = ap.parse_args()
    utc = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)  # noqa: E731
    if a.region:
        if a.cmd != "build":
            sys.exit("--region: build only")
        t_from = utc(a.t_from)
        region = dict(name=a.region.upper(), objectid=0, acres=0, cause="", discovered=int(t_from.timestamp() * 1000), bbox=REGIONS[a.region])
        build(region, t_from, utc(a.t_to), f"{a.region}-{t_from:%Y-%m-%d}")
        return
    fire = fire_info(a.fire, a.year)
    print(f"{fire['name']}: {fire['acres']} ac, {fire['cause']}, discovered "
          f"{datetime.fromtimestamp(fire['discovered'] / 1000, timezone.utc):%Y-%m-%d %H:%M} UTC, bbox {[round(v, 4) for v in fire['bbox']]}", flush=True)
    if a.cmd == "scout":
        scout(fire, a.days)
    else:
        build(fire, utc(a.t_from), utc(a.t_to), f"{fire['name'].lower()}-{a.year}")


if __name__ == "__main__":
    main()
