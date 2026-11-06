# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "rasterio", "scipy", "shapely", "tabulate", "pmtiles", "pillow", "requests", "psutil",
#                 "zarr>=3", "obstore"]
# ///
"""The agencies' maps on the risk backtest's own test: do USFS burn probability, USFS wildfire hazard potential and
CAL FIRE's fire hazard severity zones score burned ground above unburned ground, and do they beat the map's value?
And where do they agree with the value, and where not?

Layers (read per backtest chunk, cut to the chunk's 256 x 256 grid of 10 m pixels, nearest neighbor):
  usfs_bp   USFS Wildfire Risk to Communities annual burn probability, 2024 edition, 30 m (FSim at 270 m, upsampled).
            Landscape as of the end of 2020 (LANDFIRE 2020), fire occurrence and weather through 2020.
            Raw integer units; only ranks are used here.
  usfs_whp  USFS Wildfire Hazard Potential, 2023 edition, 270 m, continuous. Same end-2020 landscape.
  fhsz      CAL FIRE Fire Hazard Severity Zones (fhsz24_5: SRA zones in force April 2024, LRA zones recommended
            2024 to 2025), 1 moderate, 2 high, 3 very high, 0 unzoned. Rated only on State and Local
            Responsibility Area ground (SRA22_2); federal ground is never zoned, so it is left out, not scored 0.

Before the fire or not: both USFS layers describe the end of 2020, so every fire of 2021 on is after them, and the
2020 fires are probably inside their fuels already (burned ground looks like less fuel there). The CAL FIRE zones
were drafted in November 2022 (SRA) and 2024 (LRA); fires before that may be in their inputs.
Results are given for all years, 2021 on, and the two halves (2020 to 2022, 2023 to 2026).

Every comparison uses the same pixels for every layer: our value defined, the USFS layer defined ("usfs" set), and
for the CAL FIRE zones also rated ground ("rated" set). The value and fuel alone are scored again on each set.

Steps (resumable, data in data/agency/):
  fetch   the three layers and the responsibility areas per chunk (backtest chunks and the starts' chunks)
  score   fires (inside vs outside, 5-10 km, 20-30 km), starts, agreement -> results/agency_compare.json

RAM: under 2 GB (one chunk at a time; CAL FIRE polygons cached on disk, at most 200 held in memory).

    uv run scripts/agency_compare.py fetch|score
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
import risk_backtest as rb  # noqa: E402
from fuel_common import CHUNK, MOS_RES, MOS_X0, MOS_Y0  # noqa: E402

ROOT = rb.ROOT
A = ROOT / "data" / "agency"
IMG = "https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/{}/ImageServer/exportImage"
FS = "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/{}/FeatureServer/0/query"
RASTERS = {"usfs_bp": "USFS_EDW_RMRS_WRC_BurnProbability", "usfs_whp": "USFS_EDW_RMRS_WildfireHazardPotentialContinuous"}
RA_CODE = {"FRA": 1, "LRA": 2, "SRA": 3}


# ---------------------------------------------------------------- fetch

def get(url, params, binary=False):
    import requests
    for attempt in range(6):
        try:
            r = requests.post(url, data=params, timeout=180)
            r.raise_for_status()
            if not binary:
                j = r.json()
                if "error" in j:
                    raise RuntimeError(j["error"])
                return j
            if not r.headers.get("content-type", "").startswith("image/"):
                raise RuntimeError(r.text[:300])
            return r.content
        except Exception:
            if attempt == 5:
                raise
            time.sleep(3 + 5 * attempt)


def raster(service, box):
    import rasterio
    b = get(IMG.format(service), dict(bbox=",".join(map(str, box)), bboxSR=4326, imageSR=4326, size=f"{CHUNK},{CHUNK}",
                                      format="tiff", interpolation="RSP_NearestNeighbor", f="image"), binary=True)
    with rasterio.MemoryFile(b) as m, m.open() as ds:
        assert ds.shape == (CHUNK, CHUNK), ds.shape
        return ds.read(1)


class Polygons:
    """A feature layer's polygons by object id: on disk once, at most `keep` held in memory."""

    def __init__(self, layer, fields, keep=200):
        self.layer, self.fields, self.keep = layer, fields, keep
        self.dir = A / "poly" / layer
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mem, self.lock = OrderedDict(), threading.Lock()

    def ids(self, box):
        j = get(FS.format(self.layer), dict(geometry=",".join(map(str, box)), geometryType="esriGeometryEnvelope", inSR=4326,
                                            spatialRel="esriSpatialRelIntersects", returnIdsOnly="true", f="json"))
        return sorted(j.get("objectIds") or [])

    def fetch(self, oids):
        from shapely.geometry import shape
        for k in range(0, len(oids), 20):
            fc = get(FS.format(self.layer), dict(objectIds=",".join(map(str, oids[k:k + 20])), outFields="OBJECTID," + self.fields,
                                                 outSR=4326, geometryPrecision=6, f="geojson"))
            for f in fc["features"]:
                p = f["properties"]
                g = shapely.make_valid(shape(f["geometry"]))
                tmp = self.dir / f".{p['OBJECTID']}.{threading.get_ident()}"
                tmp.write_text(json.dumps(dict(props=p, wkb=shapely.to_wkb(g, hex=True))))
                os.replace(tmp, self.dir / f"{p['OBJECTID']}.json")

    def get(self, oid):
        with self.lock:
            if oid in self.mem:
                self.mem.move_to_end(oid)
                return self.mem[oid]
        d = json.loads((self.dir / f"{oid}.json").read_text())
        v = (d["props"], shapely.from_wkb(d["wkb"]))
        with self.lock:
            self.mem[oid] = v
            while len(self.mem) > self.keep:
                self.mem.popitem(last=False)
        return v

    def in_box(self, box):
        oids = self.ids(box)
        miss = [o for o in oids if not (self.dir / f"{o}.json").exists()]
        if miss:
            self.fetch(miss)
        clip = shapely.box(*box)
        return [(p, shapely.clip_by_rect(g, *box)) for p, g in map(self.get, oids) if g.intersects(clip)]


def burn(shapes, box):
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    shapes = [(g, v) for g, v in shapes if not g.is_empty]
    if not shapes:
        return np.zeros((CHUNK, CHUNK), np.uint8)
    return rasterize(shapes, out_shape=(CHUNK, CHUNK), transform=from_origin(box[0], box[3], MOS_RES, MOS_RES),
                     fill=0, dtype="uint8")


def all_chunks():
    return sorted(set(zip(rb.scored_chunks().cy, rb.scored_chunks().cx)) | set(rb.start_chunks()))


def fetch(workers=6, max_gb=4.0, limit=None):
    import psutil
    out = A / "chunks"
    out.mkdir(parents=True, exist_ok=True)
    zones, ra = Polygons("fhsz24_5", "FHSZ,FHSZ_7Class,SRA22_2"), Polygons("SRA22_2", "SRA")
    todo = [c for c in all_chunks() if not (out / f"{c[0]}_{c[1]}.npz").exists()][:limit]
    proc = psutil.Process(os.getpid())
    print(f"chunks {len(all_chunks())}, {len(todo)} to fetch", flush=True)

    def one(cy, cx):
        box = rb.chunk_box(cy, cx)
        r = {k: raster(s, box) for k, s in RASTERS.items()}
        # higher class drawn last wins where zones overlap
        z = sorted(zones.in_box(box), key=lambda pg: pg[0]["FHSZ"] or 0)
        r["fhsz"] = burn([(g, int(p["FHSZ"] or 0)) for p, g in z], box)
        r["ra"] = burn([(g, RA_CODE.get(p["SRA"], 0)) for p, g in ra.in_box(box)], box)
        f = out / f"{cy}_{cx}.npz"
        np.savez_compressed(out / f".{cy}_{cx}.tmp.npz", **r)
        os.replace(out / f".{cy}_{cx}.tmp.npz", f)

    t0, done = time.time(), 0
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(one, *c): c for c in todo}
        for fu in as_completed(futs):
            fu.result()
            done += 1
            rss = proc.memory_info().rss / 1e9
            if done % 25 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"fetch {done}/{len(todo)} {el / done:.2f} s/chunk, eta {(len(todo) - done) * el / done / 60:.1f} min, "
                      f"rss {rss:.2f} GB", flush=True)
            if rss > max_gb:
                print(f"rss {rss:.2f} GB over the {max_gb} GB cap; stopping (rerun resumes)", flush=True)
                os._exit(3)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


# ---------------------------------------------------------------- score

SETS = {"usfs": ["risk", "fuel", "usfs_bp", "usfs_whp"], "rated": ["risk", "fuel", "usfs_bp", "fhsz"]}
OTHERS = {"usfs": ["fuel", "usfs_bp", "usfs_whp"], "rated": ["fuel", "usfs_bp", "fhsz"]}
MIN_PX = 100
PERIODS = {"all years": (2020, 2026), "2021 on": (2021, 2026), "2020-2022": (2020, 2022), "2023-2026": (2023, 2026)}


def layers(cy, cx):
    z = np.load(A / "chunks" / f"{cy}_{cx}.npz")
    bp = z["usfs_bp"].astype(np.float32)
    whp = z["usfs_whp"].astype(np.float32)
    whp[whp < 0] = np.nan
    rated = np.isin(z["ra"], (RA_CODE["LRA"], RA_CODE["SRA"]))
    return dict(usfs_bp=bp, usfs_whp=whp, fhsz=np.where(rated, z["fhsz"], np.nan).astype(np.float32)), z["ra"]


def masks(P):
    ok = np.isfinite(P["risk"]) & np.isfinite(P["usfs_bp"]) & np.isfinite(P["usfs_whp"])
    return {"usfs": ok, "rated": ok & np.isfinite(P["fhsz"])}


def chunk_parts_factory(fires):
    cache, leaks = OrderedDict(), [0]

    def chunk_parts(cy, cx, year):
        k = (cy, cx, year)
        if k not in cache:
            fuel, cont = rb.continuity(cy, cx, year - 1)
            if fuel is None:
                cache[k] = None
            else:
                slope, aspect = rb.terrain_of(cy, cx)
                last, used = fires.last(cy, cx, before=year)
                leaks[0] += bool(used and max(used) >= year)
                P = rb.parts(fuel, cont, slope, aspect, last, now=year)
                L, ra = layers(cy, cx)
                cache[k] = ({k2: P[k2] for k2 in ("risk", "fuel")} | L, ra)
            if len(cache) > 64:
                cache.popitem(last=False)
        return cache[k]
    return chunk_parts, leaks


def paired(pf, a, b, rng):
    d = (pf[f"auc_{a}"] - pf[f"auc_{b}"]).dropna()
    m, lo, hi, n = rb.boot_ci(d, rng)
    return dict(diff=m, lo=lo, hi=hi, fires=n, share=float((d > 0).mean()) if len(d) else np.nan)


def score_fires(chunk_parts, rng, out, tabulate):
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    ch = rb.scored_chunks()
    per = pd.read_parquet(rb.FT / "perimeters.parquet").set_index("fire_id")
    rows, case_px, ctrl_px = [], {}, {}
    t0 = time.time()
    for n, r in enumerate(ch.itertuples()):
        got = chunk_parts(r.cy, r.cx, r.year)
        if got is None:
            continue
        P, _ = got
        M = masks(P)
        d = dict(fire_id=r.fire_id, year=r.year, cy=r.cy, cx=r.cx, kind=r.kind)
        if r.kind == "edge":
            lon0, lat0, lon1, lat1 = rb.chunk_box(r.cy, r.cx)
            inside = rasterize([(shapely.from_wkb(per.loc[r.fire_id, "geom"]), 1)], out_shape=(CHUNK, CHUNK),
                               transform=from_origin(lon0, lat1, MOS_RES, MOS_RES), fill=0).astype(bool)
            for s, feats in SETS.items():
                m = M[s]
                if (inside & m).sum() < MIN_PX or (~inside & m).sum() < MIN_PX:
                    continue
                for f in feats:
                    d[f"{s}|auc_{f}"] = rb.auc(P[f][inside & m], P[f][~inside & m])
                idx = np.flatnonzero(inside & m)
                pick = rng.choice(idx, min(rb.N_SAMPLE, len(idx)), replace=False)
                case_px.setdefault((s, r.fire_id), []).append(np.stack([P[f].ravel()[pick] for f in feats], 1))
        else:
            for s, feats in SETS.items():
                idx = np.flatnonzero(M[s])
                if len(idx) < MIN_PX:
                    continue
                pick = rng.choice(idx, min(rb.N_SAMPLE, len(idx)), replace=False)
                ctrl_px[(s, r.fire_id, r.kind)] = np.stack([P[f].ravel()[pick] for f in feats], 1)
        rows.append(d)
        if n % 100 == 0:
            print(f"fires: chunk {n + 1}/{len(ch)} {(time.time() - t0) / (n + 1):.2f} s/chunk", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(A / "chunk_scores.parquet", index=False)
    e = df[df.kind == "edge"]
    pf_all = e.groupby("fire_id").mean(numeric_only=True)

    out["inside_vs_outside"] = {}
    for s, feats in SETS.items():
        out["inside_vs_outside"][s] = {}
        for period, (y0, y1) in PERIODS.items():
            pf = pf_all[(pf_all.year >= y0) & (pf_all.year <= y1)].rename(columns=lambda c: c.split("|", 1)[1] if c.startswith(s + "|") else "x" + c)
            pf = pf.dropna(subset=["auc_risk"]) if "auc_risk" in pf else pf.iloc[:0]
            res, tab = {}, []
            for f in feats:
                m, lo, hi, nf = rb.boot_ci(pf[f"auc_{f}"], rng)
                res[f] = dict(auc=m, lo=lo, hi=hi, fires=nf)
                tab.append([f, nf, f"{m:.3f}", f"[{lo:.3f}, {hi:.3f}]", ""])
            for f in OTHERS[s]:
                p = paired(pf, "risk", f, rng)
                res[f"risk minus {f}"] = p
                tab.append([f"value minus {f}", p["fires"], f"{p['diff']:+.3f}", f"[{p['lo']:+.3f}, {p['hi']:+.3f}]", f"{p['share']:.0%}"])
            out["inside_vs_outside"][s][period] = res
            print(f"\n== inside vs outside the line, {s} pixels, fires {period}; AUC per fire, mean over fires")
            print(tabulate(tab, headers=["layer", "fires", "AUC / diff", "95 %", "value higher in"]))

    out["burned_vs_far"] = {}
    for s, feats in SETS.items():
        out["burned_vs_far"][s] = {}
        for ring in rb.RINGS:
            for period, (y0, y1) in [("all years", PERIODS["all years"]), ("2021 on", PERIODS["2021 on"])]:
                per_fire = {f: [] for f in feats}
                for (s2, fid, kind), cpx in ctrl_px.items():
                    if s2 != s or kind != ring or (s, fid) not in case_px:
                        continue
                    y = int(per.loc[fid, "year"])
                    if not (y0 <= y <= y1):
                        continue
                    cs = np.concatenate(case_px[(s, fid)])
                    for k, f in enumerate(feats):
                        per_fire[f].append(rb.auc(cs[:, k], cpx[:, k]))
                res, tab = {}, []
                for f in feats:
                    m, lo, hi, nf = rb.boot_ci(per_fire[f], rng)
                    res[f] = dict(auc=m, lo=lo, hi=hi, fires=nf)
                    tab.append([f, nf, f"{m:.3f}", f"[{lo:.3f}, {hi:.3f}]"])
                dd = np.array(per_fire["risk"])[:, None] - np.array([per_fire[f] for f in OTHERS[s]]).T if per_fire["risk"] else np.zeros((0, 3))
                for k, f in enumerate(OTHERS[s]):
                    m, lo, hi, nf = rb.boot_ci(dd[:, k], rng)
                    res[f"risk minus {f}"] = dict(diff=m, lo=lo, hi=hi, fires=nf)
                    tab.append([f"value minus {f}", nf, f"{m:+.3f}", f"[{lo:+.3f}, {hi:+.3f}]"])
                out["burned_vs_far"][s][f"{ring}|{period}"] = res
                print(f"\n== burned ground vs a same-year chunk {ring} away, {s} pixels, fires {period}")
                print(tabulate(tab, headers=["layer", "fires", "AUC / diff", "95 %"]))


def reach(a, r, c, m, lat):
    """The layer at the pixel (m = 0), or its highest value within m meters (a box), as the backtest scores starts."""
    if m == 0:
        return a[r, c]
    dy = MOS_RES * rb.M_LAT
    hy, hx = int(round(m / dy)), int(round(m / (dy * math.cos(math.radians(lat)))))
    w = a[max(0, r - hy):r + hy + 1, max(0, c - hx):c + hx + 1]
    return float(np.nanmax(w)) if np.isfinite(w).any() else np.nan


def score_starts(chunk_parts, rng, out, tabulate):
    st = pd.read_parquet(rb.FT / "starts.parquet")
    rows = []
    dy = MOS_RES * rb.M_LAT
    for sid, r in enumerate(st.itertuples()):
        got = chunk_parts(r.cy, r.cx, r.year)
        if got is None:
            continue
        P, _ = got
        M = masks(P)
        rr, cc = int((MOS_Y0 - r.lat) / MOS_RES) - r.cy * CHUNK, int((r.lon - MOS_X0) / MOS_RES) - r.cx * CHUNK
        for s, feats in SETS.items():
            if not M[s][rr, cc]:
                continue
            base = dict(sid=sid, year=r.year, set=s)

            def add(kind, PP, MM, i, j):
                d = dict(base, kind=kind)
                for f in feats:
                    a = np.where(MM, PP[f], np.nan)
                    for m in (0, 100):
                        d[f"{f}{m}"] = reach(a, i, j, m, r.lat)
                rows.append(d)

            add("start", P, M[s], rr, cc)
            ii, jj = np.mgrid[0:CHUNK, 0:CHUNK]
            cand = np.flatnonzero((np.hypot((ii - rr) * dy, (jj - cc) * dy * math.cos(math.radians(r.lat))) >= 300) & M[s])
            for k in rng.choice(cand, min(rb.N_CTRL, len(cand)), replace=False):
                add("near", P, M[s], *divmod(int(k), CHUNK))
            gc = chunk_parts(r.ctrl_cy, r.ctrl_cx, r.year)
            if gc is not None:
                PC, _ = gc
                MC = masks(PC)[s]
                cand = np.flatnonzero(MC)
                for k in rng.choice(cand, min(rb.N_CTRL, len(cand)), replace=False):
                    add("far", PC, MC, *divmod(int(k), CHUNK))
    df = pd.DataFrame(rows)
    df.to_parquet(A / "start_scores.parquet", index=False)

    def auc_ci(g, col, kind):
        case = g[g.kind == "start"].set_index("sid")[col]
        ctrl = g[g.kind == kind]
        sids = np.intersect1d(ctrl.sid.unique(), case.index)
        if not len(sids):
            return np.nan, np.nan, np.nan, 0
        by = {k: v[col].to_numpy(float) for k, v in ctrl.groupby("sid")}
        one = lambda pick: rb.auc(case.loc[pick].to_numpy(float), np.concatenate([by[k] for k in pick]))  # noqa: E731
        b = [one(rng.choice(sids, len(sids))) for _ in range(300)]
        lo, hi = np.nanpercentile(b, [2.5, 97.5])
        return one(sids), lo, hi, len(sids)

    out["starts"] = {}
    for s, feats in SETS.items():
        for period, (y0, y1) in [("all years", PERIODS["all years"]), ("2021 on", PERIODS["2021 on"])]:
            g = df[(df.set == s) & (df.year >= y0) & (df.year <= y1)]
            tab, res = [], {}
            for kind in ("near", "far"):
                for f in feats:
                    for m in (0, 100):
                        a, lo, hi, n = auc_ci(g, f"{f}{m}", kind)
                        res[f"{kind}|{f}|{m}"] = dict(auc=a, lo=lo, hi=hi, n=n)
                        tab.append([kind, f, m, n, f"{a:.3f}", f"[{lo:.3f}, {hi:.3f}]"])
            out["starts"][f"{s}|{period}"] = res
            print(f"\n== starts vs same-year controls, {s} pixels, starts {period} (near: 300 m to ~2 km; far: 5-30 km)")
            print(tabulate(tab, headers=["control", "layer", "reach m", "starts", "AUC", "95 %"]))


def agreement(chunk_parts, rng, out, tabulate):
    from scipy.stats import spearmanr
    ch = rb.scored_chunks()
    st = pd.read_parquet(rb.FT / "starts.parquet")
    names = pd.read_parquet(rb.FT / "perimeters.parquet").set_index("fire_id")["name"]
    todo = {(r.cy, r.cx): (r.year, r.kind, names.get(r.fire_id)) for r in ch.itertuples()}
    for r in st.itertuples():
        todo.setdefault((r.cy, r.cx), (r.year, "start", r.name))
        todo.setdefault((r.ctrl_cy, r.ctrl_cx), (r.year, "start far", r.name))
    rows, px, by_class = [], [], {c: [] for c in ("FRA (not rated)", "unzoned", "moderate", "high", "very high")}
    for (cy, cx), (year, kind, name) in todo.items():
        got = chunk_parts(cy, cx, year)
        if got is None:
            continue
        P, ra = got
        ok = masks(P)["usfs"]
        if ok.sum() < MIN_PX:
            continue
        lon0, lat0, lon1, lat1 = rb.chunk_box(cy, cx)
        rated = np.isfinite(P["fhsz"]) & ok
        d = dict(cy=cy, cx=cx, year=year, kind=kind, name=name, lon=(lon0 + lon1) / 2, lat=(lat0 + lat1) / 2,
                 risk=float(np.mean(P["risk"][ok])), fuel=float(np.mean(P["fuel"][ok])), usfs_bp=float(np.mean(P["usfs_bp"][ok])),
                 usfs_whp=float(np.mean(P["usfs_whp"][ok])), rated=float(rated.mean()),
                 fhsz=float(np.mean(P["fhsz"][rated])) if rated.sum() >= MIN_PX else np.nan,
                 vh=float(np.mean(P["fhsz"][rated] == 3)) if rated.sum() >= MIN_PX else np.nan)
        rows.append(d)
        idx = np.flatnonzero(ok)
        pick = rng.choice(idx, min(300, len(idx)), replace=False)
        px.append(np.stack([P[f].ravel()[pick] for f in ("risk", "fuel", "usfs_bp", "usfs_whp", "fhsz")] + [ra.ravel()[pick].astype(np.float32)], 1))
    df = pd.DataFrame(rows)
    df.to_parquet(A / "chunk_means.parquet", index=False)
    X = np.concatenate(px)
    cols = ["risk", "fuel", "usfs_bp", "usfs_whp", "fhsz"]

    out["agreement"] = dict(chunks=int(len(df)), pixels=int(len(X)))
    tab = []
    for a, b in [("risk", "usfs_bp"), ("risk", "usfs_whp"), ("risk", "fhsz"), ("fuel", "usfs_bp"), ("usfs_bp", "usfs_whp"), ("usfs_bp", "fhsz"),
                 ("usfs_whp", "fhsz")]:
        cm = df[[a, b]].dropna()
        rc = spearmanr(cm[a], cm[b]).statistic
        ia, ib = cols.index(a), cols.index(b)
        okp = np.isfinite(X[:, ia]) & np.isfinite(X[:, ib])
        rp = spearmanr(X[okp, ia], X[okp, ib]).statistic
        out["agreement"][f"{a} ~ {b}"] = dict(chunk_rho=float(rc), chunks=int(len(cm)), pixel_rho=float(rp), pixels=int(okp.sum()))
        tab.append([f"{a} ~ {b}", len(cm), f"{rc:.2f}", int(okp.sum()), f"{rp:.2f}"])
    print("\n== agreement: Spearman rank correlation, chunk means (~2.3 km) and single 10 m pixels")
    print(tabulate(tab, headers=["pair", "chunks", "rho (chunk)", "pixels", "rho (pixel)"]))

    # the value, fuel and USFS burn probability by CAL FIRE class (pixels)
    ra, fz = X[:, 5], X[:, 4]
    groups = {"FRA (not rated)": ra == RA_CODE["FRA"], "unzoned": fz == 0, "moderate": fz == 1, "high": fz == 2, "very high": fz == 3}
    bp_rank = pd.Series(X[:, 2]).rank(pct=True).to_numpy()
    tab, out["by_fhsz_class"] = [], {}
    for g, m in groups.items():
        r = X[m, 0]
        q = np.nanpercentile(r, [25, 50, 75]) if len(r) else [np.nan] * 3
        res = dict(pixels=int(m.sum()), risk_p25=q[0], risk_median=q[1], risk_p75=q[2], fuel_mean=float(np.nanmean(X[m, 1])),
                   usfs_bp_pct_rank_median=float(np.nanmedian(bp_rank[m])))
        out["by_fhsz_class"][g] = res
        tab.append([g, res["pixels"], f"{q[0] * 100:.0f} / {q[1] * 100:.0f} / {q[2] * 100:.0f}", f"{res['fuel_mean']:.2f}",
                    f"{res['usfs_bp_pct_rank_median']:.2f}"])
    print("\n== by CAL FIRE class (sampled pixels): the value (of 100) 25th / median / 75th, mean fuel, USFS BP median percentile rank")
    print(tabulate(tab, headers=["class", "pixels", "value p25/p50/p75", "fuel", "USFS BP rank"]))

    # where the value and USFS burn probability disagree most (chunk means, percentile ranks)
    df["risk_rank"], df["bp_rank"] = df.risk.rank(pct=True), df.usfs_bp.rank(pct=True)
    df["gap"] = df.risk_rank - df.bp_rank
    keep = ["name", "year", "kind", "lat", "lon", "risk", "fuel", "usfs_bp", "fhsz", "rated", "risk_rank", "bp_rank"]
    out["disagree"] = {}
    for label, sub in [("value high, USFS low", df.nlargest(12, "gap")), ("USFS high, value low", df.nsmallest(12, "gap"))]:
        sub = sub.assign(hash=[f"#13/{la:.4f}/{lo:.4f}" for la, lo in zip(sub.lat, sub.lon)])
        out["disagree"][label] = sub[keep + ["hash"]].to_dict("records")
        print(f"\n== {label} (chunk means; ranks among the {len(df)} chunks; value is the fire year's, from the year before)")
        print(tabulate(sub[["name", "year", "kind", "hash", "risk", "fuel", "usfs_bp", "fhsz", "rated", "risk_rank", "bp_rank"]],
                       headers="keys", showindex=False, floatfmt=".2f"))


def score():
    import psutil
    from tabulate import tabulate
    rng = np.random.default_rng(0)
    fires = rb.Fires(rb.cells_of(set(all_chunks())))
    chunk_parts, leaks = chunk_parts_factory(fires)
    out = dict(layers=dict(
        usfs_bp="USFS Wildfire Risk to Communities burn probability, 2024 edition, 30 m; landscape end of 2020",
        usfs_whp="USFS Wildfire Hazard Potential, 2023 edition, 270 m, continuous; landscape end of 2020",
        fhsz="CAL FIRE Fire Hazard Severity Zones (fhsz24_5, SRA 2024 + LRA 2024-2025), 0 unzoned to 3 very high, SRA/LRA ground only"),
        sets=dict(usfs="pixels where the value and both USFS layers are defined", rated="those on SRA or LRA ground"))
    score_fires(chunk_parts, rng, out, tabulate)
    score_starts(chunk_parts, rng, out, tabulate)
    agreement(chunk_parts, rng, out, tabulate)
    out["leaks"] = leaks[0]
    print(f"\nleak check: {leaks[0]} chunks used a fire of the test year or later in the value's history (must be 0)")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "agency_compare.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote results/agency_compare.json; rss {psutil.Process(os.getpid()).memory_info().rss / 1e9:.2f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "score"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="fetch only this many chunks (a trial)")
    a = ap.parse_args()
    fetch(a.workers, limit=a.limit) if a.step == "fetch" else score()


if __name__ == "__main__":
    main()
