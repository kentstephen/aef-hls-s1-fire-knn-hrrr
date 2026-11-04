# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "pyproj", "pillow", "tifffile"]
# ///
"""NAIP photo panels for the pixels in data/naip_check/picks.parquet (from naip_pick.py).

Source: USGS National Map NAIP image service (newest year per state; California is 2022, 0.6 m, 4 bands R G B NIR).
It reprojects on request, so each crop comes back in EPSG:5070, the LANDFIRE grid, and the LANDFIRE 30 m cell can be
outlined exactly (LANDFIRE cells sit on x = -2362425 + 30 i, y = 3267405 - 30 j).

Per pixel, one PNG with three panels, 300 px each:
  1. close-up natural color, 120 m across (0.6 m NAIP pixels, shown at 0.4 m)
  2. same close-up as greenness (NDVI from NIR and red, white = none, dark green = dense live plants)
  3. context natural color, 600 m across, with the close-up area outlined
Outlines are single strokes, no casing: yellow = the 10 m AEF cell, blue = the 30 m LANDFIRE cell, white = close-up area.
The panel shows only the chip id: the LANDFIRE class and the head's call stay out of view so labels are blind.

Raw 4-band crops go to data/naip_check/raw/<id>.npz, panels to data/naip_check/chips/<id>.png,
acquisition dates to data/naip_check/naip_dates.parquet. Skips ids already done.

    uv run scripts/naip_chips.py [--workers 4] [--only n000,n001]
"""
import argparse
import io
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "naip_check"
SVC = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer"
MOS_RES = 8.983111749910169e-5
LF_X0, LF_Y0 = -2362425, 3267405
CLOSE_M, CLOSE_PX = 120, 200
CTX_M, CTX_PX = 600, 300
SHOW = 300
YELLOW, BLUE, WHITE = (255, 214, 0), (40, 120, 255), (255, 255, 255)
TO5070 = Transformer.from_crs(4326, 5070, always_xy=True)


def fetch(params, tries=4):
    url = SVC + "/" + params.pop("_op") + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=90).read()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
            print(f"    retry {i + 1}: {e}", flush=True)


def crop(x, y, half, px):
    b = fetch(dict(_op="exportImage", bbox=f"{x - half},{y - half},{x + half},{y + half}", bboxSR=5070, imageSR=5070,
                   size=f"{px},{px}", format="tiff", pixelType="U8", bandIds="0,1,2,3",
                   renderingRule=json.dumps({"rasterFunction": "None"}), interpolation="RSP_BilinearInterpolation",
                   f="image"))
    a = tifffile.imread(io.BytesIO(b))
    if a.ndim != 3 or a.shape[-1] != 4:
        raise ValueError(f"unexpected crop shape {a.shape}")
    return a


def naip_date(lon, lat):
    g = json.dumps({"x": lon, "y": lat, "spatialReference": {"wkid": 4326}})
    d = json.loads(fetch(dict(_op="identify", geometry=g, geometryType="esriGeometryPoint", returnCatalogItems="true",
                              returnGeometry="false", f="json")))
    for f in (d.get("catalogItems") or {}).get("features", []):
        a = f["attributes"]
        if a.get("Category") == 1 and a.get("acquisition_date"):
            return a["Name"], pd.Timestamp(a["acquisition_date"], unit="ms").date().isoformat(), a.get("Year")
    return None, None, None


def stretch(rgb):
    lo, hi = np.percentile(rgb, [1, 99])
    return (np.clip((rgb.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)


def greens(v):
    """NDVI -0.1 .. 0.7 on a white -> dark green ramp (matplotlib Greens end points)."""
    t = np.clip((v + 0.1) / 0.8, 0, 1)[..., None]
    lo, hi = np.array([247, 252, 245], np.float32), np.array([0, 68, 27], np.float32)
    return (lo + (hi - lo) * t).astype(np.uint8)


def to_px(X, Y, x, y, half, px):
    s = SHOW / (2 * half)  # display px per meter
    return (X - (x - half)) * s, ((y + half) - Y) * s


def panel(rec, close, ctx):
    x, y = rec["x"], rec["y"]
    lon, lat = rec["lon"], rec["lat"]
    h = MOS_RES / 2
    aef = [TO5070.transform(a, b) for a, b in [(lon - h, lat - h), (lon + h, lat - h), (lon + h, lat + h), (lon - h, lat + h)]]
    i, j = np.floor((x - LF_X0) / 30), np.floor((LF_Y0 - y) / 30)
    lx0, ly1 = LF_X0 + 30 * i, LF_Y0 - 30 * j
    lf = [(lx0, ly1 - 30), (lx0 + 30, ly1 - 30), (lx0 + 30, ly1), (lx0, ly1)]

    r, nir = close[..., 0].astype(np.float32), close[..., 3].astype(np.float32)
    ndvi = (nir - r) / np.maximum(nir + r, 1)
    ims = [Image.fromarray(stretch(close[..., :3])), Image.fromarray(greens(ndvi)), Image.fromarray(stretch(ctx[..., :3]))]
    ims = [im.resize((SHOW, SHOW), Image.Resampling.BILINEAR) for im in ims]
    for k in (0, 1):
        d = ImageDraw.Draw(ims[k])
        d.polygon([to_px(X, Y, x, y, CLOSE_M / 2, CLOSE_PX) for X, Y in lf], outline=BLUE, width=2)
        d.polygon([to_px(X, Y, x, y, CLOSE_M / 2, CLOSE_PX) for X, Y in aef], outline=YELLOW, width=2)
    d = ImageDraw.Draw(ims[2])
    c = [to_px(X, Y, x, y, CTX_M / 2, CTX_PX) for X, Y in
         [(x - CLOSE_M / 2, y - CLOSE_M / 2), (x + CLOSE_M / 2, y - CLOSE_M / 2), (x + CLOSE_M / 2, y + CLOSE_M / 2),
          (x - CLOSE_M / 2, y + CLOSE_M / 2)]]
    d.polygon(c, outline=WHITE, width=1)
    d.polygon([to_px(X, Y, x, y, CTX_M / 2, CTX_PX) for X, Y in lf], outline=YELLOW, width=2)

    head = 22
    out = Image.new("RGB", (3 * SHOW + 8, SHOW + head), (255, 255, 255))
    ImageDraw.Draw(out).text((4, 5), f"{rec['id']}   close-up 120 m  |  greenness  |  context 600 m", fill=(0, 0, 0))
    for k, im in enumerate(ims):
        out.paste(im, (k * (SHOW + 4), head))
    return out


def do_one(rec):
    t0 = time.time()
    close = crop(rec["x"], rec["y"], CLOSE_M / 2, CLOSE_PX)
    ctx = crop(rec["x"], rec["y"], CTX_M / 2, CTX_PX)
    name, date, year = naip_date(rec["lon"], rec["lat"])
    np.savez_compressed(OUT / "raw" / f"{rec['id']}.npz", close=close, ctx=ctx)
    panel(rec, close, ctx).save(OUT / "chips" / f"{rec['id']}.png", optimize=True)
    blank = float((close[..., :3].max(axis=-1) == 0).mean())
    return dict(id=rec["id"], naip_name=name, naip_date=date, naip_year=year, blank_share=blank, sec=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    (OUT / "chips").mkdir(parents=True, exist_ok=True)
    p = pd.read_parquet(OUT / "picks.parquet")
    p["x"], p["y"] = TO5070.transform(p.lon.to_numpy(), p.lat.to_numpy())
    dates_f = OUT / "naip_dates.parquet"
    old = pd.read_parquet(dates_f) if dates_f.exists() else pd.DataFrame(columns=["id"])
    todo = p[~p.id.isin(old.id) & ~p.id.map(lambda i: (OUT / "chips" / f"{i}.png").exists())]
    if a.only:
        todo = p[p.id.isin(a.only.split(","))]
    recs = todo[["id", "x", "y", "lon", "lat"]].to_dict("records")
    print(f"{len(recs)} panels to make ({len(p) - len(todo)} done)", flush=True)
    rows, t0 = [], time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for n, r in enumerate(ex.map(do_one, recs)):
            rows.append(r)
            if (n + 1) % 25 == 0 or n + 1 == len(recs):
                print(f"  {n + 1}/{len(recs)} {time.time() - t0:.0f} s", flush=True)
                pd.concat([old, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("id", keep="last").to_parquet(dates_f, index=False)
    d = pd.read_parquet(dates_f)
    print(f"NAIP years: {d.naip_year.value_counts().to_dict()}; panels with >5 % blank: {(d.blank_share > 0.05).sum()}")


if __name__ == "__main__":
    main()
