# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "rasterio", "shapely", "pyogrio"]
# ///
"""California's Census outline (the one goes_events.py keeps fire pixels inside), simplified to about 200 m, for the
web dashboard: web/public/ca.json, a list of rings [[lon, lat], ...] (outer rings only; California has no holes).

    uv run scripts/ca_outline.py
"""
import json
import sys
from pathlib import Path

import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_fuel_windows import ca_polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
g = shapely.simplify(ca_polygon(), 0.002)
polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
rings = [[[round(x, 4), round(y, 4)] for x, y in p.exterior.coords] for p in polys]
out = ROOT / "web" / "public" / "ca.json"
out.write_text(json.dumps(rings, separators=(",", ":")))
print(f"{len(rings)} rings, {sum(map(len, rings))} points, {out.stat().st_size / 1e3:.0f} KB -> {out}")
