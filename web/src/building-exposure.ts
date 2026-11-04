import type { AefTile } from "./aef/layers.js";
import type { Rings } from "./buildings.js";

/** The fuel plane's pixel, 10 m: the step the ground around a building is sampled at. */
const STEP_M = 10;
const M_PER_DEG_LAT = 111_320;

/**
 * Per building: how much burnable ground is close. The highest P(fuel) (the fuel head, 10 m mosaic tiles only)
 * among the pixels within reachM of the building's box. NaN where no fuel plane is loaded under any of those
 * pixels yet.
 */
export function buildingExposure(rings: Rings, tiles: AefTile[], reachM: number): Float32Array {
  const out = new Float32Array(rings.count).fill(NaN);
  const fine = tiles.filter((t) => t.kind === "zarr" && !t.gone);
  if (fine.length === 0) return out;
  let last: AefTile | null = null;
  /** P(fuel) at a point, -1 where no fuel plane covers it. */
  const fuelAt = (lon: number, lat: number): number => {
    if (last) {
      const i = last.pixelAt(lon, lat);
      if (i >= 0) {
        const b = last.classes.fuel[i]!;
        if (b > 0) return (b - 1) / 254;
      }
    }
    for (const t of fine) {
      if (t === last) continue;
      const [w, s, e, n] = t.bbox;
      if (lon < w || lon > e || lat < s || lat > n) continue;
      const i = t.pixelAt(lon, lat);
      if (i < 0) continue;
      const b = t.classes.fuel[i]!;
      if (b === 0) continue;
      last = t;
      return (b - 1) / 254;
    }
    return -1;
  };

  const { coords, ringOffsets, buildingRings } = rings;
  for (let bi = 0; bi < rings.count; bi++) {
    let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
    for (let r = buildingRings[bi]!; r < buildingRings[bi + 1]!; r++) {
      for (let v = ringOffsets[r]!; v < ringOffsets[r + 1]!; v++) {
        const x = coords[2 * v]!, y = coords[2 * v + 1]!;
        if (x < w) w = x;
        if (x > e) e = x;
        if (y < s) s = y;
        if (y > n) n = y;
      }
    }
    if (!(w <= e)) continue;
    const mLon = M_PER_DEG_LAT * Math.cos(((s + n) / 2) * (Math.PI / 180));
    const dLat = STEP_M / M_PER_DEG_LAT;
    const dLon = STEP_M / mLon;
    const rLat = reachM / M_PER_DEG_LAT;
    const rLon = reachM / mLon;
    let best = NaN;
    for (let lat = s - rLat; lat <= n + rLat + 1e-12; lat += dLat) {
      const dy = lat < s ? (s - lat) * M_PER_DEG_LAT : lat > n ? (lat - n) * M_PER_DEG_LAT : 0;
      for (let lon = w - rLon; lon <= e + rLon + 1e-12; lon += dLon) {
        const dx = lon < w ? (w - lon) * mLon : lon > e ? (lon - e) * mLon : 0;
        const d = Math.hypot(dx, dy);
        if (d > reachM + 0.01) continue;
        const p = fuelAt(lon, lat);
        if (p < 0) continue;
        if (!(p <= best)) best = p;
      }
    }
    out[bi] = best;
  }
  return out;
}

/** Outline colors from exposure: dark gray below the hide line (or with no ground read), the ground's ramp above it. */
export function exposureColors(
  exp: Float32Array, pHide: number, pMax: number, stripe: Uint8Array, gray: [number, number, number], logOdds = false,
): Uint8Array {
  const out = new Uint8Array(exp.length * 4);
  // the same scale as the ground (render.ts): straight, or on the log-odds clamped to the byte's range
  const logit = (p: number) => {
    const q = Math.min(0.996, Math.max(0.002, p));
    return Math.log(q / (1 - q));
  };
  const at = logOdds
    ? (v: number) => (logit(v) - logit(pHide)) / Math.max(logit(pMax) - logit(pHide), 0.01)
    : (v: number) => (v - pHide) / Math.max(pMax - pHide, 1e-4);
  for (let i = 0; i < exp.length; i++) {
    const v = exp[i]!;
    const o = i * 4;
    if (!(v >= pHide)) {
      out[o] = gray[0];
      out[o + 1] = gray[1];
      out[o + 2] = gray[2];
      out[o + 3] = 255;
      continue;
    }
    const t = Math.min(1, Math.max(0, at(v)));
    const k = Math.round(t * 255) * 4;
    out[o] = stripe[k]!;
    out[o + 1] = stripe[k + 1]!;
    out[o + 2] = stripe[k + 2]!;
    out[o + 3] = 255;
  }
  return out;
}
