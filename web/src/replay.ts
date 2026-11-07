import { GeoJsonLayer, ScatterplotLayer } from "@deck.gl/layers";
import type { AefTile } from "./aef/layers.js";
import { RES, X0, Y0 } from "./aef/mosaic.js";
import { cutDoc, type Era, type GoesDoc, loadGoesDoc, lonLatToRowCol, readGoesFire } from "./goes.js";
import type { BurnArea } from "./perimeters.js";

/**
 * One fire played back over its ground. The fuel before it (the year before's embeddings) is drawn dim; the ground
 * GOES-West saw burning lights up hot, cools, and settles on the year after's fuel, so the burn scar follows the front.
 *
 * When each 10 m pixel burned: GOES gives the first scan that saw each of its cells (about 2 x 3 km) burning; between
 * cell centers the times are blended (bilinear over the four cells around the pixel, those seen burning only), so the
 * front moves smoothly through a cell instead of the whole cell at once. Only ground inside the fire's final outline
 * burns; inside it, ground with no seen cell around it is given the last hour read.
 */

export type ReplayFire = {
  label: string;
  /** NIFC's name for it in the burned areas (WFIGS) and its year. */
  name: string;
  year: number;
  /** The GOES reference file (web/public/goes/<doc>.json) that covers it. */
  doc: string;
  /** WFIGS's start point and discovery time. */
  start: [number, number];
  discovered: number;
  /** Embedding years: before the fire and after it. */
  before: number;
  after: number;
  /** The final outline's box (west, south, east, north), for the opening view before the outline is read. */
  bbox: [number, number, number, number];
};

export const REPLAY_FIRES: Record<string, ReplayFire> = {
  park: {
    label: "Park Fire",
    name: "Park",
    year: 2024,
    doc: "ca-2024-07-22",
    start: [-121.8026, 39.8186],
    discovered: Date.parse("2024-07-24T21:52:43Z"),
    before: 2023,
    after: 2025,
    bbox: [-122.0641, 39.771, -121.4948, 40.3946],
  },
};

/** The clock starts this many hours before discovery, so the fuel is seen before the fire. */
export const LEAD_H = 6;

/** How long ground stays hot after the front reaches it, then how long it takes to cool to the after year. */
export const HOT_H = 3;
export const COOL_H = 9;
/** No arrival: outside the final outline (not drawn in the replay). */
export const NEVER = -1;

export class Replay {
  /** The hour the clock starts at (discovery, on the hour), ms. */
  readonly t0: number;
  /** First scan that saw each cell burning, hours from t0; key `${row}:${col}` in the era's grid. */
  readonly first = new Map<string, number>();
  /** Hours read so far (the newest scan read, from t0). */
  hoursRead = 0;
  hours = 0;
  scans = 0;
  scansRead = 0;
  done = false;
  /** Bumps when the arrivals change. */
  version = 0;
  private era: Era | null = null;
  /** The outline's polygons, each its rings (NIFC's polygons can overlap, so each is filled on its own). */
  private polys: [number, number][][][] = [];
  readonly bbox: [number, number, number, number];

  constructor(readonly fire: ReplayFire, readonly area: BurnArea) {
    this.t0 = Math.floor(fire.discovered / 3600e3) * 3600e3;
    const g = area.geometry;
    const polys = g.type === "Polygon" ? [g.coordinates] : g.type === "MultiPolygon" ? g.coordinates : [];
    this.polys = polys as [number, number][][][];
    let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
    for (const r of this.polys.flat()) for (const [x, y] of r) { w = Math.min(w, x); e = Math.max(e, x); s = Math.min(s, y); n = Math.max(n, y); }
    this.bbox = [w, s, e, n];
  }

  /** Read the fire's scans (a few KB each) and keep each cell's first time; onUpdate every so often and at the end. */
  async read(onUpdate: () => void, signal: AbortSignal) {
    const full: GoesDoc = await loadGoesDoc(`${import.meta.env.BASE_URL}goes/${this.fire.doc}.json`);
    const m = 0.05;
    const doc = cutDoc(full, [this.bbox[0] - m, this.bbox[1] - m, this.bbox[2] + m, this.bbox[3] + m], this.t0 - 3600e3, Infinity);
    this.era = doc.eras[doc.scans[0]?.era ?? 0] ?? null;
    this.scans = doc.scans.length;
    this.hours = doc.scans.length ? Math.ceil((doc.scans[doc.scans.length - 1]!.t - this.t0) / 3600e3) : 0;
    let last = 0;
    await readGoesFire(doc, (i, px) => {
      this.scansRead++;
      const h = (doc.scans[i]!.t - this.t0) / 3600e3;
      this.hoursRead = Math.max(this.hoursRead, h);
      for (const p of px) {
        const [, row, col] = p.key.split(":");
        const k = `${row}:${col}`;
        const o = this.first.get(k);
        if (o === undefined || h < o) this.first.set(k, Math.max(0, h));
      }
      const now = performance.now();
      if (now - last > 1500) {
        last = now;
        this.version++;
        onUpdate();
      }
      // a few KB a scan: many at once
    }, signal, 32, false);
    this.done = true;
    this.version++;
    onUpdate();
  }

  /** Each pixel's arrival hour for a 10 m tile (NEVER outside the outline), or null when the outline misses it. */
  arrival(t: AefTile): Float32Array | null {
    const era = this.era;
    if (!era) return null;
    const [w, s, e, n] = t.bbox;
    if (e < this.bbox[0] || w > this.bbox[2] || n < this.bbox[1] || s > this.bbox[3]) return null;
    const inside = this.insideMask(t);
    if (!inside) return null;
    const W = t.width, H = t.height;
    // GOES row / col on a coarse grid over the tile, interpolated between (the projection is smooth at this scale)
    const G = 16;
    const gr = new Float32Array((G + 1) * (G + 1)), gc = new Float32Array((G + 1) * (G + 1));
    for (let j = 0; j <= G; j++)
      for (let i = 0; i <= G; i++) {
        const [r, c] = lonLatToRowCol(era, w + ((e - w) * i) / G, n - ((n - s) * j) / G);
        gr[j * (G + 1) + i] = r;
        gc[j * (G + 1) + i] = c;
      }
    const d = this.dense();
    const at = (r: number, c: number) => {
      const i = r - d.r0, j = c - d.c0;
      if (i < 0 || j < 0 || i >= d.nr || j >= d.nc) return undefined;
      const v = d.a[i * d.nc + j]!;
      return Number.isNaN(v) ? undefined : v;
    };
    const late = this.done ? this.hours : this.hoursRead;
    const out = new Float32Array(W * H).fill(NEVER);
    for (let y = 0; y < H; y++) {
      const fy = ((y + 0.5) / H) * G, j0 = Math.min(G - 1, Math.floor(fy)), ty = fy - j0;
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        if (!inside[i]) continue;
        const fx = ((x + 0.5) / W) * G, i0 = Math.min(G - 1, Math.floor(fx)), tx = fx - i0;
        const k = j0 * (G + 1) + i0;
        const lerp = (a: Float32Array) =>
          (a[k]! * (1 - tx) + a[k + 1]! * tx) * (1 - ty) + (a[k + G + 1]! * (1 - tx) + a[k + G + 2]! * tx) * ty;
        const r = lerp(gr), c = lerp(gc);
        const r0 = Math.floor(r), c0 = Math.floor(c), fr = r - r0, fc = c - c0;
        let sum = 0, wsum = 0;
        const add = (rr: number, cc: number, wt: number) => {
          const v = at(rr, cc);
          if (v === undefined || wt <= 0) return;
          sum += v * wt;
          wsum += wt;
        };
        add(r0, c0, (1 - fr) * (1 - fc));
        add(r0, c0 + 1, (1 - fr) * fc);
        add(r0 + 1, c0, fr * (1 - fc));
        add(r0 + 1, c0 + 1, fr * fc);
        out[i] = wsum > 0 ? sum / wsum : late;
      }
    }
    return out;
  }

  /** Whether a lon / lat box touches the final outline (the 10 m tiles that do not are never read). */
  touches = (b: [number, number, number, number]): boolean => {
    const [w, s, e, n] = b;
    if (e < this.bbox[0] || w > this.bbox[2] || n < this.bbox[1] || s > this.bbox[3]) return false;
    // an outline vertex inside the box, or a point of a 5 x 5 grid over the box inside the outline
    for (const ring of this.polys.flat()) for (const [x, y] of ring) if (x >= w && x <= e && y >= s && y <= n) return true;
    for (let j = 0; j <= 4; j++)
      for (let i = 0; i <= 4; i++) {
        const x = w + ((e - w) * i) / 4, y = s + ((n - s) * j) / 4;
        for (const poly of this.polys) {
          let hit = false;
          for (const r of poly)
            for (let a = 0, c = r.length - 1; a < r.length; c = a++) {
              const [xa, ya] = r[a]!, [xc, yc] = r[c]!;
              if ((ya > y) !== (yc > y) && x < ((xc - xa) * (y - ya)) / (yc - ya) + xa) hit = !hit;
            }
          if (hit) return true;
        }
      }
    return false;
  };

  /** The first times on a plain grid over the cells (NaN where none), made once per version. */
  private denseCache: { v: number; r0: number; c0: number; nr: number; nc: number; a: Float32Array } | null = null;
  denseGrid() {
    return this.dense();
  }

  /** The GOES era the scans were read in (projection and grid), once the reference file is in. */
  eraOf(): Era | null {
    return this.era;
  }

  /** One ground point's arrival hour, as the GPU blends it (replay-gpu.ts); the newest hour read where no cell is around. */
  arrivalAt(lon: number, lat: number): number {
    const era = this.era;
    const late = this.done ? this.hours : this.hoursRead;
    if (!era) return late;
    const d = this.dense();
    const [r, c] = lonLatToRowCol(era, lon, lat);
    const fr = r - d.r0, fc = c - d.c0;
    const i0 = Math.floor(fr), j0 = Math.floor(fc), a = fr - i0, b = fc - j0;
    let sum = 0, wsum = 0;
    for (const [di, dj, wt] of [[0, 0, (1 - a) * (1 - b)], [0, 1, (1 - a) * b], [1, 0, a * (1 - b)], [1, 1, a * b]] as const) {
      const i = i0 + di, j = j0 + dj;
      if (i < 0 || j < 0 || i >= d.nr || j >= d.nc) continue;
      const v = d.a[i * d.nc + j]!;
      if (Number.isNaN(v) || wt <= 0) continue;
      sum += v * wt;
      wsum += wt;
    }
    return wsum > 0 ? sum / wsum : late;
  }

  private dense() {
    if (this.denseCache?.v === this.version) return this.denseCache;
    let r0 = Infinity, r1 = -Infinity, c0 = Infinity, c1 = -Infinity;
    const cells = [...this.first].map(([k, h]) => {
      const [r, c] = k.split(":").map(Number) as [number, number];
      r0 = Math.min(r0, r); r1 = Math.max(r1, r); c0 = Math.min(c0, c); c1 = Math.max(c1, c);
      return [r, c, h] as const;
    });
    const nr = cells.length ? r1 - r0 + 1 : 0, nc = cells.length ? c1 - c0 + 1 : 0;
    const a = new Float32Array(nr * nc).fill(NaN);
    for (const [r, c, h] of cells) a[(r - r0) * nc + (c - c0)] = h;
    this.denseCache = { v: this.version, r0, c0, nr, nc, a };
    return this.denseCache;
  }

  /** The final outline: one thin stroke, pale on a dark map and dark gray on a light one, no casing. */
  outlineLayer(beforeId?: string, light = false) {
    return new GeoJsonLayer({
      id: "replay-outline",
      data: [this.area] as never,
      filled: false,
      stroked: true,
      getLineColor: light ? [70, 70, 70, 200] : [236, 236, 236, 170],
      updateTriggers: { getLineColor: [light] },
      getLineWidth: 1.2,
      lineWidthUnits: "pixels",
      beforeId,
    } as never);
  }

  /** 1 where the pixel's center is inside the final outline; null when none of the tile is. */
  private insideMask(t: AefTile): Uint8Array | null {
    const px0 = Math.round((t.bbox[0] - X0) / RES);
    const py0 = Math.round((Y0 - t.bbox[3]) / RES);
    const X = (lon: number) => (lon - X0) / RES - px0;
    const Y = (lat: number) => (Y0 - lat) / RES - py0;
    const cv = new OffscreenCanvas(t.width, t.height);
    const ctx = cv.getContext("2d", { willReadFrequently: true })!;
    ctx.fillStyle = "#fff";
    for (const poly of this.polys) {
      ctx.beginPath();
      for (const ring of poly) {
        ring.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(X(x), Y(y)) : ctx.lineTo(X(x), Y(y))));
        ctx.closePath();
      }
      ctx.fill("evenodd");
    }
    const img = ctx.getImageData(0, 0, t.width, t.height).data;
    const out = new Uint8Array(t.width * t.height);
    let any = false;
    for (let i = 0; i < out.length; i++) if ((out[i] = img[4 * i + 3]! >= 128 ? 1 : 0)) any = true;
    return any ? out : null;
  }

  /** The ignition point: a small dot in the outline's color, a faint ring of the same color around it. */
  startLayer(beforeId?: string, light = false) {
    return new ScatterplotLayer({
      id: "replay-start",
      data: [this.fire.start],
      getPosition: (d: [number, number]) => d,
      radiusUnits: "pixels",
      getRadius: 4,
      filled: true,
      stroked: true,
      getFillColor: light ? [50, 50, 50, 255] : [255, 255, 255, 255],
      getLineColor: light ? [50, 50, 50, 90] : [255, 255, 255, 120],
      updateTriggers: { getFillColor: [light], getLineColor: [light] },
      lineWidthUnits: "pixels",
      getLineWidth: 6,
      beforeId,
    } as never);
  }
}
