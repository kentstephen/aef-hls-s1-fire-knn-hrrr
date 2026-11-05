import type { AefPool } from "../aef/pool.js";
import { CHUNK, RES, X0, Y0, YEARS } from "../aef/mosaic.js";
import type { BurnArea } from "../perimeters.js";
import { pFuel } from "./head.js";

/**
 * The fuel inside one fire's outline, as the mean P(fuel) per embedding year: only the mosaic chunks in view
 * that the outline covers, each pixel counted when its center is inside one of the fire's polygons. The year on
 * screen is read first, then the years either side of it, working outward. Each chunk-year's counts are kept, so
 * panning or stepping the year reads only what is new.
 *
 * Every chunk-year read is the fuel head on that year and the two before it (3 years of embeddings), the same read
 * the map makes for its tiles.
 */

/** The fuel head reads each year with the two before it, so 2019 is the first year it can score. */
export const HIST_YEARS = YEARS.filter((y) => y - 2 >= YEARS[0]!);
/** Chunk-years read at the same time. */
const AT_ONCE = 6;

/** mean: the average P(fuel) over the pixels counted so far (null before any). */
export type YearHist = { year: number; mean: number | null; n: number; done: number; total: number };

type Edge = [x0: number, y0: number, x1: number, y1: number];
/** Each polygon of the fire as its ring edges in mosaic pixel coordinates (x right, y down from Y0). */
type Poly = { edges: Edge[]; y0: number; y1: number; x0: number; x1: number };

function polysOf(f: BurnArea): Poly[] {
  const g = f.geometry;
  const list = g.type === "Polygon" ? [g.coordinates] : g.type === "MultiPolygon" ? g.coordinates : [];
  return list.map((rings) => {
    const edges: Edge[] = [];
    let y0 = Infinity, y1 = -Infinity, x0 = Infinity, x1 = -Infinity;
    for (const r of rings)
      for (let i = 0; i < r.length - 1; i++) {
        const ax = (r[i]![0]! - X0) / RES, ay = (Y0 - r[i]![1]!) / RES;
        const bx = (r[i + 1]![0]! - X0) / RES, by = (Y0 - r[i + 1]![1]!) / RES;
        edges.push([ax, ay, bx, by]);
        y0 = Math.min(y0, ay, by); y1 = Math.max(y1, ay, by); x0 = Math.min(x0, ax, bx); x1 = Math.max(x1, ax, bx);
      }
    return { edges, y0, y1, x0, x1 };
  });
}

/**
 * 1 where the pixel center is inside any of the fire's polygons (even-odd within one polygon, so holes stay out;
 * union across polygons, which NIFC lets overlap). Null when the chunk holds none of the fire.
 */
function chunkMask(polys: Poly[], cy: number, cx: number): Uint8Array | null {
  const r0 = cy * CHUNK, c0 = cx * CHUNK;
  let mask: Uint8Array | null = null;
  const xs: number[] = [];
  for (const p of polys) {
    if (p.y1 < r0 || p.y0 > r0 + CHUNK || p.x1 < c0 || p.x0 > c0 + CHUNK) continue;
    const near = p.edges.filter((e) => Math.max(e[1], e[3]) >= r0 && Math.min(e[1], e[3]) <= r0 + CHUNK);
    for (let r = 0; r < CHUNK; r++) {
      const y = r0 + r + 0.5;
      xs.length = 0;
      for (const [ax, ay, bx, by] of near) if ((ay > y) !== (by > y)) xs.push(ax + ((y - ay) / (by - ay)) * (bx - ax));
      if (xs.length < 2) continue;
      xs.sort((a, b) => a - b);
      for (let k = 0; k + 1 < xs.length; k += 2) {
        const a = Math.max(0, Math.ceil(xs[k]! - c0 - 0.5)), b = Math.min(CHUNK - 1, Math.floor(xs[k + 1]! - c0 - 0.5));
        if (b < a) continue;
        mask ??= new Uint8Array(CHUNK * CHUNK);
        mask.fill(1, r * CHUNK + a, r * CHUNK + b + 1);
      }
    }
  }
  return mask;
}

/** The order the years are read in: the one on screen, then outward (one before, one after, two before, ...). */
export function yearOrder(center: number): number[] {
  return [...HIST_YEARS].sort((a, b) => Math.abs(a - center) - Math.abs(b - center) || a - b);
}

export class BurnHist {
  private fireKey = "";
  private polys: Poly[] = [];
  private masks = new Map<string, Uint8Array | null>();
  /** Per `${year}:${cy}:${cx}`: [pixels counted, sum of their P(fuel)]. */
  private counts = new Map<string, Float64Array>();

  constructor(private pool: AefPool) {}

  /**
   * Reads what is missing for this fire, these chunks (in view) and the years around `center`; onUpdate gets the
   * per-year totals after every chunk-year. Stops quietly when the signal aborts.
   */
  async run(fire: BurnArea, fireKey: string, view: [number, number, number, number], center: number,
    onUpdate: (h: YearHist[]) => void, signal: AbortSignal) {
    if (fireKey !== this.fireKey) {
      this.fireKey = fireKey;
      this.polys = polysOf(fire);
      this.masks.clear();
      this.counts.clear();
    }
    // the chunks in view that the outline covers
    const [w, s, e, n] = view;
    const chunks: [number, number][] = [];
    for (let cy = Math.floor((Y0 - n) / RES / CHUNK); cy <= Math.floor((Y0 - s) / RES / CHUNK); cy++)
      for (let cx = Math.floor((w - X0) / RES / CHUNK); cx <= Math.floor((e - X0) / RES / CHUNK); cx++) {
        const k = `${cy}:${cx}`;
        if (!this.masks.has(k)) this.masks.set(k, chunkMask(this.polys, cy, cx));
        if (this.masks.get(k)) chunks.push([cy, cx]);
      }
    const years = yearOrder(center);
    const report = () =>
      onUpdate(HIST_YEARS.map((year) => {
        let nPx = 0, sum = 0, done = 0;
        for (const [cy, cx] of chunks) {
          const c = this.counts.get(`${year}:${cy}:${cx}`);
          if (!c) continue;
          done++;
          nPx += c[0]!;
          sum += c[1]!;
        }
        return { year, mean: nPx ? sum / nPx : null, n: nPx, done, total: chunks.length };
      }));
    report();
    // the year on screen over every chunk first; then chunk by chunk, each chunk's other years together, so the
    // worker still holds the embedding years they share (each year's fuel reads it and the two before)
    const todo: [number, number, number][] = [];
    const want = (y: number, cy: number, cx: number) => { if (!this.counts.has(`${y}:${cy}:${cx}`)) todo.push([y, cy, cx]); };
    for (const [cy, cx] of chunks) want(years[0]!, cy, cx);
    for (const [cy, cx] of chunks) for (const y of years.slice(1)) want(y, cy, cx);
    let next = 0;
    const t0 = performance.now(), work = { read: 0, score: 0, calls: 0 };
    const worker = async () => {
      while (next < todo.length && !signal.aborted) {
        const [y, cy, cx] = todo[next++]!;
        try {
          const c = await this.pool.zarr(YEARS.indexOf(y), cy * CHUNK, cx * CHUNK, CHUNK, CHUNK, signal, [y - 2, y - 1, y].map((v) => YEARS.indexOf(v)));
          if (c.ms) { work.read += c.ms.read; work.score += c.ms.score; work.calls++; }
          const mask = this.masks.get(`${cy}:${cx}`)!;
          const h = new Float64Array(2);
          for (let i = 0; i < mask.length; i++) {
            if (!mask[i]) continue;
            const p = pFuel(c.fuel[i]!);
            if (p === null) continue;
            h[0]!++;
            h[1]! += p;
          }
          this.counts.set(`${y}:${cy}:${cx}`, h);
          report();
        } catch (err) {
          if ((err as Error).name === "AbortError") return;
          console.warn("fire histogram:", err);
        }
      }
    };
    await Promise.all(Array.from({ length: AT_ONCE }, worker));
    if (todo.length && !signal.aborted) {
      const r = { chunkYears: todo.length, wallS: (performance.now() - t0) / 1e3, readS: work.read / 1e3, scoreS: work.score / 1e3, calls: work.calls };
      (globalThis as { __burnHistTiming?: unknown }).__burnHistTiming = r;
      console.info("fire bars read:", r);
    }
  }
}
