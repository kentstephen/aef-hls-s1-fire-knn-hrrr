/// <reference lib="webworker" />
import { contours } from "d3-contour";

/**
 * How far a fire could reach from the GOES pixels burning this hour, over a grid of cells around the fire.
 *
 * A shortest-path spread (minimum travel time, as FlamMap does it, without wind or moisture): every cell costs its
 * width divided by a spread rate. The rate follows the ground:
 *   fuel   the fuel head's P(fuel) under the cell; ground that is not fuel spreads at a tenth of the rate of full fuel
 *   slope  uphill steps go faster by Rothermel's (1972) slope factor, 1 + K tan^2(slope), capped; downhill no change
 * Each burning pixel gets a budget of reach (in meters of flat full fuel) that grows the longer the pixel has burned
 * and shrinks after it stops. A cell's glow is how much budget is left when the spread arrives there (1 at the
 * fire, 0 at the edge of the reach).
 *
 * The result is relative reach, not hours: hours would need wind and fuel moisture.
 */

export type ReachInit = {
  type: "init";
  nx: number;
  ny: number;
  cellM: number;
  /** Elevation (m) per cell, row 0 at the north. */
  elev: Float32Array;
};
export type ReachFuel = { type: "fuel"; fuel: Float32Array };
/** Source cells and each one's budget (m); out to maxM at most. */
export type ReachRun = { type: "run"; id: number; cells: Int32Array; budget: Float32Array; maxM: number; steps: number[] };
export type ReachOut = { id: number; glow: Uint8Array; lines: { level: number; rings: [number, number][][] }[]; ms: number };

/** Rothermel's slope coefficient 5.275 * beta^-0.3 for a packing ratio near 0.01 (brush and timber litter): about 20. */
const K_SLOPE = 20;
/** The most an uphill step speeds the spread: three times flat ground. */
const SLOPE_CAP = 3;
/** Ground that is not fuel spreads at this share of full fuel's rate. */
const NONFUEL_RATE = 0.1;

let nx = 0, ny = 0, cellM = 0;
let elev: Float32Array = new Float32Array(0);
let fuel: Float32Array = new Float32Array(0);

/** A binary min-heap of (cost, cell) in typed arrays. */
class Heap {
  cost = new Float32Array(1 << 16);
  cell = new Int32Array(1 << 16);
  n = 0;
  push(c: number, i: number) {
    if (this.n === this.cost.length) {
      const c2 = new Float32Array(this.n * 2); c2.set(this.cost); this.cost = c2;
      const i2 = new Int32Array(this.n * 2); i2.set(this.cell); this.cell = i2;
    }
    let k = this.n++;
    while (k > 0) {
      const p = (k - 1) >> 1;
      if (this.cost[p]! <= c) break;
      this.cost[k] = this.cost[p]!; this.cell[k] = this.cell[p]!;
      k = p;
    }
    this.cost[k] = c; this.cell[k] = i;
  }
  pop(): number {
    const top = this.cell[0]!;
    const c = this.cost[--this.n]!, i = this.cell[this.n]!;
    let k = 0;
    for (;;) {
      let m = 2 * k + 1;
      if (m >= this.n) break;
      if (m + 1 < this.n && this.cost[m + 1]! < this.cost[m]!) m++;
      if (this.cost[m]! >= c) break;
      this.cost[k] = this.cost[m]!; this.cell[k] = this.cell[m]!;
      k = m;
    }
    this.cost[k] = c; this.cell[k] = i;
    return top;
  }
  topCost() { return this.cost[0]!; }
}

const NB: [number, number, number][] = [[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1], [1, 1, Math.SQRT2], [1, -1, Math.SQRT2], [-1, 1, Math.SQRT2], [-1, -1, Math.SQRT2]];

function run(m: ReachRun): ReachOut {
  const t0 = performance.now();
  const n = nx * ny;
  // cost so far, started at maxM - budget so a pixel with more budget reaches farther
  const cost = new Float32Array(n).fill(Infinity);
  const heap = new Heap();
  for (let k = 0; k < m.cells.length; k++) {
    const i = m.cells[k]!;
    const c = Math.max(0, m.maxM - m.budget[k]!);
    if (c < cost[i]!) { cost[i] = c; heap.push(c, i); }
  }
  const done = new Uint8Array(n);
  while (heap.n) {
    const c = heap.topCost();
    const i = heap.pop();
    if (done[i]) continue;
    done[i] = 1;
    if (c > m.maxM) break;
    const x = i % nx, y = (i - x) / nx;
    for (const [dx, dy, dl] of NB) {
      const x2 = x + dx, y2 = y + dy;
      if (x2 < 0 || y2 < 0 || x2 >= nx || y2 >= ny) continue;
      const j = y2 * nx + x2;
      if (done[j]) continue;
      const run = dl * cellM;
      const rise = elev[j]! - elev[i]!;
      const slope = rise > 0 ? Math.min(SLOPE_CAP, 1 + K_SLOPE * (rise / run) ** 2) : 1;
      const f = fuel[j]!;
      const rate = (NONFUEL_RATE + (1 - NONFUEL_RATE) * (Number.isNaN(f) ? 0.5 : f)) * slope;
      const c2 = c + run / rate;
      if (c2 < cost[j]!) { cost[j] = c2; heap.push(c2, j); }
    }
  }
  const glow = new Uint8Array(n);
  const val = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const v = cost[i]! <= m.maxM ? 1 - cost[i]! / m.maxM : 0;
    val[i] = v;
    glow[i] = Math.round(v * 255);
  }
  // the isochrone lines: where the spread has used up a given share of the largest budget
  const lines = m.steps.map((s) => {
    const level = 1 - s / m.maxM;
    const g = contours().size([nx, ny]).thresholds([level])(val as unknown as number[])[0]!;
    const rings: [number, number][][] = [];
    for (const poly of g.coordinates) for (const ring of poly) rings.push(ring as [number, number][]);
    return { level: s, rings };
  });
  return { id: m.id, glow, lines, ms: performance.now() - t0 };
}

self.onmessage = (e: MessageEvent<ReachInit | ReachFuel | ReachRun>) => {
  const m = e.data;
  if (m.type === "init") {
    nx = m.nx; ny = m.ny; cellM = m.cellM; elev = m.elev;
    fuel = new Float32Array(nx * ny).fill(NaN);
  } else if (m.type === "fuel") {
    fuel = m.fuel;
  } else {
    const out = run(m);
    (self as unknown as Worker).postMessage(out, [out.glow.buffer]);
  }
};
