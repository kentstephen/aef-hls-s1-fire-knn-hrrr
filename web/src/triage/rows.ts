import scoreJson from "../../../models/goes_score.json";
import type { FirePixel } from "../goes.js";
import type { GoesEvent } from "../goesboxes.js";

/**
 * One table row per GOES event (goesboxes.ts goesEvents), with the GOES-only triage score: a logistic regression on
 * the event's first scan (scripts/goes_score_fit.py -> models/goes_score.json), the chance it is a recorded wildfire.
 * Fuel columns stay empty until the embeddings are read.
 */

export type TriageRow = {
  /** Stays the same as more scans join the event: its first pixel at its first time. */
  key: string;
  ev: GoesEvent;
  lon: number;
  lat: number;
  t0: number;
  t1: number;
  scans: number;
  pixels: number;
  areaKm2: number;
  peakMw: number;
  firstMw: number;
  firstStrength: number;
  firstPx: number;
  score: number;
  fuelUnder: number | null;
  fuelAround: number | null;
  /** waiting: not read yet; reading: embeddings being read now; done; failed. */
  fuelState: "waiting" | "reading" | "done" | "failed";
  /** Chunks read of those the event touches (big events are sampled). */
  fuelChunks?: [read: number, touched: number];
};

/** FDC mask classes to strength (higher = stronger), as scripts/goes_events.py; the 30s map to their 10s. */
const STRENGTH: Record<number, number> = { 10: 6, 11: 6, 13: 5, 14: 4, 12: 3, 15: 2 };

const S = scoreJson as { mean: number[]; scale: number[]; coef: number[]; intercept: number };

export function goesScore(firstMw: number, firstStrength: number, firstPx: number): number {
  const x = [Math.log1p(Number.isNaN(firstMw) ? 0 : firstMw), firstStrength, Math.log(firstPx)];
  let z = S.intercept;
  for (let i = 0; i < 3; i++) z += (S.coef[i]! * (x[i]! - S.mean[i]!)) / S.scale[i]!;
  return 1 / (1 + Math.exp(-z));
}

/** A footprint's area in km2 (small quadrilateral: flat earth at its latitude). */
function areaKm2(poly: [number, number][]): number {
  const k = Math.cos((poly[0]![1] * Math.PI) / 180) * 111.32;
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x0, y0] = poly[i]!, [x1, y1] = poly[(i + 1) % poly.length]!;
    a += x0 * k * y1 * 111.32 - x1 * k * y0 * 111.32;
  }
  return Math.abs(a) / 2;
}

export function triageRow(ev: GoesEvent): TriageRow {
  const firstKeys = new Set<string>();
  let firstMw = NaN, firstStrength = 0, peakMw = NaN, firstKey = "";
  const fp = new Map<string, FirePixel>();
  for (const p of ev.px) {
    fp.set(p.key, p);
    // (NaN power = not computed; a NaN max takes the first real value)
    if (!Number.isNaN(p.power) && !(p.power <= peakMw)) peakMw = p.power;
    if (p.t !== ev.t0) continue;
    firstKeys.add(p.key);
    if (!firstKey || p.key < firstKey) firstKey = p.key;
    if (!Number.isNaN(p.power) && !(p.power <= firstMw)) firstMw = p.power;
    firstStrength = Math.max(firstStrength, STRENGTH[p.mask % 20] ?? 0);
  }
  let area = 0, sx = 0, sy = 0;
  for (const p of fp.values()) {
    area += areaKm2(p.polygon);
    for (const [x, y] of p.polygon) { sx += x / 4; sy += y / 4; }
  }
  return {
    key: `${ev.t0}|${firstKey}`,
    ev,
    lon: sx / fp.size,
    lat: sy / fp.size,
    t0: ev.t0,
    t1: ev.t1,
    scans: ev.scans,
    pixels: ev.pixels,
    areaKm2: area,
    peakMw,
    firstMw,
    firstStrength,
    firstPx: firstKeys.size,
    score: goesScore(firstMw, firstStrength, firstKeys.size),
    fuelUnder: null,
    fuelAround: null,
    fuelState: "waiting",
  };
}
