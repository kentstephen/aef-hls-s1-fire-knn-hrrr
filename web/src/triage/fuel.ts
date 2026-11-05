import type { AefPool } from "../aef/pool.js";
import { CHUNK, RES, X0, Y0, YEARS } from "../aef/mosaic.js";
import { pFuel } from "../fuel/head.js";
import type { TriageRow } from "./rows.js";

/**
 * Fuel under and around one GOES event, from the embeddings: the fuel head (models/fuel_head.json) on every 10 m
 * pixel of the mosaic chunks the event touches, run in the AefPool workers. As scripts/goes_fuel.py:
 *   under   pixels inside a hot GOES pixel's footprint
 *   around  pixels within 1.5 GOES pixels of one, not under (the footprint grown 3 times about its center)
 * each the mean P(fuel), i.e. the expected share of fuel. Embeddings of the year before the fires, the fuel head's
 * spread over that year and the two before it.
 *
 * Big events touch hundreds of chunks: at most MAX_CHUNKS are read, spread evenly over the ones that touch.
 */

export const MAX_CHUNKS = 24;

export type EventFuel = { under: number | null; around: number | null; chunks: number; chunksTouched: number };

type Quad = [number, number][];

/** A quad grown k times about its center. */
const grow = (q: Quad, k: number): Quad => {
  const cx = q.reduce((a, p) => a + p[0], 0) / q.length, cy = q.reduce((a, p) => a + p[1], 0) / q.length;
  return q.map(([x, y]) => [cx + (x - cx) * k, cy + (y - cy) * k]);
};

/** Fill a convex quad (mosaic pixel coordinates, relative to the chunk) into the chunk's mask with value v. */
function fillQuad(mask: Uint8Array, q: Quad, v: number) {
  const ys = q.map((p) => p[1]);
  const r0 = Math.max(0, Math.floor(Math.min(...ys))), r1 = Math.min(CHUNK - 1, Math.ceil(Math.max(...ys)));
  for (let r = r0; r <= r1; r++) {
    const y = r + 0.5;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < q.length; i++) {
      const [xa, ya] = q[i]!, [xb, yb] = q[(i + 1) % q.length]!;
      if ((ya <= y && yb > y) || (yb <= y && ya > y)) {
        const x = xa + ((y - ya) / (yb - ya)) * (xb - xa);
        if (x < lo) lo = x;
        if (x > hi) hi = x;
      }
    }
    if (hi < lo) continue;
    const c0 = Math.max(0, Math.ceil(lo - 0.5)), c1 = Math.min(CHUNK - 1, Math.floor(hi - 0.5));
    for (let c = c0; c <= c1; c++) if (mask[r * CHUNK + c]! < v) mask[r * CHUNK + c] = v;
  }
}

export async function eventFuel(pool: AefPool, row: TriageRow, embYear: number, signal?: AbortSignal): Promise<EventFuel> {
  // each hot footprint once, in mosaic pixel coordinates
  const seen = new Map<string, Quad>();
  for (const p of row.ev.px) if (!seen.has(p.key)) seen.set(p.key, p.polygon.map(([lon, lat]) => [(lon - X0) / RES, (Y0 - lat) / RES]));
  const under = [...seen.values()];
  const ring = under.map((q) => grow(q, 3));

  // the chunks any grown footprint touches
  const touched = new Set<string>();
  for (const q of ring) {
    const xs = q.map((p) => p[0]), ys = q.map((p) => p[1]);
    for (let cy = Math.floor(Math.min(...ys) / CHUNK); cy <= Math.floor(Math.max(...ys) / CHUNK); cy++)
      for (let cx = Math.floor(Math.min(...xs) / CHUNK); cx <= Math.floor(Math.max(...xs) / CHUNK); cx++) touched.add(`${cy}:${cx}`);
  }
  let chunks = [...touched].sort().map((k) => k.split(":").map(Number) as [number, number]);
  const chunksTouched = chunks.length;
  if (chunks.length > MAX_CHUNKS) {
    const step = chunks.length / MAX_CHUNKS;
    chunks = Array.from({ length: MAX_CHUNKS }, (_, i) => chunks[Math.floor(i * step)]!);
  }

  const t = YEARS.indexOf(embYear);
  const years = [embYear - 2, embYear - 1, embYear].map((y) => YEARS.indexOf(y));
  if (t < 0 || years.some((i) => i < 0)) throw new Error(`no embeddings for ${embYear}`);
  let sumU = 0, nU = 0, sumA = 0, nA = 0;
  await Promise.all(
    chunks.map(async ([cy, cx]) => {
      const c = await pool.zarr(t, cy * CHUNK, cx * CHUNK, CHUNK, CHUNK, signal, years);
      const mask = new Uint8Array(CHUNK * CHUNK);
      const rel = (q: Quad): Quad => q.map(([x, y]) => [x - cx * CHUNK, y - cy * CHUNK]);
      for (const q of ring) fillQuad(mask, rel(q), 1);
      for (const q of under) fillQuad(mask, rel(q), 2);
      for (let i = 0; i < mask.length; i++) {
        const m = mask[i]!;
        if (!m) continue;
        const p = pFuel(c.fuel[i]!);
        if (p === null) continue;
        if (m === 2) { sumU += p; nU++; }
        else { sumA += p; nA++; }
      }
    }),
  );
  return { under: nU ? sumU / nU : null, around: nA ? sumA / nA : null, chunks: chunks.length, chunksTouched };
}
