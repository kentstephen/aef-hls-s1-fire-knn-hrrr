import { CHUNK, RES, X0, Y0, YEARS } from "../aef/mosaic.js";
import type { AefPool } from "../aef/pool.js";
import { pFuel } from "../fuel/head.js";
import { type ChunkHistory, type Fire, FireHistory } from "./history.js";
import { type Parts, parts, VIRIDIS } from "./score.js";
import type { RiskGrid } from "./shade.js";
import { type ChunkTerrain, chunkTerrain } from "./terrain.js";

/**
 * The factors under every 10 m pixel of the chunks in view, read on demand as the map moves (nothing precomputed):
 *   fuel        P(fuel), the fuel head on the newest embedding year and the two before it
 *   continuity  from the fuel, on a coarse grid of 8 x 8 pixel cells (about 80 m by 60 m): the share of fuel
 *               within CONT_RADIUS_M, and the connected fuel patch (cells at least half fuel, joined side by side)
 *   terrain     slope and aspect (terrain.ts)
 *   history     last burn year and fire count (history.ts)
 */

export const EMB_YEAR = 2025;
/** Chunks read at the same time. */
const AT_ONCE = 6;
/** Pixels per coarse cell side. */
const COARSE = 8;
const NC = CHUNK / COARSE;
export const CONT_RADIUS_M = 500;
/** A coarse cell counts toward a patch when at least this share of it is fuel. */
const PATCH_SHARE = 0.5;
const M_LAT = 111_320;
/** Chunks kept (about 1 MB each); past this the ones farthest from the view are let go. */
const KEEP = 400;
const NOW_YEAR = new Date().getFullYear();
let gridSeq = 0;

export type ChunkFactors = {
  cy: number;
  cx: number;
  /** Fuel head score bytes (0 = no embedding, else 1 + round(P * 254)). */
  fuel: Uint8Array;
  terrain: ChunkTerrain;
  history: ChunkHistory;
  /** Per coarse cell: the share of fuel (NaN where no pixel has an embedding). */
  coarse: Float32Array;
  /** Per pixel: the share of fuel within CONT_RADIUS_M, from the chunks read when last drawn. */
  cont?: Float32Array;
  /** The risk per pixel: 0 = none, else 1 + round(risk * 254). */
  risk?: Uint8Array;
  /** The risk picture (256 x 256), redrawn when a neighbor arrives. */
  image?: ImageBitmap;
  bounds: [number, number, number, number];
};

export type PointFactors = {
  fuel: number | null;
  /** Share of fuel within CONT_RADIUS_M (null where nothing near is read). */
  near: number | null;
  /** The connected fuel patch: km2, and whether it runs past the chunks read (so is at least that big). */
  patchKm2: number | null;
  patchOpen: boolean;
  slope: number;
  aspect: number;
  last: number;
  count: number;
  fires: Fire[];
  /** The risk and its four parts (null where the fuel or continuity is missing). */
  parts: Parts | null;
};

const key = (cy: number, cx: number) => `${cy}:${cx}`;

function coarseOf(fuel: Uint8Array): Float32Array {
  const out = new Float32Array(NC * NC);
  for (let i = 0; i < NC; i++)
    for (let j = 0; j < NC; j++) {
      let s = 0, n = 0;
      for (let r = i * COARSE; r < (i + 1) * COARSE; r++)
        for (let c = j * COARSE; c < (j + 1) * COARSE; c++) {
          const p = pFuel(fuel[r * CHUNK + c]!);
          if (p === null) continue;
          s += p;
          n++;
        }
      out[i * NC + j] = n ? s / n : NaN;
    }
  return out;
}

export class RiskFactors {
  readonly chunks = new Map<string, ChunkFactors>();
  private reading = new Map<string, Promise<void>>();
  readonly history: FireHistory;
  /** Bumped whenever a chunk's picture changes. */
  version = 0;

  constructor(private pool: AefPool, history: FireHistory, private onDraw: () => void) {
    this.history = history;
  }

  /** The chunks a lon / lat box covers, nearest its center first. */
  static chunksIn(bbox: [number, number, number, number]): [number, number][] {
    const [w, s, e, n] = bbox;
    const cy0 = Math.floor((Y0 - n) / RES / CHUNK), cy1 = Math.floor((Y0 - s) / RES / CHUNK);
    const cx0 = Math.floor((w - X0) / RES / CHUNK), cx1 = Math.floor((e - X0) / RES / CHUNK);
    const out: [number, number][] = [];
    for (let cy = cy0; cy <= cy1; cy++) for (let cx = cx0; cx <= cx1; cx++) out.push([cy, cx]);
    const my = (cy0 + cy1) / 2, mx = (cx0 + cx1) / 2;
    return out.sort((a, b) => Math.hypot(a[0] - my, a[1] - mx) - Math.hypot(b[0] - my, b[1] - mx));
  }

  private readChunk(cy: number, cx: number, signal: AbortSignal): Promise<void> {
    const k = key(cy, cx);
    let p = this.reading.get(k);
    if (p) return p;
    const lon0 = X0 + cx * CHUNK * RES, lat1 = Y0 - cy * CHUNK * RES;
    const box: [number, number, number, number] = [lon0, lat1 - CHUNK * RES, lon0 + CHUNK * RES, lat1];
    const years = [EMB_YEAR - 2, EMB_YEAR - 1, EMB_YEAR].map((y) => YEARS.indexOf(y));
    p = Promise.all([
      this.pool.zarr(YEARS.indexOf(EMB_YEAR), cy * CHUNK, cx * CHUNK, CHUNK, CHUNK, signal, years),
      chunkTerrain(cy, cx),
      this.history.ensure(box, signal).then(() => this.history.chunk(cy, cx)),
    ]).then(async ([c, terrain, history]) => {
      this.chunks.set(k, { cy, cx, fuel: c.fuel, terrain, history, coarse: coarseOf(c.fuel), bounds: box });
      // this chunk, and its neighbors: their continuity reaches into it
      const redo: ChunkFactors[] = [];
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const n = this.chunks.get(key(cy + dy, cx + dx));
        if (n) redo.push(n);
      }
      await Promise.all(redo.map((n) => this.draw(n)));
      this.version++;
      this.onDraw();
    });
    p.catch(() => this.reading.delete(k)).finally(() => { if (this.chunks.has(k)) this.reading.delete(k); });
    this.reading.set(k, p);
    return p;
  }

  /** Reads every chunk in the box not on hand yet; onChunk after each. Stops quietly when the signal aborts. */
  async load(bbox: [number, number, number, number], onChunk: (done: number, total: number) => void, signal: AbortSignal) {
    const all = RiskFactors.chunksIn(bbox);
    const todo = all.filter(([cy, cx]) => !this.chunks.has(key(cy, cx)));
    let next = 0;
    const count = () => all.filter(([cy, cx]) => this.chunks.has(key(cy, cx))).length;
    onChunk(count(), all.length);
    const worker = async () => {
      while (next < todo.length && !signal.aborted) {
        const [cy, cx] = todo[next++]!;
        // a read shared with an earlier view can fail with that view's abort: read it again for this one
        for (let tries = 0; tries < 3 && !signal.aborted; tries++) {
          try {
            await this.readChunk(cy, cx, signal);
            onChunk(count(), all.length);
            break;
          } catch (e) {
            if (signal.aborted) return;
            if ((e as Error).name === "AbortError") continue;
            console.warn("risk factors:", e);
            break;
          }
        }
      }
    };
    await Promise.all(Array.from({ length: AT_ONCE }, worker));
    this.trim(bbox);
  }

  /** The risk over the chunks of a lon / lat box and one chunk around it, as one grid (null when too big). */
  grid(bbox: [number, number, number, number]): RiskGrid | null {
    const [w, s, e, n] = bbox;
    const cy0 = Math.floor((Y0 - n) / RES / CHUNK) - 1, cy1 = Math.floor((Y0 - s) / RES / CHUNK) + 1;
    const cx0 = Math.floor((w - X0) / RES / CHUNK) - 1, cx1 = Math.floor((e - X0) / RES / CHUNK) + 1;
    const ny = cy1 - cy0 + 1, nx = cx1 - cx0 + 1;
    if (nx * CHUNK > 8192 || ny * CHUNK > 8192) return null;
    const width = nx * CHUNK, height = ny * CHUNK;
    const data = new Uint8Array(width * height);
    for (let cy = cy0; cy <= cy1; cy++)
      for (let cx = cx0; cx <= cx1; cx++) {
        const r = this.chunks.get(key(cy, cx))?.risk;
        if (!r) continue;
        for (let row = 0; row < CHUNK; row++)
          data.set(r.subarray(row * CHUNK, (row + 1) * CHUNK), ((cy - cy0) * CHUNK + row) * width + (cx - cx0) * CHUNK);
      }
    return {
      data, width, height, seq: ++gridSeq,
      bounds: [X0 + cx0 * CHUNK * RES, Y0 - (cy1 + 1) * CHUNK * RES, X0 + (cx1 + 1) * CHUNK * RES, Y0 - cy0 * CHUNK * RES],
    };
  }

  /** Lets go of the chunks farthest from the view once more than KEEP are held. */
  private trim(bbox: [number, number, number, number]) {
    if (this.chunks.size <= KEEP) return;
    const my = (Y0 - (bbox[1] + bbox[3]) / 2) / RES / CHUNK, mx = ((bbox[0] + bbox[2]) / 2 - X0) / RES / CHUNK;
    const far = [...this.chunks.entries()].sort((a, b) => Math.hypot(b[1].cy - my, b[1].cx - mx) - Math.hypot(a[1].cy - my, a[1].cx - mx));
    for (const [k, ch] of far.slice(0, this.chunks.size - KEEP)) {
      ch.image?.close();
      this.chunks.delete(k);
    }
    this.version++;
    this.onDraw();
  }

  /** Share of fuel within CONT_RADIUS_M of a coarse cell (cell centers inside it), over the chunks read. */
  private near(gi: number, gj: number, cellH: number, cellW: number): number {
    const ri = Math.ceil(CONT_RADIUS_M / cellH), rj = Math.ceil(CONT_RADIUS_M / cellW);
    let s = 0, n = 0;
    for (let di = -ri; di <= ri; di++)
      for (let dj = -rj; dj <= rj; dj++) {
        if (Math.hypot(di * cellH, dj * cellW) > CONT_RADIUS_M) continue;
        const v = this.coarseAt(gi + di, gj + dj);
        if (v === undefined || Number.isNaN(v)) continue;
        s += v;
        n++;
      }
    return n ? s / n : NaN;
  }

  /** The chunk's continuity per pixel (smoothed between coarse cells) and its risk picture. */
  private async draw(ch: ChunkFactors) {
    const lat = (ch.bounds[1] + ch.bounds[3]) / 2;
    const cellH = COARSE * RES * M_LAT, cellW = cellH * Math.cos((lat * Math.PI) / 180);
    // coarse continuity with a one-cell margin, so pixels near the edge blend with the neighbor's cells
    const M = NC + 2, gi0 = ch.cy * NC - 1, gj0 = ch.cx * NC - 1;
    const g = new Float32Array(M * M);
    for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) g[i * M + j] = this.near(gi0 + i, gj0 + j, cellH, cellW);
    const cont = new Float32Array(CHUNK * CHUNK);
    const rgba = new Uint8ClampedArray(CHUNK * CHUNK * 4);
    const risk = new Uint8Array(CHUNK * CHUNK);
    for (let r = 0; r < CHUNK; r++) {
      const u = (r + 0.5) / COARSE - 0.5 + 1, i0 = Math.floor(u), fu = u - i0;
      for (let c = 0; c < CHUNK; c++) {
        const v = (c + 0.5) / COARSE - 0.5 + 1, j0 = Math.floor(v), fv = v - j0;
        let s = 0, w = 0;
        for (const [di, dj, wt] of [[0, 0, (1 - fu) * (1 - fv)], [0, 1, (1 - fu) * fv], [1, 0, fu * (1 - fv)], [1, 1, fu * fv]] as const) {
          const x = g[(i0 + di) * M + j0 + dj]!;
          if (Number.isNaN(x) || wt === 0) continue;
          s += x * wt;
          w += wt;
        }
        const idx = r * CHUNK + c;
        cont[idx] = w ? s / w : NaN;
        const p = pFuel(ch.fuel[idx]!);
        if (p === null || Number.isNaN(cont[idx]!)) continue;
        const t = parts(p, cont[idx]!, ch.terrain.slope[idx]!, ch.terrain.aspect[idx]!, ch.history.last[idx]!, NOW_YEAR);
        risk[idx] = 1 + Math.round(t.risk * 254);
        const q = Math.round(t.risk * 255) * 3;
        rgba[idx * 4] = VIRIDIS[q]!;
        rgba[idx * 4 + 1] = VIRIDIS[q + 1]!;
        rgba[idx * 4 + 2] = VIRIDIS[q + 2]!;
        rgba[idx * 4 + 3] = 255;
      }
    }
    const image = await createImageBitmap(new ImageData(rgba, CHUNK, CHUNK));
    if (this.chunks.get(key(ch.cy, ch.cx)) !== ch) return image.close();
    const old = ch.image;
    ch.cont = cont;
    ch.risk = risk;
    ch.image = image;
    // a draw already queued may still use the old picture
    if (old) setTimeout(() => old.close(), 1500);
  }

  /** The coarse cell's fuel share at global coarse row / column, undefined when its chunk is not read. */
  private coarseAt(gi: number, gj: number): number | undefined {
    const ch = this.chunks.get(key(Math.floor(gi / NC), Math.floor(gj / NC)));
    if (!ch) return undefined;
    return ch.coarse[(((gi % NC) + NC) % NC) * NC + (((gj % NC) + NC) % NC)];
  }

  /** Every factor at a point, or null when its chunk is not read. */
  at(lon: number, lat: number): PointFactors | null {
    const px = Math.floor((lon - X0) / RES), py = Math.floor((Y0 - lat) / RES);
    const ch = this.chunks.get(key(Math.floor(py / CHUNK), Math.floor(px / CHUNK)));
    if (!ch) return null;
    const i = (py - ch.cy * CHUNK) * CHUNK + (px - ch.cx * CHUNK);
    const gi = Math.floor(py / COARSE), gj = Math.floor(px / COARSE);
    const cellH = COARSE * RES * M_LAT, cellW = cellH * Math.cos((lat * Math.PI) / 180);

    // share of fuel within the radius, as the picture has it
    const near = ch.cont ? ch.cont[i]! : this.near(gi, gj, cellH, cellW);

    // the connected fuel patch from this cell, over the chunks read
    let patchKm2: number | null = null, patchOpen = false;
    const v0 = this.coarseAt(gi, gj);
    if (v0 !== undefined && v0 >= PATCH_SHARE) {
      const seen = new Set<string>([`${gi}:${gj}`]);
      const stack: [number, number][] = [[gi, gj]];
      let cells = 0;
      while (stack.length) {
        const [a, b] = stack.pop()!;
        cells++;
        for (const [da, db] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
          const k = `${a + da}:${b + db}`;
          if (seen.has(k)) continue;
          seen.add(k);
          const v = this.coarseAt(a + da, b + db);
          if (v === undefined) { patchOpen = true; continue; }
          if (v >= PATCH_SHARE) stack.push([a + da, b + db]);
        }
      }
      patchKm2 = (cells * cellH * cellW) / 1e6;
    } else if (v0 !== undefined) patchKm2 = 0;

    const fuel = pFuel(ch.fuel[i]!);
    return {
      fuel,
      near: Number.isNaN(near) ? null : near,
      parts: fuel === null || Number.isNaN(near) ? null : parts(fuel, near, ch.terrain.slope[i]!, ch.terrain.aspect[i]!, ch.history.last[i]!, NOW_YEAR),
      patchKm2,
      patchOpen,
      slope: ch.terrain.slope[i]!,
      aspect: ch.terrain.aspect[i]!,
      last: ch.history.last[i]!,
      count: ch.history.count[i]!,
      fires: this.history.at(lon, lat),
    };
  }
}
