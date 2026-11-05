import { FetchSource, PMTiles } from "pmtiles";
import type { AefTile } from "./aef/layers.js";
import { RES, X0, Y0 } from "./aef/mosaic.js";
import type { FirePixel } from "./goes.js";
import type { ReachOut } from "./reach.worker.js";

/**
 * One fire hour by hour: the grid of cells around it, the ground's elevation (Mapterhorn) and fuel (the fuel head)
 * under each cell, and the reach from the pixels burning each hour (reach.worker.ts). The glow becomes a mask on
 * the 10 m fuel tiles; the reach lines come back as lon / lat rings.
 */

/** Mapterhorn's global terrain (Copernicus GLO-30 up to zoom 12) on Source Cooperative, range read. */
const MAPTERHORN = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/mapterhorn/mapterhorn/planet.pmtiles";
/** Zoom 11: about 30 m a pixel over California, the GLO-30 data's own size. */
const ELEV_Z = 11;
const M_PER_DEG = 111_320;

/** Reach budgets (m of flat full fuel): a pixel's first hour, the growth per doubling of hours burned, the most. */
export const REACH = { firstM: 1500, perDoublingM: 1000, maxM: 6000, fadeHours: 6, lineStepM: 1500 };

let planet: PMTiles | null = null;

async function terrariumTile(x: number, y: number): Promise<{ x: number; y: number; h: Float32Array } | null> {
  planet ??= new PMTiles(new FetchSource(MAPTERHORN));
  const r = await planet.getZxy(ELEV_Z, x, y);
  if (!r?.data) return null;
  const bmp = await createImageBitmap(new Blob([r.data], { type: "image/webp" }));
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0);
  const px = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
  const h = new Float32Array(bmp.width * bmp.height);
  for (let i = 0; i < h.length; i++) h[i] = px[4 * i]! * 256 + px[4 * i + 1]! + px[4 * i + 2]! / 256 - 32768;
  return { x, y, h };
}

export class FireReach {
  readonly nx: number;
  readonly ny: number;
  readonly cellM: number;
  private lon0: number;
  private lat1: number;
  private mLon: number;
  /** Each GOES pixel's cells on the grid. */
  private cellsOf = new Map<string, Int32Array>();
  /** Each GOES pixel's hours seen burning (hours from t0, sorted). */
  private hoursOf = new Map<string, number[]>();
  /** Each hour's burning pixels. */
  readonly byHour: Map<number, FirePixel[]>;
  readonly t0: number;
  readonly hours: number;
  private worker: Worker;
  private nextId = 0;
  private pending = new Map<number, (o: ReachOut) => void>();
  ready: Promise<void>;
  /** The latest glow (0..255 per cell). */
  glow: Uint8Array | null = null;

  constructor(pixels: FirePixel[], bbox: [number, number, number, number]) {
    const margin = REACH.maxM + 2000;
    const latC = (bbox[1] + bbox[3]) / 2;
    this.mLon = M_PER_DEG * Math.cos((latC * Math.PI) / 180);
    this.lon0 = bbox[0] - margin / this.mLon;
    this.lat1 = bbox[3] + margin / M_PER_DEG;
    const wM = (bbox[2] - bbox[0]) * this.mLon + 2 * margin;
    const hM = (bbox[3] - bbox[1]) * M_PER_DEG + 2 * margin;
    // 60 m cells, coarser for a big fire so the grid stays near 2.5 million cells
    this.cellM = Math.max(60, Math.ceil(Math.sqrt((wM * hM) / 2.5e6)));
    this.nx = Math.ceil(wM / this.cellM);
    this.ny = Math.ceil(hM / this.cellM);
    this.t0 = Math.floor(Math.min(...pixels.map((p) => p.t)) / 3600e3) * 3600e3;
    this.hours = Math.floor((Math.max(...pixels.map((p) => p.t)) - this.t0) / 3600e3) + 1;
    this.byHour = new Map();
    for (const p of pixels) {
      const h = Math.floor((p.t - this.t0) / 3600e3);
      const hs = this.hoursOf.get(p.key);
      if (!hs) this.hoursOf.set(p.key, [h]);
      else if (!hs.includes(h)) hs.push(h);
      let l = this.byHour.get(h);
      if (!l) this.byHour.set(h, (l = []));
      if (!l.some((q) => q.key === p.key)) l.push(p);
      if (!this.cellsOf.has(p.key)) this.cellsOf.set(p.key, this.rasterize(p.polygon));
    }
    for (const hs of this.hoursOf.values()) hs.sort((a, b) => a - b);
    this.worker = new Worker(new URL("./reach.worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (e: MessageEvent<ReachOut>) => {
      const f = this.pending.get(e.data.id);
      this.pending.delete(e.data.id);
      f?.(e.data);
    };
    this.ready = this.loadElevation();
  }

  /** The grid's lon / lat box. */
  bbox(): [number, number, number, number] {
    const [e, s] = this.lonLat(this.nx, this.ny);
    return [this.lon0, s, e, this.lat1];
  }

  dispose() {
    this.worker.terminate();
  }

  /** Cell center of (x, y) as lon / lat. */
  private lonLat(x: number, y: number): [number, number] {
    return [this.lon0 + (x * this.cellM) / this.mLon, this.lat1 - (y * this.cellM) / M_PER_DEG];
  }

  /** The cells whose centers fall inside a pixel's footprint. */
  private rasterize(poly: [number, number][]): Int32Array {
    const pts = poly.map(([lon, lat]) => [((lon - this.lon0) * this.mLon) / this.cellM - 0.5, ((this.lat1 - lat) * M_PER_DEG) / this.cellM - 0.5]);
    const xs = pts.map((p) => p[0]!), ys = pts.map((p) => p[1]!);
    const out: number[] = [];
    for (let y = Math.max(0, Math.floor(Math.min(...ys))); y <= Math.min(this.ny - 1, Math.ceil(Math.max(...ys))); y++)
      for (let x = Math.max(0, Math.floor(Math.min(...xs))); x <= Math.min(this.nx - 1, Math.ceil(Math.max(...xs))); x++) {
        let inside = false;
        for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
          const [xi, yi] = pts[i]!, [xj, yj] = pts[j]!;
          if (yi! > y !== yj! > y && x < ((xj! - xi!) * (y - yi!)) / (yj! - yi!) + xi!) inside = !inside;
        }
        if (inside) out.push(y * this.nx + x);
      }
    return Int32Array.from(out);
  }

  /** Elevation for every cell from Mapterhorn's zoom-11 tiles (nearest pixel), then the worker's grid. */
  private async loadElevation() {
    const n = 2 ** ELEV_Z;
    const tileXY = (lon: number, lat: number) => {
      const fx = ((lon + 180) / 360) * n;
      const lr = (lat * Math.PI) / 180;
      const fy = ((1 - Math.log(Math.tan(lr) + 1 / Math.cos(lr)) / Math.PI) / 2) * n;
      return [fx, fy] as const;
    };
    const [wx, ny0] = tileXY(this.lon0, this.lat1);
    const [ex, sy] = tileXY(...this.lonLat(this.nx, this.ny));
    const jobs: Promise<{ x: number; y: number; h: Float32Array } | null>[] = [];
    for (let ty = Math.floor(ny0); ty <= Math.floor(sy); ty++) for (let tx = Math.floor(wx); tx <= Math.floor(ex); tx++) jobs.push(terrariumTile(tx, ty));
    const tiles = new Map<string, Float32Array>();
    let size = 512;
    for (const t of await Promise.all(jobs)) if (t) { tiles.set(`${t.x}/${t.y}`, t.h); size = Math.round(Math.sqrt(t.h.length)); }
    const elev = new Float32Array(this.nx * this.ny);
    for (let y = 0; y < this.ny; y++)
      for (let x = 0; x < this.nx; x++) {
        const [fx, fy] = tileXY(...this.lonLat(x + 0.5, y + 0.5));
        const h = tiles.get(`${Math.floor(fx)}/${Math.floor(fy)}`);
        if (!h) continue;
        const px = Math.min(size - 1, Math.floor((fx % 1) * size)), py = Math.min(size - 1, Math.floor((fy % 1) * size));
        elev[y * this.nx + x] = h[py * size + px]!;
      }
    this.worker.postMessage({ type: "init", nx: this.nx, ny: this.ny, cellM: this.cellM, elev }, [elev.buffer]);
  }

  /** P(fuel) for every cell from the 10 m tiles on hand (NaN where none is loaded yet; the worker reads it as 0.5).
   * Returns the share of cells with a reading. */
  setFuel(tiles: AefTile[]): number {
    let read = 0;
    const fuel = new Float32Array(this.nx * this.ny).fill(NaN);
    for (const t of tiles) {
      const [w, s, e, nn] = t.bbox;
      const x0 = Math.max(0, Math.floor(((w - this.lon0) * this.mLon) / this.cellM));
      const x1 = Math.min(this.nx - 1, Math.ceil(((e - this.lon0) * this.mLon) / this.cellM));
      const y0 = Math.max(0, Math.floor(((this.lat1 - nn) * M_PER_DEG) / this.cellM));
      const y1 = Math.min(this.ny - 1, Math.ceil(((this.lat1 - s) * M_PER_DEG) / this.cellM));
      for (let y = y0; y <= y1; y++)
        for (let x = x0; x <= x1; x++) {
          const [lon, lat] = this.lonLat(x + 0.5, y + 0.5);
          if (lon < w || lon > e || lat < s || lat > nn) continue;
          const i = t.pixelAt(lon, lat);
          if (i < 0) continue;
          const b = t.classes.fuel[i]!;
          fuel[y * this.nx + x] = b > 0 ? (b - 1) / 254 : 0;
          read++;
        }
    }
    this.worker.postMessage({ type: "fuel", fuel }, [fuel.buffer]);
    return read / (this.nx * this.ny);
  }

  /** The pixels burning at an hour and their budgets: more for a pixel that has burned longer, fading after it stops. */
  sources(hour: number): { cells: Int32Array; budget: Float32Array } {
    const cells: number[] = [];
    const budget: number[] = [];
    for (const [key, hs] of this.hoursOf) {
      let seen = 0, last = -1;
      for (const h of hs) {
        if (h > hour) break;
        seen++;
        last = h;
      }
      if (!seen) continue;
      const b = Math.min(REACH.maxM, REACH.firstM + REACH.perDoublingM * Math.log2(seen)) * Math.exp(-(hour - last) / REACH.fadeHours);
      if (b < 300) continue;
      for (const c of this.cellsOf.get(key)!) { cells.push(c); budget.push(b); }
    }
    return { cells: Int32Array.from(cells), budget: Float32Array.from(budget) };
  }

  /** The reach at an hour: the glow per cell (kept on this object) and the reach lines as lon / lat rings. */
  async run(hour: number): Promise<{ lines: { level: number; rings: [number, number][][] }[]; ms: number }> {
    await this.ready;
    const { cells, budget } = this.sources(hour);
    const id = this.nextId++;
    const steps: number[] = [];
    // the reach lines are off on the map, so none are traced
    // for (let s = REACH.lineStepM; s < REACH.maxM; s += REACH.lineStepM) steps.push(s);
    const out = await new Promise<ReachOut>((ok) => {
      this.pending.set(id, ok);
      this.worker.postMessage({ type: "run", id, cells, budget, maxM: REACH.maxM, steps }, [cells.buffer, budget.buffer]);
    });
    this.glow = out.glow;
    // contour coordinates are in cell corners: x across, y down from the north edge
    const lines = out.lines.map((l) => ({ level: l.level, rings: l.rings.map((r) => r.map(([x, y]) => this.lonLat(x, y))) }));
    return { lines, ms: out.ms };
  }

  /** The glow on one 10 m tile's pixels (bilinear between cells), lifted so a lone burning pixel still reads. */
  mask(t: AefTile): Uint8Array | null {
    const g = this.glow;
    if (!g) return null;
    const [w, , , nn] = t.bbox;
    const px0 = Math.round((w - X0) / RES);
    const py0 = Math.round((Y0 - nn) / RES);
    const out = new Uint8Array(t.width * t.height);
    let any = false;
    for (let r = 0; r < t.height; r++) {
      const lat = Y0 - (py0 + r + 0.5) * RES;
      const fy = ((this.lat1 - lat) * M_PER_DEG) / this.cellM - 0.5;
      if (fy < 0 || fy >= this.ny - 1) continue;
      const y = Math.floor(fy), wy = fy - y;
      for (let c = 0; c < t.width; c++) {
        const lon = X0 + (px0 + c + 0.5) * RES;
        const fx = ((lon - this.lon0) * this.mLon) / this.cellM - 0.5;
        if (fx < 0 || fx >= this.nx - 1) continue;
        const x = Math.floor(fx), wx = fx - x;
        const i = y * this.nx + x;
        const v = (g[i]! * (1 - wx) + g[i + 1]! * wx) * (1 - wy) + (g[i + this.nx]! * (1 - wx) + g[i + this.nx + 1]! * wx) * wy;
        if (v <= 0) continue;
        out[r * t.width + c] = Math.round(255 * Math.sqrt(v / 255));
        any = true;
      }
    }
    return any ? out : null;
  }
}
