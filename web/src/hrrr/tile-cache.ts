import { ignitionWeather } from "../aef/head.js";
import { ffwiFromMoisture, lagMoisture, M_SCALE, MS_TO_MPH, U16_NAN, WIND_SCALE } from "../ffwi/compute.js";
import { HRRR_GRID, HRRR_LEAD_TIME_COUNT, HRRR_SHARD } from "./metadata.js";

/**
 * JS-side cache of the per-shard Fosberg inputs, filled by getTileData (and
 * the side channel) as shards arrive. The county join and the tooltips read
 * from here, so nothing is fetched twice.
 *
 * Holds equilibrium moisture and wind rather than finished FFWI so the
 * fuel-moisture timelag (tau) can be re-applied on the fly: `setTau` re-runs
 * the recursion over every cached shard (a few ms per shard) and `sample`
 * computes FFWI from the lagged moisture and the instantaneous wind.
 *
 * Keyed by `${row},${col}` of the shard grid (row = floor(y / 265),
 * col = floor(x / 300)). Blocks are lead-major: idx = lead * (w*h) + ly*w + lx.
 */
export type FfwiTile = {
  key: string;
  rowStart: number;
  colStart: number;
  width: number;
  height: number;
  /** Equilibrium moisture %, x1000 (65535 = NaN). */
  mEq: Uint16Array;
  /** Lagged moisture for the cache's current tau (same array as mEq when tau = 0). */
  mLag: Uint16Array;
  /** Wind mph x100 (65535 = NaN). */
  wind: Uint16Array;
  rh: Uint8Array;
  /** degC + 100. */
  tc: Uint8Array;
};

/** Whole grid is 4 x 6 shards; cap at that so memory is bounded. */
const SHARD_COLS = 6;
const MAX_TILES = 24;

export type TileSample = {
  /** FFWI with the cache's timelag applied. */
  ffwi: number;
  /** FFWI as published (no lag). */
  ffwiInstant: number;
  windMph: number;
  rh: number;
  tc: number;
};

export class FfwiTileCache {
  private tiles = new Map<string, FfwiTile>();
  /** The same tiles by shard index (row * 6 + col), for the hot loops. */
  private byIdx: (FfwiTile | undefined)[] = new Array(MAX_TILES);
  /** Bumped on every change so React can memo against it. */
  version = 0;
  listeners = new Set<() => void>();
  /** Fuel-moisture timelag in hours; 0 = Fosberg as published. */
  tau = 0;

  static keyFor(hrrrX: number, hrrrY: number): string {
    return `${Math.floor(hrrrY / HRRR_SHARD.height)},${Math.floor(hrrrX / HRRR_SHARD.width)}`;
  }

  private static idxFor(hrrrX: number, hrrrY: number): number {
    return Math.floor(hrrrY / HRRR_SHARD.height) * SHARD_COLS + Math.floor(hrrrX / HRRR_SHARD.width);
  }

  private static idxOf(t: FfwiTile): number {
    return Math.floor(t.rowStart / HRRR_SHARD.height) * SHARD_COLS + Math.floor(t.colStart / HRRR_SHARD.width);
  }

  /** Whether a pixel lies on the HRRR grid at all. */
  inGrid(hrrrX: number, hrrrY: number): boolean {
    return hrrrX >= 0 && hrrrY >= 0 && hrrrX < HRRR_GRID.width && hrrrY < HRRR_GRID.height;
  }

  /** The shard holding a pixel, or undefined if not loaded. */
  tileAt(hrrrX: number, hrrrY: number): FfwiTile | undefined {
    return this.byIdx[FfwiTileCache.idxFor(hrrrX, hrrrY)];
  }

  get(key: string): FfwiTile | undefined {
    return this.tiles.get(key);
  }

  has(key: string): boolean {
    return this.tiles.has(key);
  }

  set(tile: FfwiTile): void {
    if (this.tiles.size >= MAX_TILES && !this.tiles.has(tile.key)) {
      const first = this.tiles.keys().next().value;
      if (first !== undefined) {
        const old = this.tiles.get(first);
        this.tiles.delete(first);
        if (old) this.byIdx[FfwiTileCache.idxOf(old)] = undefined;
      }
    }
    this.applyLag(tile);
    this.tiles.set(tile.key, tile);
    this.byIdx[FfwiTileCache.idxOf(tile)] = tile;
    this.bump();
  }

  /** Re-run the timelag over every cached shard. */
  setTau(tau: number): void {
    if (tau === this.tau) return;
    this.tau = tau;
    for (const t of this.tiles.values()) this.applyLag(t);
    this.bump();
  }

  private applyLag(t: FfwiTile): void {
    const perLead = t.width * t.height;
    if (!(this.tau > 0)) {
      t.mLag = t.mEq;
      return;
    }
    if (t.mLag === t.mEq) t.mLag = new Uint16Array(t.mEq.length);
    lagMoisture(t.mEq, HRRR_LEAD_TIME_COUNT, perLead, this.tau, t.mLag);
  }

  /** The shards in hand. */
  entries(): IterableIterator<FfwiTile> {
    return this.tiles.values();
  }

  /** Drop one shard (its data has been taken elsewhere). */
  delete(key: string): void {
    const old = this.tiles.get(key);
    if (!old) return;
    this.tiles.delete(key);
    this.byIdx[FfwiTileCache.idxOf(old)] = undefined;
  }

  clear(): void {
    this.tiles.clear();
    this.byIdx = new Array(MAX_TILES);
    this.bump();
  }

  get size(): number {
    return this.tiles.size;
  }

  private bump(): void {
    this.version++;
    for (const l of this.listeners) l();
  }

  /**
   * FFWI for every lead at one pixel into `out` (length 49): lagged, or
   * published (instant) when `instant` is set. NaN where not loaded / fill.
   * One tile lookup per pixel; this is what the county passes use.
   */
  ffwiSeries(hrrrX: number, hrrrY: number, out: Float32Array, instant = false): void {
    const t = this.byIdx[FfwiTileCache.idxFor(hrrrX, hrrrY)];
    if (!t) {
      out.fill(NaN);
      return;
    }
    const per = t.width * t.height;
    const off = (hrrrY - t.rowStart) * t.width + (hrrrX - t.colStart);
    const mArr = instant ? t.mEq : t.mLag;
    for (let lead = 0; lead < HRRR_LEAD_TIME_COUNT; lead++) {
      const i = lead * per + off;
      const m = mArr[i]!;
      const w = t.wind[i]!;
      out[lead] = m === U16_NAN || w === U16_NAN ? NaN : ffwiFromMoisture(m / M_SCALE, w / WIND_SCALE);
    }
  }

  /**
   * The weather part of the ignition logit for every lead at one pixel (lagged FFWI, T, RH and the
   * wind in m/s through the ignition head); NaN where not loaded.
   */
  ignitionSeries(hrrrX: number, hrrrY: number, out: Float32Array): void {
    const t = this.byIdx[FfwiTileCache.idxFor(hrrrX, hrrrY)];
    if (!t) {
      out.fill(NaN);
      return;
    }
    const per = t.width * t.height;
    const off = (hrrrY - t.rowStart) * t.width + (hrrrX - t.colStart);
    for (let lead = 0; lead < HRRR_LEAD_TIME_COUNT; lead++) {
      const i = lead * per + off;
      const m = t.mLag[i]!;
      const w = t.wind[i]!;
      if (m === U16_NAN || w === U16_NAN) {
        out[lead] = NaN;
        continue;
      }
      const mph = w / WIND_SCALE;
      out[lead] = ignitionWeather(ffwiFromMoisture(m / M_SCALE, mph), (t.tc[i] ?? 100) - 100, t.rh[i] ?? 0, mph / MS_TO_MPH);
    }
  }

  /** The weather part of the ignition logit at one pixel and lead, NaN if not loaded. */
  ignitionAt(hrrrX: number, hrrrY: number, lead: number): number {
    const t = this.byIdx[FfwiTileCache.idxFor(hrrrX, hrrrY)];
    if (!t) return NaN;
    const i = lead * t.width * t.height + (hrrrY - t.rowStart) * t.width + (hrrrX - t.colStart);
    const m = t.mLag[i]!;
    const w = t.wind[i]!;
    if (m === U16_NAN || w === U16_NAN) return NaN;
    const mph = w / WIND_SCALE;
    return ignitionWeather(ffwiFromMoisture(m / M_SCALE, mph), (t.tc[i] ?? 100) - 100, t.rh[i] ?? 0, mph / MS_TO_MPH);
  }

  /** Lagged FFWI at one pixel and lead, NaN if not loaded / fill. Fast path for the county loops. */
  ffwiAt(hrrrX: number, hrrrY: number, lead: number): number {
    const t = this.byIdx[FfwiTileCache.idxFor(hrrrX, hrrrY)];
    if (!t) return NaN;
    const i = lead * t.width * t.height + (hrrrY - t.rowStart) * t.width + (hrrrX - t.colStart);
    const m = t.mLag[i]!;
    const w = t.wind[i]!;
    if (m === U16_NAN || w === U16_NAN) return NaN;
    return ffwiFromMoisture(m / M_SCALE, w / WIND_SCALE);
  }

  /** Full readout for the tooltip. Returns null if the shard is not loaded. */
  sample(hrrrX: number, hrrrY: number, lead: number): TileSample | null {
    const t = this.byIdx[FfwiTileCache.idxFor(hrrrX, hrrrY)];
    if (!t) return null;
    const i = lead * t.width * t.height + (hrrrY - t.rowStart) * t.width + (hrrrX - t.colStart);
    const m = t.mLag[i]!;
    const m0 = t.mEq[i]!;
    const w = t.wind[i]!;
    const ok = w !== U16_NAN;
    return {
      ffwi: ok && m !== U16_NAN ? ffwiFromMoisture(m / M_SCALE, w / WIND_SCALE) : NaN,
      ffwiInstant: ok && m0 !== U16_NAN ? ffwiFromMoisture(m0 / M_SCALE, w / WIND_SCALE) : NaN,
      windMph: ok ? Math.round(w / WIND_SCALE) : 0,
      rh: t.rh[i] ?? 0,
      tc: (t.tc[i] ?? 100) - 100,
    };
  }
}
