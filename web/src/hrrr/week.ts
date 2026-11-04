import type { WindMode } from "../ffwi/compute.js";
import { dateFromInitTimeIdx, HRRR_GRID, HRRR_SHARD, initTimeIdxFromDate, lonLatToHrrrPixelF } from "./metadata.js";
import { FfwiTileCache } from "./tile-cache.js";
import { TERM_LUT } from "./week-term.js";
import type { WeekJobMsg, WeekResult } from "./week.worker.js";

export { termByte, termLogit } from "./week-term.js";

/**
 * A run of days from the 48 h forecast store, one 12Z run per day, its first 24 hours each, stitched
 * into one hourly series (day 0 at 12Z through the last day at 11Z the next morning). The forecast
 * store is chunked by run and shard (49 leads x 265 x 300 px), so a day costs a whole shard fetch
 * (about 18 MB over the wire per shard) whatever is kept; what is kept is small: per shard and day,
 * the weather part of the ignition logit, a byte per pixel and hour (1.9 MB per shard-day; the FFWI
 * itself is not kept, the term is the number the picture and the counties use).
 *
 * The fetch, decode and Fosberg pass run in a worker (`week.worker.ts`), so the map keeps moving while a
 * window loads. Only the shards under the map's bounds are fetched: zoomed in on a region, that region's
 * shards and no others. Shard-days are one queue ranked by the playhead (the day on screen first, then the
 * days after it); a few are in flight at once. The map plays over the days in hand while the rest arrive.
 */
export const RUN_HOUR_UTC = 12;
export const HOURS_PER_DAY = 24;

export type WeekSpec = {
  /** The UTC date of day 0 (its 12Z run). */
  start: Date;
  days: number;
};

/** The init index of day d's run, or -1 when the store does not have it. */
export function weekInitIdx(spec: WeekSpec, d: number, initTimeCount: number): number {
  const t = new Date(spec.start.getTime());
  t.setUTCHours(RUN_HOUR_UTC, 0, 0, 0);
  t.setUTCDate(t.getUTCDate() + d);
  const idx = initTimeIdxFromDate(t, initTimeCount - 1);
  return dateFromInitTimeIdx(idx).getTime() === t.getTime() ? idx : -1;
}

/** The valid time of hour h of the week. */
export function weekHourDate(spec: WeekSpec, h: number): Date {
  const t = new Date(spec.start.getTime());
  t.setUTCHours(RUN_HOUR_UTC + h, 0, 0, 0);
  return t;
}

type WeekShard = { term: Uint8Array; width: number; rowStart: number; colStart: number };

type Day = { initIdx: number; done: Set<string> };

/** One shard of one day, waiting its turn. */
type Job = { d: number; key: string; row: number; col: number };

/** What the panel's day strip shows for each day. */
export type DayState = "in" | "loading" | "waiting" | "none";

/** Shards in flight at once over the whole window (each about 18 MB over the wire, 50 MB in the worker while decoding). */
const IN_FLIGHT = 4;

const SHARD_COLS = Math.ceil(HRRR_GRID.width / HRRR_SHARD.width);
const SHARD_ROWS = Math.ceil(HRRR_GRID.height / HRRR_SHARD.height);
const N_SHARDS = SHARD_COLS * SHARD_ROWS;

/** One worker for every window: the store opens once, and a window replaced mid-load cancels its jobs. */
let worker: Worker | null = null;
let jobSeq = 0;
const pending = new Map<number, (r: WeekResult) => void>();
function shardWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL("./week.worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (e: MessageEvent<WeekResult>) => {
    const done = pending.get(e.data.id);
    pending.delete(e.data.id);
    done?.(e.data);
  };
  return worker;
}

let seq = 0;

export class WeekWeather {
  /** Tells one instance from another with the same spec (a tile's weather must come from the live one). */
  readonly id = ++seq;
  readonly depth: number;
  private days: Day[] = [];
  /** By day and shard index (row * SHARD_COLS + col): the lookups in the texture loops stay numeric. */
  private shards: (WeekShard | undefined)[];
  /** Bumped when anything changed (a shard arrived, a job started, the view moved): the panel re-renders. */
  version = 0;
  listeners = new Set<() => void>();
  /** Shards (by key) whose planes arrived since the last `takeArrived`: the tiles over them need new textures. */
  private arrived = new Set<string>();
  private gone = false;
  /** Every shard-day under the view not yet in hand, keyed `${d}|${shard}`; done jobs leave the map. */
  private jobs = new Map<string, Job>();
  /** In flight: `${d}|${shard}` to the worker's job id. */
  private inFlight = new Map<string, number>();
  /** The day the playhead is on: its shards go first, then the days after it in order. */
  private focusDay = 0;
  /** The shards under the map's bounds. */
  private viewKeys = new Set<string>();
  private hasView = false;

  constructor(readonly spec: WeekSpec, private windMode: WindMode, initTimeCount: number) {
    this.depth = spec.days * HOURS_PER_DAY;
    this.shards = new Array(spec.days * N_SHARDS);
    // debug hook for the headless probes
    (globalThis as unknown as { __week?: unknown }).__week = this;
    for (let d = 0; d < spec.days; d++) this.days.push({ initIdx: weekInitIdx(spec, d, initTimeCount), done: new Set() });
  }

  /** The shards whose planes arrived since the last call (then cleared). */
  takeArrived(): Set<string> {
    const a = this.arrived;
    this.arrived = new Set();
    return a;
  }

  /** Shard-days under the view: in hand, and asked for (the loading bar). */
  get progress(): { done: number; wanted: number } {
    let done = 0;
    let wanted = 0;
    for (const day of this.days) {
      if (day.initIdx < 0) continue;
      for (const key of this.viewKeys) {
        wanted++;
        if (day.done.has(key)) done++;
      }
    }
    return { done, wanted };
  }

  /** Each day of the week: in hand under the view, loading, waiting its turn, or not in the store. */
  dayStates(): DayState[] {
    return this.days.map((day, d) => {
      if (day.initIdx < 0) return "none";
      if (this.viewKeys.size === 0) return "waiting";
      let missing = 0;
      let loading = false;
      for (const key of this.viewKeys) {
        if (day.done.has(key)) continue;
        missing++;
        if (this.inFlight.has(`${d}|${key}`)) loading = true;
      }
      return missing === 0 ? "in" : loading ? "loading" : "waiting";
    });
  }

  /** Days whose shards under the view are all in hand (0 when nothing has been asked for yet). */
  daysReady(): number {
    return this.dayStates().filter((s) => s === "in").length;
  }

  /** Whether hour h's day is in hand under the view. */
  hourReady(h: number): boolean {
    const d = Math.floor(h / HOURS_PER_DAY);
    const day = this.days[d];
    if (!day || day.initIdx < 0) return false;
    for (const key of this.viewKeys) if (!day.done.has(key)) return false;
    return this.viewKeys.size > 0;
  }

  /**
   * The hour after h for the play loop: the next hour when its day is in hand under the view, else the
   * first hour of the next day that is (wrapping), else just the next hour (nothing is in yet: scrub gray).
   */
  nextHour(h: number): number {
    const n = (h + 1) % this.depth;
    if (this.hourReady(n)) return n;
    const d0 = Math.floor(n / HOURS_PER_DAY);
    for (let k = 0; k < this.days.length; k++) {
      const d = (d0 + k) % this.days.length;
      if (this.hourReady(d * HOURS_PER_DAY)) return d * HOURS_PER_DAY;
    }
    return n;
  }

  inGrid(x: number, y: number): boolean {
    return x >= 0 && y >= 0 && x < HRRR_GRID.width && y < HRRR_GRID.height;
  }

  /** The window was replaced: drop the queue and cancel what is in flight. */
  abort(): void {
    this.gone = true;
    this.jobs.clear();
    for (const id of this.inFlight.values()) worker?.postMessage({ kind: "cancel", id });
    this.inFlight.clear();
  }

  /** The playhead moved: the day it is on comes first in the queue from now on. */
  focus(hour: number): void {
    this.focusDay = Math.max(0, Math.min(this.days.length - 1, Math.floor(hour / HOURS_PER_DAY)));
  }

  /**
   * The map's bounds (west, south, east, north): the shards under them are the view, the only ones fetched.
   * Jobs for shards the view left are dropped (one already in flight finishes and is kept).
   */
  setViewBox(box: [number, number, number, number]): void {
    const [w, s, e, n] = box;
    const keys = new Set<string>();
    // a shard is 900 x 800 km: 48 x 48 samples over the bounds (edges included) find every one under them
    const N = 48;
    for (let j = 0; j <= N; j++) {
      const lat = s + ((n - s) * j) / N;
      for (let i = 0; i <= N; i++) {
        const p = lonLatToHrrrPixelF(w + ((e - w) * i) / N, lat);
        if (p) keys.add(FfwiTileCache.keyFor(Math.floor(p.x), Math.floor(p.y)));
      }
    }
    this.hasView = true;
    this.setKeys(keys);
  }

  /** HRRR pixels something on screen needs before the map's bounds are known (later the bounds rule). */
  ensure(px: Int16Array, py: Int16Array): void {
    if (this.hasView) return;
    const keys = new Set(this.viewKeys);
    for (let i = 0; i < px.length; i++) if (px[i]! >= 0) keys.add(FfwiTileCache.keyFor(px[i]!, py[i]!));
    this.setKeys(keys);
  }

  private setKeys(keys: Set<string>): void {
    const same = keys.size === this.viewKeys.size && [...keys].every((k) => this.viewKeys.has(k));
    if (same) return;
    this.viewKeys = keys;
    for (const [id, j] of this.jobs) if (!keys.has(j.key)) this.jobs.delete(id);
    for (const key of keys) {
      const [row, col] = key.split(",").map(Number) as [number, number];
      for (let d = 0; d < this.days.length; d++) {
        const id = `${d}|${key}`;
        if (this.days[d]!.initIdx < 0 || this.days[d]!.done.has(key) || this.jobs.has(id)) continue;
        this.jobs.set(id, { d, key, row, col });
      }
    }
    this.bump();
    this.pump();
  }

  /** The playhead's day first, then the days after it. */
  private rank(j: Job): number {
    const n = this.days.length;
    return (j.d - this.focusDay + n) % n;
  }

  /** Start the best waiting jobs until the in-flight cap is reached. */
  private pump(): void {
    while (!this.gone && this.inFlight.size < IN_FLIGHT) {
      let best: Job | null = null;
      let bestRank = Infinity;
      for (const [id, j] of this.jobs) {
        if (this.inFlight.has(id)) continue;
        const r = this.rank(j);
        if (r < bestRank) {
          bestRank = r;
          best = j;
        }
      }
      if (!best) return;
      void this.run(best);
    }
  }

  private async run(j: Job): Promise<void> {
    const id = `${j.d}|${j.key}`;
    const day = this.days[j.d]!;
    const wid = ++jobSeq;
    this.inFlight.set(id, wid);
    this.bump();
    const r = await new Promise<WeekResult>((resolve) => {
      pending.set(wid, resolve);
      const msg: WeekJobMsg = { kind: "job", id: wid, initIdx: day.initIdx, row: j.row, col: j.col, hours: HOURS_PER_DAY, windMode: this.windMode };
      shardWorker().postMessage(msg);
    });
    if (this.gone) return;
    this.inFlight.delete(id);
    this.jobs.delete(id);
    if (r.ok) {
      this.shards[j.d * N_SHARDS + j.row * SHARD_COLS + j.col] = { term: r.term, width: r.width, rowStart: r.rowStart, colStart: r.colStart };
      day.done.add(j.key);
      this.arrived.add(j.key);
    } else if (!r.aborted) {
      // dropped; a new window or a view change that asks for it again retries
      console.warn("week shard failed", id, r.error);
    }
    this.bump();
    this.pump();
  }

  private bump(): void {
    this.version++;
    for (const l of this.listeners) l();
  }

  private shard(d: number, x: number, y: number): WeekShard | undefined {
    return this.shards[d * N_SHARDS + Math.floor(y / HRRR_SHARD.height) * SHARD_COLS + Math.floor(x / HRRR_SHARD.width)];
  }

  /** The weather term for every hour of the week at one pixel (length depth), NaN where not in hand. */
  series(x: number, y: number, out: Float32Array): void {
    const si = Math.floor(y / HRRR_SHARD.height) * SHARD_COLS + Math.floor(x / HRRR_SHARD.width);
    for (let d = 0; d < this.days.length; d++) {
      const s = this.shards[d * N_SHARDS + si];
      const o = d * HOURS_PER_DAY;
      if (!s) {
        out.fill(NaN, o, o + HOURS_PER_DAY);
        continue;
      }
      const base = (y - s.rowStart) * s.width + (x - s.colStart);
      const stride = s.term.length / HOURS_PER_DAY;
      for (let h = 0; h < HOURS_PER_DAY; h++) out[o + h] = TERM_LUT[s.term[h * stride + base]!]!;
    }
  }

  /** WeatherSource.at: the term at one pixel and hour. */
  at = (x: number, y: number, h: number): number => this.termAt(x, y, h);

  /** WeatherSource.plane: the term over the whole HRRR grid at one hour, NaN where not in hand (the county loops). */
  plane = (h: number): Float32Array => {
    const out = new Float32Array(HRRR_GRID.width * HRRR_GRID.height).fill(NaN);
    const d = Math.floor(h / HOURS_PER_DAY);
    const hh = h - d * HOURS_PER_DAY;
    for (let si = 0; si < N_SHARDS; si++) {
      const s = this.shards[d * N_SHARDS + si];
      if (!s) continue;
      const stride = s.term.length / HOURS_PER_DAY;
      const rows = stride / s.width;
      for (let r = 0; r < rows; r++) {
        const o = (s.rowStart + r) * HRRR_GRID.width + s.colStart;
        const q = hh * stride + r * s.width;
        for (let c = 0; c < s.width; c++) out[o + c] = TERM_LUT[s.term[q + c]!]!;
      }
    }
    return out;
  };

  /** The weather term at one pixel and hour of the week, NaN if not in hand. */
  termAt(x: number, y: number, h: number): number {
    const d = Math.floor(h / HOURS_PER_DAY);
    const s = this.shard(d, x, y);
    if (!s) return NaN;
    const stride = s.term.length / HOURS_PER_DAY;
    return TERM_LUT[s.term[(h - d * HOURS_PER_DAY) * stride + (y - s.rowStart) * s.width + (x - s.colStart)]!]!;
  }
}
