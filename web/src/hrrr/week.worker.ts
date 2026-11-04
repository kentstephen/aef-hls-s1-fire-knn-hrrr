import * as zarr from "zarrita";
import { ignitionWeather } from "../aef/head.js";
import { ffwiFromMoisture, fosbergBlock, M_SCALE, MS_TO_MPH, U16_NAN, WIND_SCALE, type WindMode } from "../ffwi/compute.js";
import { FFWI_BANDS, HRRR_GRID, HRRR_SHARD } from "./metadata.js";
import { openHrrr, type HrrrArrays } from "./store.js";
import { termByte } from "./week-term.js";

/**
 * The window's shard work off the main thread: fetch one run shard (the first 24 leads of the five bands),
 * decode it, run the Fosberg pass and keep the weather part of the ignition logit as a byte per pixel and
 * hour. Only the bytes go back (1.9 MB a shard-day, transferred); the raw floats never leave the worker.
 */
export type WeekJobMsg = { kind: "job"; id: number; initIdx: number; row: number; col: number; hours: number; windMode: WindMode };
export type WeekCancelMsg = { kind: "cancel"; id: number };
export type WeekResult =
  | { id: number; ok: true; term: Uint8Array; width: number; height: number; rowStart: number; colStart: number }
  | { id: number; ok: false; aborted: boolean; error?: string };

let arraysP: Promise<HrrrArrays> | null = null;
const ctrls = new Map<number, AbortController>();

async function run(m: WeekJobMsg): Promise<WeekResult> {
  const ctrl = new AbortController();
  ctrls.set(m.id, ctrl);
  try {
    arraysP ??= openHrrr().then((s) => s.arrays);
    const arrays = await arraysP;
    const rowStart = m.row * HRRR_SHARD.height;
    const colStart = m.col * HRRR_SHARD.width;
    const height = Math.min(rowStart + HRRR_SHARD.height, HRRR_GRID.height) - rowStart;
    const width = Math.min(colStart + HRRR_SHARD.width, HRRR_GRID.width) - colStart;
    const slices = [m.initIdx, zarr.slice(0, m.hours), zarr.slice(rowStart, rowStart + height), zarr.slice(colStart, colStart + width)];
    const chunks = await Promise.all(FFWI_BANDS.map((b) => zarr.get(arrays[b], slices, { signal: ctrl.signal })));
    if (ctrl.signal.aborted) return { id: m.id, ok: false, aborted: true };
    const n = m.hours * height * width;
    const mEq = new Uint16Array(n);
    const wind = new Uint16Array(n);
    const rh = new Uint8Array(n);
    const tc = new Uint8Array(n);
    const f = (i: number) => chunks[i]!.data as Float32Array;
    fosbergBlock(n, f(0), f(1), f(2), f(3), f(4), m.windMode, mEq, wind, rh, tc);
    // the same quantized inputs as the run cache, so the week and the run agree to the byte
    const term = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const me = mEq[i]!;
      const w = wind[i]!;
      if (me === U16_NAN || w === U16_NAN) {
        term[i] = 255;
        continue;
      }
      const mph = w / WIND_SCALE;
      term[i] = termByte(ignitionWeather(ffwiFromMoisture(me / M_SCALE, mph), tc[i]! - 100, rh[i]!, mph / MS_TO_MPH));
    }
    return { id: m.id, ok: true, term, width, height, rowStart, colStart };
  } catch (e) {
    return { id: m.id, ok: false, aborted: ctrl.signal.aborted, error: ctrl.signal.aborted ? undefined : String(e) };
  } finally {
    ctrls.delete(m.id);
  }
}

self.onmessage = (e: MessageEvent<WeekJobMsg | WeekCancelMsg>) => {
  const m = e.data;
  if (m.kind === "cancel") {
    ctrls.get(m.id)?.abort();
    return;
  }
  void run(m).then((r) => {
    if (r.ok) (self as unknown as Worker).postMessage(r, [r.term.buffer]);
    else (self as unknown as Worker).postMessage(r);
  });
};
