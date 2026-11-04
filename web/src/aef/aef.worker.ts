import * as zarr from "zarrita";
import fuelJson from "../../../models/fuel_head.json";
import { type FuelHead, packWeights, scoreCpu } from "../fuel/head.js";
import { type GpuScorer, makeGpuScorer } from "../fuel/head-gpu.js";
import { classify, type Classified, NUM_BANDS } from "./head.js";
import { openAefMosaic, type AefArray, YEARS } from "./mosaic.js";

// the fuel head (scripts/export_fuel_head.py): run on the GPU (WebGPU) in this worker, the CPU without it
const FUEL = fuelJson as FuelHead;
const FUEL_W = packWeights(FUEL);
const FUEL_YEAR_IDX = FUEL.years.map((y) => YEARS.indexOf(y));
let scorer: Promise<GpuScorer | null> | null = null;
const fuelScorer = () =>
  (scorer ??= makeGpuScorer(FUEL, FUEL_W, 256 * 256).catch((e) => {
    console.warn("WebGPU fuel head unavailable, scoring on the CPU:", e);
    return null;
  }));

/**
 * The head off the page's thread: a COG block handed over as band-major int8, or a mosaic chunk read
 * here (zarrita's zstd decode of a 4 MB chunk would otherwise hold the map). Each job answers with
 * the three class planes.
 */
type Job =
  | { type: "classify"; id: number; data: Int8Array; n: number }
  | { type: "zarr"; id: number; timeIdx: number; py0: number; px0: number; h: number; w: number; fuelYearIdx?: number[] }
  | { type: "abort"; id: number };

let mosaic: Promise<AefArray> | null = null;
const running = new Map<number, AbortController>();

function reply(id: number, c: Classified) {
  postMessage({ id, ...c }, { transfer: [c.nlcd.buffer, c.esri.buffer, c.cover.buffer, c.ign.buffer, c.fuel.buffer] });
}
function fail(id: number, e: unknown) {
  const err = e as { name?: string; message?: string };
  postMessage({ id, error: { name: err?.name || "Error", message: err?.message || String(e) } });
}

self.onmessage = async (ev: MessageEvent<Job>) => {
  const m = ev.data;
  if (m.type === "abort") {
    running.get(m.id)?.abort();
    return;
  }
  if (m.type === "classify") {
    try {
      reply(m.id, classify(m.data, m.n));
    } catch (e) {
      fail(m.id, e);
    }
    return;
  }
  const ac = new AbortController();
  running.set(m.id, ac);
  try {
    if (!mosaic) mosaic = openAefMosaic().then((o) => o.arr);
    mosaic.catch(() => (mosaic = null));
    const arr = await mosaic;
    // the map's year for the class heads, and the fuel head's years (newest last): the head's own (2023 to 2025)
    // unless the job names others (the season view reads the year before its fires)
    const fy = m.fuelYearIdx ?? FUEL_YEAR_IDX;
    const ys = zarr.slice(m.py0, m.py0 + m.h), xs = zarr.slice(m.px0, m.px0 + m.w);
    const idx = [...new Set([m.timeIdx, ...fy])];
    const got = await Promise.all(idx.map((t) => zarr.get(arr, [t, null, ys, xs], { signal: ac.signal })));
    for (const c of got)
      if (c.shape.length !== 3 || c.shape[0] !== NUM_BANDS || c.shape[1] !== m.h || c.shape[2] !== m.w) {
        throw new Error(`Unexpected chunk shape [${c.shape.join(", ")}] for a ${m.w} x ${m.h} tile`);
      }
    const data = (t: number) => got[idx.indexOf(t)]!.data as Int8Array;
    const n = m.w * m.h;
    const years = fy.map(data);
    const gpu = await fuelScorer();
    const fuel = gpu ? await gpu(years, n) : scoreCpu(FUEL, FUEL_W, years, n);
    const c = classify(data(m.timeIdx), n);
    reply(m.id, { ...c, fuel });
  } catch (e) {
    fail(m.id, e);
  } finally {
    running.delete(m.id);
  }
};
