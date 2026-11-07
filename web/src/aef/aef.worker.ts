import * as zarr from "zarrita";
import fuelJson from "../../../models/fuel_head.json";
import { type FuelHead, packWeights, scoreCpu } from "../fuel/head.js";
import { type GpuScorer, makeGpuScorer } from "../fuel/head-gpu.js";
import volumeJson from "../../../models/fuel_volume.json";
import { packVolume, probeGap, volumeCpu, type VolumeHead } from "../fuel/volume.js";
import { makeVolumeScorer, type VolumeScorer } from "../fuel/volume-gpu.js";
import { classify, classifyLean, type Classified, NUM_BANDS } from "./head.js";
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

// the fuel volume head (scripts/fit_volume.py): one year's embedding -> how much stands on the pixel
const VOLUME = volumeJson as VolumeHead;
const VOLUME_W = packVolume(VOLUME);
let volScorer: Promise<VolumeScorer | null> | null = null;
const volumeScorer = () =>
  (volScorer ??= (async () => {
    const gap = probeGap(VOLUME, VOLUME_W);
    if (gap > 1e-3) console.warn(`fuel volume head: the browser's forward pass is ${gap} off the export's probes`);
    return makeVolumeScorer(VOLUME, VOLUME_W, 256 * 256);
  })().catch((e) => {
    console.warn("WebGPU fuel volume head unavailable, scoring on the CPU:", e);
    return null;
  }));
const scoreVolume = async (e: Int8Array, n: number) => {
  const g = await volumeScorer();
  return g ? g(e, n) : volumeCpu(VOLUME, VOLUME_W, e, n);
};

/**
 * The head off the page's thread: a COG block handed over as band-major int8, or a mosaic chunk read
 * here (zarrita's zstd decode of a 4 MB chunk would otherwise hold the map). Each job answers with
 * the three class planes.
 */
type Job =
  | { type: "classify"; id: number; data: Int8Array; n: number }
  | { type: "zarr"; id: number; timeIdx: number; py0: number; px0: number; h: number; w: number; fuelYearIdx?: number[]; value?: "pfuel" | "volume"; lean?: boolean }
  | { type: "raw"; id: number; years: number[]; py0: number; px0: number; h: number; w: number }
  | { type: "abort"; id: number };

let mosaic: Promise<AefArray> | null = null;
const running = new Map<number, AbortController>();

/**
 * Chunk-years read lately, decoded, least recently used dropped past CACHE_CAP: the fuel head reads each year with
 * the two before it, so the next year over (the map stepping a year, the fire bars) needs only one new read. The
 * pool sends a chunk to the same worker every time, so this worker is the one that holds it.
 */
const CACHE_CAP = 192 * 1024 ** 2;
const cache = new Map<string, Int8Array>();
let cacheBytes = 0;
/** Reads in flight, shared by every job that wants the same chunk-year; aborted once no job wants it. */
const inflight = new Map<string, { p: Promise<Int8Array>; ac: AbortController; wants: number }>();

function keep(k: string, d: Int8Array) {
  cache.set(k, d);
  cacheBytes += d.byteLength;
  for (const [old, v] of cache) {
    if (cacheBytes <= CACHE_CAP) break;
    cache.delete(old);
    cacheBytes -= v.byteLength;
  }
}

/**
 * In development, a chunk kept on disk by scripts/replay_download.py (served at /local-aef/ by vite.config.ts), the
 * same band-major int8 block the mosaic gives; null when it is not there, and the mosaic is read instead.
 */
async function fromLocal(t: number, py0: number, px0: number, h: number, w: number, signal: AbortSignal): Promise<Int8Array | null> {
  if (h !== CHUNK || w !== CHUNK || py0 % CHUNK || px0 % CHUNK) return null;
  try {
    const r = await fetch(`/local-aef/${YEARS[t]}/${py0 / CHUNK}_${px0 / CHUNK}.bin`, { signal, cache: "no-store" });
    if (!r.ok) return null;
    const b = await r.arrayBuffer();
    return b.byteLength === NUM_BANDS * h * w ? new Int8Array(b) : null;
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    return null;
  }
}
const CHUNK = 256;

/** keepIt false: the chunk is not kept for reuse (a view that reads each chunk once). */
async function chunkYear(arr: AefArray, t: number, py0: number, px0: number, h: number, w: number, signal: AbortSignal, keepIt = true): Promise<Int8Array> {
  const k = `${t}:${py0}:${px0}:${h}:${w}`;
  const hit = cache.get(k);
  if (hit) {
    cache.delete(k);
    cache.set(k, hit);
    return hit;
  }
  let f = inflight.get(k);
  if (!f || f.ac.signal.aborted) {
    const ac = new AbortController();
    const fromZarr = () => zarr.get(arr, [t, null, zarr.slice(py0, py0 + h), zarr.slice(px0, px0 + w)], { signal: ac.signal }).then((c) => {
      if (c.shape.length !== 3 || c.shape[0] !== NUM_BANDS || c.shape[1] !== h || c.shape[2] !== w) {
        throw new Error(`Unexpected chunk shape [${c.shape.join(", ")}] for a ${w} x ${h} tile`);
      }
      return c.data as Int8Array;
    });
    const p = (import.meta.env.DEV ? fromLocal(t, py0, px0, h, w, ac.signal).then((d) => d ?? fromZarr()) : fromZarr()).then((d) => {
      if (keepIt) keep(k, d);
      return d;
    });
    f = { p, ac, wants: 0 };
    inflight.set(k, f);
    const done = () => { if (inflight.get(k) === f) inflight.delete(k); };
    p.then(done, done);
  }
  const g = f;
  g.wants++;
  const drop = () => { if (--g.wants === 0) g.ac.abort(); };
  signal.addEventListener("abort", drop, { once: true });
  try {
    return await g.p;
  } finally {
    signal.removeEventListener("abort", drop);
    if (!signal.aborted) g.wants--;
  }
}

function reply(id: number, c: Classified) {
  const extra = [c.vx?.buffer, c.v0?.buffer].filter((b): b is ArrayBuffer => !!b);
  postMessage({ id, ...c }, { transfer: [c.nlcd.buffer, c.esri.buffer, c.cover.buffer, c.ign.buffer, c.fuel.buffer, ...extra] });
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
  if (m.type === "raw") {
    // the chunk's raw int8 blocks for the years asked, one after another, for a GPU pass on the page (replay-gpu.ts)
    try {
      if (!mosaic) mosaic = openAefMosaic().then((o) => o.arr);
      mosaic.catch(() => (mosaic = null));
      const arr = await mosaic;
      const got = await Promise.all(m.years.map((t) => chunkYear(arr, t, m.py0, m.px0, m.h, m.w, ac.signal, false)));
      const n = NUM_BANDS * m.h * m.w;
      const out = new Int8Array(n * got.length);
      got.forEach((d, i) => out.set(d, i * n));
      postMessage({ id: m.id, raw: out }, { transfer: [out.buffer] });
    } catch (e) {
      fail(m.id, e);
    } finally {
      running.delete(m.id);
    }
    return;
  }
  try {
    if (!mosaic) mosaic = openAefMosaic().then((o) => o.arr);
    mosaic.catch(() => (mosaic = null));
    const arr = await mosaic;
    // the map's year for the class heads, and the fuel head's years (newest last): the head's own (2023 to 2025)
    // unless the job names others (the season view reads the year before its fires)
    const fy = m.fuelYearIdx ?? FUEL_YEAR_IDX;
    const idx = [...new Set([m.timeIdx, ...fy])];
    const t0 = performance.now();
    const got = await Promise.all(idx.map((t) => chunkYear(arr, t, m.py0, m.px0, m.h, m.w, ac.signal, !m.lean)));
    const t1 = performance.now();
    const data = (t: number) => got[idx.indexOf(t)]!;
    const n = m.w * m.h;
    const years = fy.map(data);
    const c = m.lean ? classifyLean(data(m.timeIdx), n) : classify(data(m.timeIdx), n);
    if (m.value === "volume") {
      // the volume of the newest year as the drawn plane, its packed outputs for the tooltip, and the volume
      // two years before (the oldest of the years read) for what a fire took
      const vx = await scoreVolume(years[years.length - 1]!, n);
      const before = await scoreVolume(years[0]!, n);
      const fuel = new Uint8Array(n);
      const v0 = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
        fuel[i] = vx[i]! & 255;
        v0[i] = before[i]! & 255;
      }
      reply(m.id, { ...c, fuel, vx, v0, ms: { read: t1 - t0, score: performance.now() - t1 } });
      return;
    }
    const gpu = await fuelScorer();
    const fuel = gpu ? await gpu(years, n) : scoreCpu(FUEL, FUEL_W, years, n);
    reply(m.id, { ...c, fuel, ms: { read: t1 - t0, score: performance.now() - t1 } });
  } catch (e) {
    fail(m.id, e);
  } finally {
    running.delete(m.id);
  }
};
