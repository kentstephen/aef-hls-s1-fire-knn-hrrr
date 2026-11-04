import { Decompress } from "fzstd";
import proj4 from "proj4";
import { cacheGet, cachePut } from "./disk-cache.js";
import { classify, type Classified, IGN_HI, IGN_LO, NO_CLASS } from "./head.js";
import { lonLatToHrrrPixel } from "../hrrr/metadata.js";
import { type AefFile, type AefIndex, FULL_WIDTH, loadAefIndex, NUM_BANDS, RECORD_BYTES } from "./index.js";
import { AEF_YEAR } from "./mosaic.js";
import { TILE, tileBounds, toLat, toLon } from "./mercator.js";

/**
 * The zoomed-out reader. A Web Mercator tile asks for the land cover at an overview level; for
 * every COG under it the level is read as one byte range (the index gives the range and each band's
 * block inside it), decoded, classified once and kept (the class planes, 3 bytes a pixel; the
 * embeddings are dropped), then resampled onto the tile by nearest pixel through the UTM
 * projection. Memory is the class planes on hand (LRU, PLANES_CAP) plus one or two levels being
 * decoded; the byte ranges are kept on disk (disk-cache.ts).
 */
type Msg =
  | { type: "init"; baseUrl: string }
  | { type: "tile"; id: number; z: number; x: number; y: number; level: number }
  | { type: "units"; ids: Uint16Array; width: number; height: number; lon0: number; lat0: number; step: number }
  | { type: "county"; id: number; fi: number; level: number }
  | { type: "abort"; id: number };

/** The county (or other unit) under each cell of a lon/lat grid, 0 = none; installed by the "units" message. */
let units: { ids: Uint16Array; width: number; height: number; lon0: number; lat0: number; step: number } | null = null;

const RANGE_TTL = 30 * 864e5;
const PLANES_CAP = 256 * 1024 * 1024;
const FETCH_SLOTS = 16;
const DECODE_SLOTS = 2;
const NODATA = -128;
const STOP = Symbol("enough");

/**
 * The first `need` bytes of a zstd frame. A block is 1024 x 1024 whatever the level, and a 16 px
 * level uses the first 16 rows, so decoding is stopped as soon as the rows wanted are out; the
 * rest of the frame (zstd blocks of up to 128 KB) is never inflated.
 */
function decodeHead(bytes: Uint8Array, need: number): Uint8Array {
  const out = new Uint8Array(need);
  let got = 0;
  const d = new Decompress((chunk) => {
    const take = Math.min(chunk.length, need - got);
    out.set(chunk.subarray(0, take), got);
    got += take;
    if (got >= need) throw STOP;
  });
  try {
    d.push(bytes, true);
  } catch (e) {
    if (e !== STOP) throw e;
  }
  if (got < need) throw new Error(`zstd block short: ${got} of ${need} bytes`);
  return out;
}

let index: Promise<AefIndex> | null = null;
let baseUrl = "/";
const zoneRanges = new Map<string, Promise<DataView>>();
const converters = new Map<string, proj4.Converter>();
const running = new Map<number, AbortController>();

type Planes = Classified & { width: number; bytes: number };
const planes = new Map<string, Planes>();
let planesBytes = 0;
const inflight = new Map<string, Promise<Planes>>();

function keep(key: string, p: Planes) {
  planes.set(key, p);
  planesBytes += p.bytes;
  while (planesBytes > PLANES_CAP && planes.size > 1) {
    const k = planes.keys().next().value as string;
    planesBytes -= planes.get(k)!.bytes;
    planes.delete(k);
  }
}
function touch(key: string): Planes | undefined {
  const p = planes.get(key);
  if (p) {
    planes.delete(key);
    planes.set(key, p);
  }
  return p;
}

/** A counting semaphore. */
function slots(n: number) {
  let free = n;
  const waiting: (() => void)[] = [];
  return async <T>(f: () => Promise<T>): Promise<T> => {
    if (free <= 0) await new Promise<void>((ok) => waiting.push(ok));
    free--;
    try {
      return await f();
    } finally {
      free++;
      waiting.shift()?.();
    }
  };
}
const fetchSlot = slots(FETCH_SLOTS);
const decodeSlot = slots(DECODE_SLOTS);

function ranges(zone: string): Promise<DataView> {
  let p = zoneRanges.get(zone);
  if (!p) {
    p = fetch(`${baseUrl}aef-${AEF_YEAR}-ranges-${zone}.bin`).then(async (r) => {
      if (!r.ok) throw new Error(`ranges ${zone}: HTTP ${r.status}`);
      return new DataView(await r.arrayBuffer());
    });
    p.catch(() => zoneRanges.delete(zone));
    zoneRanges.set(zone, p);
  }
  return p;
}

async function rangeGet(url: string, start: number, length: number): Promise<Uint8Array> {
  const key = `cog|${url}|${start}+${length}`;
  const have = await cacheGet(key);
  if (have) return have;
  const bytes = await fetchSlot(async () => {
    for (let k = 0; ; k++) {
      try {
        const r = await fetch(url, { headers: { Range: `bytes=${start}-${start + length - 1}` } });
        if (r.status !== 206) throw new Error(`HTTP ${r.status}`);
        return new Uint8Array(await r.arrayBuffer());
      } catch (e) {
        if (k >= 3) throw e;
        await new Promise((go) => setTimeout(go, 1000 * 2 ** k));
      }
    }
  });
  cachePut(key, bytes, RANGE_TTL);
  return bytes;
}

/** The class planes of one file at one level, read once. */
function fileLevel(idx: AefIndex, f: AefFile, level: number): Promise<Planes> {
  const key = `${f.rel}:${level}`;
  const have = touch(key);
  if (have) return Promise.resolve(have);
  let p = inflight.get(key);
  if (!p) {
    p = (async () => {
      const width = idx.levels[level]!;
      const n = width * width;
      const dv = await ranges(f.zone);
      const rec = f.ri * RECORD_BYTES;
      const start = dv.getFloat64(rec + level * 8, true);
      const blk = rec + 7 * 8 + level * 64 * 8;
      let span = 0;
      for (let b = 0; b < NUM_BANDS; b++) {
        const rel = dv.getUint32(blk + b * 8, true);
        const cnt = dv.getUint32(blk + b * 8 + 4, true);
        if (cnt > 0) span = Math.max(span, rel + cnt);
      }
      let out: Classified;
      if (span === 0) {
        out = { nlcd: new Uint8Array(n).fill(NO_CLASS), esri: new Uint8Array(n).fill(NO_CLASS), cover: new Uint8Array(n), ign: new Uint8Array(n), fuel: new Uint8Array(n) };
      } else {
        const bytes = await rangeGet(idx.base + f.rel, start, span);
        out = await decodeSlot(async () => {
          const data = new Int8Array(NUM_BANDS * n);
          const block = idx.block;
          for (let b = 0; b < NUM_BANDS; b++) {
            const rel = dv.getUint32(blk + b * 8, true);
            const cnt = dv.getUint32(blk + b * 8 + 4, true);
            if (cnt === 0) {
              data.fill(NODATA, b * n, (b + 1) * n);
              continue;
            }
            const raw = decodeHead(bytes.subarray(rel, rel + cnt), width * block);
            const i8 = new Int8Array(raw.buffer, raw.byteOffset, raw.length);
            // the block is `block` px wide; the level fills its top-left width x width
            for (let r = 0; r < width; r++) data.set(i8.subarray(r * block, r * block + width), b * n + r * width);
          }
          return classify(data, n);
        });
      }
      const pl: Planes = { ...out, width, bytes: 4 * n };
      keep(key, pl);
      return pl;
    })();
    p.catch(() => undefined).finally(() => inflight.delete(key));
    inflight.set(key, p);
  }
  return p;
}

function converter(epsg: number): proj4.Converter {
  const k = String(epsg);
  let c = converters.get(k);
  if (!c) {
    c = proj4("EPSG:4326", `+proj=utm +zone=${epsg - 32600} +datum=WGS84 +units=m +no_defs`);
    converters.set(k, c);
  }
  return c;
}

async function tile(m: { id: number; z: number; x: number; y: number; level: number }, signal: AbortSignal) {
  const idx = await (index ??= loadAefIndex(baseUrl));
  const [minX, minY, maxX, maxY] = tileBounds(m.z, m.x, m.y);
  const w = toLon(minX);
  const e = toLon(maxX);
  const s = toLat(minY);
  const n = toLat(maxY);
  const files = idx.tiles.filter((f) => f.bbox[0] < e && f.bbox[2] > w && f.bbox[1] < n && f.bbox[3] > s);
  const width = idx.levels[m.level]!;
  const res = (10 * FULL_WIDTH) / width;
  const got = await Promise.all(files.map((f) => fileLevel(idx, f, m.level)));
  if (signal.aborted) throw Object.assign(new Error("aborted"), { name: "AbortError" });
  const np = TILE * TILE;
  const nlcd = new Uint8Array(np).fill(NO_CLASS);
  const esri = new Uint8Array(np).fill(NO_CLASS);
  const cover = new Uint8Array(np);
  const ign = new Uint8Array(np);
  // UTM of every pixel center by zone, from a 9 x 9 grid of projected points (UTM is smooth)
  const G = 8;
  const zones = [...new Set(files.map((f) => f.zone))];
  const grids = new Map<string, Float64Array>();
  for (const zone of zones) {
    const c = converter(idx.zones[zone]!.epsg);
    const g = new Float64Array((G + 1) * (G + 1) * 2);
    for (let j = 0; j <= G; j++) {
      for (let i = 0; i <= G; i++) {
        const lon = toLon(minX + ((maxX - minX) * i) / G);
        const lat = toLat(maxY - ((maxY - minY) * j) / G);
        const [ux, uy] = c.forward([lon, lat]);
        g[(j * (G + 1) + i) * 2] = ux!;
        g[(j * (G + 1) + i) * 2 + 1] = uy!;
      }
    }
    grids.set(zone, g);
  }
  const cell = TILE / G;
  for (let py = 0; py < TILE; py++) {
    const v = (py + 0.5) / cell;
    const j0 = Math.min(G - 1, Math.floor(v));
    const fv = v - j0;
    for (let px = 0; px < TILE; px++) {
      const u = (px + 0.5) / cell;
      const i0 = Math.min(G - 1, Math.floor(u));
      const fu = u - i0;
      const o = py * TILE + px;
      for (let k = 0; k < files.length; k++) {
        const f = files[k]!;
        const g = grids.get(f.zone)!;
        const a = (j0 * (G + 1) + i0) * 2;
        const b = a + 2;
        const c2 = a + (G + 1) * 2;
        const d = c2 + 2;
        const ux = (g[a]! * (1 - fu) + g[b]! * fu) * (1 - fv) + (g[c2]! * (1 - fu) + g[d]! * fu) * fv;
        const uy = (g[a + 1]! * (1 - fu) + g[b + 1]! * fu) * (1 - fv) + (g[c2 + 1]! * (1 - fu) + g[d + 1]! * fu) * fv;
        const col = Math.floor((ux - f.utmWest) / res);
        const row = Math.floor((uy - f.utmSouth) / res);
        if (col < 0 || row < 0 || col >= width || row >= width) continue;
        const pl = got[k]!;
        const q = row * width + col;
        const cls = pl.nlcd[q]!;
        if (cls === NO_CLASS) continue;
        nlcd[o] = cls;
        esri[o] = pl.esri[q]!;
        cover[o] = pl.cover[q]!;
        ign[o] = pl.ign[q]!;
        break;
      }
    }
  }
  postMessage({ id: m.id, nlcd, esri, cover, ign }, { transfer: [nlcd.buffer, esri.buffer, cover.buffer, ign.buffer] });
}

/** The county sums bin the ground logit: BINS bins over IGN_LO..IGN_HI (0.75 logit each). */
export const COUNTY_BINS = 16;

/**
 * One file's ignition ground by county and HRRR pixel at an overview level: for every pixel with ground, a
 * count in its (county, HRRR pixel) cell's histogram of the ground logit. The main thread folds the hour's
 * weather in later: the share of the county's pixels over the hide line, and the mean odds. Sparse arrays
 * back: county index, HRRR x, y, and COUNTY_BINS counts per cell.
 */
async function county(m: { id: number; fi: number; level: number }) {
  const idx = await (index ??= loadAefIndex(baseUrl));
  const f = idx.tiles[m.fi];
  if (!f || !units) throw new Error("no file or no units");
  const pl = await fileLevel(idx, f, m.level);
  const width = pl.width;
  const res = (10 * FULL_WIDTH) / width;
  const c = converter(idx.zones[f.zone]!.epsg);
  const cells = new Map<number, { u: number; hx: number; hy: number; hist: Uint8Array }>();
  const u = units;
  const binW = (IGN_HI - IGN_LO) / COUNTY_BINS;
  for (let row = 0; row < width; row++) {
    const uy = f.utmSouth + (row + 0.5) * res;
    for (let col = 0; col < width; col++) {
      const q = row * width + col;
      if (pl.nlcd[q] === NO_CLASS) continue;
      const [lon, lat] = c.inverse([f.utmWest + (col + 0.5) * res, uy]) as [number, number];
      const gx = Math.floor((lon - u.lon0) / u.step);
      const gy = Math.floor((u.lat0 - lat) / u.step);
      if (gx < 0 || gy < 0 || gx >= u.width || gy >= u.height) continue;
      const unit = u.ids[gy * u.width + gx]!;
      if (unit === 0) continue;
      const h = lonLatToHrrrPixel(lon, lat);
      if (!h) continue;
      const key = (unit * 2048 + h.y) * 2048 + h.x;
      let cell = cells.get(key);
      if (!cell) {
        cell = { u: unit, hx: h.x, hy: h.y, hist: new Uint8Array(COUNTY_BINS) };
        cells.set(key, cell);
      }
      // the byte is the logit over IGN_LO..IGN_HI; its bin
      const bin = Math.min(COUNTY_BINS - 1, Math.floor(((pl.ign[q]! / 255) * (IGN_HI - IGN_LO)) / binW));
      if (cell.hist[bin]! < 255) cell.hist[bin]!++;
    }
  }
  const k = cells.size;
  const unit = new Uint16Array(k);
  const hx = new Int16Array(k);
  const hy = new Int16Array(k);
  const hist = new Uint8Array(k * COUNTY_BINS);
  let i = 0;
  for (const cell of cells.values()) {
    unit[i] = cell.u;
    hx[i] = cell.hx;
    hy[i] = cell.hy;
    hist.set(cell.hist, i * COUNTY_BINS);
    i++;
  }
  postMessage({ id: m.id, county: { unit, hx, hy, hist, pixelKm2: (res * res) / 1e6 } }, { transfer: [unit.buffer, hx.buffer, hy.buffer, hist.buffer] });
}

self.onmessage = (ev: MessageEvent<Msg>) => {
  const m = ev.data;
  if (m.type === "init") {
    baseUrl = m.baseUrl;
    index ??= loadAefIndex(baseUrl);
    return;
  }
  if (m.type === "abort") {
    running.get(m.id)?.abort();
    return;
  }
  if (m.type === "units") {
    units = m;
    return;
  }
  if (m.type === "county") {
    county(m).catch((e: { name?: string; message?: string }) => postMessage({ id: m.id, error: { name: e?.name || "Error", message: e?.message || String(e) } }));
    return;
  }
  const ac = new AbortController();
  running.set(m.id, ac);
  tile(m, ac.signal)
    .catch((e: { name?: string; message?: string }) => postMessage({ id: m.id, error: { name: e?.name || "Error", message: e?.message || String(e) } }))
    .finally(() => running.delete(m.id));
};
