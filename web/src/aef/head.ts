import canopyJson from "../../../models/canopy_ridge.json";
import ignitionJson from "../../../models/ignition_logit.json";
import esriJson from "../../../models/landcover_esri_logit.json";
import nlcdJson from "../../../models/landcover_nlcd_logit.json";

/**
 * The linear heads fit offline (scripts/fit_landcover_head.py), folded so a pixel costs one dot
 * product per output: the standardization (x - mean) / scale is pushed into the weights.
 * Input per pixel: 64 AEF int8 values -> sign(q) * (q / 127.5)^2 -> unit length.
 */
export const NUM_BANDS = 64;
export const NODATA = -128;
/** Class index written where a pixel has no embedding. */
export const NO_CLASS = 255;

export type Teacher = "nlcd" | "esri";

export type HeadJson = {
  kind: string;
  teacher: string;
  mean: number[];
  scale: number[];
  coef: number[][];
  intercept: number[];
  classes?: number[];
  legend?: Record<string, { name: string; color: string }>;
  targets?: string[];
};

export type Folded = { k: number; w: Float32Array; b: Float32Array };

function fold(m: HeadJson): Folded {
  const k = m.coef.length;
  const w = new Float32Array(k * NUM_BANDS);
  const b = new Float32Array(k);
  for (let j = 0; j < k; j++) {
    let bias = m.intercept[j]!;
    for (let i = 0; i < NUM_BANDS; i++) {
      const c = m.coef[j]![i]! / m.scale[i]!;
      w[j * NUM_BANDS + i] = c;
      bias -= c * m.mean[i]!;
    }
    b[j] = bias;
  }
  return { k, w, b };
}

export type ClassHead = {
  teacher: Teacher;
  /** Teacher class codes in model order (the class index on the map is the position here). */
  classes: number[];
  names: string[];
  colors: string[];
  folded: Folded;
};

function classHead(teacher: Teacher, m: HeadJson): ClassHead {
  const classes = m.classes!;
  return {
    teacher,
    classes,
    names: classes.map((c) => m.legend![String(c)]!.name),
    colors: classes.map((c) => m.legend![String(c)]!.color),
    folded: fold(m),
  };
}

export const HEADS: Record<Teacher, ClassHead> = {
  nlcd: classHead("nlcd", nlcdJson as HeadJson),
  esri: classHead("esri", esriJson as HeadJson),
};
/** Ridge: cover 0..1, mean height m, p95 height m. Only cover is used on the map. */
export const CANOPY = fold(canopyJson as HeadJson);

/**
 * The ignition head (ignition/export_head.py): one logistic head on the pixel embedding plus the
 * hour's weather, fit on WFIGS ignitions against points 20 to 30 km away at another hour. The ground
 * part (64 bands, with the intercept) is folded here and stored per pixel as a byte over IGN_LO..IGN_HI
 * in logit units; the weather part is added per hour from the HRRR cache (tile-cache.ts). Logit in,
 * sigmoid out: the probability that this pixel at this hour is an ignition rather than a control.
 */
export type IgnitionJson = HeadJson & { weather: string[] };
export const IGN_LO = -6;
export const IGN_HI = 6;
export type IgnitionHead = {
  w: Float32Array;
  bias: number;
  weather: string[];
  wc: number[];
  wm: number[];
  /** The weather part folded to a * x + b per input (0 where the head has no such input). */
  aFfwi: number; aT: number; aRh: number; aWind: number; wBias: number;
};

function foldIgnition(m: IgnitionJson): IgnitionHead {
  const w = new Float32Array(NUM_BANDS);
  let bias = m.intercept[0]!;
  for (let i = 0; i < NUM_BANDS; i++) {
    const c = m.coef[0]![i]! / m.scale[i]!;
    w[i] = c;
    bias -= c * m.mean[i]!;
  }
  const wc = m.weather.map((_, j) => m.coef[0]![NUM_BANDS + j]! / m.scale[NUM_BANDS + j]!);
  const wm = m.weather.map((_, j) => m.mean[NUM_BANDS + j]!);
  const a: Record<string, number> = { ffwi: 0, t_c: 0, rh: 0, wind_ms: 0 };
  let wBias = 0;
  m.weather.forEach((name, j) => {
    a[name] = wc[j]!;
    wBias -= wc[j]! * wm[j]!;
  });
  return { w, bias, weather: m.weather, wc, wm, aFfwi: a.ffwi!, aT: a.t_c!, aRh: a.rh!, aWind: a.wind_ms!, wBias };
}
export const IGNITION = foldIgnition(ignitionJson as IgnitionJson);

/** The weather part of the ignition logit for one hour (wind sustained, m/s; FFWI as the map computes it). */
export function ignitionWeather(ffwi: number, tc: number, rh: number, windMs: number): number {
  const h = IGNITION;
  return h.aFfwi * ffwi + h.aT * tc + h.aRh * rh + h.aWind * windMs + h.wBias;
}

export function ignByte(logit: number): number {
  return Math.round(Math.min(1, Math.max(0, (logit - IGN_LO) / (IGN_HI - IGN_LO))) * 255);
}
export function ignLogit(byte: number): number {
  return IGN_LO + (byte / 255) * (IGN_HI - IGN_LO);
}

/**
 * The planes per pixel: two class indices, canopy cover 0..255, the ignition ground logit as a byte, and the
 * fuel head's P(fuel) as a byte (0 = none, else 1 + P * 254; fuel/head.ts). The fuel plane is filled only on
 * the 10 m mosaic, where the worker reads the fuel head's three years; the COG overviews leave it 0.
 */
export type Classified = {
  nlcd: Uint8Array; esri: Uint8Array; cover: Uint8Array; ign: Uint8Array; fuel: Uint8Array;
  /** Worker time spent reading (disk or network, and decoding) and scoring, in ms. */
  ms?: { read: number; score: number };
};

/**
 * Every pixel of a band-major int8 block [64, n] through the three heads. Argmax of the logits is
 * the class (softmax is monotone, so it is not needed for the label); cover is clamped to 0..1 and
 * stored as 0..255.
 */
export function classify(data: Int8Array, n: number): Classified {
  const nlcd = new Uint8Array(n);
  const esri = new Uint8Array(n);
  const cover = new Uint8Array(n);
  const ign = new Uint8Array(n);
  const x = new Float32Array(NUM_BANDS);
  const hn = HEADS.nlcd.folded;
  const he = HEADS.esri.folded;
  for (let p = 0; p < n; p++) {
    let norm = 0;
    let bad = false;
    for (let i = 0; i < NUM_BANDS; i++) {
      const q = data[i * n + p]!;
      if (q === NODATA) {
        bad = true;
        break;
      }
      const f = q / 127.5;
      const v = f * Math.abs(f);
      x[i] = v;
      norm += v * v;
    }
    if (bad || norm === 0) {
      nlcd[p] = NO_CLASS;
      esri[p] = NO_CLASS;
      continue;
    }
    const inv = 1 / Math.sqrt(norm);
    for (let i = 0; i < NUM_BANDS; i++) x[i]! *= inv;
    nlcd[p] = argmax(hn, x);
    esri[p] = argmax(he, x);
    const c = dot(CANOPY.w, 0, x) + CANOPY.b[0]!;
    cover[p] = Math.round(Math.min(1, Math.max(0, c)) * 255);
    ign[p] = ignByte(dot(IGNITION.w, 0, x) + IGNITION.bias);
  }
  return { nlcd, esri, cover, ign, fuel: new Uint8Array(n) };
}

function dot(w: Float32Array, j: number, x: Float32Array): number {
  const o = j * NUM_BANDS;
  let s = 0;
  for (let i = 0; i < NUM_BANDS; i++) s += w[o + i]! * x[i]!;
  return s;
}

function argmax(h: Folded, x: Float32Array): number {
  let best = 0;
  let bv = -Infinity;
  for (let j = 0; j < h.k; j++) {
    const v = dot(h.w, j, x) + h.b[j]!;
    if (v > bv) {
      bv = v;
      best = j;
    }
  }
  return best;
}
