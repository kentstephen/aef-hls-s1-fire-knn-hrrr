import { GeoJsonLayer } from "@deck.gl/layers";
import type { Feature, MultiPolygon, Polygon } from "geojson";
import { OFFSET } from "./aef/calibration.js";
import type { CountyPart } from "./aef/coarse.js";
import { rampStripe, type GroundRamp, rampStops } from "./aef/ramp.js";
import type { WeatherSource } from "./aef/weather.js";
import { HRRR_GRID, HRRR_SHARD } from "./hrrr/metadata.js";
import { IGN_HI, IGN_LO } from "./aef/head.js";

/**
 * The pixel picture averaged by county. The coarse workers bin the ground logit per county and HRRR pixel
 * over every file of the index at one overview level (coarse.worker.ts `county`); here the hour's weather
 * term is folded in per HRRR pixel, so a county's value is the mean of exp(ground + weather) over its
 * pixels over the hide line, the share of the county the raster would light, and the mean odds, which in
 * calibrated units is ignitions per area and time. The county shapes are
 * the Census 2023 cartographic boundaries (scripts/build_counties.py).
 */
export type CountyProps = { name: string; state: string; km2: number; bbox: [number, number, number, number] };
export type CountyFeature = Feature<Polygon | MultiPolygon, CountyProps> & { id: string };

/** A lon/lat grid over CONUS with the county index (1-based) under each cell, for the workers. */
export type UnitsGrid = { ids: Uint16Array; width: number; height: number; lon0: number; lat0: number; step: number };

const LON0 = -125.5;
const LAT0 = 50;
const STEP = 0.02;
const GW = Math.ceil((-66 - LON0) / STEP);
const GH = Math.ceil((LAT0 - 24) / STEP);
/** ln(1000 km2 x 24 h): mean odds per km2 per hour to ignitions per 1000 km2 per day, with the calibration offset. */
const TO_RATE = Math.exp(OFFSET + Math.log(1000 * 24));

export async function loadCounties(baseUrl: string): Promise<CountyFeature[]> {
  const r = await fetch(`${baseUrl}counties.json`);
  if (!r.ok) throw new Error(`counties: HTTP ${r.status}`);
  const j = (await r.json()) as { features: CountyFeature[] };
  return j.features;
}

/**
 * The county under each grid cell, drawn once to an offscreen canvas with the county index in the fill
 * color. Edge cells blend two counties' colors; those fail the check byte and take a neighbor's index.
 */
export function unitsGrid(features: CountyFeature[]): UnitsGrid {
  const canvas = new OffscreenCanvas(GW, GH);
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, GW, GH);
  const ring = (coords: number[][]) => {
    coords.forEach(([lon, lat], i) => {
      const x = (lon! - LON0) / STEP;
      const y = (LAT0 - lat!) / STEP;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
  };
  features.forEach((f, i) => {
    const k = i + 1;
    const r = k & 255;
    const g = k >> 8;
    ctx.fillStyle = `rgb(${r},${g},${255 - r})`;
    ctx.beginPath();
    const polys = f.geometry.type === "Polygon" ? [f.geometry.coordinates] : f.geometry.coordinates;
    for (const poly of polys) for (const rg of poly) ring(rg);
    ctx.fill("evenodd");
  });
  const img = ctx.getImageData(0, 0, GW, GH).data;
  const ids = new Uint16Array(GW * GH);
  const n = GW * GH;
  for (let i = 0; i < n; i++) {
    const r = img[i * 4]!;
    const g = img[i * 4 + 1]!;
    const b = img[i * 4 + 2]!;
    const k = r + g * 256;
    if (k === 0) continue;
    if (b === 255 - r && k <= features.length) ids[i] = k;
    else ids[i] = 0xffff; // a blended edge cell: filled from a neighbor below
  }
  for (let i = 0; i < n; i++) {
    if (ids[i] !== 0xffff) continue;
    const x = i % GW;
    const y = (i - x) / GW;
    let got = 0;
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]] as const) {
      const xx = x + dx;
      const yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= GW || yy >= GH) continue;
      const v = ids[yy * GW + xx]!;
      if (v !== 0 && v !== 0xffff) {
        got = v;
        break;
      }
    }
    ids[i] = got;
  }
  return { ids, width: GW, height: GH, lon0: LON0, lat0: LAT0, step: STEP };
}

/** One county's cells: the HRRR pixels its ground pixels fall in, with a histogram of the ground logit in each. */
type Cells = { hx: number[]; hy: number[]; hist: Uint8Array[]; total: number };

const BINS = 16;
/** A county needs this many ground pixels (about 160 km2 at 2.56 km) to get a value: Virginia's independent cities and the
 * like, a few pixels of developed ground, otherwise read 100 percent lit and top every list. */
const MIN_PIXELS = 25;
const BIN_W = (IGN_HI - IGN_LO) / BINS;
/** Bin centers as logits, and their odds. */
const CENTER = Float64Array.from({ length: BINS }, (_, b) => IGN_LO + (b + 0.5) * BIN_W);
const CENTER_ODDS = CENTER.map((z) => Math.exp(z));

/** A county's reading at one hour. */
export type CountyHour = {
  /** The share of the county's ground pixels over the hide line, NaN under half coverage. */
  share: Float32Array;
  /** The mean odds exp(ground + weather) over the same pixels. */
  odds: Float32Array;
};

export class CountyTable {
  readonly cells: Cells[];
  /** One HRRR pixel per shard a county touches, for asking the weather in. */
  private shardPix: Map<string, [number, number]>[];
  pixelKm2 = 0;
  filesDone = 0;
  /** Bumps when the sums change (a file landed). */
  version = 0;
  private hours = new Map<string, { v: number; wv: number; out: CountyHour }>();

  constructor(readonly features: CountyFeature[]) {
    this.cells = features.map(() => ({ hx: [], hy: [], hist: [], total: 0 }));
    this.shardPix = features.map(() => new Map());
  }

  add(p: CountyPart): void {
    this.pixelKm2 = p.pixelKm2;
    for (let i = 0; i < p.unit.length; i++) {
      const c = this.cells[p.unit[i]! - 1];
      if (!c) continue;
      const hx = p.hx[i]!;
      const hy = p.hy[i]!;
      const h = p.hist.subarray(i * BINS, (i + 1) * BINS);
      let n = 0;
      for (let b = 0; b < BINS; b++) n += h[b]!;
      // the same HRRR pixel may come from two files (a zone seam): add to the existing cell
      let j = -1;
      for (let k = c.hx.length - 1; k >= 0 && k >= c.hx.length - 8; k--) if (c.hx[k] === hx && c.hy[k] === hy) { j = k; break; }
      if (j < 0) {
        c.hx.push(hx);
        c.hy.push(hy);
        c.hist.push(Uint8Array.from(h));
      } else {
        const e = c.hist[j]!;
        for (let b = 0; b < BINS; b++) e[b] = Math.min(255, e[b]! + h[b]!);
      }
      c.total += n;
      const sk = `${Math.floor(hy / HRRR_SHARD.height)},${Math.floor(hx / HRRR_SHARD.width)}`;
      const sp = this.shardPix[p.unit[i]! - 1]!;
      if (!sp.has(sk)) sp.set(sk, [hx, hy]);
    }
    this.filesDone++;
    this.version++;
    this.hours.clear();
  }

  /** The HRRR pixels (one per shard) under the counties whose boxes meet a lon/lat view. */
  pixelsInView(view: [number, number, number, number]): { px: Int16Array; py: Int16Array } {
    const xs: number[] = [];
    const ys: number[] = [];
    this.features.forEach((f, i) => {
      const b = f.properties.bbox;
      if (b[2] < view[0] || b[0] > view[2] || b[3] < view[1] || b[1] > view[3]) return;
      for (const [x, y] of this.shardPix[i]!.values()) {
        xs.push(x);
        ys.push(y);
      }
    });
    return { px: Int16Array.from(xs), py: Int16Array.from(ys) };
  }

  /**
   * Every county at one hour: the share of its ground pixels the raster lights (the hour's weather term at or
   * over the gate, and ground + weather over zHide), and the mean odds, over the pixels with weather in hand;
   * NaN under half coverage.
   * Cached per hour and hide line until the sums or the weather change (weatherVersion).
   */
  hour(h: number, zHide: number, wxGate: number, src: WeatherSource, weatherVersion: number): CountyHour {
    const key = `${h}|${zHide.toFixed(3)}|${wxGate.toFixed(3)}`;
    const hit = this.hours.get(key);
    if (hit && hit.v === this.version && hit.wv === weatherVersion) return hit.out;
    const n = this.features.length;
    const out: CountyHour = { share: new Float32Array(n).fill(NaN), odds: new Float32Array(n).fill(NaN) };
    const plane = src.plane?.(h);
    const at = src.at;
    if (plane || at) {
      for (let i = 0; i < n; i++) {
        const c = this.cells[i]!;
        let lit = 0;
        let acc = 0;
        let cnt = 0;
        for (let k = 0; k < c.hx.length; k++) {
          const w = plane ? plane[c.hy[k]! * HRRR_GRID.width + c.hx[k]!]! : at!(c.hx[k]!, c.hy[k]!, h);
          if (w !== w) continue;
          const hist = c.hist[k]!;
          const ew = Math.exp(w);
          // the first bin whose center plus the weather clears the hide line; none if the weather is under the gate
          const b0 = w >= wxGate ? Math.max(0, Math.ceil((zHide - w - IGN_LO) / BIN_W - 0.5)) : BINS;
          for (let b = 0; b < BINS; b++) {
            const m = hist[b]!;
            if (m === 0) continue;
            cnt += m;
            acc += m * CENTER_ODDS[b]! * ew;
            if (b >= b0) lit += m;
          }
        }
        if (c.total >= MIN_PIXELS && cnt >= 0.5 * c.total) {
          out.share[i] = lit / cnt;
          out.odds[i] = acc / cnt;
        }
      }
    }
    this.hours.set(key, { v: this.version, wv: weatherVersion, out });
    return out;
  }

  /** Mean odds as ignitions per 1000 km2 per day. */
  static rate(odds: number): number {
    return odds * TO_RATE;
  }
}

export function countyLayer(o: {
  features: CountyFeature[];
  share0: Float32Array;
  share1: Float32Array | null;
  frac: number;
  ramp: GroundRamp;
  opacity: number;
  visible: boolean;
  beforeId?: string;
  trigger: unknown[];
  onHover: (index: number, x: number, y: number) => void;
}): GeoJsonLayer<CountyProps> {
  const stripe = rampStripe(rampStops(o.ramp));
  // the county's color is the share of its ground the raster would light: 0 clear, 1 the ramp top; fades in under 5 percent
  const color = (i: number): [number, number, number, number] => {
    const a = o.share0[i]!;
    const b = o.share1 ? o.share1[i]! : NaN;
    const t = b === b && a === a ? a + (b - a) * o.frac : a === a ? a : b;
    if (!(t > 0)) return [0, 0, 0, 0];
    const x = Math.round(Math.min(1, t) * 255) * 4;
    return [stripe[x]!, stripe[x + 1]!, stripe[x + 2]!, Math.round(255 * Math.min(1, t / 0.05))];
  };
  const index = new Map(o.features.map((f, i) => [f.id, i]));
  return new GeoJsonLayer<CountyProps>({
    id: "counties",
    data: o.features as never,
    filled: true,
    stroked: false,
    getFillColor: ((f: CountyFeature) => color(index.get(f.id)!)) as never,
    opacity: o.opacity,
    pickable: true,
    onHover: ((info: { object?: CountyFeature; x: number; y: number }) => o.onHover(info.object ? index.get(info.object.id)! : -1, info.x, info.y)) as never,
    visible: o.visible,
    beforeId: o.beforeId,
    updateTriggers: { getFillColor: o.trigger },
  } as never);
}
