import { FetchSource, PMTiles } from "pmtiles";
import { CHUNK, RES, X0, Y0 } from "../aef/mosaic.js";

/**
 * Slope and aspect under every 10 m pixel of a mosaic chunk, from Mapterhorn's terrain (Copernicus GLO-30, zoom 12,
 * 512 px Terrarium WebP tiles) range read from Source Cooperative. Slope is computed on Mapterhorn's own grid
 * (Horn's 3 x 3 method, about 15 m cells over California). GLO-30 is a surface model: a lone tree or building is a
 * spike, and Horn's window puts a steep ring around it (its south-west half then reads as sunny-facing, an L).
 * So the elevation is cleaned first: a MEDIAN x MEDIAN median drops spikes smaller than about half its window
 * (a median keeps no trace of them, where an average spreads each into a square), then a Gaussian of SIGMA cells
 * (round, no square edges) leaves the hillside. Each 10 m pixel blends the four cells around it.
 * Only data: nothing here is drawn as terrain.
 */

const MAPTERHORN = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/mapterhorn/mapterhorn/planet.pmtiles";
const Z = 12;
const N = 2 ** Z;
/** Mercator meters per tile pixel at the equator for this zoom, before the 512 is known. */
const EARTH = 40075016.686;
/** Median window (cells, odd). */
const MEDIAN = 5;
/** Gaussian width (cells) after the median, and how far it reaches. */
const SIGMA = 2;
const GR = 3 * SIGMA;
/** Cells read past the pixels' own: the median, the Gaussian, Horn's window and the blend. */
const MARGIN = (MEDIAN - 1) / 2 + GR + 2;

let planet: PMTiles | null = null;
type Tile = { h: Float32Array; size: number } | null;
const tiles = new Map<string, Promise<Tile>>();

function tile(x: number, y: number): Promise<Tile> {
  const k = `${x}/${y}`;
  let p = tiles.get(k);
  if (!p) {
    planet ??= new PMTiles(new FetchSource(MAPTERHORN));
    p = planet.getZxy(Z, x, y).then(async (r) => {
      if (!r?.data) return null;
      const bmp = await createImageBitmap(new Blob([r.data], { type: "image/webp" }), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
      const c = new OffscreenCanvas(bmp.width, bmp.height);
      const ctx = c.getContext("2d", { willReadFrequently: true })!;
      ctx.drawImage(bmp, 0, 0);
      const px = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
      const h = new Float32Array(bmp.width * bmp.height);
      for (let i = 0; i < h.length; i++) h[i] = px[4 * i]! * 256 + px[4 * i + 1]! + px[4 * i + 2]! / 256 - 32768;
      return { h, size: bmp.width };
    });
    p.catch(() => tiles.delete(k));
    tiles.set(k, p);
  }
  return p;
}

const fx = (lon: number) => ((lon + 180) / 360) * N;
const fy = (lat: number) => {
  const r = (lat * Math.PI) / 180;
  return ((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * N;
};

/** slope: degrees (NaN where no terrain); aspect: degrees clockwise from north the ground faces (NaN on flat ground). */
export type ChunkTerrain = { slope: Float32Array; aspect: Float32Array };

export async function chunkTerrain(cy: number, cx: number): Promise<ChunkTerrain> {
  const lon0 = X0 + cx * CHUNK * RES, lon1 = lon0 + CHUNK * RES;
  const lat1 = Y0 - cy * CHUNK * RES, lat0 = lat1 - CHUNK * RES;
  // the tiles under the chunk plus a margin for the cleaning and the windows
  const tx0 = Math.floor(fx(lon0) - 0.05), tx1 = Math.floor(fx(lon1) + 0.05);
  const ty0 = Math.floor(fy(lat1) - 0.05), ty1 = Math.floor(fy(lat0) + 0.05);
  const got = new Map<string, NonNullable<Tile>>();
  await Promise.all(
    Array.from({ length: (tx1 - tx0 + 1) * (ty1 - ty0 + 1) }, (_, i) => {
      const x = tx0 + (i % (tx1 - tx0 + 1)), y = ty0 + Math.floor(i / (tx1 - tx0 + 1));
      return tile(x, y).then((t) => { if (t) got.set(`${x}/${y}`, t); });
    }),
  );
  const size = got.values().next().value?.size ?? 512;

  // the window of Mapterhorn cells (global column / row at this zoom) the chunk needs
  const gx0 = Math.floor(fx(lon0) * size) - MARGIN, gx1 = Math.ceil(fx(lon1) * size) + MARGIN;
  const gy0 = Math.floor(fy(lat1) * size) - MARGIN, gy1 = Math.ceil(fy(lat0) * size) + MARGIN;
  const W = gx1 - gx0 + 1, H = gy1 - gy0 + 1;
  const raw = new Float32Array(W * H);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const gx = gx0 + x, gy = gy0 + y;
      const t = got.get(`${Math.floor(gx / size)}/${Math.floor(gy / size)}`);
      raw[y * W + x] = t ? t.h[(gy % size) * size + (gx % size)]! : NaN;
    }

  // median over MEDIAN x MEDIAN (cells with no terrain left out)
  const med = new Float32Array(W * H);
  const win = new Float32Array(MEDIAN * MEDIAN);
  const mh = (MEDIAN - 1) / 2;
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let k = 0;
      for (let dy = -mh; dy <= mh; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= H) continue;
        for (let dx = -mh; dx <= mh; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= W) continue;
          const e = raw[yy * W + xx]!;
          if (!Number.isNaN(e)) win[k++] = e;
        }
      }
      if (!k) { med[y * W + x] = NaN; continue; }
      const w = win.subarray(0, k).sort();
      med[y * W + x] = k % 2 ? w[(k - 1) / 2]! : (w[k / 2 - 1]! + w[k / 2]!) / 2;
    }

  // Gaussian, across then down (weights of cells with no terrain left out)
  const kern = Float32Array.from({ length: 2 * GR + 1 }, (_, i) => Math.exp(-((i - GR) ** 2) / (2 * SIGMA * SIGMA)));
  const blur = (src: Float32Array, dx: number, dy: number) => {
    const out = new Float32Array(W * H);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        let s = 0, w = 0;
        for (let i = -GR; i <= GR; i++) {
          const xx = x + i * dx, yy = y + i * dy;
          if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;
          const e = src[yy * W + xx]!;
          if (Number.isNaN(e)) continue;
          s += e * kern[i + GR]!;
          w += kern[i + GR]!;
        }
        out[y * W + x] = w ? s / w : NaN;
      }
    return out;
  };
  const elev = blur(blur(med, 1, 0), 0, 1);

  // rise to the east and to the south per cell (Horn), NaN at the window's edge
  const latC = (lat0 + lat1) / 2;
  const cell = (EARTH / (N * size)) * Math.cos((latC * Math.PI) / 180);
  const ex = new Float32Array(W * H).fill(NaN), es = new Float32Array(W * H).fill(NaN);
  for (let y = 1; y < H - 1; y++)
    for (let x = 1; x < W - 1; x++) {
      const e = (dx: number, dy: number) => elev[(y + dy) * W + x + dx]!;
      ex[y * W + x] = (e(1, -1) + 2 * e(1, 0) + e(1, 1) - (e(-1, -1) + 2 * e(-1, 0) + e(-1, 1))) / (8 * cell);
      es[y * W + x] = (e(-1, 1) + 2 * e(0, 1) + e(1, 1) - (e(-1, -1) + 2 * e(0, -1) + e(1, -1))) / (8 * cell);
    }

  // each 10 m pixel: the rises blended from the four cell centers around it, then slope and aspect
  const slope = new Float32Array(CHUNK * CHUNK), aspect = new Float32Array(CHUNK * CHUNK);
  const us = new Float32Array(CHUNK);
  for (let c = 0; c < CHUNK; c++) us[c] = fx(lon0 + (c + 0.5) * RES) * size - 0.5 - gx0;
  for (let r = 0; r < CHUNK; r++) {
    const v = fy(lat1 - (r + 0.5) * RES) * size - 0.5 - gy0;
    const y0 = Math.floor(v), fv = v - y0;
    for (let c = 0; c < CHUNK; c++) {
      const u = us[c]!, x0 = Math.floor(u), fu = u - x0;
      const i = y0 * W + x0;
      const lerp = (a: Float32Array) =>
        (a[i]! * (1 - fu) + a[i + 1]! * fu) * (1 - fv) + (a[i + W]! * (1 - fu) + a[i + W + 1]! * fu) * fv;
      const dzdx = lerp(ex), dzds = lerp(es);
      const g = Math.hypot(dzdx, dzds);
      slope[r * CHUNK + c] = (Math.atan(g) * 180) / Math.PI;
      // the ground faces down the slope: east part -dzdx, north part +dzds
      aspect[r * CHUNK + c] = g < 1e-3 || Number.isNaN(g) ? NaN : (((Math.atan2(-dzdx, dzds) * 180) / Math.PI) + 360) % 360;
    }
  }
  return { slope, aspect };
}
