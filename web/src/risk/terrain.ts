import { FetchSource, PMTiles } from "pmtiles";
import { CHUNK, RES, X0, Y0 } from "../aef/mosaic.js";

/**
 * Slope and aspect under every 10 m pixel of a mosaic chunk, from Mapterhorn's terrain (Copernicus GLO-30, zoom 12,
 * 512 px Terrarium WebP tiles) range read from Source Cooperative. Slope is computed on Mapterhorn's own grid
 * (Horn's 3 x 3 method, about 15 m cells over California) and each 10 m pixel takes the nearest cell. GLO-30 is a
 * surface model: a lone tree or building is a spike, and Horn's window puts a steep ring around it, so the
 * elevation is first averaged over SMOOTH x SMOOTH cells (about 75 m), the hillside's slope rather than a crown's.
 * Only data: nothing here is drawn as terrain.
 */

const MAPTERHORN = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/mapterhorn/mapterhorn/planet.pmtiles";
const Z = 12;
const N = 2 ** Z;
/** Mercator meters per tile pixel at the equator for this zoom, before the 512 is known. */
const EARTH = 40075016.686;
/** Cells averaged across (odd) before the slope. */
const SMOOTH = 5;
const HALF = (SMOOTH - 1) / 2;

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
  // the tiles under the chunk plus a margin for the smoothing and the 3 x 3 window
  const tx0 = Math.floor(fx(lon0) - 0.02), tx1 = Math.floor(fx(lon1) + 0.02);
  const ty0 = Math.floor(fy(lat1) - 0.02), ty1 = Math.floor(fy(lat0) + 0.02);
  const got = new Map<string, NonNullable<Tile>>();
  await Promise.all(
    Array.from({ length: (tx1 - tx0 + 1) * (ty1 - ty0 + 1) }, (_, i) => {
      const x = tx0 + (i % (tx1 - tx0 + 1)), y = ty0 + Math.floor(i / (tx1 - tx0 + 1));
      return tile(x, y).then((t) => { if (t) got.set(`${x}/${y}`, t); });
    }),
  );
  const size = got.values().next().value?.size ?? 512;
  // elevation at a global Mapterhorn pixel (column gx, row gy at this zoom)
  const raw = (gx: number, gy: number): number => {
    const t = got.get(`${Math.floor(gx / size)}/${Math.floor(gy / size)}`);
    if (!t) return NaN;
    return t.h[(gy % size) * size + (gx % size)]!;
  };
  // the mean over SMOOTH x SMOOTH cells around it (cells with no terrain left out)
  const smooth = new Map<number, number>();
  const elev = (gx: number, gy: number): number => {
    const k = gy * N * size + gx;
    let v = smooth.get(k);
    if (v !== undefined) return v;
    let s = 0, n = 0;
    for (let dy = -HALF; dy <= HALF; dy++)
      for (let dx = -HALF; dx <= HALF; dx++) {
        const e = raw(gx + dx, gy + dy);
        if (Number.isNaN(e)) continue;
        s += e;
        n++;
      }
    v = n ? s / n : NaN;
    smooth.set(k, v);
    return v;
  };
  const latC = (lat0 + lat1) / 2;
  const cell = (EARTH / (N * size)) * Math.cos((latC * Math.PI) / 180);

  // slope and aspect once per Mapterhorn cell the chunk lands on
  const memo = new Map<number, [number, number]>();
  const at = (gx: number, gy: number): [number, number] => {
    const k = gy * N * size + gx;
    let v = memo.get(k);
    if (v) return v;
    const a = elev(gx - 1, gy - 1), b = elev(gx, gy - 1), c = elev(gx + 1, gy - 1);
    const d = elev(gx - 1, gy), f = elev(gx + 1, gy);
    const g = elev(gx - 1, gy + 1), h = elev(gx, gy + 1), i = elev(gx + 1, gy + 1);
    const dzdx = (c + 2 * f + i - (a + 2 * d + g)) / (8 * cell); // rise to the east
    const dzds = (g + 2 * h + i - (a + 2 * b + c)) / (8 * cell); // rise to the south
    const slope = (Math.atan(Math.hypot(dzdx, dzds)) * 180) / Math.PI;
    // the ground faces down the slope: east part -dzdx, north part +dzds
    const flat = Math.hypot(dzdx, dzds) < 1e-3;
    const aspect = flat ? NaN : (((Math.atan2(-dzdx, dzds) * 180) / Math.PI) + 360) % 360;
    v = [slope, aspect];
    memo.set(k, v);
    return v;
  };

  const slope = new Float32Array(CHUNK * CHUNK), aspect = new Float32Array(CHUNK * CHUNK);
  const gxs = new Int32Array(CHUNK);
  for (let c = 0; c < CHUNK; c++) gxs[c] = Math.floor(fx(lon0 + (c + 0.5) * RES) * size);
  for (let r = 0; r < CHUNK; r++) {
    const gy = Math.floor(fy(lat1 - (r + 0.5) * RES) * size);
    for (let c = 0; c < CHUNK; c++) {
      const [s, a] = at(gxs[c]!, gy);
      slope[r * CHUNK + c] = s;
      aspect[r * CHUNK + c] = a;
    }
  }
  return { slope, aspect };
}
