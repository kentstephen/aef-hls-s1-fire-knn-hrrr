import type { Device, Texture } from "@luma.gl/core";
import { forgetData, toHalf } from "../hrrr/get-tile-data.js";
import { HRRR_LEAD_TIME_COUNT, lonLatToHrrrPixelF } from "../hrrr/metadata.js";
import type { FfwiTileCache } from "../hrrr/tile-cache.js";

/**
 * Where a weather texture's values come from: `depth` layers (the 49 leads of a run, or the hours of
 * a week) at every HRRR pixel, NaN where not in hand.
 */
export type WeatherSource = {
  depth: number;
  inGrid(x: number, y: number): boolean;
  series(x: number, y: number, out: Float32Array): void;
  /** One pixel at one layer (hour), NaN where not in hand; for the county sums. */
  at?(x: number, y: number, layer: number): number;
  /** The whole HRRR grid at one layer (row-major, width x height), NaN where not in hand; faster than `at` in a loop. */
  plane?(layer: number): Float32Array;
};

/** The run cache as a source: the FFWI, or the weather part of the ignition logit, per lead. */
export function cacheSource(cache: FfwiTileCache, kind: WeatherKind): WeatherSource {
  return {
    depth: HRRR_LEAD_TIME_COUNT,
    inGrid: (x, y) => cache.inGrid(x, y),
    series: kind === "ignition" ? (x, y, out) => cache.ignitionSeries(x, y, out) : (x, y, out) => cache.ffwiSeries(x, y, out),
    at: kind === "ignition" ? (x, y, l) => cache.ignitionAt(x, y, l) : (x, y, l) => cache.ffwiAt(x, y, l),
  };
}

/**
 * The hour's weather on a land cover tile: a grid of nodes over the tile (about one per 800 m, at
 * least 4 x 4), each the FFWI at its spot for all 49 lead hours, read from the shared cache as a
 * bilinear blend of the four HRRR pixels around it (the pixel is 3 km; blending, rather than taking
 * the nearest pixel, keeps the field continuous across tile edges and inside a tile). Uploaded as one
 * r16float texture array the shader samples at the fragment's uv; rebuilt when the source changes
 * (a shard arrives, the fuel lag changes, a week replaces the run); the layer (hour) is a uniform.
 */
export type WeatherGrid = {
  cols: number;
  rows: number;
  /** Fractional HRRR pixel coordinates per node, row-major; NaN outside the grid. */
  fx: Float32Array;
  fy: Float32Array;
  /** The integer pixel under each node (for the side channel), -1 outside. */
  px: Int16Array;
  py: Int16Array;
};

/** Nodes at cell centers; `toLonLat` maps tile fractions (u right, v down) to lon/lat. */
export function makeGrid(extentM: number, toLonLat: (u: number, v: number) => [number, number]): WeatherGrid {
  // one node per 800 m, at least 4 x 4, at most 128 x 128 (a week's texture is 6 MB a tile at 128; the shader reads the
  // nodes bicubically, so between nodes the field is a curve whatever their spacing)
  const n = Math.max(4, Math.min(128, Math.round(extentM / 800)));
  const fx = new Float32Array(n * n);
  const fy = new Float32Array(n * n);
  const px = new Int16Array(n * n);
  const py = new Int16Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const k = j * n + i;
      const [lon, lat] = toLonLat((i + 0.5) / n, (j + 0.5) / n);
      const p = Number.isFinite(lon) && Number.isFinite(lat) ? lonLatToHrrrPixelF(lon, lat) : null;
      fx[k] = p ? p.x : NaN;
      fy[k] = p ? p.y : NaN;
      px[k] = p ? Math.floor(p.x) : -1;
      py[k] = p ? Math.floor(p.y) : -1;
    }
  }
  return { cols: n, rows: n, fx, fy, px, py };
}

/** What a weather texture holds per node and lead: the FFWI, or the weather part of the ignition logit. */
export type WeatherKind = "ffwi" | "ignition";

export type WeatherTex = {
  tex: Texture;
  /** Every node on the grid had weather for every lead. */
  complete: boolean;
};


/** Catmull-Rom weights for the four samples around a fraction t (at -1, 0, 1, 2). */
function catmull(t: number, w: Float32Array): void {
  const t2 = t * t;
  const t3 = t2 * t;
  w[0] = 0.5 * (-t3 + 2 * t2 - t);
  w[1] = 0.5 * (3 * t3 - 5 * t2 + 2);
  w[2] = 0.5 * (-3 * t3 + 4 * t2 + t);
  w[3] = 0.5 * (t3 - t2);
}
const wx4 = new Float32Array(4);
const wy4 = new Float32Array(4);

/**
 * Each HRRR pixel's series read once per texture: when the tile's nodes sit closer together than the pixels
 * (zoomed in), the 16 pixels around one node are mostly the same as its neighbors', so the series of every
 * pixel under the tile are read up front (when that block is small) and the node loop reads them from there.
 */
function pixelSeries(g: WeatherGrid, src: WeatherSource): (x: number, y: number, tmp: Float32Array) => Float32Array {
  const read = (x: number, y: number, tmp: Float32Array) => {
    src.series(x, y, tmp);
    return tmp;
  };
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let k = 0; k < g.fx.length; k++) {
    const x = g.fx[k]!;
    const y = g.fy[k]!;
    if (!Number.isFinite(x)) continue;
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
  }
  if (!Number.isFinite(x0)) return read;
  // the 4 x 4 neighborhood reaches 2 pixels past the node's center on each side
  const bx = Math.floor(x0) - 2;
  const by = Math.floor(y0) - 2;
  const bw = Math.floor(x1) + 3 - bx;
  const bh = Math.floor(y1) + 3 - by;
  // worth it only when the nodes outnumber the pixels; a block of at most 4M values (16 MB, freed with the texture)
  if (bw * bh > g.fx.length || bw * bh * src.depth > 4_000_000) return read;
  const depth = src.depth;
  const block = new Float32Array(bw * bh * depth);
  const have = new Uint8Array(bw * bh);
  return (x, y, tmp) => {
    const i = (y - by) * bw + (x - bx);
    if (x < bx || y < by || x >= bx + bw || y >= by + bh) return read(x, y, tmp);
    const view = block.subarray(i * depth, (i + 1) * depth);
    if (!have[i]) {
      src.series(x, y, view);
      have[i] = 1;
    }
    return view;
  };
}

export function weatherTexture(device: Device, g: WeatherGrid, src: WeatherSource): WeatherTex {
  const per = g.cols * g.rows;
  const depth = src.depth;
  let complete = true;
  const data = new Uint16Array(depth * per);
  const nan = toHalf(NaN);
  const out = toHalf(-Infinity);
  const acc = new Float32Array(depth);
  const wsum = new Float32Array(depth);
  const s = new Float32Array(depth);
  const series = pixelSeries(g, src);
  for (let k = 0; k < per; k++) {
    const x = g.fx[k]!;
    const y = g.fy[k]!;
    if (!Number.isFinite(x)) {
      // outside the HRRR grid: -Infinity at every hour, so the shader can tell "no weather here, ever" from
      // "not in hand yet" (NaN) and draw nothing there rather than the standing ground
      for (let l = 0; l < depth; l++) data[l * per + k] = out;
      continue;
    }
    // the 4 x 4 pixel centers around the node with Catmull-Rom weights (a smooth surface through the pixel
    // values, so a threshold on it draws a rounded line, not the diamond corners of a bilinear one); where one
    // of the 16 is outside the grid, the four nearest weighted by distance; a missing hour drops out
    const cx = x - 0.5;
    const cy = y - 0.5;
    const x0 = Math.floor(cx);
    const y0 = Math.floor(cy);
    const tx = cx - x0;
    const ty = cy - y0;
    acc.fill(0);
    wsum.fill(0);
    let cubic = true;
    for (let dy = -1; dy <= 2 && cubic; dy++) for (let dx = -1; dx <= 2; dx++) if (!src.inGrid(x0 + dx, y0 + dy)) { cubic = false; break; }
    if (cubic) {
      catmull(tx, wx4);
      catmull(ty, wy4);
      for (let dy = -1; dy <= 2; dy++) {
        for (let dx = -1; dx <= 2; dx++) {
          const w = wx4[dx + 1]! * wy4[dy + 1]!;
          const sv = series(x0 + dx, y0 + dy, s);
          for (let l = 0; l < depth; l++) {
            const v = sv[l]!;
            if (v === v) {
              acc[l] = acc[l]! + w * v;
              wsum[l] = wsum[l]! + w;
            }
          }
        }
      }
    } else {
      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          const w = (dx ? tx : 1 - tx) * (dy ? ty : 1 - ty);
          if (w <= 0) continue;
          const xx = x0 + dx;
          const yy = y0 + dy;
          if (!src.inGrid(xx, yy)) continue;
          const sv = series(xx, yy, s);
          for (let l = 0; l < depth; l++) {
            const v = sv[l]!;
            if (v === v) {
              acc[l] = acc[l]! + w * v;
              wsum[l] = wsum[l]! + w;
            }
          }
        }
      }
    }
    for (let l = 0; l < depth; l++) {
      if (wsum[l]! > 1e-6) {
        data[l * per + k] = toHalf(acc[l]! / wsum[l]!);
      } else {
        data[l * per + k] = nan;
        complete = false;
      }
    }
  }
  const tex = device.createTexture({
    dimension: "2d-array",
    format: "r16float",
    width: g.cols,
    height: g.rows,
    depth,
    mipLevels: 1,
    data,
    sampler: { minFilter: "linear", magFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
  });
  return { tex: forgetData(tex), complete };
}
