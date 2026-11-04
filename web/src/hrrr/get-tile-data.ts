import type { MinimalTileData } from "@developmentseed/deck.gl-raster";
import type { GetTileDataOptions } from "@developmentseed/deck.gl-zarr";
import type { Device, Texture } from "@luma.gl/core";
import * as zarr from "zarrita";
import { fosbergBlock, M_SCALE, MS_TO_MPH, U16_NAN, WIND_SCALE, type WindMode } from "../ffwi/compute.js";
import { FFWI_BANDS, HRRR_LEAD_TIME_COUNT, HRRR_SHARD } from "./metadata.js";
import type { HrrrArrays } from "./store.js";
import type { FfwiTile, FfwiTileCache } from "./tile-cache.js";
import type { RasterVariable } from "./variables.js";

/**
 * Per-tile data handed to renderTile. For the FFWI variable: equilibrium
 * moisture (%) and wind (mph) as two r16float Texture2DArrays (49 leads),
 * combined per fragment by `gpu/lagged-ffwi.ts` so the fuel-moisture timelag
 * is a uniform, not a re-upload. For the other variables: one r32float
 * Texture2DArray of the field in native units.
 */
export type FfwiTileData = MinimalTileData & {
  texture: Texture;
  /** Present for the FFWI variable only (then `texture` is moisture). */
  windTexture?: Texture;
  tileX: number;
  tileY: number;
};

export type MakeGetTileDataArgs = {
  arrays: HrrrArrays;
  cache: FfwiTileCache;
  windMode: WindMode;
  variable: RasterVariable;
  initTimeIdx: number;
};

const halfF = new Float32Array(1);
const halfI = new Uint32Array(halfF.buffer);

/** IEEE half-float bits for a float32 (round-to-nearest-even; NaN/Inf preserved). Called per texel: no allocation. */
export function toHalf(x: number): number {
  if (Number.isNaN(x)) return 0x7e00;
  halfF[0] = x;
  const b = halfI[0]!;
  const sign = (b >>> 16) & 0x8000;
  let exp = ((b >>> 23) & 0xff) - 127 + 15;
  let mant = b & 0x7fffff;
  if (exp >= 31) return sign | 0x7c00;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - exp;
    let half = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (rem > halfway || (rem === halfway && (half & 1))) half++;
    return sign | half;
  }
  let half = sign | (exp << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (half & 1))) half++;
  return half;
}

/**
 * Let go of the array a texture was made from. luma.gl keeps every prop a resource was created with, the
 * upload data included, so without this each texture pins a CPU copy of itself for as long as it lives
 * (a week's weather texture is 2 MB a tile). The data is read once, at creation.
 */
export function forgetData(tex: Texture): Texture {
  // the resource's props, and the copy its default view keeps
  const t = tex as unknown as { props: { data?: unknown }; view?: { props?: { data?: unknown } } };
  t.props.data = undefined;
  if (t.view?.props) t.view.props.data = undefined;
  return tex;
}

function halfTexture(device: Device, width: number, height: number, values: Float32Array): Texture {
  const data = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) data[i] = toHalf(values[i]!);
  return forgetData(device.createTexture({
    dimension: "2d-array",
    format: "r16float",
    width,
    height,
    depth: HRRR_LEAD_TIME_COUNT,
    mipLevels: 1,
    data,
    sampler: { minFilter: "nearest", magFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
  }));
}

/**
 * Build the ZarrLayer `getTileData` callback. Fetches the five FFWI inputs
 * for the tile's sliceSpec in parallel, runs the Fosberg pass for all 49
 * leads on the CPU (`ffwi/compute.ts`, the single source of truth), drops the
 * block into the JS cache, and uploads the GPU textures for the selected
 * variable. The `arr` argument ZarrLayer passes is ignored: it is just the
 * layout array (temperature_2m).
 */
export function makeGetTileData(args: MakeGetTileDataArgs) {
  const { arrays, cache, windMode, variable } = args;
  return async function getTileData(
    _arr: zarr.Array<"float32", zarr.Readable>,
    options: GetTileDataOptions,
  ): Promise<FfwiTileData> {
    const { device, sliceSpec, width, height, signal } = options;

    // sliceSpec = [init, lead-slice, y-slice, x-slice]
    const spatial = sliceSpec
      .filter((s): s is zarr.Slice => !!s && typeof s === "object" && "start" in s)
      .slice(-2);
    const tileY = spatial[0]?.start ?? 0;
    const tileX = spatial[1]?.start ?? 0;
    const shardRow = Math.floor(tileY / HRRR_SHARD.height);
    const shardCol = Math.floor(tileX / HRRR_SHARD.width);

    const chunks = await Promise.all(FFWI_BANDS.map((band) => zarr.get(arrays[band], sliceSpec, { signal })));
    const first = chunks[0]!;
    if (first.shape.length !== 3 || first.shape[0] !== HRRR_LEAD_TIME_COUNT) {
      throw new Error(`Expected (49, y, x) chunk, got [${first.shape.join(", ")}]`);
    }
    if (first.shape[1] !== height || first.shape[2] !== width) {
      throw new Error(`Tile shape mismatch: expected [${HRRR_LEAD_TIME_COUNT}, ${height}, ${width}], got [${first.shape.join(", ")}]`);
    }

    const n = HRRR_LEAD_TIME_COUNT * height * width;
    const t = chunks[0]!.data as Float32Array;
    const rh = chunks[1]!.data as Float32Array;
    const u = chunks[2]!.data as Float32Array;
    const v = chunks[3]!.data as Float32Array;
    const gust = chunks[4]!.data as Float32Array;
    const mEq = new Uint16Array(n);
    const wind = new Uint16Array(n);
    const rhU8 = new Uint8Array(n);
    const tcU8 = new Uint8Array(n);
    fosbergBlock(n, t, rh, u, v, gust, windMode, mEq, wind, rhU8, tcU8);

    let texture: Texture;
    let windTexture: Texture | undefined;
    let byteLength: number;
    if (variable === "ffwi") {
      const mF = new Float32Array(n);
      const wF = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        mF[i] = mEq[i] === U16_NAN ? NaN : mEq[i]! / M_SCALE;
        wF[i] = wind[i] === U16_NAN ? NaN : wind[i]! / WIND_SCALE;
      }
      texture = halfTexture(device, width, height, mF);
      windTexture = halfTexture(device, width, height, wF);
      byteLength = n * 4;
    } else {
      let field: Float32Array;
      if (variable === "temperature") field = t;
      else if (variable === "rh") field = rh;
      else {
        field = new Float32Array(n);
        for (let i = 0; i < n; i++) field[i] = (windMode === "gust" ? gust[i]! : Math.hypot(u[i]!, v[i]!)) * MS_TO_MPH;
      }
      texture = device.createTexture({
        dimension: "2d-array",
        format: "r32float",
        width,
        height,
        depth: HRRR_LEAD_TIME_COUNT,
        mipLevels: 1,
        data: field,
        sampler: { minFilter: "nearest", magFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
      });
      byteLength = field.byteLength;
    }

    // Key by the shard grid (edge tiles are narrower than a full shard).
    const tile: FfwiTile = {
      key: `${shardRow},${shardCol}`,
      rowStart: tileY,
      colStart: tileX,
      width,
      height,
      mEq,
      mLag: mEq,
      wind,
      rh: rhU8,
      tc: tcU8,
    };
    cache.set(tile);

    return { texture, windTexture, width, height, tileX, tileY, byteLength };
  };
}
