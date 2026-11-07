import { COORDINATE_SYSTEM, type UpdateParameters } from "@deck.gl/core";
import { BitmapLayer, type BitmapLayerProps } from "@deck.gl/layers";
import type { Device, Texture } from "@luma.gl/core";
import type { ShaderModule } from "@luma.gl/shadertools";
import { IcechunkStore } from "icechunk-js";
import * as zarr from "zarrita";
import { lonLatToHrrrPixelF } from "./hrrr/metadata.js";

/**
 * The fire weather over a replayed fire, hour by hour: the Fosberg Fire Weather Index from the HRRR analysis (what the
 * weather was, not a forecast; dynamical.org's Icechunk store, read in the browser). Only the fire's window of HRRR
 * cells and hours is kept: the store's chunks are 90 days x 45 x 45 cells, so one fire is a few chunks per variable,
 * read one variable at a time.
 *
 * The index is resampled onto a grid of about 1 km over the fire's box and FIELD_MARGIN around it, a byte per cell
 * and hour (the index x FFWI_SCALE), for the GPU (r8unorm, 2d-array, a layer an hour): the fuel inside the outline
 * reads it to light up, and the field around the fire (fieldLayer) draws it.
 */

const URL = "https://dynamical-noaa-hrrr.s3.us-west-2.amazonaws.com/noaa-hrrr-analysis/v0.2.0.icechunk";
/** The store's first hour. */
const T0 = Date.UTC(2014, 9, 1);
const VARS = ["temperature_2m", "relative_humidity_2m", "wind_u_10m", "wind_v_10m"] as const;
/**
 * The Fosberg index where the fuel connects (lit to full): 30, the top tenth of the Park Fire's hours, crossed on the
 * afternoons of July 24 to 26 and not on the calmer days after.
 */
export const FFWI_GATE = 30;
/** Stored as a byte: the index x 4 (to 63.75, a quarter step). */
export const FFWI_SCALE = 4;
/** The field drawn this far around the fire's box, degrees (about 15 km). */
export const FIELD_MARGIN = 0.15;
/** Grid step over the fire, degrees (about 1 km). */
export const WX_STEP = 0.01;

export function ffwi(tC: number, rh: number, windMs: number): number {
  const tF = tC * 1.8 + 32;
  const h = Math.min(100, Math.max(0, rh));
  const m = h < 10 ? 0.03229 + 0.281073 * h - 0.000578 * h * tF
    : h < 50 ? 2.22749 + 0.160107 * h - 0.014784 * tF
    : 21.0606 + 0.005565 * h * h - 0.00035 * h * tF - 0.483199 * h;
  const mr = Math.min(30, Math.max(0, m)) / 30;
  const eta = 1 - 2 * mr + 1.5 * mr * mr - 0.5 * mr * mr * mr;
  const u = windMs * 2.2369363;
  return (eta * Math.sqrt(1 + u * u)) / 0.3002;
}

export class ReplayWeather {
  /** Grid over the fire: west, north, columns, rows; hours from h0 (relative to the replay's start), nh of them. */
  nx = 0;
  ny = 0;
  nh = 0;
  h0 = 0;
  /** The index x FFWI_SCALE per [hour, row, col]. */
  glow: Uint8Array | null = null;
  /** The mean FFWI over the fire's box per hour. */
  mean: Float32Array | null = null;
  private tex: { device: Device; tex: Texture } | null = null;

  /** The grid's box: the fire's box and FIELD_MARGIN around it. */
  readonly box: [number, number, number, number];

  constructor(readonly bbox: [number, number, number, number], readonly t0: number) {
    const m = FIELD_MARGIN;
    this.box = [bbox[0] - m, bbox[1] - m, bbox[2] + m, bbox[3] + m];
  }

  /** The texture's extent (cell edges), for the field image's bounds: the grid's points are its texel centers. */
  extent(): [number, number, number, number] {
    const [w, , , n] = this.box;
    const h = WX_STEP / 2;
    return [w - h, n - (this.ny - 1) * WX_STEP - h, w + (this.nx - 1) * WX_STEP + h, n + h];
  }

  /** Read hours fromH to toH (relative to t0) of the four inputs over the box. */
  async load(fromH: number, toH: number, signal?: AbortSignal) {
    const [w, s, e, n] = this.box;
    const corners = [[w, s], [w, n], [e, s], [e, n]].map(([x, y]) => lonLatToHrrrPixelF(x!, y!)!);
    const x0 = Math.floor(Math.min(...corners.map((c) => c.x))) - 2, x1 = Math.ceil(Math.max(...corners.map((c) => c.x))) + 2;
    const y0 = Math.floor(Math.min(...corners.map((c) => c.y))) - 2, y1 = Math.ceil(Math.max(...corners.map((c) => c.y))) + 2;
    const i0 = Math.round((this.t0 - T0) / 3600e3) + Math.floor(fromH), i1 = Math.round((this.t0 - T0) / 3600e3) + Math.ceil(toH) + 1;
    const store = await IcechunkStore.open(URL, { withRangeCoalescing: zarr.withRangeCoalescing } as never);
    const vals: Float32Array[] = [];
    for (const v of VARS) {
      if (signal?.aborted) return;
      const arr = await zarr.open(store.resolve(`/${v}`), { kind: "array" });
      const c = await zarr.get(arr as zarr.Array<"float32", zarr.Readable>, [zarr.slice(i0, i1), zarr.slice(y0, y1), zarr.slice(x0, x1)]);
      vals.push(c.data as Float32Array);
    }
    const nt = i1 - i0, wy = y1 - y0, wx = x1 - x0;
    // FFWI per HRRR cell and hour
    const f = new Float32Array(nt * wy * wx);
    for (let i = 0; i < f.length; i++) f[i] = ffwi(vals[0]![i]!, vals[1]![i]!, Math.hypot(vals[2]![i]!, vals[3]![i]!));
    // onto the fire's lon / lat grid, bilinear between HRRR cell centers
    this.nx = Math.ceil((e - w) / WX_STEP) + 1;
    this.ny = Math.ceil((n - s) / WX_STEP) + 1;
    this.nh = nt;
    this.h0 = Math.floor(fromH);
    const glow = new Uint8Array(nt * this.ny * this.nx);
    const mean = new Float32Array(nt);
    // the mean over the fire's own box
    const [fw, fs, fe, fn] = this.bbox;
    let inBox = 0;
    for (let r = 0; r < this.ny; r++) for (let c = 0; c < this.nx; c++) {
      const lon = w + c * WX_STEP, lat = n - r * WX_STEP;
      if (lon >= fw && lon <= fe && lat >= fs && lat <= fn) inBox++;
    }
    const at = (t: number, yy: number, xx: number) => f[(t * wy + yy) * wx + xx]!;
    for (let r = 0; r < this.ny; r++)
      for (let c = 0; c < this.nx; c++) {
        const p = lonLatToHrrrPixelF(w + c * WX_STEP, n - r * WX_STEP);
        if (!p) continue;
        const fx = p.x - 0.5 - x0, fy = p.y - 0.5 - y0;
        const xa = Math.max(0, Math.min(wx - 2, Math.floor(fx))), ya = Math.max(0, Math.min(wy - 2, Math.floor(fy)));
        const tx = Math.min(1, Math.max(0, fx - xa)), ty = Math.min(1, Math.max(0, fy - ya));
        for (let t = 0; t < nt; t++) {
          const v = (at(t, ya, xa) * (1 - tx) + at(t, ya, xa + 1) * tx) * (1 - ty) + (at(t, ya + 1, xa) * (1 - tx) + at(t, ya + 1, xa + 1) * tx) * ty;
          const lon = w + c * WX_STEP, lat = n - r * WX_STEP;
          if (lon >= fw && lon <= fe && lat >= fs && lat <= fn) mean[t]! += v / inBox;
          glow[(t * this.ny + r) * this.nx + c] = Math.round(Math.min(255, Math.max(0, v * FFWI_SCALE)));
        }
      }
    this.glow = glow;
    this.mean = mean;
  }

  /** The mean FFWI over the fire's box at an hour (between hours, the two around it mixed); null before it is read. */
  meanAt(hour: number): number | null {
    if (!this.mean) return null;
    const l = Math.min(this.nh - 1, Math.max(0, hour - this.h0));
    const a = Math.floor(l), b = Math.min(this.nh - 1, a + 1);
    return this.mean[a]! + (this.mean[b]! - this.mean[a]!) * (l - a);
  }

  /** The glow as a texture (one layer an hour), made once per device; null before it is read. */
  texture(device: Device): Texture | null {
    if (!this.glow) return null;
    if (this.tex?.device === device) return this.tex.tex;
    const tex = device.createTexture({
      dimension: "2d-array", format: "r8unorm", width: this.nx, height: this.ny, depth: this.nh, mipLevels: 1, data: this.glow,
      sampler: { minFilter: "linear", magFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
    });
    this.tex = { device, tex };
    return tex;
  }
}

// ---- the field around the fire: the hour's Fosberg in gray, dim, under the fuel

export type FieldProps = {
  fieldTex: Texture; hour: number; h0: number; nh: number; gate: number; scale: number; alpha: number;
  /** The image's extent and the fire's box (west, south, east, north): the field fades out away from the fire. */
  extent: [number, number, number, number];
  fire: [number, number, number, number];
};

const FM = "replayField";
const FieldModule = {
  name: FM,
  fs: `\
uniform ${FM}Uniforms {
  float hour;
  float h0;
  float nh;
  float gate;
  float scale;
  float alpha;
  vec4 extent;
  vec4 fire;
} ${FM};
`,
  inject: {
    "fs:#decl": `
precision highp sampler2DArray;
uniform sampler2DArray fieldTex;
`,
    "fs:DECKGL_FILTER_COLOR": /* glsl */ `
      // between hours the two around it mixed, so the field drifts rather than jumps
      float l = clamp(${FM}.hour - ${FM}.h0, 0.0, ${FM}.nh - 1.0);
      float la = floor(l);
      float f = mix(texture(fieldTex, vec3(geometry.uv, la)).r, texture(fieldTex, vec3(geometry.uv, min(la + 1.0, ${FM}.nh - 1.0))).r, l - la) * 255.0 / ${FM}.scale;
      // dark where calm, pale toward the line where the fuel connects
      float t = clamp((f - 5.0) / max(${FM}.gate - 5.0, 1.0), 0.0, 1.0);
      // faded out with distance from the fire's box (degrees, east-west shrunk by latitude): no edge of its own
      float lon = mix(${FM}.extent.x, ${FM}.extent.z, geometry.uv.x);
      float lat = mix(${FM}.extent.w, ${FM}.extent.y, geometry.uv.y);
      float dx = max(max(${FM}.fire.x - lon, lon - ${FM}.fire.z), 0.0) * cos(radians(lat));
      float dy = max(max(${FM}.fire.y - lat, lat - ${FM}.fire.w), 0.0);
      float fade = 1.0 - smoothstep(0.03, 0.13, length(vec2(dx, dy)));
      color = vec4(vec3(0.06 + 0.36 * t), ${FM}.alpha * fade);
    `,
  },
  uniformTypes: { hour: "f32", h0: "f32", nh: "f32", gate: "f32", scale: "f32", alpha: "f32", extent: "vec4<f32>", fire: "vec4<f32>" },
  getUniforms: (p: Partial<FieldProps>) => ({
    hour: p.hour ?? 0, h0: p.h0 ?? 0, nh: p.nh ?? 1, gate: p.gate ?? FFWI_GATE, scale: p.scale ?? FFWI_SCALE, alpha: p.alpha ?? 0.5,
    extent: p.extent ?? [0, 0, 1, 1], fire: p.fire ?? [0, 0, 1, 1],
    fieldTex: p.fieldTex,
  }),
} as const satisfies ShaderModule<FieldProps>;

type FieldLayerProps = BitmapLayerProps & { wx: ReplayWeather; hour: number; fieldAlpha: number };

/** One image over the field's box, colored per fragment from the hour's index (the base image is a white pixel). */
class FieldLayer extends BitmapLayer<FieldLayerProps> {
  static layerName = "ReplayFieldLayer";
  getShaders() {
    const s = super.getShaders();
    return { ...s, modules: [...s.modules, FieldModule] };
  }
  updateState(params: UpdateParameters<this>) {
    super.updateState(params);
  }
  draw(opts: Parameters<BitmapLayer["draw"]>[0]) {
    const wx = this.props.wx;
    const tex = wx.texture(this.context.device);
    if (!tex) return;
    const field: FieldProps = {
      fieldTex: tex, hour: this.props.hour, h0: wx.h0, nh: wx.nh, gate: FFWI_GATE, scale: FFWI_SCALE, alpha: this.props.fieldAlpha,
      extent: wx.extent(), fire: wx.bbox,
    };
    (this.state.model as unknown as { shaderInputs: { setProps: (p: Record<string, unknown>) => void } } | undefined)?.shaderInputs.setProps({ [FM]: field });
    super.draw(opts);
  }
}

const WHITE = typeof ImageData === "undefined" ? null : new ImageData(new Uint8ClampedArray([255, 255, 255, 255]), 1, 1);

/** The field layer for an hour (its texture made from the weather on the layer's own device), or null without one. */
export function fieldLayer(wx: ReplayWeather, hour: number, alpha: number, beforeId?: string) {
  if (!wx.glow || !WHITE) return null;
  return new FieldLayer({
    id: "replay-field",
    image: WHITE,
    bounds: wx.extent(),
    _imageCoordinateSystem: COORDINATE_SYSTEM.LNGLAT,
    wx,
    hour,
    fieldAlpha: alpha,
    beforeId,
  } as never);
}
