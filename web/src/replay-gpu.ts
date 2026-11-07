import type { Layer, LayerExtension } from "@deck.gl/core";
import { GeoJsonLayer } from "@deck.gl/layers";
import { ZarrLayer, type GetTileDataOptions } from "@developmentseed/deck.gl-zarr";
import type { Device, Framebuffer, Texture } from "@luma.gl/core";
import { Model } from "@luma.gl/engine";
import type { ShaderModule } from "@luma.gl/shadertools";
import type * as zarr from "zarrita";
import volumeJson from "../../models/fuel_volume.json";
import { RES, X0, Y0, YEARS } from "./aef/mosaic.js";
import type { AefPool } from "./aef/pool.js";
import { rampStops, rampTexture, type GroundRamp } from "./aef/ramp.js";
import type { VolumeHead } from "./fuel/volume.js";
import type { ViewGrid } from "./fuel/viewHist.js";
import { burnGrid } from "./perimeters.js";
import type { Replay } from "./replay.js";
import { FFWI_GATE, FFWI_SCALE, type ReplayWeather, WX_STEP } from "./replay-wx.js";

/**
 * The replay on the GPU. Each 10 m tile's two embedding years go up as one int8 texture array (128 layers); one
 * render pass runs the volume head for both years into a small rgba8 texture (r the before year's volume byte, g the
 * after year's), and the 8 MB embedding texture is freed. Drawing (replayDraw below) reads that texture and, per
 * fragment, projects the pixel into the GOES grid to blend its arrival hour from the cells around it. The outline
 * masks on the GPU (deck's MaskExtension), and tiles the outline does not touch are never requested.
 */

const VOLUME = volumeJson as VolumeHead;
const NB = 64;
/** The volume byte: 0 no embedding, else 1 + the scaled log volume x 254 (volume.ts). */

// ---- the volume pass

const PASS_VS = /* glsl */ `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

/**
 * Weights as rgba32float, 16 texels (64 floats) a row: row 0 the mean, 1 one over the scale, 2..65 the first layer's
 * rows, 66 its bias, 67 the second layer's volume row, 68 its bias (first float).
 */
function weightRows(h: VolumeHead): Float32Array {
  const out = new Float32Array(69 * NB);
  out.set(h.mean, 0);
  out.set(h.scale.map((s) => 1 / s), NB);
  out.set(h.w1, 2 * NB);
  out.set(h.b1, 66 * NB);
  out.set(h.w2.slice(0, h.hidden), 67 * NB);
  out[68 * NB] = h.b2[0]!;
  return out;
}

const PASS_FS = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp isampler2DArray;
uniform isampler2DArray embTex;
uniform sampler2D wTex;
out vec4 fragColor;

vec4 W(int k, int row) { return texelFetch(wTex, ivec2(k, row), 0); }

/** The volume head on one year's 64 layers from base: the scaled log volume 0..1, or -1 with no embedding. */
float vol(ivec2 px, int base) {
  vec4 x[16];
  int miss = 0;
  for (int k = 0; k < 16; k++) {
    vec4 v;
    for (int c = 0; c < 4; c++) {
      int q = texelFetch(embTex, ivec3(px, base + 4 * k + c), 0).r;
      if (q == -128) miss++;
      float f = float(q) / 127.5;
      v[c] = f * abs(f);
    }
    x[k] = (v - W(k, 0)) * W(k, 1);
  }
  if (miss == 64) return -1.0;
  float o = W(0, 68).x;
  for (int j = 0; j < 64; j++) {
    float s = W(j / 4, 66)[j % 4];
    for (int k = 0; k < 16; k++) s += dot(W(k, 2 + j), x[k]);
    o += max(s, 0.0) * W(j / 4, 67)[j % 4];
  }
  return clamp(o, 0.0, 1.0);
}

float byteOf(float v) { return v < 0.0 ? 0.0 : (1.0 + floor(v * 254.0 + 0.5)) / 255.0; }

void main() {
  ivec2 px = ivec2(gl_FragCoord.xy);
  fragColor = vec4(byteOf(vol(px, 0)), byteOf(vol(px, 64)), 0.0, 1.0);
}`;

class VolumePass {
  private model: Model;
  private wTex: Texture;
  constructor(readonly device: Device) {
    this.wTex = device.createTexture({
      format: "rgba32float", width: 16, height: 69, mipLevels: 1, data: weightRows(VOLUME),
      sampler: { minFilter: "nearest", magFilter: "nearest" },
    });
    this.model = new Model(device, {
      id: "replay-volume-pass",
      vs: PASS_VS,
      fs: PASS_FS,
      topology: "triangle-list",
      vertexCount: 3,
      bufferLayout: [],
      parameters: { depthWriteEnabled: false, depthCompare: "always" },
      disableWarnings: true,
    });
  }

  /** Two years' int8 blocks [128, h, w] -> the volume texture (mipmapped) and its bytes on the CPU. */
  run(raw: Int8Array, w: number, h: number): { tex: Texture; v0: Uint8Array; v1: Uint8Array } {
    const d = this.device;
    const emb = d.createTexture({
      dimension: "2d-array", format: "r8sint", width: w, height: h, depth: 2 * NB, mipLevels: 1, data: raw,
      sampler: { minFilter: "nearest", magFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
    });
    const levels = Math.floor(Math.log2(Math.max(w, h))) + 1;
    const tex = d.createTexture({
      format: "rgba8unorm", width: w, height: h, mipLevels: levels,
      sampler: { minFilter: "linear", magFilter: "linear", mipmapFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
    });
    const fb: Framebuffer = d.createFramebuffer({ width: w, height: h, colorAttachments: [tex] });
    this.model.setBindings({ embTex: emb, wTex: this.wTex });
    const pass = d.beginRenderPass({ framebuffer: fb, clearColor: [0, 0, 0, 0], parameters: { viewport: [0, 0, w, h] } });
    this.model.draw(pass);
    pass.end();
    const px = d.readPixelsToArrayWebGL(fb) as Uint8Array;
    (tex as unknown as { generateMipmapsWebGL?: () => void }).generateMipmapsWebGL?.();
    fb.destroy();
    emb.destroy();
    const v0 = new Uint8Array(w * h), v1 = new Uint8Array(w * h);
    for (let i = 0; i < v0.length; i++) {
      v0[i] = px[4 * i]!;
      v1[i] = px[4 * i + 1]!;
    }
    return { tex, v0, v1 };
  }
}

// ---- drawing a tile

export type ReplayDrawProps = {
  volTex: Texture;
  rampTex: Texture;
  gridTex: Texture;
  hour: number;
  hotH: number;
  coolH: number;
  beforeAlpha: number;
  pHide: number;
  pMax: number;
  /** 1: the ramp over the log-odds of the scaled volume (as the fuel map), 0 straight. */
  scale: number;
  /** The arrival hour given to ground with no seen cell around it (the newest hour read). */
  late: number;
  /** The tile's west, south, east, north. */
  bbox: [number, number, number, number];
  /** The arrival grid's first row and column and its size in the GOES grid. */
  grid: [number, number, number, number];
  /** GOES: semi-major, semi-minor axis, satellite distance from the earth's center (m), sub-satellite lon (rad). */
  proj: [number, number, number, number];
  /** GOES grid: x0, dx, y0, dy (scan angles, rad). */
  scan: [number, number, number, number];
  /** The fire weather's glow (replay-wx.ts): r8 layers an hour over a lon / lat grid. */
  glowTex: Texture;
  /** Its grid: west, north, step (deg), and 1 when it is in. */
  wxBox: [number, number, number, number];
  /** Its columns, rows, hours, and the first hour (from the replay's start). */
  wxDims: [number, number, number, number];
  /** The unburned fuel's opacity where the hour's Fosberg is under the line (full where it connects). */
  baseAlpha: number;
  /** The line (Fosberg) and the stored byte's scale. */
  gate: number;
  wxScale: number;
  /** 1: GOES arrival drawn (burning, cooling, the after year); 0: the fuel alone with the weather. */
  goes: number;
};

const RD = "replayDraw";
export const ReplayDraw = {
  name: RD,
  fs: `\
uniform ${RD}Uniforms {
  float hour;
  float hotH;
  float coolH;
  float beforeAlpha;
  float pHide;
  float pMax;
  float scale;
  float late;
  vec4 bbox;
  vec4 grid;
  vec4 proj;
  vec4 scan;
  vec4 wxBox;
  vec4 wxDims;
  float baseAlpha;
  float gate;
  float wxScale;
  float goes;
} ${RD};
`,
  inject: {
    "fs:#decl": `
uniform sampler2D volTex;
uniform sampler2D rampTex;
uniform sampler2D gridTex;
precision highp sampler2DArray;
uniform sampler2DArray glowTex;
`,
    "fs:DECKGL_FILTER_COLOR": /* glsl */ `
      vec4 vt = texture(volTex, geometry.uv);
      float fb0 = vt.r * 255.0, fb1 = vt.g * 255.0;
      if (fb0 < 0.5 && fb1 < 0.5) discard;
      // the pixel in the GOES grid (GOES-R PUG forward projection), then its arrival from the four cells around it
      float lonD = mix(${RD}.bbox.x, ${RD}.bbox.z, geometry.uv.x);
      float latD = mix(${RD}.bbox.w, ${RD}.bbox.y, geometry.uv.y);
      float lon = radians(lonD);
      float lat = radians(latD);
      float req = ${RD}.proj.x, rpol = ${RD}.proj.y, H = ${RD}.proj.z;
      float e2 = 1.0 - (rpol * rpol) / (req * req);
      float latc = atan((rpol * rpol) / (req * req) * tan(lat));
      float rc = rpol / sqrt(1.0 - e2 * cos(latc) * cos(latc));
      float sx = H - rc * cos(latc) * cos(lon - ${RD}.proj.w);
      float sy = -rc * cos(latc) * sin(lon - ${RD}.proj.w);
      float sz = rc * sin(latc);
      float xa = asin(-sy / sqrt(sx * sx + sy * sy + sz * sz));
      float ya = atan(sz / sx);
      vec2 rcf = vec2((ya - ${RD}.scan.z) / ${RD}.scan.w, (xa - ${RD}.scan.x) / ${RD}.scan.y) - ${RD}.grid.xy;
      vec2 b0 = floor(rcf);
      vec2 f = rcf - b0;
      float sum = 0.0, wsum = 0.0;
      for (int k = 0; k < 4; k++) {
        vec2 o = vec2(float(k / 2), float(k % 2));
        ivec2 at = ivec2(b0 + o);
        if (at.x < 0 || at.y < 0 || float(at.x) >= ${RD}.grid.z || float(at.y) >= ${RD}.grid.w) continue;
        float v = texelFetch(gridTex, ivec2(at.y, at.x), 0).r;
        if (v < 0.0) continue;
        float wt = (o.x > 0.5 ? f.x : 1.0 - f.x) * (o.y > 0.5 ? f.y : 1.0 - f.y);
        sum += v * wt;
        wsum += wt;
      }
      float ar = ${RD}.goes > 0.5 ? (wsum > 0.0 ? sum / wsum : ${RD}.late) : 1e9;
      // each year's volume on the ramp, as the fuel map places it
      float place[2];
      for (int k = 0; k < 2; k++) {
        float v = ((k == 0 ? fb0 : fb1) - 1.0) / 254.0;
        if (${RD}.scale > 0.5) {
          float h0 = clamp(${RD}.pHide, 0.002, 0.996), h1 = clamp(${RD}.pMax, 0.002, 0.996), pv = clamp(v, 0.002, 0.996);
          place[k] = clamp((log(pv / (1.0 - pv)) - log(h0 / (1.0 - h0))) / max(log(h1 / (1.0 - h1)) - log(h0 / (1.0 - h0)), 0.01), 0.0, 1.0);
        } else {
          place[k] = clamp((v - ${RD}.pHide) / max(${RD}.pMax - ${RD}.pHide, 0.0001), 0.0, 1.0);
        }
      }
      float now = ${RD}.hour;
      if (now < ar) {
        if (fb0 < 0.5) discard;
        vec3 rgb = texture(rampTex, vec2(place[0], 0.5)).rgb;
        // the hour's Fosberg at the pixel: under the line the fuel shows at baseAlpha, never gone; where it connects,
        // in full. Before the fire starts (the lead-in) it is in full regardless, so the fuel is seen first.
        float a = ${RD}.baseAlpha;
        if (${RD}.wxBox.w > 0.5) {
          vec2 wuv = vec2(((lonD - ${RD}.wxBox.x) / ${RD}.wxBox.z + 0.5) / ${RD}.wxDims.x, ((${RD}.wxBox.y - latD) / ${RD}.wxBox.z + 0.5) / ${RD}.wxDims.y);
          float l = clamp(now - ${RD}.wxDims.w, 0.0, ${RD}.wxDims.z - 1.0);
          float la = floor(l);
          float fw = mix(texture(glowTex, vec3(wuv, la)).r, texture(glowTex, vec3(wuv, min(la + 1.0, ${RD}.wxDims.z - 1.0))).r, l - la) * 255.0 / ${RD}.wxScale;
          a = mix(a, 1.0, smoothstep(${RD}.gate - 4.0, ${RD}.gate + 4.0, fw));
        }
        a = max(a, clamp(-now / 3.0, 0.0, 1.0));
        color = vec4(rgb, a * ${RD}.beforeAlpha);
        return;
      }
      // burning: amber to near white by how much stood there before, whitest at the front; light carries it, not red
      float dt = now - ar;
      float heat = fb0 < 0.5 ? 0.3 : (fb0 - 1.0) / 254.0;
      vec3 hot = mix(vec3(1.0, 0.66, 0.22), vec3(1.0, 0.98, 0.90), heat) * (0.55 + 0.45 * heat);
      hot = mix(hot, vec3(1.0), 0.5 * exp(-dt / 0.75));
      if (dt < ${RD}.hotH) {
        color = vec4(hot, 1.0);
        return;
      }
      float kk = smoothstep(0.0, 1.0, clamp((dt - ${RD}.hotH) / max(${RD}.coolH, 0.01), 0.0, 1.0));
      color = vec4(mix(hot, texture(rampTex, vec2(place[1], 0.5)).rgb, kk), 1.0);
    `,
  },
  uniformTypes: {
    hour: "f32", hotH: "f32", coolH: "f32", beforeAlpha: "f32", pHide: "f32", pMax: "f32", scale: "f32", late: "f32",
    bbox: "vec4<f32>", grid: "vec4<f32>", proj: "vec4<f32>", scan: "vec4<f32>", wxBox: "vec4<f32>", wxDims: "vec4<f32>", baseAlpha: "f32", gate: "f32", wxScale: "f32", goes: "f32",
  },
  getUniforms: (p: Partial<ReplayDrawProps>) => ({
    hour: p.hour ?? 0, hotH: p.hotH ?? 3, coolH: p.coolH ?? 9, beforeAlpha: p.beforeAlpha ?? 1,
    pHide: p.pHide ?? 0, pMax: p.pMax ?? 1, scale: p.scale ?? 1, late: p.late ?? 0,
    bbox: p.bbox ?? [0, 0, 0, 0], grid: p.grid ?? [0, 0, 0, 0], proj: p.proj ?? [0, 0, 0, 0], scan: p.scan ?? [0, 0, 0, 0],
    wxBox: p.wxBox ?? [0, 0, 1, 0], wxDims: p.wxDims ?? [1, 1, 1, 0], baseAlpha: p.baseAlpha ?? 0.45, gate: p.gate ?? 30, wxScale: p.wxScale ?? 4, goes: p.goes ?? 1,
    volTex: p.volTex, rampTex: p.rampTex, gridTex: p.gridTex, glowTex: p.glowTex,
  }),
} as const satisfies ShaderModule<ReplayDrawProps>;

// ---- the tiles

type ReplayTile = { tex: Texture; v0: Uint8Array; v1: Uint8Array; bbox: [number, number, number, number]; width: number; height: number; byteLength: number; key: string };

export type ReplayLayerOpts = {
  hour: number;
  hotH: number;
  coolH: number;
  beforeAlpha: number;
  pHide: number;
  pMax: number;
  logit: boolean;
  ramp: GroundRamp;
  maskExt: LayerExtension;
  /** The fire weather that lights the unburned fuel (none until it is read). */
  wx: ReplayWeather | null;
  /** The unburned fuel's opacity under the line. */
  baseAlpha: number;
  /** GOES arrival drawn (burning, the after year), or the fuel alone with the weather. */
  goes: boolean;
  beforeId?: string;
  visible: boolean;
};

const MASK_ID = "replay-outline-mask";

export class ReplayTiles {
  private pass: VolumePass | null = null;
  private device: Device | null = null;
  private ramp: { id: GroundRamp; tex: Texture } | null = null;
  private grid: { v: number; tex: Texture; dims: [number, number, number, number] } | null = null;
  readonly tiles = new Map<string, ReplayTile>();
  /** Tiles asked and back. */
  readonly work = { reading: 0, done: 0 };
  private outlineGrid: { key: string; g: Uint8Array } | null = null;
  private noGlow: Texture | null = null;

  constructor(
    private replay: Replay,
    private pool: AefPool,
    private mosaic: { arr: zarr.Array<"int8", zarr.Readable>; attrs: unknown },
    private onTile: () => void,
  ) {}

  private getTileData = async (_arr: zarr.Array<"int8", zarr.Readable>, o: GetTileDataOptions): Promise<ReplayTile> => {
    const ys = o.sliceSpec[2] as zarr.Slice, xs = o.sliceSpec[3] as zarr.Slice;
    const py0 = ys.start ?? 0, px0 = xs.start ?? 0;
    const { width, height } = o;
    const bbox: [number, number, number, number] = [X0 + px0 * RES, Y0 - (py0 + height) * RES, X0 + (px0 + width) * RES, Y0 - py0 * RES];
    if (!this.replay.touches(bbox)) return null as unknown as ReplayTile;
    this.work.reading++;
    try {
      const f = this.replay.fire;
      const raw = await this.pool.raw([YEARS.indexOf(f.before), YEARS.indexOf(f.after)], py0, px0, height, width, o.signal);
      if (this.device !== o.device || !this.pass) {
        this.device = o.device;
        this.pass = new VolumePass(o.device);
      }
      const { tex, v0, v1 } = this.pass.run(raw, width, height);
      const t: ReplayTile = { tex, v0, v1, bbox, width, height, byteLength: 2 * width * height, key: `${py0},${px0}` };
      this.tiles.set(t.key, t);
      this.work.done++;
      this.onTile();
      return t;
    } finally {
      this.work.reading--;
    }
  };

  private onUnload = (tile: { content?: ReplayTile | null }) => {
    const t = tile.content;
    if (!t) return;
    this.tiles.delete(t.key);
    setTimeout(() => t.tex.destroy(), 1500);
  };

  /** The arrival grid as a texture (r32float, -1 where a cell was not seen burning), remade when the replay reads more. */
  private gridTex(device: Device) {
    const r = this.replay;
    if (this.grid?.v === r.version) return this.grid;
    const d = r.denseGrid();
    const a = new Float32Array(Math.max(1, d.nr * d.nc)).fill(-1);
    d.a.forEach((v, i) => { if (!Number.isNaN(v)) a[i] = v; });
    const tex = device.createTexture({ format: "r32float", width: Math.max(1, d.nc), height: Math.max(1, d.nr), mipLevels: 1, data: a,
      sampler: { minFilter: "nearest", magFilter: "nearest" } });
    if (this.grid) { const old = this.grid.tex; setTimeout(() => old.destroy(), 1500); }
    this.grid = { v: r.version, tex, dims: [d.r0, d.c0, d.nr, d.nc] };
    return this.grid;
  }

  layers(o: ReplayLayerOpts): Layer[] {
    const r = this.replay;
    const era = r.eraOf();
    const drawKey = `${o.hour}|${o.hotH}|${o.coolH}|${o.beforeAlpha}|${o.pHide}|${o.pMax}|${o.logit}|${o.ramp}|${r.version}|${o.wx?.glow ? 1 : 0}|${o.baseAlpha}|${o.goes}`;
    const renderTile = (d: ReplayTile) => {
      const device = this.device;
      if (!device || (o.goes && !era)) return { renderPipeline: [] };
      if (this.ramp?.id !== o.ramp) {
        if (this.ramp) { const old = this.ramp.tex; setTimeout(() => old.destroy(), 1500); }
        this.ramp = { id: o.ramp, tex: rampTexture(device, rampStops(o.ramp)) };
      }
      const g = this.gridTex(device);
      const p = era?.projection ?? { semi_major_axis: 1, semi_minor_axis: 1, perspective_point_height: 1, longitude_of_projection_origin: 0 };
      const gr = era?.grid ?? { x0: 0, dx: 1, y0: 0, dy: 1 };
      const wt = o.wx?.texture(device) ?? null;
      this.noGlow ??= device.createTexture({ dimension: "2d-array", format: "r8unorm", width: 1, height: 1, depth: 1, mipLevels: 1, data: new Uint8Array(1) });
      return {
        renderPipeline: [{
          module: ReplayDraw,
          props: {
            volTex: d.tex, rampTex: this.ramp.tex, gridTex: g.tex,
            hour: o.hour, hotH: o.hotH, coolH: o.coolH, beforeAlpha: o.beforeAlpha, pHide: o.pHide, pMax: o.pMax,
            scale: o.logit ? 1 : 0, late: r.done ? r.hours : r.hoursRead,
            bbox: d.bbox, grid: g.dims,
            proj: [p.semi_major_axis, p.semi_minor_axis, p.perspective_point_height + p.semi_major_axis, (p.longitude_of_projection_origin * Math.PI) / 180],
            scan: [gr.x0, gr.dx, gr.y0, gr.dy],
            glowTex: wt ?? this.noGlow,
            wxBox: [o.wx?.box[0] ?? 0, o.wx?.box[3] ?? 0, WX_STEP, wt ? 1 : 0],
            wxDims: [o.wx?.nx ?? 1, o.wx?.ny ?? 1, o.wx?.nh ?? 1, o.wx?.h0 ?? 0],
            baseAlpha: o.baseAlpha, gate: FFWI_GATE, wxScale: FFWI_SCALE, goes: o.goes ? 1 : 0,
          } satisfies ReplayDrawProps,
        }],
      };
    };
    const m = 0.01;
    return [
      new GeoJsonLayer({
        id: MASK_ID,
        data: [r.area] as never,
        filled: true,
        stroked: false,
        getFillColor: [255, 255, 255, 255],
        operation: "mask",
        // in the same slot of the basemap as the layer it masks (an interleaved mask elsewhere does not apply)
        beforeId: o.beforeId,
      } as never),
      new ZarrLayer<zarr.Readable, "int8", ReplayTile>({
        id: "replay-volume",
        node: this.mosaic.arr,
        metadata: this.mosaic.attrs,
        selection: { time: 0, band: null },
        getTileData: this.getTileData,
        renderTile,
        minZoom: 9,
        extent: [r.bbox[0] - m, r.bbox[1] - m, r.bbox[2] + m, r.bbox[3] + m],
        maxRequests: 24,
        maxCacheSize: 1000,
        updateTriggers: { renderTile: [drawKey] },
        onTileUnload: this.onUnload,
        extensions: [o.maskExt],
        maskId: MASK_ID,
        maskByInstance: false,
        visible: o.visible,
        beforeId: o.beforeId,
      } as never) as unknown as Layer,
    ];
  }

  /**
   * Grid cells per volume byte as drawn at this hour, for the view histogram: inside the outline only, the before year
   * until the pixel's arrival, the after year once it has cooled, none while it is hot.
   */
  onGrid(g: ViewGrid, view: [number, number, number, number], hour: number, hotH: number, coolH: number, goes = true): { counts: Float64Array; unread: number } {
    const counts = new Float64Array(256);
    const key = `${view.join(",")}|${g.gw}x${g.gh}`;
    if (this.outlineGrid?.key !== key) this.outlineGrid = { key, g: burnGrid([this.replay.area], view, g.gw, g.gh) };
    const inside = this.outlineGrid.g;
    let unread = 0;
    for (let r = 0; r < g.gh; r++) {
      const lat = g.lats[r]!;
      const cy = Math.floor((Y0 - lat) / RES / 256);
      for (let c = 0; c < g.gw; c++) {
        const i = r * g.gw + c;
        if (!inside[i]) continue;
        const lon = g.lons[c]!;
        const cx = Math.floor((lon - X0) / RES / 256);
        const t = this.tiles.get(`${cy * 256},${cx * 256}`);
        if (!t) { unread++; continue; }
        const p = Math.min(t.height - 1, Math.floor((Y0 - lat) / RES) - cy * 256) * t.width + Math.min(t.width - 1, Math.floor((lon - X0) / RES) - cx * 256);
        const a = goes ? this.replay.arrivalAt(lon, lat) : Infinity;
        const v = hour < a ? t.v0[p]! : hour < a + hotH + coolH ? -1 : t.v1[p]!;
        if (v > 0) counts[v]!++;
      }
    }
    return { counts, unread };
  }
}
