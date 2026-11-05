import { type Layer, LayerExtension, project } from "@deck.gl/core";
import type { Device, Texture } from "@luma.gl/core";
import { VIRIDIS } from "./score.js";

/**
 * Lines and outlines colored on the GPU by the risk value under them: each fragment finds its own lon / lat from
 * its common-space position, looks the value up in one texture of the area read (RiskGrid, row 0 the north edge,
 * rows even in latitude like the mosaic), and takes viridis from a higher start, so a feature stands out a step
 * brighter than the ground drawn under it. Gray where nothing is read.
 */

/** The value per 10 m pixel over a block of mosaic chunks: 0 = not read, else 1 + round(risk * 254). */
export type RiskGrid = { data: Uint8Array; width: number; height: number; bounds: [w: number, s: number, e: number, n: number]; seq: number };

/** Where on viridis a value of 0 lands for features (the ground's own starts at 0): a step up the ramp, so a feature
 * is a shade brighter than the ground under it and still reads the risk across the whole ramp. */
export const FEATURE_START = 0.3;
/** How far feature colors are mixed toward white (0: none; whitening washed thin lines into one pale gray). */
export const FEATURE_LIFT = 0;
/** Fill opacity of buildings and other polygons (their outlines are opaque). */
export const FILL_ALPHA = 26;
/** Thinnest a line is drawn, in pixels, at any zoom. */
export const MIN_LINE_PX = 2;
/** Features where no value is read. */
export const FEATURE_GRAY = [0.43, 0.43, 0.46];

const fs = /* glsl */ `\
layout(std140) uniform riskShadeUniforms {
  vec4 bounds;
  vec3 gray;
  float enabled;
  float start;
  float lift;
} riskShade;
uniform sampler2D riskShade_grid;
uniform sampler2D riskShade_ramp;
`;

const WORLD_SCALE = (512 / (2 * Math.PI)).toFixed(10);

const riskShade = {
  name: "riskShade",
  dependencies: [project],
  vs: "",
  fs,
  inject: {
    "vs:#decl": "out vec2 riskShade_common;\n",
    "vs:#main-end": "  riskShade_common = geometry.position.xy + project.commonOrigin.xy;\n",
    "fs:#decl": "in vec2 riskShade_common;\n",
    "fs:DECKGL_FILTER_COLOR": /* glsl */ `
  {
    const float PI = 3.141592653589793;
    float lon = degrees(riskShade_common.x / ${WORLD_SCALE} - PI);
    float lat = degrees(2.0 * atan(exp(riskShade_common.y / ${WORLD_SCALE} - PI)) - PI / 2.0);
    vec2 uv = vec2((lon - riskShade.bounds.x) / (riskShade.bounds.z - riskShade.bounds.x),
                   (riskShade.bounds.w - lat) / (riskShade.bounds.w - riskShade.bounds.y));
    vec3 c = riskShade.gray;
    if (riskShade.enabled > 0.5 && uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0) {
      float b = texture(riskShade_grid, uv).r * 255.0;
      if (b > 0.5) {
        float t = (b - 1.0) / 254.0;
        c = texture(riskShade_ramp, vec2(riskShade.start + (1.0 - riskShade.start) * t, 0.5)).rgb;
        c = mix(c, vec3(1.0), riskShade.lift);
      }
    }
    color.rgb = c;
  }
`,
  },
  getUniforms: (o: { grid?: Texture; ramp?: Texture; bounds?: number[]; enabled?: number }) => {
    const out: Record<string, unknown> = {
      bounds: o.bounds ?? [0, 0, 1, 1],
      gray: FEATURE_GRAY,
      enabled: o.enabled ?? 0,
      start: FEATURE_START,
      lift: FEATURE_LIFT,
    };
    if (o.grid) out.riskShade_grid = o.grid;
    if (o.ramp) out.riskShade_ramp = o.ramp;
    return out;
  },
  uniformTypes: { bounds: "vec4<f32>", gray: "vec3<f32>", enabled: "f32", start: "f32", lift: "f32" },
};

// one texture per grid, shared by every layer; the ones before the newest are let go a moment later (a layer
// may still hold the last grid for a frame or two)
const gridTex = new Map<RiskGrid, Texture>();
let newest: RiskGrid | null = null;
let rampTex: Texture | null = null;
let blank: Texture | null = null;

function textures(device: Device, g: RiskGrid | null) {
  rampTex ??= (() => {
    const rgba = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) {
      rgba.set([VIRIDIS[i * 3]!, VIRIDIS[i * 3 + 1]!, VIRIDIS[i * 3 + 2]!, 255], i * 4);
    }
    return device.createTexture({ format: "rgba8unorm", width: 256, height: 1, mipLevels: 1, data: rgba,
      sampler: { minFilter: "linear", magFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" } });
  })();
  blank ??= device.createTexture({ format: "r8unorm", width: 1, height: 1, mipLevels: 1, data: new Uint8Array(1),
    sampler: { minFilter: "nearest", magFilter: "nearest" } });
  if (!g) return { grid: blank, ramp: rampTex };
  let t = gridTex.get(g);
  if (!t) {
    t = device.createTexture({ format: "r8unorm", width: g.width, height: g.height, mipLevels: 1, data: g.data,
      sampler: { minFilter: "nearest", magFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" } });
    gridTex.set(g, t);
    if (!newest || g.seq > newest.seq) newest = g;
    setTimeout(() => {
      for (const [k, v] of gridTex) if (newest && k.seq < newest.seq) { v.destroy(); gridTex.delete(k); }
    }, 3000);
  }
  return { grid: t, ramp: rampTex };
}

type ShadeProps = { riskGrid: RiskGrid | null };

export class RiskShadeExtension extends LayerExtension {
  static extensionName = "RiskShadeExtension";
  // compared by reference (one level): a new grid object reaches every sublayer
  static defaultProps = { riskGrid: { type: "object", value: null, compare: 1 } };

  getShaders() {
    return { modules: [riskShade] };
  }

  draw(this: Layer<ShadeProps>) {
    const g = this.props.riskGrid;
    const { grid, ramp } = textures(this.context.device, g);
    this.setShaderModuleProps({ riskShade: { grid, ramp, bounds: g?.bounds ?? [0, 0, 1, 1], enabled: g ? 1 : 0 } });
  }
}

export const RISK_SHADE = new RiskShadeExtension();
