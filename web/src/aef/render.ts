import type { RenderTileResult } from "@developmentseed/deck.gl-raster";
import type { Texture } from "@luma.gl/core";
import type { ShaderModule } from "@luma.gl/shadertools";
import { HOLD_SCALE } from "./fuel.js";
import { IGN_HI, IGN_LO } from "./head.js";

/** What the ground layer draws: land cover lit by FFWI, the classes, or the ignition head. */
export type FuelMode = "lit" | "cover" | "ignition";

export type FuelLightProps = {
  classTex: Texture;
  /** The ignition ground logit per pixel (a byte over IGN_LO..IGN_HI). */
  groundTex: Texture;
  /** The fuel head's P(fuel) per pixel (0 = none, else 1 + P * 254): the fuel weight in the lit mode. */
  fuelTex: Texture;
  /** Per node and lead: the FFWI (modes 0, 1) or the weather part of the ignition logit (mode 2). */
  ffwiTex: Texture;
  lutTex: Texture;
  rampTex: Texture;
  /** 0 = lit by the hour's FFWI, 1 = land cover classes, 2 = ignition probability. */
  mode: number;
  /** Ignition mode, probability scale: hidden below pHide, ramp top at pMax. */
  pHide: number;
  pMax: number;
  /** Ignition mode: 0 = the ramp over the probability (pHide..pMax), 1 = over the logit (zLo..zHi). */
  scale: number;
  zLo: number;
  zHi: number;
  /** 0 = the ground alone (the weather term left out), 1 = ground plus the hour's weather. */
  useWeather: number;
  /**
   * The standing picture: above 0, every pixel with ground is drawn at this opacity colored by the ground
   * alone, and goes opaque, colored by ground plus the hour, where the weather term crosses wxGate.
   * At 0 the hide line applies instead (nothing drawn below it).
   */
  standingAlpha: number;
  wxGate: number;
  /**
   * The weather field under everything (0 = off): the hour's weather term drawn as gray at this opacity over
   * every pixel the weather covers, ground or not, dark where calm and pale at the gate, so the weather's
   * currents show in the dead space; the ground's color sits on top where the gate is crossed.
   */
  fieldAlpha: number;
  layerIndex: number;
  hideBelow: number;
  rescaleMax: number;
  /** How many hours back the afterglow looks (0 = this hour only). */
  glowHours: number;
  /** Grain: 0 = a flat fill; above 0 the fill breaks into that many cells across the tile, each lit by the value. */
  grain: number;
  /**
   * The season view (fuel alone): above 0, the fuel draws at this share of its opacity, and at full opacity where
   * seasonTex is set (near a fire start that found fuel, inside a burn). 0 = off.
   */
  seasonDim: number;
  /** With seasonDim: 1 = the fuel outside the mask in gray (its own lightness), color only inside. */
  seasonGray: number;
  /** 1 = every fuel pixel at full alpha. */
  solid: number;
  /**
   * The imagery under the fuel (fuel alone, no weather): 0 = off (the ramp's colors); 1 = a cover in the basemap's
   * color over it, fading out from the hide line to the ramp top, so the imagery shows in full on the fuel and
   * faintly (VEIL_FLOOR) on the rest. Every pixel of a tile is drawn, even where nothing covers, so the layer
   * as a mask marks the whole tile.
   */
  veil: number;
  /** 1 = the fuel colored by stretchTex (each fuel byte's place on the ramp in this view). */
  stretch: number;
  stretchTex: Texture;
  /**
   * The fuel as a tint on the imagery under it (fuel alone): above 0, the ramp's color multiplied onto the imagery's
   * shade (layers.ts), so the colors are the Colors pane's and the imagery adds its light and dark; the value is the
   * tint's strength.
   */
  tint: number;
  /** The season mask per tile pixel (r8, 0..1). */
  seasonTex: Texture;
  /**
   * The replay (fuel alone): 1 = each pixel drawn by its arrival hour (arriveTex, below 0 = never burns): the before
   * year's fuel (v0Tex) at beforeAlpha until the front reaches it, hot for hotH hours (brighter the more stood there
   * before), then cooling over coolH to the after year's fuel (fuelTex) at full opacity.
   */
  replay: number;
  replayHour: number;
  hotH: number;
  coolH: number;
  beforeAlpha: number;
  v0Tex: Texture;
  arriveTex: Texture;
};

const MODULE_NAME = "fuelLight";
/** The cover over the imagery: CARTO Dark Matter's background (#0e0e0e), so covered ground looks like bare map. */
const VEIL_RGB = "0.0549, 0.0549, 0.0549";
/** How much of the imagery shows on ground that is not fuel. */
const VEIL_FLOOR = "0.15";

/**
 * Per fragment: the pixel's class (r8, nearest) -> its column in the lookup texture (row 0 rgb the
 * class color, a its fuel weight; row 1 r its hold hours) -> either the class color, or the lit
 * value: the hour's FFWI at this spot (the weather texture, linear) times the fuel weight, and with
 * the afterglow on, the brightest of the hours behind it, each held for the class's hold hours (from
 * the fires' own durations: wildland days, built ground a day or less). Hidden below the hide
 * line and ramped (viridis) from there to the ramp top, fading in over the first part of the ramp
 * so the ground comes up as the weather worsens instead of switching on. With grain on, the fill
 * breaks into ground-fixed grain whose density follows the value, so the ramp reads as texture.
 */
export const FuelLight = {
  name: MODULE_NAME,
  fs: `\
uniform ${MODULE_NAME}Uniforms {
  float mode;
  float layerIndex;
  float hideBelow;
  float rescaleMax;
  float glowHours;
  float grain;
  float pHide;
  float pMax;
  float scale;
  float zLo;
  float zHi;
  float useWeather;
  float standingAlpha;
  float wxGate;
  float fieldAlpha;
  float seasonDim;
  float seasonGray;
  float solid;
  float veil;
  float stretch;
  float tint;
  float replay;
  float replayHour;
  float hotH;
  float coolH;
  float beforeAlpha;
} ${MODULE_NAME};
`,
  inject: {
    "fs:#decl": `
precision highp sampler2DArray;
uniform sampler2D classTex;
uniform sampler2D groundTex;
uniform sampler2D fuelTex;
uniform sampler2DArray ffwiTex;
uniform sampler2D lutTex;
uniform sampler2D rampTex;
uniform sampler2D seasonTex;
uniform sampler2D stretchTex;
uniform sampler2D v0Tex;
uniform sampler2D arriveTex;

const float WX_NAN = intBitsToFloat(0x7fc00000);
/** A node outside the HRRR grid holds -Infinity at every hour: no weather here, ever. */
const float WX_OUT = uintBitsToFloat(0xff800000u);

/** Catmull-Rom weights for the four samples around a fraction t (at -1, 0, 1, 2). */
vec4 wxCatmull(float t) {
  float t2 = t * t;
  float t3 = t2 * t;
  return 0.5 * vec4(-t3 + 2.0 * t2 - t, 3.0 * t3 - 5.0 * t2 + 2.0, -3.0 * t3 + 4.0 * t2 + t, t3 - t2);
}

/**
 * One layer of the weather texture read bicubically (Catmull-Rom over the 4 x 4 nodes around the point), so a
 * threshold on the field is a curve, not the facets and diamond corners a bilinear read makes between nodes.
 * A missing node drops out and the rest are renormalized; with no node, NaN.
 */
float wxLayer(vec2 uv, int l) {
  ivec3 sz = textureSize(ffwiTex, 0);
  vec2 p = uv * vec2(sz.xy) - 0.5;
  vec2 p0 = floor(p);
  vec2 f = p - p0;
  vec4 wx = wxCatmull(f.x);
  vec4 wy = wxCatmull(f.y);
  float acc = 0.0;
  float wsum = 0.0;
  bool outside = false;
  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < 4; i++) {
      ivec2 q = clamp(ivec2(p0) + ivec2(i - 1, j - 1), ivec2(0), sz.xy - 1);
      float v = texelFetch(ffwiTex, ivec3(q, l), 0).r;
      if (isnan(v)) continue;
      if (isinf(v)) { outside = true; continue; }
      float w = wx[i] * wy[j];
      acc += w * v;
      wsum += w;
    }
  }
  return wsum > 0.05 ? acc / wsum : (outside ? WX_OUT : WX_NAN);
}

/**
 * The weather at a fractional hour: the two hours around it mixed by the fraction, so the field glides between
 * the forecast's hours while playing instead of jumping on the hour. A missing hour yields to the other one.
 */
float wxAt(vec2 uv, float l) {
  float l0 = floor(l);
  float f = l - l0;
  float depth = float(textureSize(ffwiTex, 0).z);
  float a = wxLayer(uv, int(l0));
  if (f <= 0.0 || l0 + 1.0 >= depth) return a;
  float b = wxLayer(uv, int(l0) + 1);
  if (isnan(a)) return b;
  if (isnan(b)) return a;
  if (isinf(a) || isinf(b)) return WX_OUT;
  return mix(a, b, f);
}

float fuelHash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
`,
    "fs:DECKGL_FILTER_COLOR": /* glsl */ `
      float ci = texture(classTex, geometry.uv).r * 255.0;
      bool hasGround = ci <= 254.5;
      // in the risk view open water is not ground: the near-shore water the embeddings cover then reads the same
      // as the open water past them (the weather field alone), with no band ending at the edge of the AEF files
      if (hasGround && ${MODULE_NAME}.mode > 1.5 && texelFetch(lutTex, ivec2(int(ci + 0.5), 1), 0).g > 0.5) hasGround = false;
      bool field = ${MODULE_NAME}.mode > 1.5 && ${MODULE_NAME}.fieldAlpha > 0.0 && ${MODULE_NAME}.useWeather > 0.5;
      if (!hasGround && !field) discard;
      vec4 lut = hasGround ? texelFetch(lutTex, ivec2(int(ci + 0.5), 0), 0) : vec4(0.0);
      if (${MODULE_NAME}.mode > 1.5) {
        // the ignition head: ground logit (per pixel) plus the hour's weather logit (per node), sigmoid;
        // with the afterglow on, the brightest of the hours behind, each losing 0.3 logit per hour
        float g = hasGround ? texture(groundTex, geometry.uv).r * ${IGN_HI - IGN_LO}.0 + (${IGN_LO}.0) : 0.0;
        float z = g;
        // the hour's weather term, two ways back through the afterglow: the field keeps the weather's own
        // short memory (the strongest recent hour losing 0.3 logit per hour); the ground holds the hour by
        // what it is (row 1 of the lookup: hold hours per land cover): a past hour's term slides from its
        // value down to half a logit under the gate over the hold, so the ground stays lit for that long
        // after its peak whatever the peak was; timber, shrub and grass for days, built ground a day or less
        float wx = -99.0;
        float wxG = -99.0;
        float hold = hasGround ? texelFetch(lutTex, ivec2(int(ci + 0.5), 1), 0).r * 255.0 / ${HOLD_SCALE}.0 : 0.0;
        if (${MODULE_NAME}.useWeather > 0.5) {
          float wl = wxAt(geometry.uv, ${MODULE_NAME}.layerIndex);
          // the ground and the weather are one footprint: outside the HRRR grid nothing draws; a pixel whose
          // shard is not in hand yet (NaN) keeps the standing gray until it arrives
          if (isinf(wl)) discard;
          if (!isnan(wl)) { wx = wl; wxG = wl; }
          int back = int(min(${MODULE_NAME}.glowHours, 48.0));
          for (int k = 1; k <= 48; k++) {
            if (k > back) break;
            float l = ${MODULE_NAME}.layerIndex - float(k);
            if (l < 0.0) break;
            float wk = wxAt(geometry.uv, l);
            if (isnan(wk)) continue;
            wx = max(wx, wk - 0.3 * float(k));
            if (hold > 0.0) wxG = max(wxG, wk - max(wk - (${MODULE_NAME}.wxGate - 0.5), 0.0) * min(float(k) / hold, 1.0));
          }
          z = g + wxG;
        }
        // the weather field under everything: the term as gray, dark 3 logits below the gate, pale at it
        vec4 under = vec4(0.0);
        if (field && wx > -90.0) {
          float tf = clamp((wx - (${MODULE_NAME}.wxGate - 3.0)) / 4.0, 0.0, 1.0);
          under = vec4(vec3(0.10 + 0.80 * tf), ${MODULE_NAME}.fieldAlpha);
        }
        vec3 rgb = vec3(0.0);
        float a = 0.0;
        if (hasGround) {
          bool standing = ${MODULE_NAME}.standingAlpha > 0.0 && ${MODULE_NAME}.useWeather > 0.5;
          // standing: how far the weather is across the gate (0 = calm, 1 = lit), over half a logit
          float lit = standing ? smoothstep(${MODULE_NAME}.wxGate, ${MODULE_NAME}.wxGate + 0.5, wxG) : 1.0;
          float zc = standing ? mix(g, z, lit) : z;
          float t;
          bool hide = false;
          if (${MODULE_NAME}.scale > 0.5) {
            if (!standing && zc < ${MODULE_NAME}.zLo) hide = true;
            t = clamp((zc - ${MODULE_NAME}.zLo) / max(${MODULE_NAME}.zHi - ${MODULE_NAME}.zLo, 0.01), 0.0, 1.0);
          } else {
            float pr = 1.0 / (1.0 + exp(-zc));
            if (!standing && pr < ${MODULE_NAME}.pHide) hide = true;
            t = clamp((pr - ${MODULE_NAME}.pHide) / max(${MODULE_NAME}.pMax - ${MODULE_NAME}.pHide, 0.0001), 0.0, 1.0);
          }
          a = hide ? 0.0 : standing ? mix(${MODULE_NAME}.standingAlpha, 1.0, lit) : min(1.0, t / 0.15);
          if (${MODULE_NAME}.grain > 0.0) {
            vec2 cell = floor(geometry.uv * ${MODULE_NAME}.grain);
            float h = fuelHash(cell + vec2(0.37, 0.91));
            a *= smoothstep(h - 0.08, h + 0.08, 0.1 + 0.9 * t);
          }
          rgb = texture(rampTex, vec2(t, 0.5)).rgb;
          if (standing) {
            // the standing picture in gray (the ramp's luminance), color only where the hour lifts the pixel
            float lum = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
            rgb = mix(vec3(lum), rgb, lit);
          }
        }
        if (under.a > 0.0) {
          // the ground over the field
          float ao = a + under.a * (1.0 - a);
          rgb = (rgb * a + under.rgb * under.a * (1.0 - a)) / max(ao, 1e-4);
          a = ao;
        }
        if (a < 0.01) discard;
        color = vec4(rgb, a);
      } else if (${MODULE_NAME}.mode > 0.5) {
        color = vec4(lut.rgb, 1.0);
      } else {
        // the fuel weight: the fuel head's P(fuel) (fuel/head.ts) in place of the land cover class's weight;
        // no fuel plane (the COG overviews) draws nothing
        float fb = texture(fuelTex, geometry.uv).r * 255.0;
        if (${MODULE_NAME}.veil > 0.5 && ${MODULE_NAME}.useWeather < 0.5) {
          // the imagery under: how much of it shows here (0 to 1), the fuel's place on the ramp; the cover over it
          // is the basemap's own color. No embedding: covered as non-fuel
          float show = 0.0;
          if (fb >= 0.5) {
            float fv = (fb - 1.0) / 254.0;
            if (fv >= ${MODULE_NAME}.pHide) {
              if (${MODULE_NAME}.scale > 0.5) {
                float h0 = clamp(${MODULE_NAME}.pHide, 0.002, 0.996);
                float h1 = clamp(${MODULE_NAME}.pMax, 0.002, 0.996);
                float pv = clamp(fv, 0.002, 0.996);
                show = clamp((log(pv / (1.0 - pv)) - log(h0 / (1.0 - h0))) / max(log(h1 / (1.0 - h1)) - log(h0 / (1.0 - h0)), 0.01), 0.0, 1.0);
              } else {
                show = clamp((fv - ${MODULE_NAME}.pHide) / max(${MODULE_NAME}.pMax - ${MODULE_NAME}.pHide, 0.0001), 0.0, 1.0);
              }
            }
          }
          color = vec4(vec3(${VEIL_RGB}), (1.0 - show) * (1.0 - ${VEIL_FLOOR}));
          return;
        }
        if (fb < 0.5) discard;
        float fw = (fb - 1.0) / 254.0;
        if (${MODULE_NAME}.useWeather < 0.5 && ${MODULE_NAME}.replay > 0.5) {
          // the replay: the before year's fuel and the after year's on the ramp, as the fuel alone is placed below
          float fb0 = texture(v0Tex, geometry.uv).r * 255.0;
          float place[2];
          for (int k = 0; k < 2; k++) {
            float b = k == 0 ? fb0 : fb;
            float v = (b - 1.0) / 254.0;
            if (${MODULE_NAME}.scale > 0.5) {
              float h0 = clamp(${MODULE_NAME}.pHide, 0.002, 0.996);
              float h1 = clamp(${MODULE_NAME}.pMax, 0.002, 0.996);
              float pv = clamp(v, 0.002, 0.996);
              place[k] = clamp((log(pv / (1.0 - pv)) - log(h0 / (1.0 - h0))) / max(log(h1 / (1.0 - h1)) - log(h0 / (1.0 - h0)), 0.01), 0.0, 1.0);
            } else {
              place[k] = clamp((v - ${MODULE_NAME}.pHide) / max(${MODULE_NAME}.pMax - ${MODULE_NAME}.pHide, 0.0001), 0.0, 1.0);
            }
          }
          float ar = texture(arriveTex, geometry.uv).r;
          float now = ${MODULE_NAME}.replayHour;
          // outside the fire's final outline: not drawn
          if (ar < 0.0) discard;
          if (now < ar) {
            // not reached yet: the before year (full before the fire starts, dim once it has)
            if (fb0 < 0.5 || (fb0 - 1.0) / 254.0 < ${MODULE_NAME}.pHide) discard;
            color = vec4(texture(rampTex, vec2(place[0], 0.5)).rgb, ${MODULE_NAME}.beforeAlpha);
            return;
          }
          // burning: amber to near white by how much stood there before, whitest right at the front; light, not hue,
          // carries it, so it reads apart from every fuel ramp without red
          float dt = now - ar;
          float heat = fb0 < 0.5 ? 0.3 : (fb0 - 1.0) / 254.0;
          vec3 hot = mix(vec3(1.0, 0.66, 0.22), vec3(1.0, 0.98, 0.90), heat) * (0.55 + 0.45 * heat);
          hot = mix(hot, vec3(1.0), 0.5 * exp(-dt / 0.75));
          if (dt < ${MODULE_NAME}.hotH) {
            color = vec4(hot, 1.0);
            return;
          }
          // cooling to the after year's fuel, then that alone
          float k = smoothstep(0.0, 1.0, clamp((dt - ${MODULE_NAME}.hotH) / max(${MODULE_NAME}.coolH, 0.01), 0.0, 1.0));
          color = vec4(mix(hot, texture(rampTex, vec2(place[1], 0.5)).rgb, k), 1.0);
          return;
        }
        if (${MODULE_NAME}.useWeather < 0.5) {
          // the weather left out: P(fuel) alone, hidden below pHide, ramped from there to pMax, straight or on the
          // log-odds (scale 1), which gives the crowded top (90, 96, 99 %) room on the ramp
          if (fw < ${MODULE_NAME}.pHide) discard;
          float tp;
          if (${MODULE_NAME}.scale > 0.5) {
            float lo = log(clamp(${MODULE_NAME}.pHide, 0.002, 0.996) / (1.0 - clamp(${MODULE_NAME}.pHide, 0.002, 0.996)));
            float pm = clamp(${MODULE_NAME}.pMax, 0.002, 0.996);
            float hi = log(pm / (1.0 - pm));
            float pf = clamp(fw, 0.002, 0.996);
            tp = clamp((log(pf / (1.0 - pf)) - lo) / max(hi - lo, 0.01), 0.0, 1.0);
          } else {
            tp = clamp((fw - ${MODULE_NAME}.pHide) / max(${MODULE_NAME}.pMax - ${MODULE_NAME}.pHide, 0.0001), 0.0, 1.0);
          }
          // stretched to the view: the byte's place among the view's values, read between texels as the fuel is
          if (${MODULE_NAME}.stretch > 0.5) tp = texture(stretchTex, vec2((fb + 0.5) / 256.0, 0.5)).r;
          // the season view: dim except where the tile's mask says a fire touched it
          float mk = texture(seasonTex, geometry.uv).r;
          float sm = ${MODULE_NAME}.seasonDim > 0.0 ? mix(${MODULE_NAME}.seasonDim, 1.0, mk) : 1.0;
          vec3 rc = texture(rampTex, vec2(tp, 0.5)).rgb;
          if (${MODULE_NAME}.tint > 0.0) {
            // the ramp's own color, multiplied onto the imagery's shade (layers.ts): the Colors pane's colors,
            // darkened where the ground is dark
            color = vec4(mix(vec3(1.0), rc, ${MODULE_NAME}.tint), 1.0);
            return;
          }
          // gray outside the mask: the ramp's own lightness, no hue; the color comes in with the mask
          if (${MODULE_NAME}.seasonGray > 0.5) rc = mix(vec3(dot(rc, vec3(0.299, 0.587, 0.114))), rc, mk);
          color = vec4(rc, (${MODULE_NAME}.solid > 0.5 ? 1.0 : min(1.0, 0.35 + tp)) * sm);
          return;
        }
        float f = wxAt(geometry.uv, ${MODULE_NAME}.layerIndex);
        float v = isnan(f) ? -1.0 : f * fw;
        float hold = texelFetch(lutTex, ivec2(int(ci + 0.5), 1), 0).r * 255.0 / ${HOLD_SCALE}.0;
        int back = int(min(${MODULE_NAME}.glowHours, 48.0));
        if (hold > 0.0) {
          for (int k = 1; k <= 48; k++) {
            if (k > back) break;
            float l = ${MODULE_NAME}.layerIndex - float(k);
            if (l < 0.0) break;
            float g = wxAt(geometry.uv, l);
            if (isnan(g)) continue;
            v = max(v, g * fw * exp(-float(k) / hold));
          }
        }
        if (v < ${MODULE_NAME}.hideBelow) discard;
        float t = clamp((v - ${MODULE_NAME}.hideBelow) / max(${MODULE_NAME}.rescaleMax - ${MODULE_NAME}.hideBelow, 1.0), 0.0, 1.0);
        float a = min(1.0, t / 0.15);
        if (${MODULE_NAME}.grain > 0.0) {
          // grain fixed to the tile: each cell lit when its hash falls under the value, so density follows the value
          vec2 cell = floor(geometry.uv * ${MODULE_NAME}.grain);
          float h = fuelHash(cell + vec2(0.37, 0.91));
          float keep = smoothstep(h - 0.08, h + 0.08, 0.1 + 0.9 * t);
          a *= keep;
          if (a < 0.01) discard;
        }
        color = vec4(texture(rampTex, vec2(t, 0.5)).rgb, a);
      }
    `,
  },
  uniformTypes: { mode: "f32", layerIndex: "f32", hideBelow: "f32", rescaleMax: "f32", glowHours: "f32", grain: "f32", pHide: "f32", pMax: "f32", scale: "f32", zLo: "f32", zHi: "f32", useWeather: "f32", standingAlpha: "f32", wxGate: "f32", fieldAlpha: "f32", seasonDim: "f32", seasonGray: "f32", solid: "f32", veil: "f32", stretch: "f32", tint: "f32", replay: "f32", replayHour: "f32", hotH: "f32", coolH: "f32", beforeAlpha: "f32" },
  getUniforms: (p: Partial<FuelLightProps>) => ({
    mode: p.mode ?? 0,
    layerIndex: p.layerIndex ?? 0,
    hideBelow: p.hideBelow ?? 0,
    rescaleMax: p.rescaleMax ?? 100,
    glowHours: p.glowHours ?? 0,
    grain: p.grain ?? 0,
    pHide: p.pHide ?? 0.5,
    pMax: p.pMax ?? 0.95,
    scale: p.scale ?? 0,
    zLo: p.zLo ?? 0,
    zHi: p.zHi ?? 4,
    useWeather: p.useWeather ?? 1,
    standingAlpha: p.standingAlpha ?? 0,
    wxGate: p.wxGate ?? 1,
    fieldAlpha: p.fieldAlpha ?? 0,
    seasonDim: p.seasonDim ?? 0,
    seasonGray: p.seasonGray ?? 0,
    solid: p.solid ?? 0,
    veil: p.veil ?? 0,
    stretch: p.stretch ?? 0,
    tint: p.tint ?? 0,
    replay: p.replay ?? 0,
    replayHour: p.replayHour ?? 0,
    hotH: p.hotH ?? 3,
    coolH: p.coolH ?? 9,
    beforeAlpha: p.beforeAlpha ?? 0.35,
    v0Tex: p.v0Tex,
    arriveTex: p.arriveTex,
    stretchTex: p.stretchTex,
    seasonTex: p.seasonTex,
    classTex: p.classTex,
    groundTex: p.groundTex,
    fuelTex: p.fuelTex,
    ffwiTex: p.ffwiTex,
    lutTex: p.lutTex,
    rampTex: p.rampTex,
  }),
} as const satisfies ShaderModule<FuelLightProps>;

export function fuelPipeline(p: FuelLightProps): RenderTileResult {
  return { renderPipeline: [{ module: FuelLight, props: p }] };
}
