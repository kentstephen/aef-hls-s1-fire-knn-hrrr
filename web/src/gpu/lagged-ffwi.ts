import type { Texture } from "@luma.gl/core";
import type { ShaderModule } from "@luma.gl/shadertools";

/**
 * FFWI on the GPU from two Texture2DArrays (49 leads each): equilibrium
 * moisture (%) and wind (mph). With alpha = exp(-1 / tau) > 0 the fragment
 * walks the leads from +0 h to the current one applying the fuel-moisture
 * timelag recursion (`ffwi/compute.ts lagMoisture`); with alpha = 0 it reads
 * the current lead only (Fosberg as published). Then eta(m) and the wind
 * term, anchored at 0.3002: this is the GLSL copy of `ffwiFromMoisture`.
 * Keep the two in sync. EMC itself never lives in GLSL.
 */
export type LaggedFfwiProps = {
  mEqTex: Texture;
  windTex: Texture;
  layerIndex: number;
  alpha: number;
};

const MODULE_NAME = "laggedFfwi";

export const LaggedFfwi = {
  name: MODULE_NAME,
  fs: `\
uniform ${MODULE_NAME}Uniforms {
  float layerIndex;
  float alpha;
} ${MODULE_NAME};
`,
  inject: {
    "fs:#decl": `
precision highp sampler2DArray;
uniform sampler2DArray mEqTex;
uniform sampler2DArray windTex;
`,
    "fs:DECKGL_FILTER_COLOR": /* glsl */ `
      float L = ${MODULE_NAME}.layerIndex;
      float a = ${MODULE_NAME}.alpha;
      float m;
      if (a <= 0.0) {
        m = texture(mEqTex, vec3(geometry.uv, L)).r;
      } else {
        m = texture(mEqTex, vec3(geometry.uv, 0.0)).r;
        for (int t = 1; t < 49; t++) {
          if (float(t) > L + 0.5) break;
          float me = texture(mEqTex, vec3(geometry.uv, float(t))).r;
          m = me + (m - me) * a;
        }
      }
      float U = texture(windTex, vec3(geometry.uv, L)).r;
      if (isnan(m) || isnan(U)) discard;
      float mr = m / 30.0;
      float eta = 1.0 - 2.0 * mr + 1.5 * mr * mr - 0.5 * mr * mr * mr;
      float v = eta * sqrt(1.0 + U * U) / 0.3002;
      color = vec4(v, v, v, 1.0);
    `,
  },
  uniformTypes: {
    layerIndex: "f32",
    alpha: "f32",
  },
  getUniforms: (props: Partial<LaggedFfwiProps>) => ({
    layerIndex: props.layerIndex ?? 0,
    alpha: props.alpha ?? 0,
    mEqTex: props.mEqTex,
    windTex: props.windTex,
  }),
} as const satisfies ShaderModule<LaggedFfwiProps>;
