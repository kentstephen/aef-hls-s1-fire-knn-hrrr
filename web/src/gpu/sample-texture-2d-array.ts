import type { Texture } from "@luma.gl/core";
import type { ShaderModule } from "@luma.gl/shadertools";

export type SampleTexture2DArrayProps = {
  dataTex: Texture;
  layerIndex: number;
};

const MODULE_NAME = "sampleTexture2DArray";

/**
 * Samples a sampler2DArray at (uv, layerIndex), writes scalar to color.rgb.
 * Inlined from deck.gl-raster/examples/dynamical-zarr-ecmwf.
 */
export const SampleTexture2DArray = {
  name: MODULE_NAME,
  fs: `\
uniform ${MODULE_NAME}Uniforms {
  float layerIndex;
} ${MODULE_NAME};
`,
  inject: {
    "fs:#decl": `
precision highp sampler2DArray;
uniform sampler2DArray dataTex;
`,
    "fs:DECKGL_FILTER_COLOR": /* glsl */ `
      float v = texture(dataTex, vec3(geometry.uv, ${MODULE_NAME}.layerIndex)).r;
      if (isnan(v)) discard;
      color = vec4(v, v, v, 1.0);
    `,
  },
  uniformTypes: {
    layerIndex: "f32",
  },
  getUniforms: (props: Partial<SampleTexture2DArrayProps>) => ({
    layerIndex: props.layerIndex ?? 0,
    dataTex: props.dataTex,
  }),
} as const satisfies ShaderModule<SampleTexture2DArrayProps>;
