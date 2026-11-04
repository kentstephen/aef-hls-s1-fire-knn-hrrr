import type { ShaderModule } from "@luma.gl/shadertools";

/**
 * Discards fragments whose scalar value (`color.r`) falls outside
 * [filterMin, filterMax]. Runs after SampleTexture2DArray (raw zarr value
 * in color.r) and before LinearRescale. Lifted verbatim from
 * deck.gl-raster/examples/dynamical-zarr-ecmwf, kept local so we don't
 * pull a one-file dependency.
 */
export type FilterRangeProps = {
  filterMin: number;
  filterMax: number;
};

const MODULE_NAME = "filterRange";

export const FilterRange = {
  name: MODULE_NAME,
  fs: `\
uniform ${MODULE_NAME}Uniforms {
  float filterMin;
  float filterMax;
} ${MODULE_NAME};
`,
  inject: {
    "fs:DECKGL_FILTER_COLOR": /* glsl */ `
      if (color.r < ${MODULE_NAME}.filterMin || color.r > ${MODULE_NAME}.filterMax) {
        discard;
      }
    `,
  },
  uniformTypes: {
    filterMin: "f32",
    filterMax: "f32",
  },
  getUniforms: (props: Partial<FilterRangeProps>) => {
    return {
      filterMin: props.filterMin ?? Number.NEGATIVE_INFINITY,
      filterMax: props.filterMax ?? Number.POSITIVE_INFINITY,
    };
  },
} as const satisfies ShaderModule<FilterRangeProps>;
