import type { RenderTileResult } from "@developmentseed/deck.gl-raster";
import { Colormap, LinearRescale } from "@developmentseed/deck.gl-raster/gpu-modules";
import type { Texture } from "@luma.gl/core";
import { FilterRange } from "../gpu/filter-range.js";
import { LaggedFfwi } from "../gpu/lagged-ffwi.js";
import { SampleTexture2DArray } from "../gpu/sample-texture-2d-array.js";
import type { FfwiTileData } from "./get-tile-data.js";

export type MakeRenderTileArgs = {
  layerIndex: number;
  colormapTexture: Texture;
  colormapIndex: number;
  reversed: boolean;
  /** Native-unit value below which pixels are discarded (the map shows through). */
  hideBelow: number;
  /** Colormap domain, native units. */
  rescaleMin: number;
  rescaleMax: number;
  /** Fuel-moisture timelag in hours (FFWI only); 0 = Fosberg as published. */
  tauHours: number;
};

/**
 * GPU pipeline: the field at the current lead (FFWI computed per fragment
 * from moisture + wind with the timelag; other variables sampled directly)
 * -> discard low values -> rescale to the colormap domain -> colormap.
 */
export function makeRenderTile(a: MakeRenderTileArgs) {
  const alpha = a.tauHours > 0 ? Math.exp(-1 / a.tauHours) : 0;
  return function renderTile(data: FfwiTileData): RenderTileResult {
    const field = data.windTexture
      ? { module: LaggedFfwi, props: { mEqTex: data.texture, windTex: data.windTexture, layerIndex: a.layerIndex, alpha } }
      : { module: SampleTexture2DArray, props: { dataTex: data.texture, layerIndex: a.layerIndex } };
    return {
      renderPipeline: [
        field,
        { module: FilterRange, props: { filterMin: a.hideBelow, filterMax: Number.POSITIVE_INFINITY } },
        { module: LinearRescale, props: { rescaleMin: a.rescaleMin, rescaleMax: a.rescaleMax } },
        { module: Colormap, props: { colormapTexture: a.colormapTexture, colormapIndex: a.colormapIndex, reversed: a.reversed } },
      ],
    };
  };
}
