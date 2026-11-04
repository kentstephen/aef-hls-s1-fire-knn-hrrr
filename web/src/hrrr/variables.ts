import type { WindMode } from "../ffwi/compute.js";

/**
 * What the raster can show: FFWI or any of its HRRR inputs, one per view.
 * All come from the same five-band tile fetch; the selected one is written
 * to the tile's Texture2DArray (49 leads) in getTileData. Ranges are the
 * colormap's domain in native units. RH is reversed so that dry (dangerous)
 * is the bright end, keeping "bright = worse" across every variable.
 * Temperature gets the diverging cool-warm map (blue cold, red hot; a
 * blue-red axis survives protan vision, red-green would not).
 */
export type RasterVariable = "ffwi" | "temperature" | "rh" | "wind";

export type RasterVariableDef = {
  id: RasterVariable;
  label: string;
  units: string;
  min: number;
  max: number;
  reversed: boolean;
  /** Colormap applied when this variable is selected (the select still overrides). */
  defaultColormap: "solar" | "coolwarm";
  format: (v: number) => string;
};

export const RASTER_VARIABLES: Record<RasterVariable, RasterVariableDef> = {
  ffwi: { id: "ffwi", label: "Fosberg Fire Weather Index", units: "", min: 0, max: 100, reversed: false, defaultColormap: "solar", format: (v) => v.toFixed(0) },
  temperature: { id: "temperature", label: "Temperature (2 m)", units: "°C", min: -10, max: 45, reversed: false, defaultColormap: "coolwarm", format: (v) => `${v.toFixed(0)} °C` },
  rh: { id: "rh", label: "Relative humidity (2 m)", units: "%", min: 0, max: 100, reversed: true, defaultColormap: "solar", format: (v) => `${v.toFixed(0)} %` },
  wind: { id: "wind", label: "Wind (10 m)", units: "mph", min: 0, max: 50, reversed: false, defaultColormap: "solar", format: (v) => `${v.toFixed(0)} mph` },
};

export function windLabel(mode: WindMode): string {
  return mode === "gust" ? "Wind gust (surface)" : "Wind (10 m sustained)";
}
