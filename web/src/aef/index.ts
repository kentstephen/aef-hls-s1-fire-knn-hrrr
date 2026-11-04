import { AEF_YEAR } from "./mosaic.js";
/**
 * The browser's index of the AlphaEarth COGs of AEF_YEAR over CONUS (web/public/aef-<year>-conus.json and
 * one aef-<year>-ranges-<zone>.bin per UTM zone, both from scripts/build_aef_index.py). The JSON has
 * every file's bounds; the binary has, per file, the byte range of each coarse overview level and
 * each band's block inside it, so a level is one range request with no header round trip.
 */
export type AefFile = {
  rel: string;
  bbox: [number, number, number, number];
  zone: string;
  /** Row in the zone's ranges file. */
  ri: number;
  utmWest: number;
  utmSouth: number;
};

export type AefIndex = {
  base: string;
  /** Overview level widths in px, coarsest last; factor = 8192 / width, resolution = 10 m x factor. */
  levels: number[];
  block: number;
  zones: Record<string, { epsg: number; n: number }>;
  tiles: AefFile[];
};

export const FULL_WIDTH = 8192;
export const NUM_BANDS = 64;
/** Bytes per file record in a zone's ranges file: float64 start x 7 levels, then uint32 (rel, count) x 64 x 7. */
export const RECORD_BYTES = 7 * 8 + 7 * 64 * 8;

export async function loadAefIndex(baseUrl: string): Promise<AefIndex> {
  const r = await fetch(`${baseUrl}aef-${AEF_YEAR}-conus.json`);
  if (!r.ok) throw new Error(`AEF index: HTTP ${r.status}`);
  const j = (await r.json()) as {
    base: string;
    levels: number[];
    block: number;
    zones: Record<string, { epsg: number; n: number }>;
    tiles: [string, number, number, number, number, string, number, number, number][];
  };
  return {
    base: j.base,
    levels: j.levels,
    block: j.block,
    zones: j.zones,
    tiles: j.tiles.map(([rel, w, s, e, n, zone, ri, utmWest, utmSouth]) => ({ rel, bbox: [w, s, e, n], zone, ri, utmWest, utmSouth })),
  };
}

/** The overview level for a tile: the one whose resolution is nearest the tile's ground resolution. */
export function levelFor(index: AefIndex, groundResM: number): number {
  const factor = Math.max(8, Math.min(512, 2 ** Math.round(Math.log2(Math.max(groundResM, 10) / 10))));
  const width = FULL_WIDTH / factor;
  const i = index.levels.indexOf(width);
  return i >= 0 ? i : index.levels.length - 1;
}
