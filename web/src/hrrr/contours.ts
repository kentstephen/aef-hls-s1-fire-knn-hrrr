import { contours } from "d3-contour";
import { HRRR_GRID, hrrrPixelToLonLat, lonLatToHrrrPixelF } from "./metadata.js";
import type { FfwiTileCache } from "./tile-cache.js";

/**
 * Contours of the hour's FFWI field over the view, drawn as lines over the lit ground so the
 * weather has edges and not only a tint. Marching squares (d3-contour) over the HRRR pixels the
 * view covers, from the shared cache (the lagged value, as the ground uses); decimated so a
 * continent-wide view stays under a quarter million cells and a frame change costs tens of ms.
 * Pixels not loaded or outside the grid count as below every level.
 */
export type ContourPath = { level: number; path: [number, number][] };

const MAX_CELLS = 250_000;

export function ffwiContours(
  cache: FfwiTileCache,
  lead: number,
  /** [west, south, east, north] of the view. */
  view: [number, number, number, number],
  levels: number[],
): ContourPath[] {
  if (cache.size === 0) return [];
  const [w, s, e, n] = view;
  // the view's outline in HRRR pixels (the grid is a conic, so the edges bow): sample each edge
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const take = (lon: number, lat: number) => {
    const p = lonLatToHrrrPixelF(Math.max(-180, Math.min(180, lon)), Math.max(-85, Math.min(85, lat)));
    if (!p) return;
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
  };
  for (let i = 0; i <= 8; i++) {
    const f = i / 8;
    take(w + (e - w) * f, s);
    take(w + (e - w) * f, n);
    take(w, s + (n - s) * f);
    take(e, s + (n - s) * f);
  }
  if (!Number.isFinite(minX)) {
    // the view's edges are off the grid: if it holds the grid, take all of it
    if (w < -130 && e > -60 && s < 25 && n > 50) { minX = 0; minY = 0; maxX = HRRR_GRID.width; maxY = HRRR_GRID.height; } else return [];
  }
  const colStart = Math.max(0, Math.floor(minX) - 2);
  const rowStart = Math.max(0, Math.floor(minY) - 2);
  const colEnd = Math.min(HRRR_GRID.width, Math.ceil(maxX) + 2);
  const rowEnd = Math.min(HRRR_GRID.height, Math.ceil(maxY) + 2);
  const fw = colEnd - colStart;
  const fh = rowEnd - rowStart;
  if (fw < 2 || fh < 2) return [];
  const dec = Math.max(1, Math.ceil(Math.sqrt((fw * fh) / MAX_CELLS)));
  const gw = Math.ceil(fw / dec);
  const gh = Math.ceil(fh / dec);
  const values = new Float64Array(gw * gh);
  for (let j = 0; j < gh; j++) {
    const y = rowStart + j * dec;
    for (let i = 0; i < gw; i++) {
      const v = cache.ffwiAt(colStart + i * dec, y, lead);
      values[j * gw + i] = v === v ? v : -1;
    }
  }
  const gen = contours().size([gw, gh]).thresholds(levels).smooth(true);
  const out: ContourPath[] = [];
  for (const mp of gen(values as unknown as number[])) {
    for (const poly of mp.coordinates) {
      for (const ring of poly) {
        const path: [number, number][] = [];
        for (const [gx, gy] of ring) {
          path.push(hrrrPixelToLonLat(colStart + gx! * dec - 0.5, rowStart + gy! * dec - 0.5));
        }
        if (path.length > 2) out.push({ level: mp.value, path });
      }
    }
  }
  return out;
}
