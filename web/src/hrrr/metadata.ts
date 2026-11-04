import proj4 from "proj4";

/**
 * Dynamical NOAA HRRR 48-hour forecast store. Grid facts verified 2026-08-21
 * (docs/data-sources.md). Lambert Conformal Conic on a sphere R=6371229 m,
 * central meridian -97.5, latitude of origin 38.5, both standard parallels
 * 38.5. Shape 1059 (y) x 1799 (x), 3 km pixels. Dim order on the variable
 * arrays: (init_time, lead_time, y, x).
 */

/** Icechunk v2 repo on AWS Open Data (CORS *). Read with icechunk-js. */
export const HRRR_ICECHUNK_URL =
  "https://dynamical-noaa-hrrr.s3.us-west-2.amazonaws.com/noaa-hrrr-forecast-48-hour/v0.1.0.icechunk";
/** Plain Zarr v3 mirror (deprecated upstream, still in sync). Fallback only. */
export const HRRR_ZARR_FALLBACK_URL =
  "https://data.source.coop/dynamical/noaa-hrrr-forecast-48-hour/v0.1.0.zarr";

const HRRR_WKT = `PROJCS["unnamed",GEOGCS["Coordinate System imported from GRIB file",DATUM["unnamed",SPHEROID["Sphere",6371229,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["latitude_of_origin",38.5],PARAMETER["central_meridian",-97.5],PARAMETER["standard_parallel_1",38.5],PARAMETER["standard_parallel_2",38.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]`;

/** proj4 string equivalent of HRRR_WKT, for the per-cell pixel lookup. */
export const HRRR_PROJ4 =
  "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs";

/** Synthetic GeoZarr attrs (the store is not GeoZarr-compliant). Affine is
 *  top-left edge based: x_out = 3000*col + x0, y_out = -3000*row + y0. */
export const HRRR_GEOZARR_ATTRS = {
  "spatial:dimensions": ["y", "x"],
  "spatial:transform": [3000, 0, -2699020.142521929, 0, -3000, 1588193.847443335],
  "spatial:shape": [1059, 1799],
  "proj:wkt2": HRRR_WKT,
} as const;

export const HRRR_GRID = {
  x0: -2699020.142521929,
  dx: 3000,
  width: 1799,
  y0: 1588193.847443335,
  dy: -3000,
  height: 1059,
} as const;

/** Inner shard (chunk) shape on the spatial dims. zarrita only exposes the
 *  outer chunk grid, which for this store is the whole grid, so this is
 *  hardcoded (verified from `temperature_2m/zarr.json`). */
export const HRRR_SHARD = { height: 265, width: 300 } as const;

export const HRRR_LEAD_TIME_COUNT = 49;
export const HRRR_LEAD_TIME_HOURS: readonly number[] = Array.from(
  { length: HRRR_LEAD_TIME_COUNT },
  (_, i) => i,
);

/** Inits every 6 h since 2018-07-13 12:00 UTC. */
export const INIT_TIME_ORIGIN = new Date("2018-07-13T12:00:00Z");
const MS_PER_INIT_STEP = 6 * 60 * 60 * 1000;

export function dateFromInitTimeIdx(idx: number): Date {
  return new Date(INIT_TIME_ORIGIN.getTime() + idx * MS_PER_INIT_STEP);
}

export function initTimeIdxFromDate(date: Date, maxIdx: number): number {
  const stepDiff = Math.round(
    (date.getTime() - INIT_TIME_ORIGIN.getTime()) / MS_PER_INIT_STEP,
  );
  return Math.max(0, Math.min(maxIdx, stepDiff));
}

export function isoDateString(date: Date): string {
  return date.toISOString().slice(0, 10);
}

/** The five FFWI inputs. Order is fixed and matches the tile fetch. */
export const FFWI_BANDS = [
  "temperature_2m",
  "relative_humidity_2m",
  "wind_u_10m",
  "wind_v_10m",
  "wind_gust_surface",
] as const;
export type FfwiBand = (typeof FFWI_BANDS)[number];

const lccConverter = proj4("EPSG:4326", HRRR_PROJ4);
const lccForward = lccConverter.forward;
const lccInverse = lccConverter.inverse;

/**
 * HRRR pixel (col, row) for a lon/lat, or null when outside the grid.
 * Uses the top-left-edge affine, so `floor` gives the containing pixel.
 */
export function lonLatToHrrrPixel(
  lon: number,
  lat: number,
): { x: number; y: number } | null {
  const [px, py] = lccForward([lon, lat]);
  const col = Math.floor((px - HRRR_GRID.x0) / HRRR_GRID.dx);
  const row = Math.floor((py - HRRR_GRID.y0) / HRRR_GRID.dy);
  if (col < 0 || col >= HRRR_GRID.width || row < 0 || row >= HRRR_GRID.height) {
    return null;
  }
  return { x: col, y: row };
}

/** Fractional HRRR pixel coordinates for a lon/lat (pixel centers at .5), or null outside the grid. */
export function lonLatToHrrrPixelF(lon: number, lat: number): { x: number; y: number } | null {
  const [px, py] = lccForward([lon, lat]);
  const x = (px - HRRR_GRID.x0) / HRRR_GRID.dx;
  const y = (py - HRRR_GRID.y0) / HRRR_GRID.dy;
  if (!(x >= 0 && x < HRRR_GRID.width && y >= 0 && y < HRRR_GRID.height)) return null;
  return { x, y };
}

/** Lon/lat of the center of HRRR pixel (col, row). */
export function hrrrPixelToLonLat(col: number, row: number): [number, number] {
  const x = HRRR_GRID.x0 + (col + 0.5) * HRRR_GRID.dx;
  const y = HRRR_GRID.y0 + (row + 0.5) * HRRR_GRID.dy;
  const [lon, lat] = lccInverse([x, y]);
  return [lon, lat];
}
