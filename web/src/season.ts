import { PolygonLayer, ScatterplotLayer } from "@deck.gl/layers";
import type { AefTile } from "./aef/layers.js";
import { RES, X0, Y0 } from "./aef/mosaic.js";

/**
 * A fire season over the fuel map: every California wildfire start of the year (NIFC WFIGS incident locations)
 * and every burn perimeter (WFIGS interagency perimeters), read from NIFC's services when the view opens.
 * The fuel is the year before (the embeddings of the season's own year can already see its burn scars).
 *
 * Each start gets a circle of the reach (30 m by default). A start "finds fuel" when the most burnable ground
 * inside its circle is at or above the hide line. Where it does, and inside a perimeter, the tile's season mask
 * is set and the ground there draws at full strength; elsewhere the fuel stays dim.
 */
export const SEASON_YEAR = 2025;
const SERVICES = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services";
const STARTS_URL = `${SERVICES}/WFIGS_Incident_Locations/FeatureServer/0/query`;
const PERIMS_URL = `${SERVICES}/WFIGS_Interagency_Perimeters/FeatureServer/0/query`;
const PAGE = 2000;
const M_PER_DEG_LAT = 111_320;
/** The step the ground inside a circle is sampled at: the fuel plane's pixel. */
const STEP_M = 10;

export type SeasonStarts = {
  n: number;
  lon: Float64Array;
  lat: Float64Array;
  /** Day of the season, 0 = Jan 1, in California's own date. Sorted. */
  day: Int16Array;
  acres: Float32Array;
  /** 0 unknown, 1 natural, 2 human. */
  cause: Uint8Array;
  name: string[];
};

export type SeasonPerimeter = {
  name: string;
  day: number;
  acres: number;
  /** One polygon (outer ring first, then holes), lon/lat. */
  polygon: [number, number][][];
  bbox: [number, number, number, number];
};

const caDate = new Intl.DateTimeFormat("en-CA", { timeZone: "America/Los_Angeles", year: "numeric", month: "2-digit", day: "2-digit" });

/** Day of the season (0 = Jan 1 of the year) of a UTC time, in California's date. */
export function dayOf(ms: number, year = SEASON_YEAR): number {
  const [y, m, d] = caDate.format(new Date(ms)).split("-").map(Number) as [number, number, number];
  return Math.round((Date.UTC(y, m - 1, d) - Date.UTC(year, 0, 1)) / 864e5);
}

export function dateOfDay(day: number, year = SEASON_YEAR): Date {
  return new Date(Date.UTC(year, 0, 1 + day));
}

export const daysIn = (year = SEASON_YEAR) => Math.round((Date.UTC(year + 1, 0, 1) - Date.UTC(year, 0, 1)) / 864e5);

async function getJson(url: string, params: Record<string, string>, signal?: AbortSignal) {
  const r = await fetch(`${url}?${new URLSearchParams(params)}`, { signal });
  if (!r.ok) throw new Error(`${url.split("/services/")[1]}: ${r.status}`);
  const j = await r.json();
  if (j.error) throw new Error(`${url.split("/services/")[1]}: ${j.error.message ?? "error"}`);
  return j;
}

const yearWhere = (field: string, year: number) =>
  `${field} >= timestamp '${year}-01-01 08:00:00' AND ${field} < timestamp '${year + 1}-01-01 08:00:00'`;

/** Every California wildfire start of the year, the pages fetched side by side. */
export async function loadSeasonStarts(year = SEASON_YEAR, signal?: AbortSignal): Promise<SeasonStarts> {
  const where = `POOState = 'US-CA' AND IncidentTypeCategory = 'WF' AND ${yearWhere("FireDiscoveryDateTime", year)}`;
  const { count } = await getJson(STARTS_URL, { where, returnCountOnly: "true", f: "json" }, signal);
  const pages = await Promise.all(
    Array.from({ length: Math.ceil(count / PAGE) }, (_, k) =>
      getJson(STARTS_URL, {
        where,
        outFields: "IncidentName,FireDiscoveryDateTime,FireCause,InitialLatitude,InitialLongitude,IncidentSize,DiscoveryAcres,FinalAcres",
        // the record's own point (kept up to date) rather than the initial reported one, which can sit km away
        returnGeometry: "true",
        outSR: "4326",
        orderByFields: "FireDiscoveryDateTime, OBJECTID",
        resultOffset: String(k * PAGE),
        resultRecordCount: String(PAGE),
        f: "json",
      }, signal),
    ),
  );
  type A = { IncidentName: string; FireDiscoveryDateTime: number; FireCause: string | null; InitialLatitude: number; InitialLongitude: number;
    IncidentSize: number | null; DiscoveryAcres: number | null; FinalAcres: number | null };
  const rows = pages.flatMap((p) => p.features as { attributes: A; geometry?: { x: number; y: number } | null }[])
    .map((f) => {
      const a = f.attributes;
      const g = f.geometry && Number.isFinite(f.geometry.x) && Number.isFinite(f.geometry.y) ? f.geometry : null;
      return { a, lon: g ? g.x : a.InitialLongitude, lat: g ? g.y : a.InitialLatitude, day: dayOf(a.FireDiscoveryDateTime, year) };
    })
    .filter(({ lon, lat, day }) => day >= 0 && day < daysIn(year) && Number.isFinite(lat) && Number.isFinite(lon))
    .sort((p, q) => p.day - q.day);
  const n = rows.length;
  const out: SeasonStarts = { n, lon: new Float64Array(n), lat: new Float64Array(n), day: new Int16Array(n), acres: new Float32Array(n), cause: new Uint8Array(n), name: [] };
  rows.forEach(({ a, lon, lat, day }, i) => {
    out.lon[i] = lon;
    out.lat[i] = lat;
    out.day[i] = day;
    out.acres[i] = Math.max(a.FinalAcres ?? 0, a.IncidentSize ?? 0, a.DiscoveryAcres ?? 0);
    out.cause[i] = a.FireCause === "Natural" ? 1 : a.FireCause === "Human" ? 2 : 0;
    out.name.push(a.IncidentName ?? "");
  });
  return out;
}

/** Every California wildfire perimeter of the year, one entry per polygon. */
export async function loadSeasonPerimeters(year = SEASON_YEAR, signal?: AbortSignal): Promise<SeasonPerimeter[]> {
  const where = `attr_POOState = 'US-CA' AND attr_IncidentTypeCategory = 'WF' AND ${yearWhere("attr_FireDiscoveryDateTime", year)}`;
  const out: SeasonPerimeter[] = [];
  for (let off = 0; ; off += PAGE) {
    const j = await getJson(PERIMS_URL, {
      where,
      outFields: "poly_IncidentName,attr_FireDiscoveryDateTime,attr_IncidentSize,poly_GISAcres",
      outSR: "4326",
      maxAllowableOffset: "0.00005",
      geometryPrecision: "6",
      resultOffset: String(off),
      resultRecordCount: String(PAGE),
      f: "geojson",
    }, signal);
    type F = { geometry: { type: string; coordinates: unknown } | null; properties: Record<string, unknown> };
    for (const f of j.features as F[]) {
      if (!f.geometry) continue;
      const p = f.properties;
      const day = dayOf(p.attr_FireDiscoveryDateTime as number, year);
      const polys = (f.geometry.type === "Polygon" ? [f.geometry.coordinates] : f.geometry.coordinates) as [number, number][][][];
      for (const polygon of polys) {
        let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
        for (const [x, y] of polygon[0]!) {
          if (x < w) w = x;
          if (x > e) e = x;
          if (y < s) s = y;
          if (y > n) n = y;
        }
        out.push({ name: String(p.poly_IncidentName ?? ""), day, acres: Number(p.attr_IncidentSize ?? p.poly_GISAcres ?? 0), polygon, bbox: [w, s, e, n] });
      }
    }
    if ((j.features as unknown[]).length < PAGE) break;
  }
  return out.sort((a, b) => a.day - b.day);
}

/**
 * The most burnable ground (P(fuel)) within reachM of a point, sampled every 10 m from the 10 m tiles on hand.
 * NaN when no tile covers the point itself yet.
 */
export function fuelNear(lon: number, lat: number, reachM: number, tiles: AefTile[]): number {
  let last: AefTile | null = null;
  const fuelAt = (x: number, y: number): number => {
    if (last) {
      const i = last.pixelAt(x, y);
      if (i >= 0) return last.classes.fuel[i]!;
    }
    for (const t of tiles) {
      const [w, s, e, n] = t.bbox;
      if (x < w || x > e || y < s || y > n) continue;
      const i = t.pixelAt(x, y);
      if (i < 0) continue;
      last = t;
      return t.classes.fuel[i]!;
    }
    return -1;
  };
  if (fuelAt(lon, lat) < 0) return NaN;
  const mLon = M_PER_DEG_LAT * Math.cos((lat * Math.PI) / 180);
  const k = Math.ceil(reachM / STEP_M);
  let best = -1;
  for (let j = -k; j <= k; j++) {
    for (let i = -k; i <= k; i++) {
      if (Math.hypot(i, j) * STEP_M > reachM + 0.01) continue;
      const b = fuelAt(lon + (i * STEP_M) / mLon, lat + (j * STEP_M) / M_PER_DEG_LAT);
      if (b > best) best = b;
    }
  }
  return best > 0 ? (best - 1) / 254 : 0;
}

let canvas: OffscreenCanvas | null = null;

/**
 * A tile's season mask: 255 inside each circle (reachM around a start that found fuel) and inside each perimeter,
 * 0 elsewhere. Null when nothing touches the tile.
 */
export function seasonMask(t: AefTile, circles: [number, number][], reachM: number, perims: SeasonPerimeter[]): Uint8Array | null {
  if (!circles.length && !perims.length) return null;
  const px0 = Math.round((t.bbox[0] - X0) / RES);
  const py0 = Math.round((Y0 - t.bbox[3]) / RES);
  const X = (lon: number) => (lon - X0) / RES - px0;
  const Y = (lat: number) => (Y0 - lat) / RES - py0;
  if (!canvas || canvas.width !== t.width || canvas.height !== t.height) canvas = new OffscreenCanvas(t.width, t.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.clearRect(0, 0, t.width, t.height);
  ctx.fillStyle = "#fff";
  for (const p of perims) {
    ctx.beginPath();
    for (const ring of p.polygon) {
      ring.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(X(x), Y(y)) : ctx.lineTo(X(x), Y(y))));
      ctx.closePath();
    }
    ctx.fill("evenodd");
  }
  const ry = reachM / (RES * M_PER_DEG_LAT);
  for (const [lon, lat] of circles) {
    const rx = ry / Math.cos((lat * Math.PI) / 180);
    ctx.beginPath();
    ctx.ellipse(X(lon), Y(lat), Math.max(rx, 0.5), Math.max(ry, 0.5), 0, 0, 2 * Math.PI);
    ctx.fill();
  }
  const img = ctx.getImageData(0, 0, t.width, t.height).data;
  const out = new Uint8Array(t.width * t.height);
  for (let i = 0; i < out.length; i++) out[i] = img[4 * i + 3]!;
  return out;
}

/** Starts that found fuel: a light blue, apart from every ground ramp's colors; the rest gray. */
const FOUND: [number, number, number] = [110, 205, 255];
const NONE: [number, number, number] = [160, 160, 170];
const UNREAD: [number, number, number] = [120, 120, 130];
const PERIM: [number, number, number] = [236, 236, 236];

export type SeasonLayerOpts = {
  starts: SeasonStarts;
  /** Start indices on the map (discovered by the day), and each one's fuel within reach (NaN = not read). */
  shown: Int32Array;
  fuel: Float32Array;
  perims: SeasonPerimeter[];
  day: number;
  reachM: number;
  pHide: number;
  beforeId?: string;
  onHover: (i: number, x: number, y: number) => void;
};

export function seasonLayers(o: SeasonLayerOpts) {
  const { starts: st, day, pHide } = o;
  const color = (k: number): [number, number, number] => {
    const f = o.fuel[k]!;
    return Number.isNaN(f) ? UNREAD : f >= pHide ? FOUND : NONE;
  };
  const trig = [o.shown, o.fuel, pHide, day];
  const shown = Array.from(o.shown, (i, k) => ({ i, k }));
  type D = { i: number; k: number };
  // the starts of the last two days flare: a ring in screen pixels that shrinks as the start ages
  const fresh = shown.filter((d) => day - st.day[d.i]! <= 2);
  return [
    new PolygonLayer<SeasonPerimeter>({
      id: "season-perimeters",
      data: o.perims,
      getPolygon: (d: SeasonPerimeter) => d.polygon,
      filled: false,
      stroked: true,
      getLineColor: [...PERIM, 210],
      lineWidthUnits: "pixels",
      getLineWidth: 1.5,
      beforeId: o.beforeId,
    } as never),
    new ScatterplotLayer<D>({
      id: "season-starts",
      data: shown,
      getPosition: (d: D) => [st.lon[d.i]!, st.lat[d.i]!],
      radiusUnits: "meters",
      getRadius: o.reachM,
      radiusMinPixels: 3,
      stroked: true,
      filled: true,
      lineWidthUnits: "pixels",
      getLineWidth: 1.5,
      getLineColor: (d: D) => [...color(d.k), 235],
      getFillColor: (d: D) => [...color(d.k), Number.isNaN(o.fuel[d.k]!) || o.fuel[d.k]! < pHide ? 0 : 60],
      pickable: true,
      onHover: (info: { index: number; x: number; y: number; object?: D }) => o.onHover(info.object ? info.object.i : -1, info.x, info.y),
      updateTriggers: { getLineColor: trig, getFillColor: trig, getRadius: [o.reachM] },
      beforeId: o.beforeId,
    } as never),
    new ScatterplotLayer<D>({
      id: "season-fresh",
      data: fresh,
      getPosition: (d: D) => [st.lon[d.i]!, st.lat[d.i]!],
      radiusUnits: "pixels",
      getRadius: (d: D) => 6 + 10 * Math.exp(-(day - st.day[d.i]!)),
      stroked: true,
      filled: false,
      lineWidthUnits: "pixels",
      getLineWidth: 1.5,
      getLineColor: (d: D) => [...color(d.k), 200],
      updateTriggers: { getRadius: [day], getLineColor: trig },
      beforeId: o.beforeId,
    } as never),
  ];
}
