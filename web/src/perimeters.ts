import { GeoJsonLayer } from "@deck.gl/layers";

/**
 * The real fires as areas, the frame the ignition picture is judged in: a burn perimeter per fire
 * (NIFC WFIGS for the current year, MTBS for earlier ones; ignition/pull_nifc_perimeters.py and
 * pull_perimeters.py write perimeters-<year>.geojson). A perimeter is an end state, the ground the
 * fire took, so it is drawn as an outline only, and only while the fire burned: from its discovery
 * hour until it was called out (or contained, when no out time is recorded; NO_END_H after discovery
 * when neither is). The ground shows through it untouched: was the ground inside lit before the fire,
 * and was the ground outside not. No fill, nothing drawn before the fire's hour or after its end.
 */
export type Perimeter = {
  type: "Feature";
  properties: {
    name: string | null;
    acres: number | null;
    cause: string | null;
    discovered_ms: number | null;
    contained_ms?: number | null;
    out_ms?: number | null;
    ig_date?: string;
    /** When the fire stops being drawn, filled in on load. */
    end_ms?: number;
  };
  geometry: GeoJSON.Geometry;
};

export async function loadPerimeters(baseUrl: string, year: number): Promise<Perimeter[]> {
  const r = await fetch(`${baseUrl}perimeters-${year}.geojson`);
  if (!r.ok) throw new Error(`perimeters-${year}.geojson: ${r.status}`);
  const fc = (await r.json()) as { features: Perimeter[] };
  for (const f of fc.features) {
    const p = f.properties as Perimeter["properties"] & { discovered?: string | null };
    // the MTBS file carries an ISO discovery time where WFIGS joined, else the ignition date at 00Z
    if (p.discovered_ms == null) {
      const iso = p.discovered ?? (p.ig_date ? `${p.ig_date}T00:00:00Z` : null);
      p.discovered_ms = iso ? Date.parse(iso) : null;
    }
  }
  const out = fc.features.filter((f) => f.properties.discovered_ms != null);
  for (const f of out) {
    const p = f.properties;
    p.end_ms = p.out_ms ?? p.contained_ms ?? p.discovered_ms! + NO_END_H * 3600e3;
  }
  return out;
}

/** A fire with no out or containment time on record is drawn this long after discovery (1145 of 7548 in 2025). */
const NO_END_H = 72;

/** Perimeters of fires burning at some hour of the run (+0 to +48 h). */
export function perimetersInRun(all: Perimeter[], initTime: Date): Perimeter[] {
  const t0 = initTime.getTime();
  const t1 = t0 + 48 * 3600e3;
  return all.filter((f) => f.properties.discovered_ms! <= t1 && f.properties.end_ms! >= t0);
}

/** Warm white, one stroke, no casing: reads over viridis and over the dark map alike. */
const LINE: [number, number, number, number] = [255, 228, 190, 230];

export function perimeterLayer(o: { perimeters: Perimeter[]; initTime: Date; leadHours: number; visible: boolean; beforeId?: string }) {
  const now = o.initTime.getTime() + o.leadHours * 3600e3;
  return new GeoJsonLayer<Perimeter["properties"]>({
    id: "fire-perimeters",
    data: o.perimeters.filter((f) => f.properties.discovered_ms! <= now && f.properties.end_ms! >= now) as never,
    filled: false,
    stroked: true,
    getLineColor: LINE,
    getLineWidth: 1.5,
    lineWidthUnits: "pixels",
    lineWidthMinPixels: 1,
    pickable: true,
    visible: o.visible,
    beforeId: o.beforeId,
    updateTriggers: { data: [now] },
  } as never);
}

// ---- California's burned areas, 2020 on, read live from NIFC's WFIGS perimeter service

const WFIGS = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters/FeatureServer/0/query";
const WFIGS_WHERE = "attr_FireDiscoveryDateTime >= timestamp '2020-01-01 00:00:00' AND attr_IncidentTypeCategory = 'WF' " +
  "AND attr_POOState = 'US-CA' AND attr_IncidentSize >= 100";

export type BurnArea = {
  type: "Feature";
  properties: { poly_IncidentName: string | null; attr_IncidentSize: number | null; attr_FireDiscoveryDateTime: number | null; attr_IrwinID: string | null };
  geometry: GeoJSON.Geometry;
};

/**
 * Every California wildfire of 100 acres and up discovered from 2020 on (about 590), outlines simplified on the
 * server to about 30 m: about 4 MB, a few seconds. The service can hold a fire twice; the largest is kept.
 */
export async function loadCaBurnAreas(signal?: AbortSignal): Promise<BurnArea[]> {
  const q = new URLSearchParams({
    where: WFIGS_WHERE,
    outFields: "poly_IncidentName,attr_IncidentSize,attr_FireDiscoveryDateTime,attr_IrwinID",
    outSR: "4326",
    maxAllowableOffset: "0.0003",
    geometryPrecision: "5",
    resultRecordCount: "2000",
    f: "geojson",
  });
  const r = await fetch(`${WFIGS}?${q}`, { signal });
  if (!r.ok) throw new Error(`WFIGS perimeters: ${r.status}`);
  const fc = (await r.json()) as { features: BurnArea[] };
  const byFire = new Map<string, BurnArea>();
  const out: BurnArea[] = [];
  for (const f of fc.features) {
    if (!f.geometry) continue;
    const id = f.properties.attr_IrwinID;
    if (!id) { out.push(f); continue; }
    const had = byFire.get(id);
    if (!had || (f.properties.attr_IncidentSize ?? 0) > (had.properties.attr_IncidentSize ?? 0)) byFire.set(id, f);
  }
  return out.concat([...byFire.values()]);
}

/** Bright cyan: no fuel ramp (viridis, inferno, magma, solar, fire) holds it, and it reads without red vision. */
const BURN_LINE: [number, number, number, number] = [0, 229, 255, 255];

/** The burned areas as outlines only: one stroke, no fill, no casing. */
export function burnAreaLayer(o: { areas: BurnArea[]; visible: boolean; beforeId?: string }) {
  return new GeoJsonLayer({
    id: "ca-burn-areas",
    data: o.areas as never,
    filled: false,
    stroked: true,
    getLineColor: BURN_LINE,
    getLineWidth: 1.5,
    lineWidthUnits: "pixels",
    lineWidthMinPixels: 1,
    visible: o.visible,
    beforeId: o.beforeId,
  } as never);
}

const boxes = new WeakMap<BurnArea, [number, number, number, number]>();
function boxOf(f: BurnArea): [number, number, number, number] {
  let b = boxes.get(f);
  if (!b) {
    b = [Infinity, Infinity, -Infinity, -Infinity];
    for (const ring of rings(f.geometry))
      for (const [x, y] of ring) { if (x! < b[0]) b[0] = x!; if (y! < b[1]) b[1] = y!; if (x! > b[2]) b[2] = x!; if (y! > b[3]) b[3] = y!; }
    boxes.set(f, b);
  }
  return b;
}
function rings(g: GeoJSON.Geometry): GeoJSON.Position[][] {
  if (g.type === "Polygon") return g.coordinates;
  if (g.type === "MultiPolygon") return g.coordinates.flat();
  return [];
}
/** Even-odd over one polygon's rings (outer ring and its holes). */
function inPolygon(poly: GeoJSON.Position[][], lon: number, lat: number): boolean {
  let hit = false;
  for (const r of poly)
    for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
      const [xi, yi] = r[i]!, [xj, yj] = r[j]!;
      if ((yi! > lat) !== (yj! > lat) && lon < ((xj! - xi!) * (lat - yi!)) / (yj! - yi!) + xi!) hit = !hit;
    }
  return hit;
}
/**
 * Inside any one of the fire's polygons. Each polygon is tested on its own: NIFC's multipolygons can overlap
 * (the Park fire's do near Cohasset), and one even-odd pass over all of them would call the overlap outside.
 */
function inside(f: BurnArea, lon: number, lat: number): boolean {
  const g = f.geometry;
  if (g.type === "Polygon") return inPolygon(g.coordinates, lon, lat);
  if (g.type === "MultiPolygon") return g.coordinates.some((p) => inPolygon(p, lon, lat));
  return false;
}

export type BurnHit = { name: string; year: number | null; acres: number | null; f: BurnArea };
/** The fires that burned this point, newest first. */
export function burnAreasAt(areas: BurnArea[], lon: number, lat: number): BurnHit[] {
  const out: BurnHit[] = [];
  for (const f of areas) {
    const b = boxOf(f);
    if (lon < b[0] || lon > b[2] || lat < b[1] || lat > b[3] || !inside(f, lon, lat)) continue;
    const p = f.properties;
    out.push({
      name: p.poly_IncidentName ?? "unnamed",
      year: p.attr_FireDiscoveryDateTime == null ? null : new Date(p.attr_FireDiscoveryDateTime).getUTCFullYear(),
      acres: p.attr_IncidentSize,
      f,
    });
  }
  return out.sort((a, b) => (b.year ?? 0) - (a.year ?? 0));
}

export const BURN_MASK_ID = "ca-burn-mask";

/** The burned areas filled, as a mask (never drawn): the fuel shown only inside them. */
export function burnMaskLayer(o: { areas: BurnArea[]; beforeId?: string }) {
  return new GeoJsonLayer({
    id: BURN_MASK_ID,
    data: o.areas as never,
    filled: true,
    stroked: false,
    getFillColor: [255, 255, 255, 255],
    operation: "mask",
    beforeId: o.beforeId,
  } as never);
}

/** The selected fire: the same cyan, a heavier stroke. */
export function selectedBurnLayer(o: { area: BurnArea; visible: boolean; beforeId?: string }) {
  return new GeoJsonLayer({
    id: "ca-burn-selected",
    data: [o.area] as never,
    filled: false,
    stroked: true,
    getLineColor: BURN_LINE,
    getLineWidth: 4,
    lineWidthUnits: "pixels",
    visible: o.visible,
    beforeId: o.beforeId,
  } as never);
}
