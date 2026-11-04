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
