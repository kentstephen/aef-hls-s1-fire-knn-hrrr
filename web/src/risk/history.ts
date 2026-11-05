import { CHUNK, RES, X0, Y0 } from "../aef/mosaic.js";
import { chunkMask, type Poly, polysOf } from "../fuel/burnHist.js";
import type { BurnArea } from "../perimeters.js";

/**
 * Where and when the ground last burned, from two NIFC perimeter services read live:
 *   before 2020  InterAgencyFirePerimeterHistory (all years; the record starts about 1900), read for the area in
 *                view in 0.25 degree cells as the map moves
 *   2020 on      WFIGS (the burned areas the map already loads for California, perimeters.ts)
 * Both at 100 acres and up. Each chunk gets, per 10 m pixel, the last year a fire covered it and how many fires did.
 */

const HISTORY = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/InterAgencyFirePerimeterHistory_All_Years_View/FeatureServer/0/query";
const CELL = 0.25;
/** The history service's years; WFIGS takes over from here. */
const WFIGS_FROM = 2020;

export type Fire = { name: string; year: number; acres: number | null; polys: Poly[] };
export type ChunkHistory = {
  /** The last year a fire covered the pixel (0 = none on record). */
  last: Uint16Array;
  /** Years on record a fire covered it (one per year: the history holds some fires twice, from CAL FIRE and USFS). */
  count: Uint8Array;
};

type HistFeature = { properties: { OBJECTID: number; INCIDENT: string | null; FIRE_YEAR_INT: number | null; GIS_ACRES: number | null; UNQE_FIRE_ID: string | null }; geometry: GeoJSON.Geometry | null };

async function readCell(i: number, j: number, signal?: AbortSignal): Promise<HistFeature[]> {
  const out: HistFeature[] = [];
  const xmin = -180 + i * CELL, ymin = -90 + j * CELL;
  for (let offset = 0; ; ) {
    const q = new URLSearchParams({
      where: `GIS_ACRES >= 100 AND FIRE_YEAR_INT < ${WFIGS_FROM}`,
      geometry: JSON.stringify({ xmin, ymin, xmax: xmin + CELL, ymax: ymin + CELL, spatialReference: { wkid: 4326 } }),
      geometryType: "esriGeometryEnvelope",
      inSR: "4326",
      spatialRel: "esriSpatialRelIntersects",
      outFields: "OBJECTID,INCIDENT,FIRE_YEAR_INT,GIS_ACRES,UNQE_FIRE_ID",
      outSR: "4326",
      maxAllowableOffset: "0.0003",
      geometryPrecision: "5",
      resultOffset: String(offset),
      resultRecordCount: "1000",
      f: "geojson",
    });
    const r = await fetch(`${HISTORY}?${q}`, { signal });
    if (!r.ok) throw new Error(`fire history: ${r.status}`);
    const fc = (await r.json()) as { features: HistFeature[]; properties?: { exceededTransferLimit?: boolean }; exceededTransferLimit?: boolean };
    out.push(...fc.features);
    if (!(fc.properties?.exceededTransferLimit || fc.exceededTransferLimit) || !fc.features.length) break;
    offset += fc.features.length;
  }
  return out;
}

export class FireHistory {
  readonly fires: Fire[] = [];
  private seen = new Set<string>();
  private cells = new Map<string, Promise<void>>();

  constructor(recent: BurnArea[]) {
    for (const f of recent) {
      const ms = f.properties.attr_FireDiscoveryDateTime;
      if (ms == null) continue;
      this.fires.push({ name: f.properties.poly_IncidentName ?? "unnamed", year: new Date(ms).getUTCFullYear(), acres: f.properties.attr_IncidentSize, polys: polysOf(f) });
    }
  }

  /** Reads the history cells under this lon / lat box that are not read yet. */
  ensure(bbox: [number, number, number, number], signal?: AbortSignal): Promise<void> {
    const jobs: Promise<void>[] = [];
    for (let i = Math.floor((bbox[0] + 180) / CELL); i <= Math.floor((bbox[2] + 180) / CELL); i++)
      for (let j = Math.floor((bbox[1] + 90) / CELL); j <= Math.floor((bbox[3] + 90) / CELL); j++) {
        const k = `${i}:${j}`;
        let p = this.cells.get(k);
        if (!p) {
          p = readCell(i, j, signal).then((fs) => {
            for (const f of fs) {
              const pr = f.properties;
              // the same fire from two sources has no shared id: name, year and acres
              const id = pr.UNQE_FIRE_ID || `${(pr.INCIDENT ?? "").toUpperCase()}|${pr.FIRE_YEAR_INT}|${Math.round(pr.GIS_ACRES ?? 0)}`;
              if (!f.geometry || pr.FIRE_YEAR_INT == null || this.seen.has(id)) continue;
              this.seen.add(id);
              this.fires.push({ name: pr.INCIDENT ?? "unnamed", year: pr.FIRE_YEAR_INT, acres: pr.GIS_ACRES, polys: polysOf({ geometry: f.geometry }) });
            }
          });
          p.catch(() => this.cells.delete(k));
          this.cells.set(k, p);
        }
        jobs.push(p);
      }
    return Promise.all(jobs).then(() => undefined);
  }

  /** Last burn year and fire count per pixel of a chunk, from the fires read so far (call ensure first). */
  chunk(cy: number, cx: number): ChunkHistory {
    const last = new Uint16Array(CHUNK * CHUNK), count = new Uint8Array(CHUNK * CHUNK);
    // oldest first, so a pixel counts a year once however many records cover it that year
    for (const f of [...this.fires].sort((a, b) => a.year - b.year)) {
      const m = chunkMask(f.polys, cy, cx);
      if (!m) continue;
      for (let i = 0; i < m.length; i++) {
        if (!m[i] || f.year <= last[i]!) continue;
        last[i] = f.year;
        if (count[i]! < 255) count[i]!++;
      }
    }
    return { last, count };
  }

  /** The fires on record that covered this point, newest first. */
  at(lon: number, lat: number): Fire[] {
    const x = (lon - X0) / RES, y = (Y0 - lat) / RES;
    return this.fires
      .filter((f) => f.polys.some((p) => {
        if (x < p.x0 || x > p.x1 || y < p.y0 || y > p.y1) return false;
        let hit = false;
        for (const [ax, ay, bx, by] of p.edges) if ((ay > y) !== (by > y) && x < ax + ((y - ay) / (by - ay)) * (bx - ax)) hit = !hit;
        return hit;
      }))
      .sort((a, b) => b.year - a.year);
  }
}
