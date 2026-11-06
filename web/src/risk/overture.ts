import type { Layer } from "@deck.gl/core";
import { TileLayer } from "@deck.gl/geo-layers";
import { GeoJsonLayer } from "@deck.gl/layers";
import { parse } from "@loaders.gl/core";
import { MVTLoader } from "@loaders.gl/mvt";
import type { StyleSpecification } from "maplibre-gl";
import { PMTiles } from "pmtiles";
import { FILL_ALPHA, MIN_LINE_PX } from "./shade.js";

/**
 * Overture's map over the risk picture, CARTO's under it (riskMapStyle): roads and rail from the
 * transportation PMTiles, water, land use (parks among it) and infrastructure (power lines among it) from the base
 * PMTiles, buildings from the buildings PMTiles, read by range from Overture's bucket and parsed here so deck.gl
 * draws them, each colored by what it is (KIND_COLORS).
 */

const RELEASE = "2026-10-21.0";

const OVERTURE = `https://overturemaps-extras-us-west-2.s3.us-west-2.amazonaws.com/tiles/${RELEASE}`;
const MAP_STYLE_URL = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
/** Read from this zoom in, the zoom the risk is read from. */
export const OVERTURE_MIN_ZOOM = 12;
/** The deepest zoom of each archive (from their headers, release 2026-10-21.0). */
const BASE_DEEPEST = 13;
const TRANSPORTATION_DEEPEST = 14;
const BUILDINGS_DEEPEST = 14;
/** Buildings drawn from this zoom in (the rest from OVERTURE_MIN_ZOOM). */
export const BUILDINGS_MIN_ZOOM = 14;

/**
 * CARTO's dark matter in two bands, with deck.gl's layers between them: everything but its labels under the risk
 * picture, at every zoom, so it shows only where the picture is not drawn (not read yet, or no embedding there) and
 * never over the data; its labels on top of everything. Water over the picture is Overture's (overtureLayers).
 */
export async function riskMapStyle(): Promise<{ style: StyleSpecification | string; groundBefore?: string; linesBefore?: string }> {
  try {
    const style = (await (await fetch(MAP_STYLE_URL)).json()) as StyleSpecification;
    const labels = style.layers.filter((l) => l.type === "symbol");
    style.layers = [...style.layers.filter((l) => l.type !== "symbol"), ...labels];
    return { style, groundBefore: labels[0]?.id, linesBefore: labels[0]?.id };
  } catch {
    return { style: MAP_STYLE_URL };
  }
}

const archives = new Map<string, PMTiles>();
const archive = (name: string) => {
  let a = archives.get(name);
  if (!a) archives.set(name, (a = new PMTiles(`${OVERTURE}/${name}.pmtiles`)));
  return a;
};

type Feature = GeoJSON.Feature<GeoJSON.Geometry, Record<string, unknown>>;

async function tileFeatures(name: string, layers: string[], x: number, y: number, z: number, signal?: AbortSignal): Promise<Feature[]> {
  const r = await archive(name).getZxy(z, x, y, signal);
  if (!r?.data) return [];
  const out = (await parse(r.data, MVTLoader, {
    mvt: { shape: "geojson", coordinates: "wgs84", tileIndex: { x, y, z }, layers },
  } as never)) as unknown as Feature[];
  // a polygon twice: as itself for the faint fill, and as its outline without the tile's cut
  return out.flatMap((f) => {
    if (!f.geometry.type.endsWith("Polygon")) return [f];
    const line = outlineOf(f, x, y, z);
    return line ? [f, line] : [f];
  });
}

/**
 * A polygon from a tile as its outline only, without the cut the tile makes: a big area (a land use zone) is cut at
 * the tile's edge (and its buffer past it), and stroking that cut draws the tile grid across the map. Segments that
 * run along an edge, or lie past one, are dropped; the rest stay as lines.
 */
function outlineOf(f: Feature, x: number, y: number, z: number): Feature | null {
  const n = 2 ** z;
  // position inside the tile, 0..1 (past the edge in the buffer: below 0 or above 1)
  const tx = (lon: number) => ((lon + 180) / 360) * n - x;
  const ty = (lat: number) => ((1 - Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360)) / Math.PI) / 2) * n - y;
  const EPS = 2 / 4096;
  const onEdge = (a: GeoJSON.Position, b: GeoJSON.Position) => {
    const ax = tx(a[0]!), bx = tx(b[0]!), ay = ty(a[1]!), by = ty(b[1]!);
    return (ax <= EPS && bx <= EPS) || (ax >= 1 - EPS && bx >= 1 - EPS) || (ay <= EPS && by <= EPS) || (ay >= 1 - EPS && by >= 1 - EPS);
  };
  const g = f.geometry;
  const rings = g.type === "Polygon" ? g.coordinates : g.type === "MultiPolygon" ? g.coordinates.flat() : [];
  const lines: GeoJSON.Position[][] = [];
  for (const ring of rings) {
    let run: GeoJSON.Position[] = [];
    for (let i = 0; i + 1 < ring.length; i++) {
      if (onEdge(ring[i]!, ring[i + 1]!)) {
        if (run.length > 1) lines.push(run);
        run = [];
        continue;
      }
      if (!run.length) run.push(ring[i]!);
      run.push(ring[i + 1]!);
    }
    if (run.length > 1) lines.push(run);
  }
  if (!lines.length) return null;
  return { ...f, properties: { ...f.properties, __outline: true }, geometry: { type: "MultiLineString", coordinates: lines } };
}

/** Road widths by Overture class, in pixels. */
const ROAD_W: Record<string, number> = {
  motorway: 3.5, trunk: 3.2, primary: 3, secondary: 2.6, tertiary: 2.3,
  residential: 1.8, unclassified: 1.8, living_street: 1.6, service: 1.4, track: 1.2,
};

type RGB = [number, number, number];
/**
 * Features colored by what they are (the risk under a feature is on its hover and its card, not its color). None of
 * these is in viridis, so they stand off the ground, and none is told apart by red against green.
 */
export const KIND_COLORS: { name: string; rgb: RGB }[] = [
  // asphalt: a cool gray, darker and bluer than the buildings' warm light gray
  { name: "Roads and rail", rgb: [134, 139, 147] },
  { name: "Buildings", rgb: [200, 191, 174] },
  { name: "Power lines and infrastructure", rgb: [255, 154, 31] },
  // outline only, no fill (big areas); brown, apart from the cyan burned areas and from viridis
  { name: "Parks and protected land", rgb: [185, 135, 85] },
  // lakes filled, streams as lines; a saturated blue, apart from viridis' grayish blues
  { name: "Water", rgb: [52, 128, 255] },
];
const [ROADS, BUILDINGS, INFRA, PARKS, WATER] = KIND_COLORS.map((k) => k.rgb) as [RGB, RGB, RGB, RGB, RGB];
/** Land use other than parks, off (it cluttered the map); its color if it comes back. */
const OTHER_LAND_USE = false;
const LAND_USE: RGB = [138, 138, 146];
const PARK_SUBTYPES = new Set(["park", "protected", "recreation"]);

/** zoom: the map's; each layer shows from its own first zoom. */
type Common = { beforeId?: string; zoom: number };

export type OvertureFeature = Feature;

/**
 * How a feature is drawn: line width in pixels and color, or null to leave it out. fill: a polygon is filled at this
 * alpha (0 to 255); its outline is drawn too unless the style leaves out features marked __outline.
 */
type Style = (f: Feature) => { w: number; c: RGB; fill?: number } | null;
/** A fill alpha too faint to see, so an unfilled area still picks from anywhere inside it. */
const PICK_ONLY = 1;

/**
 * Always the archive's deepest tiles (`deepest`), from the zoom the risk is read from on: Overture's shallower tiles
 * leave out small features, so reading them would bring each layer in at its own zoom. The zoom offset reaches the
 * deepest zoom at OVERTURE_MIN_ZOOM and past it deck.gl holds the tiles at the deepest.
 * Polygons are outlines here, unless their style fills them (water); buildings are filled too (buildings.ts).
 */
function overtureLayer(id: string, name: string, layers: string[], deepest: number, o: Common, style: Style, showFrom = OVERTURE_MIN_ZOOM) {
  return new TileLayer<Feature[]>({
    id,
    minZoom: deepest,
    maxZoom: deepest,
    zoomOffset: deepest - OVERTURE_MIN_ZOOM,
    tileSize: 512,
    maxCacheSize: 200,
    visible: o.zoom >= showFrom,
    getTileData: ({ index, signal }: { index: { x: number; y: number; z: number }; signal?: AbortSignal }) => tileFeatures(name, layers, index.x, index.y, index.z, signal),
    renderSubLayers: (p: { id: string; data: Feature[] | null }) => {
      const data = (p.data ?? []).filter((f: Feature) => style(f) !== null);
      // polygons come twice (tileFeatures): as themselves, drawn only where filled (unstroked: their stroke would
      // draw the tile's cut), and as their outlines, drawn only where not filled
      const isPoly = (f: Feature) => f.geometry.type.endsWith("Polygon");
      const filled = data.filter((f) => isPoly(f) && style(f)!.fill !== undefined);
      const lines = data.filter((f) => !isPoly(f));
      return [
        new GeoJsonLayer({
          id: `${p.id}-fill`,
          data: filled as never,
          filled: true,
          stroked: false,
          getFillColor: (f: Feature) => [...style(f)!.c, style(f)!.fill!],
          pickable: true,
        } as never),
        new GeoJsonLayer({
          id: `${p.id}-line`,
          data: lines as never,
          filled: true,
          stroked: true,
          getLineColor: (f: Feature) => [...style(f)!.c, 255],
          getLineWidth: (f: Feature) => style(f)!.w,
          lineWidthUnits: "pixels",
          lineWidthMinPixels: MIN_LINE_PX,
          pointType: "circle",
          getPointRadius: 3,
          pointRadiusUnits: "pixels",
          pointRadiusMinPixels: 2.5,
          getFillColor: (f: Feature) => [...style(f)!.c, 255],
          pickable: true,
        } as never),
      ];
    },
    beforeId: o.beforeId,
  } as never);
}

/** Parks and protected land, water, buildings, roads and rail, then infrastructure (power lines among it) on top. */
export function overtureLayers(o: Common): Layer[] {
  return [
    // outlined; the fill is there only so the pointer anywhere inside picks the area (its hover and card)
    overtureLayer("ov-land-use", "base", ["land_use"], BASE_DEEPEST, o, (f) =>
      PARK_SUBTYPES.has(String(f.properties.subtype ?? "")) ? { w: 1.4, c: PARKS, fill: PICK_ONLY } : OTHER_LAND_USE ? { w: 1.2, c: LAND_USE, fill: PICK_ONLY } : null),
    // lakes and wide rivers filled (their outlines left out), streams and canals as lines
    overtureLayer("ov-water", "base", ["water"], BASE_DEEPEST, o, (f) =>
      f.geometry.type.endsWith("Polygon") ? { w: 0, c: WATER, fill: 255 } : f.properties.__outline ? null : { w: 1.6, c: WATER }),
    // filled faintly, outlined in the same color
    overtureLayer("ov-buildings", "buildings", ["building"], BUILDINGS_DEEPEST, o, () => ({ w: 1, c: BUILDINGS, fill: FILL_ALPHA }), BUILDINGS_MIN_ZOOM),
    overtureLayer("ov-roads", "transportation", ["segment"], TRANSPORTATION_DEEPEST, o, (f) => {
      const sub = String(f.properties.subtype ?? "");
      if (sub === "rail") return { w: 1.6, c: ROADS };
      if (sub !== "road") return null;
      return { w: ROAD_W[String(f.properties.class ?? "")] ?? 1.1, c: ROADS };
    }),
    // every infrastructure feature (power lines and poles, bridges, pipelines, towers, ...)
    overtureLayer("ov-infrastructure", "base", ["infrastructure"], BASE_DEEPEST, o, (f) => {
      const c = String(f.properties.class ?? "");
      if (c === "power_line") return { w: 2.6, c: INFRA };
      if (c === "minor_line") return { w: 2, c: INFRA };
      if (f.geometry.type.endsWith("LineString")) return { w: 2, c: INFRA };
      return { w: f.geometry.type.endsWith("Point") ? 1.5 : 1.4, c: INFRA };
    }),
  ] as unknown as Layer[];
}
