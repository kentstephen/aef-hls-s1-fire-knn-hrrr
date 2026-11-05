import type { Layer } from "@deck.gl/core";
import { TileLayer } from "@deck.gl/geo-layers";
import { GeoJsonLayer } from "@deck.gl/layers";
import { parse } from "@loaders.gl/core";
import { MVTLoader } from "@loaders.gl/mvt";
import type { StyleSpecification } from "maplibre-gl";
import { PMTiles } from "pmtiles";
import { RELEASE } from "../buildings.js";
import { MIN_LINE_PX } from "./shade.js";

/**
 * Overture's map over the risk picture, CARTO's under it (riskMapStyle): roads and rail from the
 * transportation PMTiles, land use (parks among it) and infrastructure (power lines among it) from the base
 * PMTiles, read by range from Overture's bucket and parsed here so deck.gl draws them, each colored by what it is
 * (KIND_COLORS). Buildings come from the GeoParquet instead (buildings.ts).
 */

const OVERTURE = `https://overturemaps-extras-us-west-2.s3.us-west-2.amazonaws.com/tiles/${RELEASE}`;
const MAP_STYLE_URL = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
/** Read from this zoom in, the zoom the risk is read from. */
export const OVERTURE_MIN_ZOOM = 12;

/**
 * CARTO's dark matter in three bands, with deck.gl's layers between them: its roads, buildings and land under the
 * risk picture, shown only zoomed out past where the picture and Overture are drawn (context for finding a town),
 * its water over the picture, and its labels on top of everything. Overture's features go between the water and
 * the labels.
 */
export async function riskMapStyle(): Promise<{ style: StyleSpecification | string; groundBefore?: string; linesBefore?: string }> {
  try {
    const style = (await (await fetch(MAP_STYLE_URL)).json()) as StyleSpecification;
    const isWater = (l: StyleSpecification["layers"][number]) => l.type !== "symbol" && "source-layer" in l && (l["source-layer"] === "water" || l["source-layer"] === "waterway");
    // CARTO's roads, buildings and land only below the zoom the risk and Overture are drawn from
    const under = style.layers
      .filter((l) => l.type !== "symbol" && !isWater(l))
      .map((l) => (l.type === "background" ? l : { ...l, maxzoom: Math.min(l.maxzoom ?? 24, OVERTURE_MIN_ZOOM) }));
    const water = style.layers.filter(isWater);
    const labels = style.layers.filter((l) => l.type === "symbol");
    style.layers = [...under, ...water, ...labels];
    return { style, groundBefore: water[0]?.id ?? labels[0]?.id, linesBefore: labels[0]?.id };
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
  return { ...f, geometry: { type: "MultiLineString", coordinates: lines } };
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
  { name: "Roads and rail", rgb: [240, 240, 240] },
  { name: "Buildings", rgb: [200, 191, 174] },
  { name: "Power lines and infrastructure", rgb: [255, 154, 31] },
  // outline only, no fill (big areas); brown, apart from the cyan burned areas and from viridis
  { name: "Parks and protected land", rgb: [185, 135, 85] },
];
const [ROADS, BUILDINGS, INFRA, PARKS] = KIND_COLORS.map((k) => k.rgb) as [RGB, RGB, RGB, RGB];
/** Land use other than parks, off (it cluttered the map); its color if it comes back. */
const OTHER_LAND_USE = false;
const LAND_USE: RGB = [138, 138, 146];
export const BUILDING_RGB = BUILDINGS;
const PARK_SUBTYPES = new Set(["park", "protected", "recreation"]);

type Common = { beforeId?: string; visible: boolean };

export type OvertureFeature = Feature;

/** How a feature is drawn: line width in pixels and color, or null to leave it out. */
type Style = (f: Feature) => { w: number; c: RGB } | null;

/**
 * zoomOffset: tiles this many zooms deeper than the map's (a layer Overture only tiles from a deeper zoom).
 * Polygons are outlines only here (only buildings are filled, buildings.ts).
 */
function overtureLayer(id: string, name: string, layers: string[], minZoom: number, maxZoom: number, o: Common, style: Style,
  { zoomOffset = 0 }: { zoomOffset?: number } = {}) {
  return new TileLayer<Feature[]>({
    id,
    minZoom,
    maxZoom,
    zoomOffset,
    tileSize: 512,
    maxCacheSize: 200,
    visible: o.visible,
    getTileData: ({ index, signal }: { index: { x: number; y: number; z: number }; signal?: AbortSignal }) => tileFeatures(name, layers, index.x, index.y, index.z, signal),
    renderSubLayers: (p: { id: string; data: Feature[] | null }) => {
      const data = (p.data ?? []).filter((f: Feature) => style(f) !== null);
      // polygons come as their outlines (tileFeatures); the polygons themselves are left out (unfilled, and their own
      // stroke would draw the tile's cut)
      const lines = data.filter((f) => !f.geometry.type.endsWith("Polygon"));
      return [
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

/** Parks and protected land, roads and rail, then infrastructure (power lines among it) on top. */
export function overtureLayers(o: Common): Layer[] {
  return [
    overtureLayer("ov-land-use", "base", ["land_use"], OVERTURE_MIN_ZOOM, 13, o, (f) =>
      PARK_SUBTYPES.has(String(f.properties.subtype ?? "")) ? { w: 1.4, c: PARKS } : OTHER_LAND_USE ? { w: 1.2, c: LAND_USE } : null),
    // Overture's tiles drop roads below zoom 14 to stay small, so the map reads the zoom 14 tiles from zoom 12 on
    overtureLayer("ov-roads", "transportation", ["segment"], OVERTURE_MIN_ZOOM, 14, o, (f) => {
      const sub = String(f.properties.subtype ?? "");
      if (sub === "rail") return { w: 1.6, c: ROADS };
      if (sub !== "road") return null;
      return { w: ROAD_W[String(f.properties.class ?? "")] ?? 1.1, c: ROADS };
    }, { zoomOffset: 2 }),
    // every infrastructure feature (power lines and poles, bridges, pipelines, towers, ...); Overture only tiles
    // this layer at zoom 13, so the map at zoom 12 (where the risk starts) reads the zoom 13 tiles
    overtureLayer("ov-infrastructure", "base", ["infrastructure"], 13, 13, o, (f) => {
      const c = String(f.properties.class ?? "");
      if (c === "power_line") return { w: 2.6, c: INFRA };
      if (c === "minor_line") return { w: 2, c: INFRA };
      if (f.geometry.type.endsWith("LineString")) return { w: 2, c: INFRA };
      return { w: f.geometry.type.endsWith("Point") ? 1.5 : 1.4, c: INFRA };
    }, { zoomOffset: 1 }),
  ] as unknown as Layer[];
}
