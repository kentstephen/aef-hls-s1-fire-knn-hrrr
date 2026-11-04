import { COGLayer, type GetTileDataOptions, MosaicLayer } from "@developmentseed/deck.gl-geotiff";
import { CreateTexture } from "@developmentseed/deck.gl-raster/gpu-modules";
import { DecoderPool, GeoTIFF, type Overview } from "@developmentseed/geotiff";
import GeoTiffWorker from "@developmentseed/geotiff/pool/worker?worker";
import type { ShaderModule } from "@luma.gl/shadertools";

/**
 * Sentinel-2 true color under the fuel map: Earth Genome's yearly mosaics (and the fill mosaics under them for
 * cloud gaps), found with Earth Genome's STAC search from the page and read straight from each footprint's TCI
 * COG on Source Cooperative, decoded in workers. Forked from aef-earthwork-web (web/src/s2.ts and ui.js
 * s2CogLayers).
 */
const STAC = "https://stac.earthgenome.org/search";
const YEARLY = "sentinel2-yearly-mosaics";
const FILL = "sentinel2-temporal-mosaics";
/** The years the mosaics cover well in California. */
export const S2_YEARS = [2022, 2023, 2024, 2025];
/** Footprints are looked for from this zoom in. */
export const S2_MIN_ZOOM = 9;

type StacFeature = {
  id: string;
  collection: string;
  bbox: [number, number, number, number];
  assets: Record<string, { href: string }>;
};

/** One footprint, as the MosaicLayer takes it. */
export type S2Item = { id: string; url: string; bbox: [number, number, number, number]; fill: boolean };

const searches = new Map<string, Promise<StacFeature[]>>();
function search(bbox: number[]): Promise<StacFeature[]> {
  const k = bbox.map((v) => v.toFixed(3)).join(",");
  let p = searches.get(k);
  if (!p) {
    p = fetch(STAC, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ collections: [YEARLY, FILL], bbox, limit: 200 }),
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`STAC ${r.status}`))))
      .then((j) => j.features as StacFeature[]);
    p.catch(() => searches.delete(k));
    searches.set(k, p);
  }
  return p;
}

/** The year's footprints under one web mercator tile: yearly first, the fill after them. */
async function s2Items(year: number, z: number, x: number, y: number): Promise<S2Item[]> {
  const n = 2 ** z;
  const lon = (i: number) => (i / n) * 360 - 180;
  const lat = (j: number) => (Math.atan(Math.sinh(Math.PI * (1 - (2 * j) / n))) * 180) / Math.PI;
  const features = await search([lon(x), lat(y + 1), lon(x + 1), lat(y)]);
  const span = `${year}-01-01_${year + 1}-01-01`;
  const yearly: S2Item[] = [];
  const fill: S2Item[] = [];
  for (const f of features) {
    if (!f.id.endsWith(span) || !f.assets.TCI || !f.bbox) continue;
    if (f.collection === FILL) fill.push({ id: `${f.id}#fill`, url: f.assets.TCI.href, bbox: f.bbox, fill: true });
    else yearly.push({ id: f.id, url: f.assets.TCI.href, bbox: f.bbox, fill: false });
  }
  if (!yearly.length) {
    // the STAC lags the bucket (2025 is in the bucket for tiles the search does not know yet): the same MGRS
    // tile's path with the year swapped. A tile that is not there reads as empty, not as an error
    const seen = new Set<string>();
    for (const f of features) {
      if (f.collection !== YEARLY || !f.assets.TCI) continue;
      const tile = f.id.split("_")[0]!;
      if (seen.has(tile)) continue;
      seen.add(tile);
      const id = `${tile}_${span}`;
      const base = f.assets.TCI.href.split("/").slice(0, -2).join("/");
      yearly.push({ id, url: `${base}/${id}/TCI.tif`, bbox: f.bbox, fill: false });
    }
  }
  return [...yearly, ...fill];
}

/** One answer of the search (a web mercator tile's footprints): its own MosaicLayer, which looks at once. */
type Batch = { key: string; yearly: S2Item[]; fill: S2Item[]; bbox: [number, number, number, number] };

/**
 * TCI's black (0, 0, 0) is nodata, given alpha 0 when the tile is read. At a footprint edge the linear filter
 * blends the edge pixel with black nodata; dividing by the filtered alpha gives the pixel back with no dark
 * fringe, and alpha under a half ends the footprint midway through the texel.
 */
const S2Look = {
  name: "s2Look",
  fs: `uniform s2LookUniforms {
  float gamma;
} s2Look;
`,
  inject: {
    "fs:DECKGL_FILTER_COLOR": `
  if (color.a < 0.5) discard;
  color = vec4(pow(color.rgb / color.a, vec3(1.0 / s2Look.gamma)), 1.0);
`,
  },
  uniformTypes: { gamma: "f32" },
  getUniforms: (q: { gamma?: number }) => ({ gamma: q.gamma ?? 1 }),
} as const satisfies ShaderModule<{ gamma: number }>;

/**
 * The footprints found so far per year and the layers that draw them. One instance for the page; each map
 * (the main one and the pair's left) asks it for its own layers.
 */
export class S2Imagery {
  private tiffs = new Map<string, Promise<GeoTIFF>>();
  private years = new Map<number, { ids: Set<string>; batches: Batch[] }>();
  private asked = new Set<string>();
  private pool: DecoderPool | null = null;
  version = 0;
  constructor(private onChange: () => void) {}

  private tiff(url: string): Promise<GeoTIFF> {
    let p = this.tiffs.get(url);
    if (!p) {
      p = GeoTIFF.fromUrl(url);
      this.tiffs.set(url, p);
      // a failed read is asked again next time
      p.catch(() => { if (this.tiffs.get(url) === p) this.tiffs.delete(url); });
    }
    return p;
  }

  /** Look for the year's footprints under the view (and half of it again on every side). */
  discover(year: number, view: [number, number, number, number], zoom: number) {
    if (zoom < S2_MIN_ZOOM) return;
    const [w0, s0, e0, n0] = view;
    const dx = (e0 - w0) / 2, dy = (n0 - s0) / 2;
    const W = w0 - dx, E = e0 + dx, S = Math.max(-85, s0 - dy), N = Math.min(85, n0 + dy);
    const z = Math.max(7, Math.min(9, Math.floor(zoom)));
    const n = 2 ** z;
    const tx = (lon: number) => Math.min(n - 1, Math.max(0, Math.floor(((lon + 180) / 360) * n)));
    const ty = (lat: number) => {
      const v = Math.sin((lat * Math.PI) / 180);
      return Math.min(n - 1, Math.max(0, Math.floor((0.5 - Math.log((1 + v) / (1 - v)) / (4 * Math.PI)) * n)));
    };
    const x0 = tx(W), x1 = tx(E), y0 = ty(N), y1 = ty(S);
    if ((x1 - x0 + 1) * (y1 - y0 + 1) > 64) return;
    for (let x = x0; x <= x1; x++)
      for (let y = y0; y <= y1; y++) {
        const k = `${year}/${z}/${x}/${y}`;
        if (this.asked.has(k)) continue;
        this.asked.add(k);
        s2Items(year, z, x, y).then(
          (items) => {
            let e = this.years.get(year);
            if (!e) this.years.set(year, (e = { ids: new Set(), batches: [] }));
            const bt: Batch = { key: k.replace(/\//g, "-"), yearly: [], fill: [], bbox: [180, 90, -180, -90] };
            for (const it of items) {
              if (e.ids.has(it.id)) continue;
              e.ids.add(it.id);
              (it.fill ? bt.fill : bt.yearly).push(it);
              bt.bbox = [Math.min(bt.bbox[0], it.bbox[0]), Math.min(bt.bbox[1], it.bbox[1]), Math.max(bt.bbox[2], it.bbox[2]), Math.max(bt.bbox[3], it.bbox[3])];
            }
            if (bt.yearly.length || bt.fill.length) {
              e.batches.push(bt);
              this.version++;
              this.onChange();
            }
          },
          () => this.asked.delete(k),
        );
      }
  }

  /** The year's layers under the view: every batch's fill first, its yearly footprints over them. */
  layers(o: { year: number; view: [number, number, number, number]; visible: boolean; prefix: string; beforeId?: string }) {
    const e = this.years.get(o.year);
    if (!e) return [];
    this.pool ??= new DecoderPool({ size: Math.min(6, navigator.hardwareConcurrency || 4), createWorker: () => new GeoTiffWorker() });
    const pool = this.pool;
    const [w0, s0, e0, n0] = o.view;
    const dx = (e0 - w0) / 2, dy = (n0 - s0) / 2;
    const live = e.batches.filter((b) => b.bbox[0] < e0 + dx && b.bbox[2] > w0 - dx && b.bbox[1] < n0 + dy && b.bbox[3] > s0 - dy);
    const parts: [Batch, "fill" | "yearly"][] = [
      ...live.filter((b) => b.fill.length).map((b) => [b, "fill"] as [Batch, "fill"]),
      ...live.filter((b) => b.yearly.length).map((b) => [b, "yearly"] as [Batch, "yearly"]),
    ];
    return parts.map(
      ([b, part]) =>
        new MosaicLayer<S2Item, GeoTIFF | null>({
          id: `${o.prefix}s2-${b.key}-${part}`,
          sources: b[part],
          minZoom: S2_MIN_ZOOM,
          visible: o.visible,
          beforeId: o.beforeId,
          // a footprint that cannot be read (the STAC lists it but the bucket lacks it) is nothing there
          getSource: (src: S2Item) => this.tiff(src.url).catch(() => null),
          renderSource: (src: S2Item, { data, signal }: { data?: GeoTIFF | null; signal?: AbortSignal }) =>
            !data
              ? null
              : new COGLayer({
                  id: `${o.prefix}s2-${o.year}-${src.id}`,
                  geotiff: data,
                  pool,
                  signal,
                  refinementStrategy: "best-available",
                  maxRequests: 16,
                  getTileData: async (image: GeoTIFF | Overview, { device, x, y, signal: sig, pool: p }: GetTileDataOptions) => {
                    const { array } = await image.fetchTile(x, y, { boundless: false, pool: p, signal: sig });
                    const { width, height } = array;
                    const data = (array as unknown as { data: Uint8Array }).data;
                    const px = width * height;
                    let rgba = data;
                    if (data.length === 3 * px) {
                      rgba = new Uint8Array(4 * px);
                      for (let i = 0; i < px; i++) {
                        const r = data[3 * i]!, g = data[3 * i + 1]!, bl = data[3 * i + 2]!;
                        rgba[4 * i] = r;
                        rgba[4 * i + 1] = g;
                        rgba[4 * i + 2] = bl;
                        rgba[4 * i + 3] = r || g || bl ? 255 : 0;
                      }
                    }
                    return {
                      texture: device.createTexture({ data: rgba, format: "rgba8unorm", width, height, sampler: { magFilter: "linear", minFilter: "linear" } }),
                      width,
                      height,
                    };
                  },
                  renderTile: (d: { texture: unknown }) => ({
                    renderPipeline: [{ module: CreateTexture, props: { textureName: d.texture } }, { module: S2Look, props: { gamma: 1 } }],
                  }),
                } as never),
        } as never),
    );
  }
}
