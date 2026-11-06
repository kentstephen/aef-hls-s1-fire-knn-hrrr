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

/** One footprint, as the MosaicLayer takes it: its true color image and each raw band's COG. */
export type S2Item = { id: string; url: string; bands: Record<string, string>; bbox: [number, number, number, number]; fill: boolean };

/**
 * How the imagery is colored. True color is Earth Genome's own image; the rest stack three raw bands
 * into red, green and blue, each over a fixed stretch (BAND_RANGE).
 */
export type S2Comp = "tci" | "live" | "blueyellow" | "swir";
export const S2_COMPS: { id: S2Comp; label: string; bands: string; about: string }[] = [
  { id: "tci", label: "True color", bands: "TCI", about: "Earth Genome's true color image." },
  { id: "live", label: "Live, dry, burned", bands: "B8A B8A B12", about: "Near infrared in red and green, shortwave infrared 2 in blue: green plants yellow, dry grass blue-gray, burned and bare fields deep blue, water black." },
  { id: "blueyellow", label: "Blue-yellow", bands: "B11 B11 B2", about: "Shortwave infrared 1 in red and green, blue in blue: dry grass, sand and bare soil yellow, roofs and concrete blue, trees dark." },
  { id: "swir", label: "SWIR", bands: "B12 B8A B4", about: "The usual fire composite: green plants bright green, dry grass pink, burned and bare fields crimson, water black." },
];
const COMP_BANDS: Record<Exclude<S2Comp, "tci">, [string, string, string]> = {
  live: ["B8A", "B8A", "B12"],
  blueyellow: ["B11", "B11", "B02"],
  swir: ["B12", "B8A", "B04"],
};
/**
 * Each band's stretch, in the mosaics' raw units (reflectance x 10,000 plus the 1,000 offset): the 1st and 99th
 * percentiles over five 2024 California samples (Park Fire foothills, Sierra forest, Central Valley farms, Southern
 * California chaparral, Chico). Fixed, so a color means the same ground in every view and year.
 */
const BAND_RANGE: Record<string, [number, number]> = {
  B02: [1060, 2330],
  B04: [1150, 3300],
  B8A: [1700, 5000],
  B11: [1900, 4800],
  B12: [1450, 4000],
};
const BANDS_20M = new Set(["B11", "B12", "B8A"]);
/** The band whose tiles a composite walks: its first 20 m band (the others are read onto its grid). */
const primOf = (comp: Exclude<S2Comp, "tci">) => COMP_BANDS[comp].find((b) => BANDS_20M.has(b))!;
/** sqrt(reflectance) in 8 bits: fine steps in the dark ground, where most of it lies. */
const SQRT8 = new Uint8Array(65536);
for (let v = 0; v < 65536; v++) SQRT8[v] = Math.round(255 * Math.sqrt(Math.min(v, 10000) / 10000));

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
  const bandsOf = (f: StacFeature) => Object.fromEntries(Object.entries(f.assets).map(([k, a]) => [k, a.href]));
  for (const f of features) {
    if (!f.id.endsWith(span) || !f.assets.TCI || !f.bbox) continue;
    if (f.collection === FILL) fill.push({ id: `${f.id}#fill`, url: f.assets.TCI.href, bands: bandsOf(f), bbox: f.bbox, fill: true });
    else yearly.push({ id: f.id, url: f.assets.TCI.href, bands: bandsOf(f), bbox: f.bbox, fill: false });
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
      const bands = Object.fromEntries(Object.keys(f.assets).map((k) => [k, `${base}/${id}/${k}.tif`]));
      yearly.push({ id, url: `${base}/${id}/TCI.tif`, bands, bbox: f.bbox, fill: false });
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

/** As S2Look, then back from sqrt to raw units (over 10,000) and each band over its stretch, lo to hi. */
const S2Stretch = {
  name: "s2Stretch",
  fs: `uniform s2StretchUniforms {
  vec3 lo;
  vec3 hi;
} s2Stretch;
`,
  inject: {
    "fs:DECKGL_FILTER_COLOR": `
  if (color.a < 0.5) discard;
  vec3 v = color.rgb / color.a;
  color = vec4(clamp((v * v - s2Stretch.lo) / (s2Stretch.hi - s2Stretch.lo), 0.0, 1.0), 1.0);
`,
  },
  uniformTypes: { lo: "vec3<f32>", hi: "vec3<f32>" },
  getUniforms: (q: { lo?: [number, number, number]; hi?: [number, number, number] }) => ({ lo: q.lo ?? [0, 0, 0], hi: q.hi ?? [1, 1, 1] }),
} as const satisfies ShaderModule<{ lo: [number, number, number]; hi: [number, number, number] }>;

type Block = { d: Uint16Array; w: number; h: number };

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

  /** A block of a band's COG, read once for every tile it lies under (no tile's abort, so one let go does not fail the rest). */
  private blocks = new Map<string, Promise<Block | null>>();
  private block(url: string, im: GeoTIFF | Overview, bx: number, by: number, pool: DecoderPool): Promise<Block | null> {
    const k = `${url}|${im.width}|${bx}|${by}`;
    let p = this.blocks.get(k);
    if (!p) {
      p = im.fetchTile(bx, by, { boundless: false, pool }).then(
        ({ array }) => {
          const a = array as unknown as { data?: Uint16Array; bands?: Uint16Array[]; width: number; height: number };
          return { d: (a.data ?? a.bands![0])!, w: a.width, h: a.height };
        },
        () => { if (this.blocks.get(k) === p) this.blocks.delete(k); return null; },
      );
      this.blocks.set(k, p);
      if (this.blocks.size > 256) this.blocks.delete(this.blocks.keys().next().value!);
    }
    return p;
  }

  /**
   * One tile of a band stack: the tile of the composite's first 20 m band, each other band read at the level of
   * the same pixel size and placed by its own geotransform (the fill footprints put the 10 m bands on another
   * origin and other blocks). Packed as sqrt(reflectance) in 8 bits. Nodata where every band is 0; a pixel with
   * some bands at 0 (water's shortwave infrared) is ground where it lies among data, nodata where it touches an
   * empty pixel (the seams at a footprint's edge).
   */
  private async bandTile(image: GeoTIFF | Overview, o: GetTileDataOptions, src: S2Item, comp: Exclude<S2Comp, "tci">) {
    const { device, x, y, signal, pool } = o;
    const bands = COMP_BANDS[comp], prim = primOf(comp);
    const tp = image.transform as unknown as number[];
    const tw = image.tileWidth, th = image.tileHeight;
    const X0 = tp[2]! + x * tw * tp[0]!, Y0 = tp[5]! + y * th * tp[4]!;
    const own = async (im: GeoTIFF | Overview, bx: number, by: number): Promise<Block | null> => {
      try {
        const { array } = await im.fetchTile(bx, by, { boundless: false, pool, signal });
        const a = array as unknown as { data?: Uint16Array; bands?: Uint16Array[]; width: number; height: number };
        return { d: (a.data ?? a.bands![0])!, w: a.width, h: a.height };
      } catch (e) {
        if (signal?.aborted) throw e;
        return null;
      }
    };
    const one = async (b: string): Promise<Block | null> => {
      if (b === prim) return own(image, x, y);
      const url = src.bands[b];
      const g = url ? await this.tiff(url).catch(() => null) : null;
      const im = g && [g, ...g.overviews].find((q) => Math.abs((q.transform as unknown as number[])[0]! / tp[0]! - 1) < 0.01);
      if (!im || !url) return null;
      const q = im.transform as unknown as number[];
      const c0 = Math.round((X0 - q[2]!) / q[0]!), r0 = Math.round((Y0 - q[5]!) / q[4]!);
      const bw = im.tileWidth, bh = im.tileHeight;
      if (bw === tw && bh === th && c0 === x * tw && r0 === y * th) return own(im, x, y);
      const out = new Uint16Array(tw * th);
      const jobs: Promise<void>[] = [];
      for (let by = Math.max(0, Math.floor(r0 / bh)); by <= Math.min(Math.ceil(im.height / bh) - 1, Math.floor((r0 + th - 1) / bh)); by++)
        for (let bx = Math.max(0, Math.floor(c0 / bw)); bx <= Math.min(Math.ceil(im.width / bw) - 1, Math.floor((c0 + tw - 1) / bw)); bx++)
          jobs.push(
            this.block(url, im, bx, by, pool as DecoderPool).then((t) => {
              if (!t) return;
              const ox = bx * bw - c0, oy = by * bh - r0;
              for (let r = Math.max(0, -oy); r < t.h && r + oy < th; r++) {
                const c = Math.max(0, -ox), n = Math.min(t.w, tw - ox) - c;
                if (n > 0) out.set(t.d.subarray(r * t.w + c, r * t.w + c + n), (r + oy) * tw + c + ox);
              }
            }),
          );
      await Promise.all(jobs);
      return { d: out, w: tw, h: th };
    };
    const uniq = [...new Set(bands)];
    const got = new Map(await Promise.all(uniq.map(async (b) => [b, await one(b)] as const)));
    const p = got.get(prim);
    if (!p) throw new Error(`no ${prim} tile`);
    const { w: width, h: height } = p;
    const chans = bands.map((b) => got.get(b));
    const rgba = new Uint8Array(4 * width * height);
    if (chans.every(Boolean)) {
      const state = new Uint8Array(width * height); // 0 empty, 1 some bands, 2 every band
      for (let r = 0; r < height; r++)
        for (let c = 0; c < width; c++) {
          const i = r * width + c, oo = 4 * i;
          let n = 0;
          for (let k = 0; k < 3; k++) {
            const a = chans[k]!;
            const v = r < a.h && c < a.w ? a.d[r * a.w + c]! : 0;
            rgba[oo + k] = SQRT8[v]!;
            if (v) n++;
          }
          state[i] = n === 0 ? 0 : n === 3 ? 2 : 1;
        }
      for (let r = 0; r < height; r++)
        for (let c = 0; c < width; c++) {
          const i = r * width + c;
          let ok = state[i] === 2;
          if (state[i] === 1) {
            ok = true;
            for (let dr = -1; dr <= 1 && ok; dr++)
              for (let dc = -1; dc <= 1; dc++) {
                const rr = r + dr, cc = c + dc;
                if (rr >= 0 && cc >= 0 && rr < height && cc < width && state[rr * width + cc] === 0) { ok = false; break; }
              }
          }
          if (ok) rgba[4 * i + 3] = 255;
          else rgba[4 * i] = rgba[4 * i + 1] = rgba[4 * i + 2] = 0;
        }
    }
    return {
      texture: device.createTexture({ data: rgba, format: "rgba8unorm", width, height, sampler: { magFilter: "linear", minFilter: "linear" } }),
      width,
      height,
    };
  }

  /**
   * The year's layers under the view: every batch's fill first, its yearly footprints over them. comp: true color
   * or a band stack; extensions (a mask) go on every footprint's layer.
   */
  layers(o: { year: number; view: [number, number, number, number]; visible: boolean; prefix: string; beforeId?: string; comp?: S2Comp; extensions?: Record<string, unknown> }) {
    const comp = o.comp ?? "tci";
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
          id: `${o.prefix}s2-${comp}-${b.key}-${part}`,
          sources: b[part],
          minZoom: S2_MIN_ZOOM,
          visible: o.visible,
          beforeId: o.beforeId,
          // a footprint that cannot be read (the STAC lists it but the bucket lacks it) is nothing there
          getSource: (src: S2Item) => {
            const url = comp === "tci" ? src.url : src.bands[primOf(comp)];
            return url ? this.tiff(url).catch(() => null) : Promise.resolve(null);
          },
          renderSource: (src: S2Item, { data, signal }: { data?: GeoTIFF | null; signal?: AbortSignal }) =>
            !data
              ? null
              : comp !== "tci"
                ? new COGLayer({
                    id: `${o.prefix}s2-${o.year}-${comp}-${src.id}`,
                    geotiff: data,
                    pool,
                    signal,
                    refinementStrategy: "best-available",
                    maxRequests: 16,
                    ...o.extensions,
                    getTileData: (image: GeoTIFF | Overview, opts: GetTileDataOptions) => this.bandTile(image, opts, src, comp),
                    renderTile: (d: { texture: unknown }) => ({
                      renderPipeline: [
                        { module: CreateTexture, props: { textureName: d.texture } },
                        {
                          module: S2Stretch,
                          props: { lo: COMP_BANDS[comp].map((b) => BAND_RANGE[b]![0] / 10000), hi: COMP_BANDS[comp].map((b) => BAND_RANGE[b]![1] / 10000) },
                        },
                      ],
                    }),
                  } as never)
                : new COGLayer({
                  id: `${o.prefix}s2-${o.year}-${src.id}`,
                  ...o.extensions,
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
