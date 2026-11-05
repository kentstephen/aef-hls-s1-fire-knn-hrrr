import type { Layer } from "@deck.gl/core";
import { GeoArrowPathLayer, GeoArrowSolidPolygonLayer } from "@geoarrow/deck.gl-geoarrow";
import * as duckdb from "@haybarn/haybarn-wasm";
import ehWorker from "@haybarn/haybarn-wasm/dist/duckdb-browser-eh.worker.js?url";
import mvpWorker from "@haybarn/haybarn-wasm/dist/duckdb-browser-mvp.worker.js?url";
import ehWasm from "@haybarn/haybarn-wasm/dist/duckdb-eh.wasm?url";
import mvpWasm from "@haybarn/haybarn-wasm/dist/duckdb-mvp.wasm?url";
import * as arrow from "apache-arrow";

/**
 * Overture building footprints read in the browser with DuckDB-WASM (haybarn) straight from the release's
 * GeoParquet on S3, turned from WKB into GeoArrow multipolygons and drawn as outlines.
 *
 * Which of the 512 files to read comes from the release's own STAC GeoParquet (one row per file with its
 * bbox, about 150 KB), read once per session. Inside a file DuckDB skips row groups by their bbox stats.
 *
 * The ground is cut into fixed tiles (Web Mercator zoom TILE_Z) so a pan only reads the new tiles, and each
 * building belongs to the one tile that holds its bbox center, so tiles never repeat a building.
 */

export const RELEASE = "2026-10-21.0";
const STAC = `https://stac.overturemaps.org/${RELEASE}/collections.parquet`;

/** Draw buildings from this zoom in. */
export const BUILDINGS_MIN_ZOOM = 14;
/** Tile zoom: a z13 tile is about 3.7 x 2.8 km at Redding's latitude, roughly 2 to 3 row groups of the files. */
export const TILE_Z = 13;
/** At most this many tiles per view, nearest the center first. */
export const MAX_TILES = 16;
/** Stop adding tiles to a view once this many buildings are loaded for it. */
export const MAX_BUILDINGS = 80_000;
/** Tiles kept in memory across views. */
const KEEP_TILES = 64;

export type Bbox = [w: number, s: number, e: number, n: number];
export type RGBA = [number, number, number, number];

/** Rings of every building in one batch: building i owns rings buildingRings[i] .. buildingRings[i+1]-1, ring r
 * owns vertices ringOffsets[r] .. ringOffsets[r+1]-1, vertex v is (coords[2v], coords[2v+1]) in lon, lat. Rings
 * are closed (last vertex repeats the first); a building's exterior ring comes first, holes and other parts after.
 * isExterior[r] is 1 for each polygon's exterior ring. All views onto the batch's own buffers, no copies. */
export type Rings = { count: number; coords: Float64Array; ringOffsets: Int32Array; buildingRings: Int32Array; isExterior: Uint8Array };

export type BuildingTile = { key: string; bbox: Bbox; batch: arrow.RecordBatch; rings: Rings };

export type LoadStats = { queries: number; queryMs: number; parseMs: number; buildings: number; tiles: number; cachedTiles: number; skippedTiles: number };

type LayerOpts = {
  table: arrow.Table;
  /** One color for all, or per batch of the table a Uint8Array of 4 bytes (RGBA) per building, in row order. */
  getLineColor?: RGBA | ((batch: arrow.RecordBatch, batchIndex: number) => Uint8Array | RGBA);
  lineWidthMinPixels?: number;
  beforeId?: string;
  id?: string;
  opacity?: number;
  /** Change this to recompute colors from the getLineColor function. */
  colorKey?: unknown;
  /** A fill under each outline in this color (none when unset), triangulated on the page's thread. */
  fillColor?: RGBA;
  /** deck.gl LayerExtensions for the outline layer (and the fill). */
  extensions?: unknown[];
  /** Extra props passed straight to each GeoArrowPathLayer (e.g. an extension's props). */
  extraProps?: Record<string, unknown> | ((batch: arrow.RecordBatch, batchIndex: number) => Record<string, unknown>);
};

const PALE: RGBA = [236, 232, 214, 230];
const abortError = () => Object.assign(new Error("aborted"), { name: "AbortError" });

/** Wait for p, but reject with an AbortError as soon as the signal fires (p itself keeps running). */
function untilAborted<T>(p: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return p;
  if (signal.aborted) return Promise.reject(abortError());
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(abortError());
    signal.addEventListener("abort", onAbort, { once: true });
    p.then(resolve, reject).finally(() => signal.removeEventListener("abort", onAbort));
  });
}

// tiles in Web Mercator at TILE_Z
const N = 2 ** TILE_Z;
const lonToX = (lon: number) => ((lon + 180) / 360) * N;
const latToY = (lat: number) => ((1 - Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360)) / Math.PI) / 2) * N;
const xToLon = (x: number) => (x / N) * 360 - 180;
const yToLat = (y: number) => (Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / N))) * 180) / Math.PI;
const tileBbox = (x: number, y: number): Bbox => [xToLon(x), yToLat(y + 1), xToLon(x + 1), yToLat(y)];
const num = (v: number) => v.toFixed(7);

export async function createBuildings(opts: { spatial?: boolean } = {}) {
  const t0 = performance.now();
  const bundle = await duckdb.selectBundle({ mvp: { mainModule: mvpWasm, mainWorker: mvpWorker }, eh: { mainModule: ehWasm, mainWorker: ehWorker } });
  const worker = new Worker(bundle.mainWorker!);
  const db = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), worker);
  await db.instantiate(new URL(bundle.mainModule, location.href).href, bundle.pthreadWorker);
  const conn = await db.connect();
  // keep each file's footer once read; reading it is the slow part of a first query
  await conn.query("SET parquet_metadata_cache = true; SET enable_http_metadata_cache = true; SET enable_object_cache = true;");
  // the parquet reader hands back the GeoParquet geometry as geoarrow.wkb without spatial; load it only if asked
  if (opts.spatial) await conn.query("INSTALL spatial; LOAD spatial;");
  const initMs = performance.now() - t0;

  // DuckDB runs one statement at a time; chain them
  let chain: Promise<unknown> = Promise.resolve();
  function run(sql: string, signal?: AbortSignal): Promise<arrow.Table> {
    const p = chain.then(async () => {
      if (signal?.aborted) throw abortError();
      const onAbort = () => void conn.cancelSent().catch(() => {});
      signal?.addEventListener("abort", onAbort, { once: true });
      try {
        const r = (await conn.query(sql)) as unknown as arrow.Table;
        if (signal?.aborted) throw abortError();
        return r;
      } catch (e) {
        throw signal?.aborted ? abortError() : e;
      } finally {
        signal?.removeEventListener("abort", onAbort);
      }
    });
    chain = p.catch(() => {});
    return p;
  }

  // the release's files and their bboxes, once per session
  let files: Promise<{ href: string; bbox: Bbox }[]> | null = null;
  const fileList = () =>
    (files ??= run(
      `SELECT assets.aws.href AS href, bbox.xmin AS w, bbox.ymin AS s, bbox.xmax AS e, bbox.ymax AS n FROM '${STAC}' WHERE collection = 'building'`,
    ).then((t) => {
      const href = t.getChild("href")!, w = t.getChild("w")!, s = t.getChild("s")!, e = t.getChild("e")!, n = t.getChild("n")!;
      return Array.from({ length: t.numRows }, (_, i) => ({ href: href.get(i) as string, bbox: [w.get(i), s.get(i), e.get(i), n.get(i)] as Bbox }));
    }));

  const tiles = new Map<string, BuildingTile>();
  const pending = new Map<string, Promise<void>>();
  const stats: LoadStats = { queries: 0, queryMs: 0, parseMs: 0, buildings: 0, tiles: 0, cachedTiles: 0, skippedTiles: 0 };
  let schema: arrow.Schema | null = null;

  /** Read the buildings of a rectangle of tiles in one query and split them into those tiles. */
  async function fetchRect(x0: number, x1: number, y0: number, y1: number, want: Set<string>, signal?: AbortSignal) {
    const [w, , , n] = tileBbox(x0, y0);
    const [, s, e] = tileBbox(x1, y1);
    const hit = (await fileList()).filter((f) => f.bbox[0] <= e && f.bbox[2] >= w && f.bbox[1] <= n && f.bbox[3] >= s);
    const keys = [...want];
    if (!hit.length) {
      for (const k of keys) store(k, emptyTile(k));
      return;
    }
    const list = hit.map((f) => `'${f.href}'`).join(", ");
    const sql = `SELECT id, height, (bbox.xmin + bbox.xmax) / 2 AS cx, (bbox.ymin + bbox.ymax) / 2 AS cy, geometry
      FROM read_parquet([${list}])
      WHERE bbox.xmax >= ${num(w)} AND bbox.xmin <= ${num(e)} AND bbox.ymax >= ${num(s)} AND bbox.ymin <= ${num(n)}
        AND (bbox.xmin + bbox.xmax) / 2 >= ${num(w)} AND (bbox.xmin + bbox.xmax) / 2 < ${num(e)}
        AND (bbox.ymin + bbox.ymax) / 2 >= ${num(s)} AND (bbox.ymin + bbox.ymax) / 2 < ${num(n)}`;
    const tq = performance.now();
    const t = await run(sql, signal);
    stats.queries++;
    stats.queryMs = performance.now() - tq;
    const tp = performance.now();
    const parts = splitByTile(t, x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    for (const k of keys) store(k, parts.get(k) ?? emptyTile(k));
    stats.parseMs = performance.now() - tp;
  }

  function store(k: string, tile: BuildingTile) {
    tiles.delete(k);
    tiles.set(k, tile);
    while (tiles.size > KEEP_TILES) tiles.delete(tiles.keys().next().value as string);
  }

  function emptyTile(k: string): BuildingTile {
    const [x, y] = k.split("/").slice(1).map(Number) as [number, number];
    return { key: k, bbox: tileBbox(x, y), ...toGeoArrow([], [], [], 0) };
  }

  /** Rows of the query grouped by tile, WKB parsed into one GeoArrow batch per tile. */
  function splitByTile(t: arrow.Table, x0: number, y0: number, nx: number, ny: number): Map<string, BuildingTile> {
    const ids: string[][] = Array.from({ length: nx * ny }, () => []);
    const heights: (number | null)[][] = Array.from({ length: nx * ny }, () => []);
    const wkbs: Uint8Array[][] = Array.from({ length: nx * ny }, () => []);
    const idCol = t.getChild("id")!, hCol = t.getChild("height")!, cx = t.getChild("cx")!, cy = t.getChild("cy")!, g = t.getChild("geometry")!;
    // walk the result's chunks without making per-row objects other than the WKB views
    let row = 0;
    for (const chunk of g.data) {
      const off = chunk.valueOffsets as Int32Array, bytes = chunk.values as Uint8Array;
      for (let i = 0; i < chunk.length; i++, row++) {
        const tx = Math.floor(lonToX(cx.get(row))) - x0, ty = Math.floor(latToY(cy.get(row))) - y0;
        if (tx < 0 || ty < 0 || tx >= nx || ty >= ny) continue;
        const b = tx + ty * nx;
        ids[b]!.push(idCol.get(row));
        heights[b]!.push(hCol.get(row));
        wkbs[b]!.push(bytes.subarray(off[chunk.offset + i]!, off[chunk.offset + i + 1]!));
      }
    }
    const out = new Map<string, BuildingTile>();
    for (let ty = 0; ty < ny; ty++)
      for (let tx = 0; tx < nx; tx++) {
        const b = tx + ty * nx, x = x0 + tx, y = y0 + ty;
        const key = `${TILE_Z}/${x}/${y}`;
        out.set(key, { key, bbox: tileBbox(x, y), ...toGeoArrow(wkbs[b]!, ids[b]!, heights[b]!, wkbs[b]!.length) });
      }
    return out;
  }

  function toGeoArrow(wkbs: Uint8Array[], ids: string[], heights: (number | null)[], n: number) {
    const geom = parseWkb(wkbs);
    const geomField = new arrow.Field("geometry", geom.data.type, false, new Map([["ARROW:extension:name", "geoarrow.multipolygon"]]));
    const idVec = arrow.vectorFromArray(ids, new arrow.Utf8());
    const hVec = arrow.vectorFromArray(heights, new arrow.Float64());
    schema ??= new arrow.Schema([new arrow.Field("id", new arrow.Utf8(), false), new arrow.Field("height", new arrow.Float64(), true), geomField]);
    const struct = arrow.makeData({
      type: new arrow.Struct(schema.fields),
      length: n,
      nullCount: 0,
      children: [idVec.data[0]!, hVec.data[0]!, geom.data],
    });
    return { batch: new arrow.RecordBatch(schema, struct), rings: geom.rings };
  }

  /**
   * Buildings for a bbox as one Arrow table: id, height, geometry (geoarrow.multipolygon), one record batch per
   * tile. Tiles already read are reused. The tiles nearest the center come first and the list stops at
   * MAX_TILES or MAX_BUILDINGS. Rejects with an AbortError when the signal fires.
   */
  async function load(bbox: Bbox, signal?: AbortSignal): Promise<arrow.Table> {
    const ts = await loadTiles(bbox, signal);
    if (!ts.length) return new arrow.Table(schema ?? emptyTile(`${TILE_Z}/0/0`).batch.schema);
    return new arrow.Table(ts.map((t) => t.batch));
  }

  async function loadTiles(bbox: Bbox, signal?: AbortSignal): Promise<BuildingTile[]> {
    const [w, s, e, n] = bbox;
    const x0 = Math.floor(lonToX(w)), x1 = Math.floor(lonToX(e)), y0 = Math.floor(latToY(n)), y1 = Math.floor(latToY(s));
    const cx = (x0 + x1 + 1) / 2, cy = (y0 + y1 + 1) / 2;
    const all: { x: number; y: number; d: number }[] = [];
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) all.push({ x, y, d: Math.hypot(x + 0.5 - cx, y + 0.5 - cy) });
    all.sort((a, b) => a.d - b.d);
    const chosen = all.slice(0, MAX_TILES);
    stats.skippedTiles = all.length - chosen.length;

    // read the missing ones as one rectangle (each row group of the files is read once)
    const key = (t: { x: number; y: number }) => `${TILE_Z}/${t.x}/${t.y}`;
    const missing = chosen.filter((t) => !tiles.has(key(t)) && !pending.has(key(t)));
    stats.cachedTiles = chosen.length - missing.length;
    if (missing.length) {
      const mx0 = Math.min(...missing.map((t) => t.x)), mx1 = Math.max(...missing.map((t) => t.x));
      const my0 = Math.min(...missing.map((t) => t.y)), my1 = Math.max(...missing.map((t) => t.y));
      const want = new Set(missing.map(key));
      // the read is shared: a later view waits on the same tiles, so one view's abort must not cancel it (a
      // cancelled first read, ~25 s, left every later view waiting on a failed job and no outlines at all)
      const job = fetchRect(mx0, mx1, my0, my1, want);
      for (const k of want) pending.set(k, job);
      job.finally(() => want.forEach((k) => pending.get(k) === job && pending.delete(k))).catch(() => {});
    }
    await untilAborted(Promise.all(chosen.map((t) => pending.get(key(t))).filter(Boolean)), signal);
    if (signal?.aborted) throw abortError();

    const out: BuildingTile[] = [];
    let count = 0;
    for (const t of chosen) {
      const tile = tiles.get(key(t));
      if (!tile) continue;
      if (count && count + tile.rings.count > MAX_BUILDINGS) {
        stats.skippedTiles++;
        continue;
      }
      count += tile.rings.count;
      out.push(tile);
    }
    stats.buildings = count;
    stats.tiles = out.length;
    return out;
  }

  /** Rings of one batch of a table from load() (views, no copies). */
  function ringsOf(batch: arrow.RecordBatch): Rings {
    for (const t of tiles.values()) if (t.batch === batch) return t.rings;
    return ringsFromData(batch.getChild("geometry")!.data[0]!);
  }

  /** One outline layer per batch (tile) so deck.gl keeps the GPU buffers of tiles that stay in view. */
  function layer(o: LayerOpts): Layer[] {
    const color = o.getLineColor ?? PALE;
    const batches = o.table.batches.filter((b) => b.numRows > 0);
    const fills = !o.fillColor
      ? []
      : batches.map((batch, bi) =>
          new GeoArrowSolidPolygonLayer({
            id: `${o.id ?? "buildings"}-fill-${tileKeyOf(batch) ?? bi}`,
            data: batch,
            getPolygon: batch.getChild("geometry")!.data[0]! as never,
            getFillColor: o.fillColor,
            earcutWorkerUrl: null,
            opacity: o.opacity ?? 1,
            ...(o.extensions ? { extensions: o.extensions } : {}),
            ...(typeof o.extraProps === "function" ? o.extraProps(batch, bi) : o.extraProps),
            beforeId: o.beforeId,
          } as never) as unknown as Layer);
    return fills.concat(batches
      .map((batch, bi) => {
        const rings = ringsOf(batch);
        const c = typeof color === "function" ? color(batch, bi) : color;
        const geomData = batch.getChild("geometry")!.data[0]!;
        return new GeoArrowPathLayer({
          id: `${o.id ?? "buildings"}-${tileKeyOf(batch) ?? bi}`,
          data: batch,
          getPath: outlinesOf(geomData, rings) as never,
          getColor: (c instanceof Uint8Array ? colorData(c) : c) as never,
          getWidth: 1,
          widthUnits: "pixels",
          widthMinPixels: o.lineWidthMinPixels ?? 1.5,
          // rings are closed already
          _pathType: "open",
          jointRounded: false,
          capRounded: false,
          opacity: o.opacity ?? 1,
          pickable: false,
          updateTriggers: { getColor: [o.colorKey] },
          ...(o.extensions ? { extensions: o.extensions } : {}),
          ...(typeof o.extraProps === "function" ? o.extraProps(batch, bi) : o.extraProps),
          // read by the MapLibre overlay
          beforeId: o.beforeId,
        } as never) as unknown as Layer;
      }));
  }

  function tileKeyOf(batch: arrow.RecordBatch): string | null {
    for (const t of tiles.values()) if (t.batch === batch) return t.key;
    return null;
  }

  const memory = () => ({ tiles: tiles.size, buildings: [...tiles.values()].reduce((a, t) => a + t.rings.count, 0), bytes: [...tiles.values()].reduce((a, t) => a + byteSize(t), 0) });

  return { load, loadTiles, layer, ringsOf, stats, memory, initMs, db, conn, run, fileList };
}

export type Buildings = Awaited<ReturnType<typeof createBuildings>>;


// ---- WKB to GeoArrow ----

const coordType = new arrow.FixedSizeList(2, new arrow.Field("xy", new arrow.Float64(), false));
const ringType = new arrow.List(new arrow.Field("vertices", coordType, false));
const polyType = new arrow.List(new arrow.Field("rings", ringType, false));
const multiType = new arrow.List(new arrow.Field("polygons", polyType, false));
const lineStringsType = new arrow.List(new arrow.Field("linestrings", ringType, false));

type Geom = { data: arrow.Data<arrow.List>; rings: Rings };

/** Parse WKB Polygon and MultiPolygon (2D; Z and M are dropped) into a geoarrow.multipolygon column. */
export function parseWkb(wkbs: Uint8Array[]): Geom {
  const n = wkbs.length;
  // first pass: sizes
  let nPolys = 0, nRings = 0, nCoords = 0;
  for (const b of wkbs) {
    const r = walk(b, null);
    nPolys += r.polys;
    nRings += r.rings;
    nCoords += r.coords;
  }
  const coords = new Float64Array(nCoords * 2);
  const ringOffsets = new Int32Array(nRings + 1);
  const polyOffsets = new Int32Array(nPolys + 1);
  const geomOffsets = new Int32Array(n + 1);
  const isExterior = new Uint8Array(nRings);
  const out = { coords, ringOffsets, polyOffsets, isExterior, c: 0, r: 0, p: 0 };
  for (let i = 0; i < n; i++) {
    walk(wkbs[i]!, out);
    geomOffsets[i + 1] = out.p;
  }
  const coordData = arrow.makeData({ type: coordType, length: nCoords, nullCount: 0, child: arrow.makeData({ type: new arrow.Float64(), length: nCoords * 2, nullCount: 0, data: coords }) });
  const ringData = arrow.makeData({ type: ringType, length: nRings, nullCount: 0, valueOffsets: ringOffsets, child: coordData });
  const polyData = arrow.makeData({ type: polyType, length: nPolys, nullCount: 0, valueOffsets: polyOffsets, child: ringData });
  const data = arrow.makeData({ type: multiType, length: n, nullCount: 0, valueOffsets: geomOffsets, child: polyData });
  const buildingRings = new Int32Array(n + 1);
  for (let i = 0; i <= n; i++) buildingRings[i] = polyOffsets[geomOffsets[i]!]!;
  return { data, rings: { count: n, coords, ringOffsets, buildingRings, isExterior } };
}

type Out = { coords: Float64Array; ringOffsets: Int32Array; polyOffsets: Int32Array; isExterior: Uint8Array; c: number; r: number; p: number };

/** Walk one WKB geometry; with out null only count, else write. */
function walk(b: Uint8Array, out: Out | null): { polys: number; rings: number; coords: number } {
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let pos = 0, polys = 0, rings = 0, coords = 0;
  const header = () => {
    const le = dv.getUint8(pos) === 1;
    let type = dv.getUint32(pos + 1, le);
    pos += 5;
    // EWKB flags: Z 0x80000000, M 0x40000000, SRID 0x20000000
    let dims = 2;
    if (type & 0x80000000) dims++;
    if (type & 0x40000000) dims++;
    if (type & 0x20000000) pos += 4;
    type &= 0x0fffffff;
    // ISO: 1000s Z, 2000s M, 3000s ZM
    const iso = Math.floor(type / 1000);
    if (iso === 1 || iso === 2) dims = 3;
    else if (iso === 3) dims = 4;
    return { le, type: type % 1000, dims };
  };
  const polygon = (le: boolean, dims: number) => {
    const nr = dv.getUint32(pos, le);
    pos += 4;
    polys++;
    for (let k = 0; k < nr; k++) {
      const np = dv.getUint32(pos, le);
      pos += 4;
      rings++;
      coords += np;
      if (out) {
        out.isExterior[out.r] = k === 0 ? 1 : 0;
        for (let j = 0; j < np; j++) {
          out.coords[out.c * 2] = dv.getFloat64(pos, le);
          out.coords[out.c * 2 + 1] = dv.getFloat64(pos + 8, le);
          out.c++;
          pos += 8 * dims;
        }
        out.ringOffsets[++out.r] = out.c;
      } else pos += 8 * dims * np;
    }
    if (out) out.polyOffsets[++out.p] = out.r;
  };
  const h = header();
  if (h.type === 3) polygon(h.le, h.dims);
  else if (h.type === 6) {
    const np = dv.getUint32(pos, h.le);
    pos += 4;
    for (let k = 0; k < np; k++) {
      const hh = header();
      polygon(hh.le, hh.dims);
    }
  }
  // anything else (points, lines) is skipped: the building keeps zero polygons
  return { polys, rings, coords };
}

/** Rings view from a geoarrow.multipolygon Data. */
function ringsFromData(d: arrow.Data): Rings {
  const geomOffsets = d.valueOffsets as Int32Array;
  const poly = d.children[0]!, ring = poly.children[0]!, coord = ring.children[0]!;
  const polyOffsets = poly.valueOffsets as Int32Array;
  const n = d.length;
  const buildingRings = new Int32Array(n + 1);
  for (let i = 0; i <= n; i++) buildingRings[i] = polyOffsets[geomOffsets[d.offset + i]!]!;
  const isExterior = new Uint8Array(ring.length);
  for (let p = 0; p < poly.length; p++) isExterior[polyOffsets[p]!] = 1;
  return { count: n, coords: coord.children[0]!.values as Float64Array, ringOffsets: ring.valueOffsets as Int32Array, buildingRings, isExterior };
}

/** Every ring of a building as one multilinestring, sharing the polygon buffers. */
function outlinesOf(geom: arrow.Data, rings: Rings): arrow.Data {
  const ringData = geom.children[0]!.children[0]!;
  return arrow.makeData({ type: lineStringsType, length: rings.count, nullCount: 0, valueOffsets: rings.buildingRings, child: ringData } as never);
}

function colorData(rgba: Uint8Array): arrow.Data {
  const t = new arrow.FixedSizeList(4, new arrow.Field("rgba", new arrow.Uint8(), false));
  return arrow.makeData({ type: t, length: rgba.length / 4, nullCount: 0, child: arrow.makeData({ type: new arrow.Uint8(), length: rgba.length, nullCount: 0, data: rgba }) });
}

function byteSize(t: BuildingTile): number {
  const r = t.rings;
  return r.coords.byteLength + r.ringOffsets.byteLength + r.buildingRings.byteLength + r.isExterior.byteLength + r.count * 60;
}
