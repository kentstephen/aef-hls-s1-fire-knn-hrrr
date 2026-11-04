import type { Layer } from "@deck.gl/core";
import type { _TileLoadProps as TileLoadProps } from "@deck.gl/geo-layers";
import { AffineTileset, AffineTilesetLevel, type GetTileDataOptions, RasterTileLayer } from "@developmentseed/deck.gl-raster";
import { ZarrLayer, type GetTileDataOptions as ZarrTileOptions } from "@developmentseed/deck.gl-zarr";
import type { Device, Texture } from "@luma.gl/core";
import type * as zarr from "zarrita";
import { FfwiTileCache } from "../hrrr/tile-cache.js";
import type { CoarsePool } from "./coarse.js";
import { lutTexture } from "./fuel.js";
import { type Classified, type Teacher } from "./head.js";
import { type AefIndex, levelFor } from "./index.js";
import { E, fromLat, fromLon, TILE, tileBounds, tileRes, toLat, toLon } from "./mercator.js";
import { type AefArray, MOSAIC_MIN_ZOOM, RES, X0, Y0, YEAR_IDX } from "./mosaic.js";
import type { AefPool } from "./pool.js";
import { rampStops, rampTexture, type GroundRamp } from "./ramp.js";
import { fuelPipeline, type FuelMode } from "./render.js";
import { forgetData } from "../hrrr/get-tile-data.js";
import { cacheSource, makeGrid, type WeatherGrid, type WeatherSource, weatherTexture } from "./weather.js";
import { ignLogit, NO_CLASS } from "./head.js";

/**
 * The land cover under the weather, two readers by zoom with the same head and the same shader:
 * zoomed out (below MOSAIC_MIN_ZOOM) the AlphaEarth COG overviews through a Web Mercator tile
 * layer whose tiles are resampled in a worker from one byte range per file and level
 * (coarse.worker.ts); zoomed in the 10 m Zarr mosaic. Each tile carries its class planes on the CPU
 * for the tooltip, two class textures on the GPU, and a small weather texture of the weather for every
 * layer of the time source (the 49 leads of a run, or the hours of a week), rebuilt when the source changes. Drawing is a per-fragment lookup; switching
 * the mode, the teacher or the hour reads nothing.
 */
export type AefTile = {
  key: string;
  kind: "coarse" | "zarr";
  width: number;
  height: number;
  byteLength: number;
  classes: Classified;
  classTex: Record<Teacher, Texture>;
  /** The ignition ground logit plane. */
  groundTex: Texture;
  /** The fuel head's P(fuel) plane (0 = none). */
  fuelTex: Texture;
  grid: WeatherGrid;
  /** The HRRR shards under the grid, made on first use. */
  shardKeys?: Set<string>;
  /** Per node and layer: the FFWI, or in ignition mode the weather part of the ignition logit. */
  ffwiTex: Texture;
  /** Which source and kind the weather texture was built from. */
  weatherKey: string;
  /** Every node had weather when the texture was built (nothing to rebuild for a new shard). */
  weatherComplete: boolean;
  /** The fuel lag the texture was built with. */
  weatherTau: number;
  bbox: [number, number, number, number];
  /** Tile width in ground meters. */
  extentM: number;
  /** Pixel index under a lon/lat, or -1. */
  pixelAt: (lon: number, lat: number) => number;
  gone?: boolean;
};

export type AefLayerOpts = {
  mode: FuelMode;
  /** The color ramp the ground is lit with. */
  ramp: GroundRamp;
  /** The map's zoom, to show one reader at a time. */
  zoom: number;
  teacher: Teacher;
  /** The layer of the weather texture to draw: the lead hour of a run, or the hour of a week. */
  leadTimeIdx: number;
  /** Where the weather comes from, and a key that changes when it does (kind, run or week). */
  weather: WeatherSource;
  weatherKey: string;
  hideBelow: number;
  rescaleMax: number;
  /** Ignition mode: hidden below this probability, ramp top at pMax (probability scale). */
  pHide: number;
  pMax: number;
  /** Ignition mode: the ramp over the probability or over the logit (zLo..zHi); the weather in or out. */
  scale: "probability" | "logit";
  zLo: number;
  zHi: number;
  useWeather: boolean;
  /** The standing picture (0 = off): faint everywhere, opaque where the weather term crosses wxGate. */
  standingAlpha: number;
  wxGate: number;
  /** The weather field as gray under everything (0 = off). */
  fieldAlpha: number;
  /** Hours of afterglow (0 = this hour only). */
  glowHours: number;
  /** Grain cells across a tile (0 = flat fill). */
  grain: number;
  /** Draw as light: additive over the map, so bright ground adds up instead of covering. */
  light: boolean;
  opacity: number;
  beforeId?: string;
  visible: boolean;
};

/** Textures freed a moment later: a draw already queued may still bind them. */
function later(...tex: Texture[]) {
  setTimeout(() => tex.forEach((t) => t.destroy()), 1500);
}

/** Web Mercator levels 0..MOSAIC_MIN_ZOOM, 256 px tiles: the XYZ grid as an affine tileset. */
let coarseTilesetCache: AffineTileset | null = null;
function coarseTileset(): AffineTileset {
  if (coarseTilesetCache) return coarseTilesetCache;
  const levels = [];
  for (let z = 0; z <= MOSAIC_MIN_ZOOM; z++) {
    const res = tileRes(z);
    levels.push(new AffineTilesetLevel({ affine: [res, 0, -E, 0, -res, E], arrayWidth: TILE * 2 ** z, arrayHeight: TILE * 2 ** z, tileWidth: TILE, tileHeight: TILE, mpu: 1 }));
  }
  coarseTilesetCache = new AffineTileset({
    levels,
    projectTo3857: (x, y) => [x, y],
    projectFrom3857: (x, y) => [x, y],
    projectTo4326: (x, y) => [toLon(x), toLat(y)],
    projectFrom4326: (lon, lat) => [fromLon(lon), fromLat(lat)],
  });
  return coarseTilesetCache;
}

/** Whether any node of the tile's weather grid lies in one of these HRRR shards (keys as FfwiTileCache.keyFor). */
function touches(t: AefTile, shards: Set<string>): boolean {
  if (!t.shardKeys) {
    t.shardKeys = new Set();
    const g = t.grid;
    for (let i = 0; i < g.px.length; i++) if (g.px[i]! >= 0) t.shardKeys.add(FfwiTileCache.keyFor(g.px[i]!, g.py[i]!));
  }
  for (const k of t.shardKeys) if (shards.has(k)) return true;
  return false;
}

export class AefLayers {
  readonly live = new Map<string, AefTile>();
  /** Tiles asked and not back yet, and how many have come back. */
  readonly work = { reading: 0, done: 0 };
  private device: Device | null = null;
  private ramp: Texture | null = null;
  private rampId: GroundRamp | null = null;
  private luts: Record<Teacher, Texture> | null = null;
  private opts: AefLayerOpts | null = null;
  private version = 0;
  private lastKey = "";
  private source: WeatherSource;
  private weatherKey = "ignition|run";

  constructor(
    private pool: AefPool,
    private coarse: CoarsePool,
    private cache: FfwiTileCache,
    private index: AefIndex,
    private mosaic: { arr: AefArray; attrs: unknown },
    /** HRRR pixels a new tile needs the forecast for (the side channel fetches the shards). */
    private onNeedWeather: (px: Int16Array, py: Int16Array) => void,
    private redraw: () => void,
  ) {
    this.source = cacheSource(cache, "ignition");
  }

  private textures(device: Device, rampId: GroundRamp) {
    if (this.device !== device || !this.ramp || !this.luts || this.rampId !== rampId) {
      if (this.device === device && this.ramp) later(this.ramp);
      this.device = device;
      this.ramp = rampTexture(device, rampStops(rampId));
      this.rampId = rampId;
      this.luts = this.device === device && this.luts ? this.luts : { nlcd: lutTexture(device, "nlcd"), esri: lutTexture(device, "esri") };
    }
    return { ramp: this.ramp, luts: this.luts };
  }

  /** Tiles whose weather texture is waiting to be rebuilt (rebuilt a few at a time, see `drain`). */
  private stale = new Set<AefTile>();
  private draining = false;
  /** The map's bounds: the tiles under them are rebuilt first. */
  private viewBox: [number, number, number, number] | null = null;

  setViewBox(box: [number, number, number, number]) {
    this.viewBox = box;
  }

  /**
   * The weather changed (a shard arrived, the fuel lag moved, the source switched): the weather texture anew
   * for the tiles it concerns. A tile whose nodes all had weather, at this lag and source, keeps its texture:
   * a shard arriving elsewhere does not touch it. The rebuilds are queued and done a few milliseconds at a
   * time, the tiles on screen first, so the map keeps moving while a window loads.
   */
  refreshWeather(shards?: Set<string>) {
    if (!this.device) return;
    for (const t of this.live.values()) {
      if (t.weatherComplete && t.weatherTau === this.cache.tau && t.weatherKey === this.weatherKey) continue;
      // only the shards named arrived: a tile over none of them keeps its texture
      if (shards && t.weatherKey === this.weatherKey && !touches(t, shards)) continue;
      this.stale.add(t);
    }
    if (this.stale.size > 0 && !this.draining) {
      this.draining = true;
      setTimeout(this.drain, 0);
    }
  }

  /** Rebuild stale weather textures for about 8 ms, on-screen tiles first, then yield to the next frame. */
  private drain = () => {
    const t0 = performance.now();
    const vb = this.viewBox;
    const onScreen = (t: AefTile) => !vb || !(t.bbox[2] < vb[0] || t.bbox[0] > vb[2] || t.bbox[3] < vb[1] || t.bbox[1] > vb[3]);
    let changed = 0;
    for (const pass of [true, false]) {
      for (const t of this.stale) {
        if (performance.now() - t0 > 8) break;
        if (t.gone || !this.device) {
          this.stale.delete(t);
          continue;
        }
        if (onScreen(t) !== pass) continue;
        this.stale.delete(t);
        const old = t.ffwiTex;
        const w = weatherTexture(this.device, t.grid, this.source);
        t.ffwiTex = w.tex;
        t.weatherComplete = w.complete;
        t.weatherTau = this.cache.tau;
        t.weatherKey = this.weatherKey;
        later(old);
        changed++;
      }
    }
    if (changed > 0) this.version++;
    // the redraw re-renders the app: at most every 150 ms while the queue drains, and once when it is empty
    const now = performance.now();
    if (this.version !== this.drawnVersion && (this.stale.size === 0 || now - this.lastRedraw > 150)) {
      this.drawnVersion = this.version;
      this.lastRedraw = now;
      this.redraw();
    }
    if (this.stale.size > 0) setTimeout(this.drain, 0);
    else this.draining = false;
  };
  private drawnVersion = -1;
  private lastRedraw = 0;

  private finish(
    device: Device,
    key: string,
    kind: AefTile["kind"],
    width: number,
    height: number,
    classes: Classified,
    toLonLat: (u: number, v: number) => [number, number],
    extentM: number,
    pixelAt: AefTile["pixelAt"],
  ): AefTile {
    this.textures(device, this.opts?.ramp ?? "viridis");
    const tex = (plane: Uint8Array) =>
      forgetData(device.createTexture({
        format: "r8unorm",
        width,
        height,
        mipLevels: 1,
        data: plane,
        sampler: { minFilter: "nearest", magFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
      }));
    const grid = makeGrid(extentM, toLonLat);
    const w = weatherTexture(device, grid, this.source);
    const corners = [toLonLat(0, 0), toLonLat(1, 0), toLonLat(0, 1), toLonLat(1, 1)];
    const bbox: AefTile["bbox"] = [
      Math.min(...corners.map((c) => c[0])), Math.min(...corners.map((c) => c[1])),
      Math.max(...corners.map((c) => c[0])), Math.max(...corners.map((c) => c[1])),
    ];
    const t: AefTile = {
      key, kind, width, height,
      byteLength: 5 * width * height + 2 * this.source.depth * grid.cols * grid.rows,
      classes,
      classTex: { nlcd: tex(classes.nlcd), esri: tex(classes.esri) },
      groundTex: tex(classes.ign),
      fuelTex: tex(classes.fuel),
      grid,
      ffwiTex: w.tex,
      weatherKey: this.weatherKey,
      weatherComplete: w.complete,
      weatherTau: this.cache.tau,
      bbox,
      extentM,
      pixelAt,
    };
    const old = this.live.get(key);
    if (old) this.drop(old);
    this.live.set(key, t);
    this.onNeedWeather(grid.px, grid.py);
    return t;
  }

  private drop(t: AefTile) {
    t.gone = true;
    if (this.live.get(t.key) === t) this.live.delete(t.key);
    later(t.classTex.nlcd, t.classTex.esri, t.groundTex, t.fuelTex, t.ffwiTex);
  }

  private async read<T>(f: () => Promise<T>): Promise<T> {
    this.work.reading++;
    try {
      const r = await f();
      this.work.done++;
      return r;
    } finally {
      this.work.reading--;
      this.redraw();
    }
  }

  // ---- the COG overviews on a Web Mercator grid
  private coarseGetTileData = async (tile: TileLoadProps, o: GetTileDataOptions): Promise<AefTile> => {
    return this.read(async () => {
      const { x, y, z } = tile.index;
      const [minX, minY, maxX, maxY] = tileBounds(z, x, y);
      const latC = toLat((minY + maxY) / 2);
      const lonC = toLon((minX + maxX) / 2);
      const groundRes = tileRes(z) * Math.cos((latC * Math.PI) / 180);
      const level = levelFor(this.index, groundRes);
      const classes = await this.coarse.tile(z, x, y, level, lonC, o.signal);
      const toLonLat = (u: number, v: number): [number, number] => [toLon(minX + (maxX - minX) * u), toLat(maxY - (maxY - minY) * v)];
      const pixelAt = (lon: number, lat: number) => {
        const ci = Math.floor(((fromLon(lon) - minX) / (maxX - minX)) * TILE);
        const ri = Math.floor(((maxY - fromLat(lat)) / (maxY - minY)) * TILE);
        return ci < 0 || ri < 0 || ci >= TILE || ri >= TILE ? -1 : ri * TILE + ci;
      };
      return this.finish(o.device, `coarse:${z}/${x}/${y}`, "coarse", TILE, TILE, classes, toLonLat, TILE * groundRes, pixelAt);
    });
  };

  // ---- the 10 m mosaic
  private zarrGetTileData = async (_arr: zarr.Array<"int8", zarr.Readable>, o: ZarrTileOptions): Promise<AefTile> => {
    return this.read(async () => {
      const ys = o.sliceSpec[2] as zarr.Slice;
      const xs = o.sliceSpec[3] as zarr.Slice;
      const py0 = ys.start ?? 0;
      const px0 = xs.start ?? 0;
      const { width, height } = o;
      const classes = await this.pool.zarr(YEAR_IDX, py0, px0, height, width, o.signal);
      const toLonLat = (u: number, v: number): [number, number] => [X0 + (px0 + u * width) * RES, Y0 - (py0 + v * height) * RES];
      const pixelAt = (lon: number, lat: number) => {
        const ci = Math.floor((lon - X0) / RES) - px0;
        const ri = Math.floor((Y0 - lat) / RES) - py0;
        return ci < 0 || ri < 0 || ci >= width || ri >= height ? -1 : ri * width + ci;
      };
      return this.finish(o.device, `zarr:${py0},${px0}`, "zarr", width, height, classes, toLonLat, width * 10, pixelAt);
    });
  };

  private renderTile = (d: AefTile) => {
    const o = this.opts;
    if (!o || !this.device || d.gone) return { renderPipeline: [] };
    const { ramp, luts } = this.textures(this.device, o.ramp);
    return fuelPipeline({
      classTex: d.classTex[o.teacher],
      groundTex: d.groundTex,
      fuelTex: d.fuelTex,
      ffwiTex: d.ffwiTex,
      lutTex: luts[o.teacher],
      rampTex: ramp,
      mode: o.mode === "ignition" ? 2 : o.mode === "cover" ? 1 : 0,
      layerIndex: o.leadTimeIdx,
      hideBelow: o.hideBelow,
      rescaleMax: o.rescaleMax,
      pHide: o.pHide,
      pMax: o.pMax,
      scale: o.scale === "logit" ? 1 : 0,
      zLo: o.zLo,
      zHi: o.zHi,
      useWeather: o.useWeather ? 1 : 0,
      standingAlpha: o.standingAlpha,
      wxGate: o.wxGate,
      fieldAlpha: o.fieldAlpha,
      glowHours: o.glowHours,
      grain: o.grain,
    });
  };

  private onUnload = (tile: { content?: AefTile | null }) => {
    if (tile.content) this.drop(tile.content);
  };

  layers(opts: AefLayerOpts): Layer[] {
    this.opts = opts;
    if (opts.weatherKey !== this.weatherKey || opts.weather !== this.source) {
      this.weatherKey = opts.weatherKey;
      this.source = opts.weather;
      this.refreshWeather();
    }
    // debug hook for the headless probe (this instance is the one deck draws from)
    (globalThis as unknown as { __aef?: unknown }).__aef = { layers: this, cache: this.cache };
    const key = `${opts.mode}|${opts.ramp}|${opts.teacher}|${opts.leadTimeIdx}|${opts.hideBelow}|${opts.rescaleMax}|${opts.pHide}|${opts.pMax}|${opts.scale}|${opts.zLo}|${opts.zHi}|${opts.useWeather}|${opts.standingAlpha}|${opts.wxGate}|${opts.fieldAlpha}|${opts.glowHours}|${opts.grain}`;
    if (key !== this.lastKey) {
      this.lastKey = key;
      this.version++;
    }
    const v = this.version;
    const common = {
      renderTile: this.renderTile,
      opacity: opts.opacity,
      visible: opts.visible,
      updateTriggers: { renderTile: [v] },
      onTileUnload: this.onUnload,
      beforeId: opts.beforeId,
      parameters: opts.light && opts.mode !== "cover"
        ? { blend: true, blendColorOperation: "add", blendColorSrcFactor: "src-alpha", blendColorDstFactor: "one", blendAlphaOperation: "add", blendAlphaSrcFactor: "one", blendAlphaDstFactor: "one-minus-src-alpha" }
        : {},
    };
    const coarse = new RasterTileLayer<AefTile>({
      id: "aef-coarse",
      tilesetDescriptor: coarseTileset(),
      getTileData: this.coarseGetTileData,
      maxRequests: 12,
      maxCacheSize: 300,
      refinementStrategy: "best-available",
      ...common,
      visible: opts.visible && opts.zoom < MOSAIC_MIN_ZOOM,
    } as never);
    const fine = new ZarrLayer<zarr.Readable, "int8", AefTile>({
      id: "aef-zarr",
      node: this.mosaic.arr,
      metadata: this.mosaic.attrs,
      selection: { time: YEAR_IDX, band: null },
      getTileData: this.zarrGetTileData,
      minZoom: MOSAIC_MIN_ZOOM,
      maxRequests: 6,
      maxCacheSize: 400,
      ...common,
    } as never);
    return [coarse, fine];
  }

  /**
   * The ignition logit at given quantiles over the pixels of the tiles on hand inside a lon/lat view, at one
   * lead (every 4th pixel each way, the weather read from the source at the HRRR pixel under the pixel's
   * nearest node; the afterglow left out). For the rank scale: the ramp set so the view's brightest fraction
   * is the ramp top, whatever the weather.
   */
  quantiles(view: [number, number, number, number], lead: number, qs: number[], useWeather: boolean, kind: AefTile["kind"]): number[] | null {
    const vals: number[] = [];
    const [vw, vs, ve, vn] = view;
    const src = this.source;
    const series = new Float32Array(src.depth);
    const wxAt = src.at
      ? (x: number, y: number) => src.at!(x, y, lead)
      : (x: number, y: number) => {
          src.series(x, y, series);
          return series[lead]!;
        };
    for (const t of this.live.values()) {
      if (t.kind !== kind) continue;
      const [w, s, e, n] = t.bbox;
      if (e < vw || w > ve || n < vs || s > vn) continue;
      const { cols, rows, px, py } = t.grid;
      for (let y = 2; y < t.height; y += 4) {
        const ny = Math.min(rows - 1, Math.floor((y / t.height) * rows));
        for (let x = 2; x < t.width; x += 4) {
          const i = y * t.width + x;
          if (t.classes.nlcd[i] === NO_CLASS) continue;
          let z = ignLogit(t.classes.ign[i]!);
          if (useWeather) {
            const nx = Math.min(cols - 1, Math.floor((x / t.width) * cols));
            const k = ny * cols + nx;
            const hx = px[k]!;
            if (hx < 0) continue;
            const wv = wxAt(hx, py[k]!);
            if (!Number.isFinite(wv)) continue;
            z += wv;
          }
          vals.push(z);
        }
      }
    }
    if (vals.length < 100) return null;
    vals.sort((a, b) => a - b);
    return qs.map((q) => vals[Math.min(vals.length - 1, Math.floor(q * vals.length))]!);
  }

  /** The classes under a lon/lat from the tiles on hand (the finer reader first), or null. */
  at(lon: number, lat: number): { tile: AefTile; nlcd: number; esri: number; cover: number; ign: number } | null {
    let best: { tile: AefTile; i: number } | null = null;
    for (const t of this.live.values()) {
      const [w, s, e, n] = t.bbox;
      if (lon < w || lon > e || lat < s || lat > n) continue;
      const i = t.pixelAt(lon, lat);
      if (i < 0) continue;
      if (!best || (t.kind === "zarr" && best.tile.kind === "coarse") || (t.kind === best.tile.kind && t.width > best.tile.width)) best = { tile: t, i };
    }
    if (!best) return null;
    const { tile, i } = best;
    return { tile, nlcd: tile.classes.nlcd[i]!, esri: tile.classes.esri[i]!, cover: tile.classes.cover[i]! / 255, ign: ignLogit(tile.classes.ign[i]!) };
  }
}
