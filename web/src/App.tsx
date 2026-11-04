import {
  COLORMAP_INDEX,
  createColormapTexture,
  decodeColormapSprite,
} from "@developmentseed/deck.gl-raster/gpu-modules";
import colormapsPngUrl from "@developmentseed/deck.gl-raster/gpu-modules/colormaps.png";
import { ZarrLayer } from "@developmentseed/deck.gl-zarr";
import { PathLayer } from "@deck.gl/layers";
import type { Device, Texture } from "@luma.gl/core";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MapboxOverlay } from "@deck.gl/mapbox";
import type { MapRef } from "react-map-gl/maplibre";
import { Map as MaplibreMap } from "react-map-gl/maplibre";
import type * as zarr from "zarrita";
import { fuelOf } from "./aef/fuel.js";
import { ignitionRate, rateLabel } from "./aef/calibration.js";
// import { fireLayers, firesInRun, loadFires, type Fires } from "./fires.js";
// the burn perimeters (perimeters.ts): not drawn for now; TODO draw them from GeoArrow (deck.gl-geoarrow) rather
// than a 7 MB GeoJSON parsed on the main thread, and only while each fire burned
// import { loadPerimeters, type Perimeter, perimeterLayer, perimetersInRun } from "./perimeters.js";
import { HEADS, NO_CLASS, type Teacher } from "./aef/head.js";
import { AefLayers } from "./aef/layers.js";
import { type AefArray, MOSAIC_MIN_ZOOM, openAefMosaic } from "./aef/mosaic.js";
import { AefPool } from "./aef/pool.js";
import { CoarsePool } from "./aef/coarse.js";
import { type AefIndex, loadAefIndex } from "./aef/index.js";
import type { WindMode } from "./ffwi/compute.js";
import { makeGetTileData, type FfwiTileData } from "./hrrr/get-tile-data.js";
import { dateFromInitTimeIdx, HRRR_GEOZARR_ATTRS, HRRR_LEAD_TIME_COUNT, initTimeIdxFromDate, lonLatToHrrrPixel } from "./hrrr/metadata.js";
import { makeRenderTile } from "./hrrr/render-tile.js";
import { buildSelection } from "./hrrr/selection.js";
import { type ContourPath, ffwiContours } from "./hrrr/contours.js";
import { SideChannel } from "./hrrr/side-channel.js";
import { openHrrr, type HrrrSource } from "./hrrr/store.js";
import { FfwiTileCache } from "./hrrr/tile-cache.js";
import { RASTER_VARIABLES, type RasterVariable } from "./hrrr/variables.js";
import { HOURS_PER_DAY, WeekWeather, weekHourDate, type WeekSpec } from "./hrrr/week.js";
import type { GroundRamp } from "./aef/ramp.js";
import { cacheSource } from "./aef/weather.js";
import { CONUS_MEAN } from "./aef/calibration.js";
import { CountyTable, countyLayer, loadCounties, unitsGrid, type CountyFeature } from "./counties.js";
import { DeckGlOverlay } from "./lib/deckgl-overlay.js";
import type * as arrow from "apache-arrow";
import { BUILDINGS_MIN_ZOOM, createBuildings } from "./buildings.js";
import { buildingExposure, exposureColors } from "./building-exposure.js";
import { rampStops, rampStripe } from "./aef/ramp.js";
import { ControlPanel, loadingSummary, type ColormapChoiceId, type FuelChoice, type IgnScale, type Status, type TimeMode } from "./ui/ControlPanel.js";

/**
 * The HRRR weather off for now: no forecast is read, nothing plays, and the ground is the fuel head's P(fuel) alone.
 * True brings back FFWI x P(fuel) hour by hour and the Time panel.
 */
const WEATHER_ON = false;
const MAP_STYLE_URL = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
/** First basemap layer above water in CARTO styles; data goes under roads/labels. */
const BEFORE_ID = "aeroway-runway";
/** How long the window's dates hold still before its weather starts loading. */
const WINDOW_SETTLE_MS = 700;
/** 100 ms a frame: a week in 17 s. */
const INITIAL_FRAME_MS = 200; // 72 hours in about 14 s
/**
 * Opens on the whole country. The single-run mode opens on the 12Z run of August 1, 2026, the day of
 * Washington's first PDS red flag warning, playing from +0 h so the afternoon peak comes up over the first
 * frames and the afterglow carries it into the 2nd. (March 14, 2025 12Z is another: the Southern Plains
 * outbreak, extreme wind and single-digit humidity from Texas into the Mid-South.)
 */
// Redding, California: the ground draws from the 10 m mosaic only (zoom MOSAIC_MIN_ZOOM in), where the fuel head runs
const START_VIEW = { longitude: -122.3917, latitude: 40.5865, zoom: 12.5 };
const START_INIT = new Date("2026-08-10T12:00:00Z");
const START_LEAD = 0;
/**
 * The week the map opens on, over the whole country: August 7 to 9, 2026, three days in the heart of the
 * interior West's season, hot and dry every afternoon. Three days rather than a full week because the whole
 * country's shards take minutes to arrive. The ground is the 2025 embeddings. (By the HRRR
 * analysis, ignition/rank_weeks.py, the head's weather term over a sample of the country: July 26 to
 * August 1 had the most area-hours above the gate; August 20 was the single widest day; April 21 and 22
 * were the High Plains wind days; February 17 the Ranger Road day.)
 */
const START_WEEK = new Date("2026-08-07T00:00:00Z");
const START_DAYS = 3;
/** Terrain under the ground: AWS Terrain Tiles (Mapzen Terrarium), public, 256 px, to zoom 15. */
const DEM_TILES = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";
const HILLSHADE_ID = "aef-hillshade";
/** A building with no burnable ground near it. */
const BUILDING_GRAY: [number, number, number] = [69, 69, 77];

/** CARTO's own building footprints off: the buildings drawn are Overture's (buildings.ts). */
function hideBasemapBuildings(map: import("maplibre-gl").Map) {
  for (const l of map.getStyle().layers) {
    if ("source-layer" in l && l["source-layer"] === "buildings") map.setLayoutProperty(l.id, "visibility", "none");
  }
}

/** Relief under everything: a hillshade layer from the terrain tiles, below the first data layer. */
function addHillshade(map: import("maplibre-gl").Map, on: boolean) {
  if (!map.getSource("aef-dem")) {
    map.addSource("aef-dem", { type: "raster-dem", tiles: [DEM_TILES], encoding: "terrarium", tileSize: 256, maxzoom: 15 });
  }
  if (!map.getLayer(HILLSHADE_ID)) {
    const before = map.getLayer("aef-coarse") ? "aef-coarse" : BEFORE_ID;
    map.addLayer(
      {
        id: HILLSHADE_ID,
        type: "hillshade",
        source: "aef-dem",
        paint: {
          "hillshade-shadow-color": "#000000",
          "hillshade-highlight-color": "#9aa7b8",
          "hillshade-accent-color": "#000000",
          "hillshade-exaggeration": 0.55,
          "hillshade-illumination-direction": 335,
        },
      },
      before,
    );
  }
  map.setLayoutProperty(HILLSHADE_ID, "visibility", on ? "visible" : "none");
}

export default function App() {
  const mapRef = useRef<MapRef>(null);
  /** Whether the cursor picks (the tooltip); Q toggles it. */
  const pickRef = useRef(true);
  const cacheRef = useRef(new FfwiTileCache());
  const [cacheVersion, setCacheVersion] = useState(0);
  useEffect(() => {
    const cache = cacheRef.current;
    const l = () => setCacheVersion(cache.version);
    cache.listeners.add(l);
    return () => {
      cache.listeners.delete(l);
    };
  }, []);

  // ---- HRRR store
  const [source, setSource] = useState<HrrrSource | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [initTimeIdx, setInitTimeIdx] = useState(0);
  useEffect(() => {
    if (!WEATHER_ON) return;
    let cancelled = false;
    openHrrr()
      .then((s) => {
        if (cancelled) return;
        setSource(s);
        setInitTimeIdx(initTimeIdxFromDate(START_INIT, s.initTimeCount - 1));
        setLeadTimeIdx(START_LEAD);
      })
      .catch((e) => setError(`HRRR store: ${String(e)}`));
    return () => {
      cancelled = true;
    };
  }, []);

  // ---- the embeddings: the tile index, the mosaic, the worker pool, the layer controller
  const poolRef = useRef<AefPool | null>(null);
  const coarseRef = useRef<CoarsePool | null>(null);
  const [aefIndex, setAefIndex] = useState<AefIndex | null>(null);
  const [mosaic, setMosaic] = useState<{ arr: AefArray; attrs: unknown } | null>(null);
  useEffect(() => {
    let gone = false;
    loadAefIndex(import.meta.env.BASE_URL)
      .then((i) => { if (!gone) setAefIndex(i); })
      .catch((e) => { if (!gone) setError(`AEF index: ${String(e)}`); });
    openAefMosaic()
      .then((m) => { if (!gone) setMosaic(m); })
      .catch((e) => { if (!gone) setError(`AEF mosaic: ${String(e)}`); });
    return () => { gone = true; };
  }, []);
  // the real ignitions as points over the ground (fires.ts, fires.bin): not drawn; the picture is the
  // ground and the weather
  // const [fires, setFires] = useState<Fires | null>(null);
  // useEffect(() => { loadFires(import.meta.env.BASE_URL).then(setFires).catch((e) => setError(`fires: ${String(e)}`)); }, []);
  // const [showFires, setShowFires] = useState(true);
  // const fireList = useMemo(() => (fires && source ? firesInRun(fires, dateFromInitTimeIdx(initTimeIdx)) : []), [fires, source, initTimeIdx]);
  // ---- the burn perimeters of the fires (the year after the embeddings), the frame for the picture
  // const [perimeters, setPerimeters] = useState<Perimeter[]>([]);
  // useEffect(() => { loadPerimeters(import.meta.env.BASE_URL, AEF_YEAR + 1).then(setPerimeters).catch((e) => setError(`perimeters: ${String(e)}`)); }, []);
  // const [showPerimeters, setShowPerimeters] = useState(true);
  // const runPerimeters = useMemo(() => (source ? perimetersInRun(perimeters, dateFromInitTimeIdx(initTimeIdx)) : []), [perimeters, source, initTimeIdx]);
  const [aefVersion, setAefVersion] = useState(0);
  const needWeatherRef = useRef<(px: Int16Array, py: Int16Array) => void>(() => {});
  const aefLayersRef = useRef<AefLayers | null>(null);
  const aefLayers = useMemo(() => {
    if (!aefIndex || !mosaic) return null;
    if (!poolRef.current) poolRef.current = new AefPool();
    if (!coarseRef.current) coarseRef.current = new CoarsePool(import.meta.env.BASE_URL);
    const l = new AefLayers(
      poolRef.current,
      coarseRef.current,
      cacheRef.current,
      aefIndex,
      mosaic,
      (px, py) => needWeatherRef.current(px, py),
      () => setAefVersion((v) => v + 1),
    );
    return l;
  }, [aefIndex, mosaic]);
  // StrictMode runs the memo twice in development and keeps the first result: the ref follows the kept one
  useEffect(() => {
    aefLayersRef.current = aefLayers;
  }, [aefLayers]);

  // ---- animation
  const timeModeRef = useRef<TimeMode>("run");
  /** Week mode: the hour after this one for the play loop (skips days not yet in hand). */
  const nextHourRef = useRef<(h: number) => number>((h) => (h + 1) % depthRef.current);
  const depthRef = useRef(START_DAYS * HOURS_PER_DAY);
  const [leadTimeIdx, setLeadTimeIdx] = useState(START_LEAD);
  const [isPlaying, setIsPlaying] = useState(WEATHER_ON);
  const [frameDurationMs, setFrameDurationMs] = useState(INITIAL_FRAME_MS);
  /**
   * How far into the hour the playhead is, 0..1 in 24 steps, so the ground glides between hours while playing.
   * Kept out of React state: it changes nearly every frame, and only the map needs it. The play loop hands the
   * map new layers itself (`pushFrac`); the app re-renders only when the hour steps.
   */
  const hourFracRef = useRef(0);
  const overlayRef = useRef<MapboxOverlay | null>(null);
  const buildLayersRef = useRef<(frac: number) => unknown[]>(() => []);
  const pushFrac = (frac: number) => {
    if (frac === hourFracRef.current) return;
    hourFracRef.current = frac;
    overlayRef.current?.setProps({ layers: buildLayersRef.current(frac) as never });
  };
  useEffect(() => {
    if (!isPlaying) {
      pushFrac(0);
      return;
    }
    let raf = 0;
    let last = performance.now();
    const loop = (now: number) => {
      if (now - last >= frameDurationMs) {
        // the hour steps: React draws the new hour at fraction 0 (pushing 0 here first would show the old hour
        // at its start for a frame)
        hourFracRef.current = 0;
        if (timeModeRef.current === "week") setHourIdx((i) => nextHourRef.current(i));
        else setLeadTimeIdx((i) => (i + 1) % HRRR_LEAD_TIME_COUNT);
        last = now;
      } else {
        pushFrac(Math.min(23, Math.floor(((now - last) / frameDurationMs) * 24)) / 24);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying, frameDurationMs]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const space = e.code === "Space" || e.key === " ";
      const full = e.key === "f" || e.key === "F";
      const labels = e.key === "l" || e.key === "L";
      const pick = e.key === "q" || e.key === "Q";
      const ground = e.key === "g" || e.key === "G";
      if (!space && !full && !labels && !pick && !ground) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const el = e.target as HTMLElement | null;
      const tag = el?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "BUTTON" || tag === "SELECT" || el?.isContentEditable) return;
      e.preventDefault();
      if (space) setIsPlaying((p) => !p);
      // G switched the old ignition head between pixels and counties: both off for now
      else if (ground) return;
      else if (pick) {
        // the tooltip under the cursor on or off (off for a clean recording)
        pickRef.current = !pickRef.current;
        if (!pickRef.current) {
          setHover(null);
          setCountyHover(null);
        }
      } else if (labels) {
        // the basemap's text and icons (every symbol layer of the style) on or off
        const m = mapRef.current?.getMap();
        if (!m) return;
        const syms = m.getStyle().layers.filter((l) => l.type === "symbol");
        const on = syms.some((l) => m.getLayoutProperty(l.id, "visibility") !== "none");
        for (const l of syms) m.setLayoutProperty(l.id, "visibility", on ? "none" : "visible");
      } else if (document.fullscreenElement) void document.exitFullscreen();
      else void document.documentElement.requestFullscreen();
    };
    // the browser's real full screen (F): the map takes the new size
    const onFull = () => mapRef.current?.getMap().resize();
    window.addEventListener("keydown", onKey, { capture: true });
    document.addEventListener("fullscreenchange", onFull);
    return () => {
      window.removeEventListener("keydown", onKey, { capture: true });
      document.removeEventListener("fullscreenchange", onFull);
    };
  }, []);

  // ---- styling
  const [windMode, setWindMode] = useState<WindMode>("sustained");
  const [colormapId, setColormapId] = useState<ColormapChoiceId>("solar");
  /** Ramp top for the ground and the field. */
  const [rescaleMax, setRescaleMax] = useState(100);
  /** Below this nothing is drawn (ground or field). */
  const [hideBelow, setHideBelow] = useState(20);
  const [showRasterState, setShowRasterState] = useState(false);
  const showRaster = showRasterState;
  const [rasterOpacity, setRasterOpacity] = useState(0.7);
  const [rasterVariable, setRasterVariable] = useState<RasterVariable>("ffwi");
  const varDef = RASTER_VARIABLES[rasterVariable];
  useEffect(() => {
    setColormapId(varDef.defaultColormap);
  }, [varDef]);
  /** Fuel-moisture timelag in hours; 0 = Fosberg as published. */
  const [tauHours, setTauHours] = useState(0);
  useEffect(() => {
    cacheRef.current.setTau(tauHours);
  }, [tauHours]);
  const [fuel, setFuelState] = useState<FuelChoice>("lit");
  /** Ignition mode: hidden below this probability, ramp top at pMax. */
  const [pHide, setPHide] = useState(0.5);
  const [pMax, setPMax] = useState(WEATHER_ON ? 0.95 : 1);
  /** How the ignition logit meets the ramp: probability, logit, or rank among the pixels in view this hour. */
  const [ignScale, setIgnScale] = useState<IgnScale>("probability");
  const [zLo, setZLo] = useState(0);
  const [zHi, setZHi] = useState(4);
  /** Rank scale: the quantiles in view that sit at the hide line and the ramp top. */
  const [qLo, setQLo] = useState(0.5);
  const [qHi, setQHi] = useState(0.99);
  /** The weather term left out: the ground alone. */
  const [useWeather, setUseWeather] = useState(true);
  /** The standing picture: the ground's probability faint all the time, opaque where the hour's weather crosses the gate. */
  const [standingAlpha, setStandingAlpha] = useState(0.25);
  const [wxGate, setWxGate] = useState(1.0);
  const [fieldAlpha, setFieldAlpha] = useState(0.35);
  // one picture at a time: drawing the field puts the ground off, and the ground back on puts the field away
  const setShowRaster = useCallback((on: boolean) => {
    setShowRasterState(on);
    if (on) setFuelState("off");
  }, []);
  const setFuel = useCallback((f: FuelChoice) => {
    setFuelState(f);
    if (f !== "off") setShowRasterState(false);
  }, []);
  const [teacher, setTeacher] = useState<Teacher>("nlcd");
  // ---- the counties: the pixel picture averaged by county (counties.ts), built once the mode is asked for
  const [counties, setCounties] = useState<CountyFeature[] | null>(null);
  useEffect(() => {
    loadCounties(import.meta.env.BASE_URL).then(setCounties).catch((e) => setError(`counties: ${String(e)}`));
  }, []);
  const countyTable = useMemo(() => (counties ? new CountyTable(counties) : null), [counties]);
  const [countyVersion, setCountyVersion] = useState(0);
  const countyStarted = useRef(false);
  useEffect(() => {
    if (fuel !== "counties" || !countyTable || !counties || !aefIndex || countyStarted.current) return;
    countyStarted.current = true;
    if (!coarseRef.current) coarseRef.current = new CoarsePool(import.meta.env.BASE_URL);
    const pool = coarseRef.current;
    pool.setUnits(unitsGrid(counties));
    // 32 px per file (2.56 km pixels): a median county is 250 of them, enough for a sum; one range request a file
    const level = Math.max(0, aefIndex.levels.indexOf(32));
    let pending = 0;
    let timer = 0;
    const bump = () => {
      if (timer) return;
      timer = window.setTimeout(() => {
        timer = 0;
        setCountyVersion((v) => v + 1);
      }, 300);
    };
    aefIndex.tiles.forEach((f, fi) => {
      pending++;
      pool
        .county(fi, level, (f.bbox[0] + f.bbox[2]) / 2)
        .then((part) => {
          countyTable.add(part);
          bump();
        })
        .catch((e) => console.warn("county sum failed", f.rel, e))
        .finally(() => {
          pending--;
          if (pending === 0) bump();
        });
    });
  }, [fuel, countyTable, counties, aefIndex]);

  const [fuelOpacity, setFuelOpacity] = useState(0.9);
  const [groundRamp, setGroundRamp] = useState<GroundRamp>("inferno");
  /** Hours of afterglow on the ground (0 = this hour only). */
  // off by default: at the same point the past day's peak adds nothing to this hour (ignition/hour_memory.py)
  const [glowHours, setGlowHours] = useState(0);
  /** Grain cells across a tile (0 = flat fill). */
  const [grain, setGrain] = useState(0);
  /** The ground drawn as light (additive) rather than paint. */
  const [light, setLight] = useState(false);
  // relief (AWS Terrarium hillshade) under the ground: off, it only makes the ground harder to read
  const [relief, setRelief] = useState(false);
  /** Contours of the FFWI field over the ground (0 = none). */
  const [contourStep, setContourStep] = useState(0);
  const [zoom, setZoom] = useState(START_VIEW.zoom);
  const [view, setView] = useState<[number, number, number, number] | null>(null);
  const viewRef = useRef<[number, number, number, number] | null>(null);
  const readView = (m: import("maplibre-gl").Map) => {
    const b = m.getBounds();
    const v: [number, number, number, number] = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
    viewRef.current = v;
    setView(v);
  };
  useEffect(() => {
    const m = mapRef.current?.getMap();
    if (m && m.isStyleLoaded()) addHillshade(m, relief);
  }, [relief]);

  const [device, setDevice] = useState<Device | null>(null);
  const [colormapImage, setColormapImage] = useState<ImageData | null>(null);
  const [colormapTexture, setColormapTexture] = useState<Texture | null>(null);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const resp = await fetch(colormapsPngUrl);
      const image = await decodeColormapSprite(await resp.arrayBuffer());
      if (!cancelled) setColormapImage(image);
    })();
    return () => {
      cancelled = true;
    };
  }, []);
  useEffect(() => {
    if (!device || !colormapImage) return;
    setColormapTexture(createColormapTexture(device, colormapImage));
  }, [device, colormapImage]);
  const colormapIndex = COLORMAP_INDEX[colormapId];

  // New run or wind mode => tiles refetch (layer id changes); drop the cache.
  useEffect(() => {
    cacheRef.current.clear();
  }, [initTimeIdx, windMode]);

  // ---- time: one 48 h run, or a week of 12Z runs stitched hour by hour
  // one 48 h run: the lit ground needs the FFWI itself, and the week's store keeps only the ignition head's
  // weather term (hrrr/week.ts); a week of FFWI for the lit ground is not built yet
  const [timeMode, setTimeMode] = useState<TimeMode>("run");
  const [weekStart, setWeekStart] = useState<Date>(START_WEEK);
  const [weekDays, setWeekDays] = useState(START_DAYS);
  const [hourIdx, setHourIdx] = useState(0);
  const [week, setWeek] = useState<WeekWeather | null>(null);
  const [weekVersion, setWeekVersion] = useState(0);
  // The window the weather loads for settles first: a date field fires a change on every step of its picker
  // or every typed digit, and each change would start a window's fetches. The panel shows the dates at once;
  // the loading starts once they have held still for WINDOW_SETTLE_MS.
  const [weekSpec, setWeekSpec] = useState<WeekSpec>(() => ({ start: weekStart, days: weekDays }));
  useEffect(() => {
    if (weekSpec.start.getTime() === weekStart.getTime() && weekSpec.days === weekDays) return;
    const t = setTimeout(() => setWeekSpec({ start: weekStart, days: weekDays }), WINDOW_SETTLE_MS);
    return () => clearTimeout(t);
  }, [weekStart, weekDays, weekSpec]);
  useEffect(() => {
    if (!source || timeMode !== "week") return;
    const w = new WeekWeather(weekSpec, windMode, source.initTimeCount);
    // the week ticks on every shard started and finished: the app re-renders at most every 250 ms for it
    let timer = 0;
    const l = () => {
      if (timer) return;
      timer = window.setTimeout(() => {
        timer = 0;
        setWeekVersion(w.version);
      }, 250);
    };
    w.listeners.add(l);
    setWeek(w);
    setHourIdx((h) => Math.min(h, w.depth - 1));
    nextHourRef.current = (h) => w.nextHour(h);
    needWeatherRef.current = (px, py) => w.ensure(px, py);
    if (viewRef.current) w.setViewBox(viewRef.current);
    return () => {
      clearTimeout(timer);
      needWeatherRef.current = () => {};
      nextHourRef.current = (h) => (h + 1) % depthRef.current;
      w.listeners.delete(l);
      w.abort();
      setWeek(null);
    };
  }, [source, timeMode, weekSpec, windMode]);
  // the playhead moved: its day goes first in the week's queue
  useEffect(() => {
    week?.focus(hourIdx);
  }, [week, hourIdx]);
  // the view moved: only the shards under the map's bounds are fetched (zoomed in on a region, that region),
  // and the tiles on screen get their weather textures first
  useEffect(() => {
    if (!view) return;
    week?.setViewBox(view);
    aefLayers?.setViewBox(view);
  }, [week, aefLayers, view]);

  // The ground needs the forecast under every tile it holds, whether or not the field is drawn:
  // a side channel fetches the shards the tiles ask for (it skips cached shards).
  useEffect(() => {
    if (!source || timeMode !== "run") return;
    const sc = new SideChannel(source.arrays, cacheRef.current);
    needWeatherRef.current = (px, py) => void sc.ensure(px, py, initTimeIdx, windMode);
    // tiles already on hand asked under an older run
    if (aefLayers) for (const t of aefLayers.live.values()) void sc.ensure(t.grid.px, t.grid.py, initTimeIdx, windMode);
    return () => {
      needWeatherRef.current = () => {};
      sc.abort();
    };
  }, [source, timeMode, initTimeIdx, windMode, aefLayers]);
  // The weather changed: the weather texture anew for the tiles it touches (a run shard: every tile).
  useEffect(() => {
    aefLayers?.refreshWeather();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aefLayers, cacheVersion]);
  useEffect(() => {
    if (!week || !aefLayers) return;
    const keys = week.takeArrived();
    if (keys.size > 0) aefLayers.refreshWeather(keys);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aefLayers, week, weekVersion]);
  useEffect(() => {
    if (fuel !== "counties" || !countyTable || !view) return;
    const { px, py } = countyTable.pixelsInView(view);
    if (px.length > 0) needWeatherRef.current(px, py);
    // countyVersion: more counties have their pixels known
  }, [fuel, countyTable, view, countyVersion, week, timeMode, initTimeIdx]);
  /** The layer of the weather texture on screen: the lead hour of the run or the hour of the week. */
  const timeIdx = timeMode === "week" ? hourIdx : leadTimeIdx;
  const timeDepth = timeMode === "week" ? weekSpec.days * HOURS_PER_DAY : HRRR_LEAD_TIME_COUNT;
  /** Where the tiles' weather textures read from, and a key that changes when that does. */
  const ignitionLike = fuel === "ignition" || fuel === "counties";
  const runSource = useMemo(() => cacheSource(cacheRef.current, ignitionLike ? "ignition" : "ffwi"), [ignitionLike]);
  const weatherSrc = timeMode === "week" && week ? week : runSource;
  timeModeRef.current = timeMode;
  depthRef.current = timeDepth;
  const weatherKey = timeMode === "week" && week ? `week|${week.id}` : `${ignitionLike ? "ignition" : "ffwi"}|run`;

  // ---- hover: lon/lat -> class under the cursor and the hour's weather term at the HRRR pixel (the FFWI itself only in
  // the run mode; the week keeps the term alone)
  const [hover, setHover] = useState<{ lon: number; lat: number; x: number; y: number } | null>(null);
  const hoverInfo = useMemo(() => {
    if (!hover) return null;
    const px = lonLatToHrrrPixel(hover.lon, hover.lat);
    const inWeek = timeMode === "week";
    const wx = px && !inWeek ? cacheRef.current.sample(px.x, px.y, leadTimeIdx) : null;
    const ground = aefLayers?.at(hover.lon, hover.lat) ?? null;
    const ci = ground ? (teacher === "nlcd" ? ground.nlcd : ground.esri) : NO_CLASS;
    const className = ci === NO_CLASS ? null : HEADS[teacher].names[ci]!;
    const weight = ci === NO_CLASS ? 0 : fuelOf(teacher, ci);
    const ffwi = inWeek ? NaN : (wx?.ffwi ?? NaN);
    const wl = px
      ? inWeek
        ? (week ? week.termAt(px.x, px.y, hourIdx) : NaN)
        : cacheRef.current.ignitionAt(px.x, px.y, leadTimeIdx)
      : NaN;
    const wx0 = wl;
    const ignP = ground && Number.isFinite(wl) ? 1 / (1 + Math.exp(-(ground.ign + wl))) : NaN;
    const groundP = ground ? 1 / (1 + Math.exp(-ground.ign)) : NaN;
    // the calibrated reading: the head's logit as ignitions per area and time (calibration.ts)
    const rate = ground && Number.isFinite(wl) ? ignitionRate(ground.ign + wl, teacher, ci) : NaN;
    return { wx, ffwi, ground, className, weight, lit: Number.isFinite(ffwi) ? ffwi * weight : NaN, ignP, groundP, wx0, rate };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hover, leadTimeIdx, hourIdx, timeMode, week, cacheVersion, weekVersion, aefLayers, aefVersion, teacher]);

  // ---- the rank scale: the view's quantiles of the logit this hour set the logit ramp
  const rankRamp = useMemo<[number, number] | null>(() => {
    if (ignScale !== "rank" || !view || !aefLayers) return null;
    const kind = zoom < MOSAIC_MIN_ZOOM ? "coarse" : "zarr";
    const q = aefLayers.quantiles(view, timeIdx, [qLo, qHi], useWeather, kind);
    return q ? [q[0]!, Math.max(q[1]!, q[0]! + 0.05)] : null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ignScale, view, zoom, timeIdx, qLo, qHi, useWeather, aefLayers, aefVersion]);

  // ---- contours of the field: recomputed when the hour, the weather, the view or the lag changes
  const contourLines = useMemo<ContourPath[]>(() => {
    if (!view || contourStep <= 0) return [];
    const levels: number[] = [];
    for (let v = Math.max(contourStep, hideBelow); v <= 200; v += contourStep) levels.push(v);
    return ffwiContours(cacheRef.current, leadTimeIdx, view, levels);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, contourStep, hideBelow, leadTimeIdx, cacheVersion, tauHours]);

  // ---- layers
  const selection = useMemo(() => buildSelection(initTimeIdx), [initTimeIdx]);
  const getTileData = useMemo(
    () => (source ? makeGetTileData({ arrays: source.arrays, cache: cacheRef.current, windMode, variable: rasterVariable, initTimeIdx }) : null),
    [source, windMode, rasterVariable, initTimeIdx],
  );
  const renderTile = useCallback(
    (data: FfwiTileData) => {
      if (!colormapTexture) return { renderPipeline: [] };
      const isFfwi = rasterVariable === "ffwi";
      return makeRenderTile({
        layerIndex: leadTimeIdx,
        colormapTexture,
        colormapIndex,
        reversed: varDef.reversed,
        hideBelow: isFfwi ? hideBelow : Number.NEGATIVE_INFINITY,
        rescaleMin: varDef.min,
        rescaleMax: isFfwi ? rescaleMax : varDef.max,
        tauHours,
      })(data);
    },
    [leadTimeIdx, colormapTexture, colormapIndex, hideBelow, rescaleMax, rasterVariable, varDef, tauHours],
  );

  const [countyHover, setCountyHover] = useState<{ i: number; x: number; y: number } | null>(null);
  /** The hide line as a logit: what the raster lights, so the county's share is of the same pixels. */
  const zHideCounty = ignScale === "probability" ? Math.log(pHide / (1 - pHide)) : ignScale === "rank" && rankRamp ? rankRamp[0] : zLo;
  /** The counties in view ranked by this hour's share lit, for the panel's table. */
  const countyRows = useMemo(() => {
    if (fuel !== "counties" || !countyTable || !counties || !view) return [];
    const ch = countyTable.hour(Math.floor(timeIdx), zHideCounty, wxGate, weatherSrc, cacheVersion * 1e6 + weekVersion);
    const rows: { id: string; name: string; state: string; share: number; rate: number; starts: number; bbox: [number, number, number, number] }[] = [];
    counties.forEach((f, i) => {
      const b = f.properties.bbox;
      if (b[2] < view[0] || b[0] > view[2] || b[3] < view[1] || b[1] > view[3]) return;
      const sh = ch.share[i]!;
      if (!(sh > 0)) return;
      const rate = CountyTable.rate(ch.odds[i]!);
      rows.push({ id: f.id, name: f.properties.name, state: f.properties.state, share: sh, rate, starts: (rate * f.properties.km2) / 1000, bbox: b });
    });
    rows.sort((a, b) => b.share - a.share);
    return rows;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fuel, countyTable, counties, view, timeIdx, zHideCounty, wxGate, weatherSrc, cacheVersion, weekVersion, countyVersion]);
  const countyInfo = useMemo(() => {
    if (!countyHover || !countyTable || !counties) return null;
    const f = counties[countyHover.i];
    if (!f) return null;
    const ch = countyTable.hour(Math.floor(timeIdx), zHideCounty, wxGate, weatherSrc, cacheVersion * 1e6 + weekVersion);
    const share = ch.share[countyHover.i]!;
    const rate = share === share ? CountyTable.rate(ch.odds[countyHover.i]!) : NaN;
    return { f, share, rate, starts: (rate * f.properties.km2) / 1000, covered: countyTable.cells[countyHover.i]!.total > 0 };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [countyHover, countyTable, counties, timeIdx, zHideCounty, wxGate, weatherSrc, cacheVersion, weekVersion, countyVersion]);
  const pickCounty = useCallback((bbox: [number, number, number, number]) => {
    mapRef.current?.getMap().fitBounds([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], { padding: 80, duration: 900 });
  }, []);

  // ---- Overture buildings (buildings.ts: DuckDB-WASM via haybarn, GeoArrow outlines), from zoom 14 in. For now
  // every outline is the dark gray of "not in danger"; the GPU join to the ground under each outline is next
  const buildingsRef = useRef<ReturnType<typeof createBuildings> | null>(null);
  const [bTable, setBTable] = useState<arrow.Table | null>(null);
  const [buildingsApi, setBuildingsApi] = useState<Awaited<ReturnType<typeof createBuildings>> | null>(null);
  useEffect(() => {
    if (!view || zoom < BUILDINGS_MIN_ZOOM) return;
    buildingsRef.current ??= createBuildings();
    const ac = new AbortController();
    buildingsRef.current
      .then(async (b) => {
        setBuildingsApi(b);
        const t = await b.load(view, ac.signal);
        if (!ac.signal.aborted) setBTable(t);
      })
      .catch((e) => {
        if ((e as Error).name !== "AbortError") console.warn("buildings:", e);
      });
    return () => ac.abort();
  }, [view, zoom]);

  // ---- each building's outline from the burnable ground near it (building-exposure.ts): the highest P(fuel)
  // within the reach, nearer ground counting more; dark gray under the hide line. Recomputed a moment after the
  // buildings, the ground tiles, the reach or the ramp change.
  const [buildingReach, setBuildingReach] = useState(30);
  const [bColors, setBColors] = useState<{ colors: Uint8Array[]; key: number } | null>(null);
  useEffect(() => {
    if (!bTable || !buildingsApi || !aefLayers) return;
    const t = window.setTimeout(() => {
      const tiles = [...aefLayers.live.values()];
      const stripe = rampStripe(rampStops(groundRamp));
      const colors = bTable.batches.map((b) =>
        exposureColors(buildingExposure(buildingsApi.ringsOf(b), tiles, buildingReach), pHide, pMax, stripe, BUILDING_GRAY, ignScale === "logit"),
      );
      setBColors((c) => ({ colors, key: (c?.key ?? 0) + 1 }));
    }, 250);
    return () => clearTimeout(t);
  }, [bTable, buildingsApi, aefLayers, aefVersion, buildingReach, pHide, pMax, groundRamp, ignScale]);

  const buildLayers = useCallback((hourFrac: number) => {
    const out: unknown[] = [];
    if (showRaster && source && getTileData && colormapTexture) {
      out.push(
        new ZarrLayer<zarr.Readable, "float32", FfwiTileData>({
          id: `hrrr-zarr-${initTimeIdx}-${windMode}-${rasterVariable}`,
          node: source.arrays.temperature_2m,
          metadata: HRRR_GEOZARR_ATTRS,
          selection,
          getTileData,
          renderTile,
          maxRequests: 20,
          maxCacheSize: 12,
          opacity: rasterOpacity,
          updateTriggers: { renderTile: [leadTimeIdx, colormapIndex, hideBelow, rescaleMax, rasterVariable, tauHours] },
          beforeId: BEFORE_ID,
        } as never),
      );
    }
    if (aefLayers) {
      out.push(
        ...aefLayers.layers({
          mode: fuel === "cover" ? "cover" : ignitionLike ? "ignition" : "lit",
          zoom,
          teacher,
          leadTimeIdx: timeIdx + hourFrac,
          weather: weatherSrc,
          weatherKey,
          hideBelow,
          rescaleMax,
          pHide,
          pMax,
          scale: ignScale === "probability" ? "probability" : "logit",
          zLo: ignScale === "rank" ? (rankRamp ? rankRamp[0] : 99) : zLo,
          zHi: ignScale === "rank" ? (rankRamp ? rankRamp[1] : 100) : zHi,
          useWeather: WEATHER_ON && useWeather,
          standingAlpha,
          wxGate,
          fieldAlpha,
          glowHours,
          grain,
          light,
          opacity: fuelOpacity,
          ramp: groundRamp,
          visible: fuel !== "off" && fuel !== "counties",
          beforeId: BEFORE_ID,
        }),
      );
    }
    if (buildingsApi && bTable && zoom >= BUILDINGS_MIN_ZOOM) {
      const cs = bColors && bColors.colors.length === bTable.batches.length ? bColors : null;
      out.push(
        ...buildingsApi.layer({
          table: bTable,
          getLineColor: cs ? (_b, bi) => cs.colors[bi]! : [...BUILDING_GRAY, 255],
          colorKey: cs?.key ?? 0,
          beforeId: BEFORE_ID,
        }),
      );
    }
    if (fuel === "counties" && countyTable && counties) {
      const h0 = Math.floor(timeIdx);
      const wv = cacheVersion * 1e6 + weekVersion;
      const c0 = countyTable.hour(h0, zHideCounty, wxGate, weatherSrc, wv);
      const c1 = hourFrac > 0 && h0 + 1 < timeDepth ? countyTable.hour(h0 + 1, zHideCounty, wxGate, weatherSrc, wv) : null;
      out.push(
        countyLayer({
          features: counties,
          share0: c0.share,
          share1: c1 ? c1.share : null,
          frac: hourFrac,
          ramp: groundRamp,
          opacity: fuelOpacity,
          visible: true,
          beforeId: BEFORE_ID,
          trigger: [h0, hourFrac, countyVersion, wv, zHideCounty, wxGate, groundRamp],
          onHover: (i, x, y) => setCountyHover(i < 0 || !pickRef.current ? null : { i, x, y }),
        }),
      );
    }
    // if (runPerimeters.length > 0) out.push(perimeterLayer({ perimeters: runPerimeters, initTime: dateFromInitTimeIdx(initTimeIdx), leadHours: leadTimeIdx, visible: showPerimeters, beforeId: BEFORE_ID }));
    // if (fireList.length > 0) out.push(...fireLayers({ fires: fireList, initTime: dateFromInitTimeIdx(initTimeIdx), leadHours: leadTimeIdx, visible: showFires, beforeId: BEFORE_ID }));
    if (contourLines.length > 0) {
      out.push(
        new PathLayer<ContourPath>({
          id: "ffwi-contours",
          data: contourLines,
          getPath: (d: ContourPath) => d.path,
          getColor: (d: ContourPath) => [226, 236, 255, Math.min(255, 90 + d.level * 1.4)],
          getWidth: (d: ContourPath) => 0.6 + d.level / 60,
          widthUnits: "pixels",
          jointRounded: true,
          capRounded: true,
          beforeId: BEFORE_ID,
          updateTriggers: { getColor: [contourLines], getWidth: [contourLines] },
        } as never),
      );
    }
    return out;
    // aefVersion: the controller's tiles or weather changed under the same props
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    showRaster, source, getTileData, colormapTexture, initTimeIdx, windMode, rasterVariable, selection, renderTile, rasterOpacity,
    leadTimeIdx, colormapIndex, hideBelow, rescaleMax, tauHours, aefLayers, aefVersion, fuel, teacher, fuelOpacity, groundRamp, zoom,
    glowHours, grain, light, contourLines, pHide, pMax, ignScale, zLo, zHi, rankRamp, useWeather, standingAlpha, wxGate, fieldAlpha,
    timeIdx, weatherSrc, weatherKey, timeMode, countyTable, counties, countyVersion, cacheVersion, weekVersion, timeDepth, zHideCounty, wxGate,
    buildingsApi, bTable, bColors,
  ]);
  buildLayersRef.current = buildLayers;
  const deckLayers = useMemo(() => buildLayers(hourFracRef.current), [buildLayers]);

  const status: Status = {
    backend: source?.backend ?? null,
    weatherOn: WEATHER_ON,
    tilesLoaded: cacheRef.current.size,
    tilesTotal: Math.max(cacheRef.current.size, showRaster ? 24 : 0),
    aefIndexReady: aefIndex !== null && mosaic !== null,
    aefReading: aefLayers?.work.reading ?? 0,
    aefDone: aefLayers?.work.done ?? 0,
    weekDone: timeMode === "week" && week ? week.progress.done : 0,
    weekWanted: timeMode === "week" && week ? week.progress.wanted : 0,
    weekDaysIn: timeMode === "week" && week ? week.daysReady() : 0,
    weekDaysTotal: timeMode === "week" ? weekSpec.days : 0,
    error,
  };
  const loading = loadingSummary(status);

  /** The model score in words: low below the hide line, elevated from there, high in the top band. */
  const riskWord = (p: number, hide: number) => (!Number.isFinite(p) ? "unknown" : p < hide ? "low" : p < 0.8 ? "elevated" : "high");
  const tooltipStyle = (x: number, y: number): React.CSSProperties => ({
    position: "absolute",
    left: x + 14,
    top: y + 14,
    pointerEvents: "none",
    background: "rgba(20,20,24,0.92)",
    color: "#eee",
    fontSize: 12,
    lineHeight: 1.4,
    padding: "6px 9px",
    borderRadius: 6,
    zIndex: 2000,
    maxWidth: 300,
    whiteSpace: "normal",
  });

  const runLabel = source ? dateFromInitTimeIdx(initTimeIdx).toISOString().slice(0, 13).replace("T", " ") + "Z" : "";
  const weekHourLabel = weekHourDate(weekSpec, hourIdx).toISOString().slice(0, 13).replace("T", " ") + "Z";

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <MaplibreMap
        ref={mapRef}
        initialViewState={START_VIEW}
        hash
        mapStyle={MAP_STYLE_URL}
        attributionControl={false}
        // a flat, north-up map: no tilt or turn from a two-finger trackpad drag, right drag or touch
        maxPitch={0}
        dragRotate={false}
        pitchWithRotate={false}
        touchPitch={false}
        onLoad={(e) => {
          setZoom(e.target.getZoom());
          hideBasemapBuildings(e.target);
          addHillshade(e.target, relief);
          readView(e.target);
        }}
        onMove={(e) => setZoom(e.viewState.zoom)}
        onMoveEnd={(e) => readView(e.target)}
        onMouseMove={(e) => { if (pickRef.current) setHover({ lon: e.lngLat.lng, lat: e.lngLat.lat, x: e.point.x, y: e.point.y }); }}
        onMouseOut={() => setHover(null)}
      >
        <DeckGlOverlay layers={deckLayers as never} interleaved onDeviceInitialized={setDevice} overlayRef={overlayRef} />
      </MaplibreMap>
      {fuel === "lit" && !WEATHER_ON && zoom < MOSAIC_MIN_ZOOM ? (
        // the fuel head runs on the 10 m mosaic only; the coarse tiles below MOSAIC_MIN_ZOOM carry no fuel plane
        <div
          style={{
            position: "absolute",
            bottom: 28,
            left: "50%",
            transform: "translateX(-50%)",
            pointerEvents: "none",
            background: "rgba(20,20,24,0.88)",
            color: "#eee",
            fontSize: 13,
            padding: "8px 14px",
            borderRadius: 8,
            zIndex: 1500,
            textAlign: "center",
          }}
        >
          Zoom in past {MOSAIC_MIN_ZOOM} to see the fuel (now {zoom.toFixed(1)}).
          <br />
          <span style={{ color: "#aaa", fontSize: 12 }}>The zoomed-out view is not built yet.</span>
        </div>
      ) : null}
      {loading ? (
        <div
          style={{
            position: "absolute",
            top: 14,
            left: "50%",
            transform: "translateX(-50%)",
            pointerEvents: "none",
            background: "rgba(20,20,24,0.85)",
            color: "#eee",
            fontSize: 12,
            padding: "6px 12px",
            borderRadius: 999,
            zIndex: 1500,
            whiteSpace: "nowrap",
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          <span
            style={{
              width: 90,
              height: 4,
              borderRadius: 2,
              background: "rgba(255,255,255,0.18)",
              overflow: "hidden",
              display: "inline-block",
            }}
          >
            <span style={{ display: "block", height: "100%", width: `${Math.round(loading.fraction * 100)}%`, background: "#f6c343", transition: "width 300ms" }} />
          </span>
          {loading.label}
        </div>
      ) : null}
      <ControlPanel
        leadTimeIdx={leadTimeIdx}
        initTimeIdx={initTimeIdx}
        initTimeCount={source?.initTimeCount ?? 0}
        isPlaying={isPlaying}
        frameDurationMs={frameDurationMs}
        onInitTimeIdxChange={setInitTimeIdx}
        onLeadTimeIdxChange={setLeadTimeIdx}
        onPlayPauseToggle={() => setIsPlaying((p) => !p)}
        onFrameDurationMsChange={setFrameDurationMs}
        timeMode={timeMode}
        onTimeModeChange={setTimeMode}
        weekStart={weekStart}
        onWeekStartChange={setWeekStart}
        weekDays={weekDays}
        onWeekDaysChange={setWeekDays}
        hourIdx={hourIdx}
        onHourIdxChange={setHourIdx}
        weekDepth={timeMode === "week" ? timeDepth : weekDays * HOURS_PER_DAY}
        weekHourLabel={weekHourLabel}
        weekDayStates={timeMode === "week" && week ? week.dayStates() : []}
        fuel={fuel}
        onFuelChange={setFuel}
        countyRows={countyRows}
        countyProgress={countyTable && aefIndex ? [countyTable.filesDone, aefIndex.tiles.length] : null}
        onCountyPick={pickCounty}
        pHide={pHide}
        onPHideChange={setPHide}
        pMax={pMax}
        onPMaxChange={setPMax}
        ignScale={ignScale}
        onIgnScaleChange={setIgnScale}
        zLo={zLo}
        onZLoChange={setZLo}
        zHi={zHi}
        onZHiChange={setZHi}
        qLo={qLo}
        onQLoChange={setQLo}
        qHi={qHi}
        onQHiChange={setQHi}
        rankRamp={rankRamp}
        useWeather={useWeather}
        onUseWeatherChange={setUseWeather}
        standingAlpha={standingAlpha}
        onStandingAlphaChange={setStandingAlpha}
        wxGate={wxGate}
        fieldAlpha={fieldAlpha}
        onFieldAlphaChange={setFieldAlpha}
        onWxGateChange={setWxGate}
        wxNow={hoverInfo && Number.isFinite(hoverInfo.wx0) ? hoverInfo.wx0 : null}


        teacher={teacher}
        onTeacherChange={setTeacher}
        fuelOpacity={fuelOpacity}
        onFuelOpacityChange={setFuelOpacity}
        groundRamp={groundRamp}
        onGroundRampChange={setGroundRamp}
        glowHours={glowHours}
        onGlowHoursChange={setGlowHours}
        grain={grain}
        onGrainChange={setGrain}
        light={light}
        onLightChange={setLight}
        relief={relief}
        onReliefChange={setRelief}
        contourStep={contourStep}
        onContourStepChange={setContourStep}
        zoom={zoom}
        windMode={windMode}
        onWindModeChange={setWindMode}
        colormapId={colormapId}
        onColormapChange={setColormapId}
        rescaleMax={rescaleMax}
        onRescaleMaxChange={setRescaleMax}
        hideBelow={hideBelow}
        onHideBelowChange={setHideBelow}
        showRaster={showRaster}
        onShowRasterChange={setShowRaster}
        rasterVariable={rasterVariable}
        onRasterVariableChange={setRasterVariable}
        rasterOpacity={rasterOpacity}
        onRasterOpacityChange={setRasterOpacity}
        tauHours={tauHours}
        onTauHoursChange={setTauHours}
        status={status}
        weatherOn={WEATHER_ON}
        buildingReach={buildingReach}
        onBuildingReachChange={setBuildingReach}
      />
      {fuel === "counties" && countyHover && countyInfo ? (
        <div style={tooltipStyle(countyHover.x, countyHover.y)}>
          <div style={{ fontWeight: 600 }}>
            {countyInfo.f.properties.name}, {countyInfo.f.properties.state}
          </div>
          <div>
            {Number.isFinite(countyInfo.share)
              ? `elevated ignition risk over ${Math.round(countyInfo.share * 100)}% of its ground this hour (a share of area, not a chance of fire)`
              : countyInfo.covered
                ? "forecast not loaded here yet"
                : "ground not summed yet"}
          </div>
          <div style={{ color: "#bbb" }}>
            {Number.isFinite(countyInfo.rate)
              ? `about ${countyInfo.rate >= 1 ? countyInfo.rate.toFixed(1) : countyInfo.rate.toFixed(3)} ignitions per 1000 km² per day, ${(countyInfo.rate / CONUS_MEAN).toFixed(1)}x the CONUS mean · ${countyInfo.starts.toFixed(2)} starts expected today over ${Math.round(countyInfo.f.properties.km2).toLocaleString()} km²`
              : ""}
          </div>
          <div style={{ color: "#aaa" }}>
            {timeMode === "week" ? `${weekHourLabel} (12Z run, +${hourIdx % HOURS_PER_DAY} h)` : `${runLabel} +${leadTimeIdx} h`}
          </div>
        </div>
      ) : null}
      {fuel === "lit" && !WEATHER_ON && hover && hoverInfo ? (
        // the weather off: the fuel head's reading alone
        <div style={tooltipStyle(hover.x, hover.y)}>
          <div style={{ fontWeight: 600 }}>
            {hoverInfo.ground && Number.isFinite(hoverInfo.ground.fuel)
              ? `chance of fuel ${Math.round(hoverInfo.ground.fuel * 100)}%`
              : "no fuel reading here (zoom in past " + MOSAIC_MIN_ZOOM + ")"}
          </div>
          <div style={{ color: "#aaa" }}>
            {hover.lat.toFixed(3)}, {hover.lon.toFixed(3)}
          </div>
        </div>
      ) : fuel !== "counties" && hover && hoverInfo && (hoverInfo.className || Number.isFinite(hoverInfo.wx0)) ? (
        <div style={tooltipStyle(hover.x, hover.y)}>
          <div style={{ fontWeight: 600 }}>
            {hoverInfo.ground
              ? `ignition risk ${riskWord(hoverInfo.ignP, pHide)} this hour (score ${Number.isFinite(hoverInfo.ignP) ? `${Math.round(hoverInfo.ignP * 100)}%` : "?"}) · ground alone ${Math.round(hoverInfo.groundP * 100)}%${Number.isFinite(hoverInfo.wx0) ? ` · weather term ${hoverInfo.wx0 >= 0 ? "+" : ""}${hoverInfo.wx0.toFixed(1)}` : ""}`
              : "no embedding here"}
          </div>
          <div style={{ color: "#bbb" }}>
            {Number.isFinite(hoverInfo.rate) ? `about ${rateLabel(hoverInfo.rate)}` : ""}
          </div>
          <div>
            {hoverInfo.className ? `reads as ${hoverInfo.className}` : ""}
            {hoverInfo.ground ? ` · canopy ${Math.round(hoverInfo.ground.cover * 100)}%` : ""}
          </div>
          <div>
            {hoverInfo.wx && Number.isFinite(hoverInfo.wx.ffwi)
              ? `FFWI ${hoverInfo.wx.ffwi.toFixed(0)} · wind ${hoverInfo.wx.windMph} mph · RH ${hoverInfo.wx.rh}% · ${hoverInfo.wx.tc} °C`
              : Number.isFinite(hoverInfo.wx0)
                ? ""
                : "forecast not loaded here yet"}
            {tauHours > 0 ? ` (fuel lag ${tauHours} h)` : ""}
          </div>
          <div style={{ color: "#aaa" }}>
            {hover.lat.toFixed(3)}, {hover.lon.toFixed(3)} ·{" "}
            {timeMode === "week" ? `${weekHourLabel} (12Z run, +${hourIdx % HOURS_PER_DAY} h)` : `${runLabel} +${leadTimeIdx} h`}
          </div>
        </div>
      ) : null}
    </div>
  );
}
