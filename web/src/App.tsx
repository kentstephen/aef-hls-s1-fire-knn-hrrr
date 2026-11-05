import {
  COLORMAP_INDEX,
  createColormapTexture,
  decodeColormapSprite,
} from "@developmentseed/deck.gl-raster/gpu-modules";
import colormapsPngUrl from "@developmentseed/deck.gl-raster/gpu-modules/colormaps.png";
import { ZarrLayer } from "@developmentseed/deck.gl-zarr";
import { PathLayer, PolygonLayer } from "@deck.gl/layers";
import type { Device, Texture } from "@luma.gl/core";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MapboxOverlay } from "@deck.gl/mapbox";
import type { MapRef } from "react-map-gl/maplibre";
import type { StyleSpecification } from "maplibre-gl";
import { Map as MaplibreMap } from "react-map-gl/maplibre";
import type * as zarr from "zarrita";
import { fuelOf } from "./aef/fuel.js";
import { ignitionRate, rateLabel } from "./aef/calibration.js";
// import { fireLayers, firesInRun, loadFires, type Fires } from "./fires.js";
// the burn perimeters (perimeters.ts): not drawn for now; TODO draw them from GeoArrow (deck.gl-geoarrow) rather
// than a 7 MB GeoJSON parsed on the main thread, and only while each fire burned
// import { loadPerimeters, type Perimeter, perimeterLayer, perimetersInRun } from "./perimeters.js";
import { type BurnArea, burnAreaLayer, burnAreasAt, loadCaBurnAreas, selectedBurnLayer } from "./perimeters.js";
import { BurnHist, type YearHist } from "./fuel/burnHist.js";
import { FireHist } from "./ui/FireHist.js";
import { HEADS, NO_CLASS, type Teacher } from "./aef/head.js";
import { AefLayers } from "./aef/layers.js";
import { AEF_YEAR, type AefArray, MOSAIC_MIN_ZOOM, openAefMosaic, YEARS } from "./aef/mosaic.js";
import {
  daysIn, type FirePick, fireRecord, fuelNear, goesMask, loadSeasonPerimeters, loadSeasonStarts, SEASON_YEAR, seasonLayers, seasonMask,
  type SeasonPerimeter, type SeasonStarts,
} from "./season.js";
import { S2_YEARS, S2Imagery } from "./s2.js";
import { byHour, type FirePixel, type GoesDoc, goesLayers, loadGoesDoc, loadGoesPerimeter, readGoesFire } from "./goes.js";
import { fireOutline, goesBoxLayers, type GoesEvent, goesEvents, ringBox } from "./goesboxes.js";
import { FireReach } from "./reach.js";

/** The one fire GOES-West is shown for (web/public/goes/<slug>.json from scripts/goes_fire_refs.py). */
const GOES_FIRE = "green-2025";
/** GOES-West over all of California for a window (web/public/goes/<slug>.json, scripts/goes_fire_refs.py --region ca):
 * the fires as boxes. July 22 to Aug 2, 2024: the busiest week of 2023 to 2026 (the Park and Borel fires). */
const BOXES_DOC = "ca-2024-07-22";
const BOXES_YEAR = 2024;
/** All of California, where the boxes view opens. */
const CA_VIEW = { longitude: -119.4, latitude: 37.3, zoom: 5.6 };
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
/**
 * The fuel alone: no GOES boxes, no fire season, no buildings. The fuel is the claim; the fire tests that back it
 * are in the README. False brings the other views back.
 */
const FUEL_ONLY = true;
/** NIFC names come in capitals ("PARK"). */
const titleCase = (s: string) => s.toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
/** The embeddings' years the fuel can be read for: the fuel model reads each year with the two before it. */
const EMB_YEARS = YEARS.filter((y) => y - 2 >= YEARS[0]!);
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
// const START_VIEW = { longitude: -122.3917, latitude: 40.5865, zoom: 12.5 };
/** The fire the map opens on and plays: the Park Fire (Butte and Tehama counties, July 24, 2024 on). */
const START_FIRE = { name: "Park Fire", lon: -121.806, lat: 40.09 };
const START_VIEW = { longitude: START_FIRE.lon, latitude: START_FIRE.lat, zoom: 11 };
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

/** CARTO's own building footprints off (a backstop: loadMapStyle drops them first); the buildings are Overture's. */
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

/** The CARTO style without its own building footprints, so they never draw (the buildings are Overture's). */
async function loadMapStyle(): Promise<StyleSpecification | string> {
  try {
    const style = (await (await fetch(MAP_STYLE_URL)).json()) as StyleSpecification;
    style.layers = style.layers.filter((l) => !("source-layer" in l && l["source-layer"] === "buildings"));
    return style;
  } catch {
    return MAP_STYLE_URL;
  }
}

const pacificHour = (t: number) => new Date(t).toLocaleString("en-US", { timeZone: "America/Los_Angeles", month: "short", day: "numeric", hour: "numeric", timeZoneName: "short" });

/** An event in a line: when GOES first and last saw it (Pacific time, the fires' own), scans and pixels. */
function boxLine(ev: GoesEvent): string {
  const f = (t: number) => new Date(t).toLocaleString("en-US", { timeZone: "America/Los_Angeles", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
  const pt = new Date(ev.t0).toLocaleString("en-US", { timeZone: "America/Los_Angeles", timeZoneName: "short" }).split(" ").pop();
  return `${f(ev.t0)} to ${f(ev.t1)} ${pt} · ${ev.scans} scan${ev.scans === 1 ? "" : "s"} · ${ev.pixels} GOES pixel${ev.pixels === 1 ? "" : "s"}`;
}

export default function App() {
  const mapRef = useRef<MapRef>(null);
  /** The pair's left map (P): the imagery beside the fuel, moving together. */
  const leftRef = useRef<MapRef>(null);
  const [pair, setPair] = useState(false);
  /** Space held: the imagery in place of the fuel. */
  const [holding, setHolding] = useState(false);
  const holdingRef = useRef(false);
  holdingRef.current = holding;
  const stepImgYearRef = useRef<(d: number) => void>(() => {});
  /** [ and ]: the embeddings a year back or forward (the key handler is set up once). */
  const stepEmbYearRef = useRef<(d: number) => void>(() => {});
  /** The key handler is set up once; it turns the season on and off through this. */
  const setSeasonOnRef = useRef<React.Dispatch<React.SetStateAction<boolean>>>(() => {});
  /** The picked GOES box (Esc flies back out), read by the key handler set up once. */
  const boxPickRef = useRef<number | null>(null);
  const backOutRef = useRef<() => void>(() => {});
  const stepImgYear = (d: number) => stepImgYearRef.current(d);
  const [mapStyle, setMapStyle] = useState<StyleSpecification | string | null>(null);
  useEffect(() => {
    void loadMapStyle().then(setMapStyle);
  }, []);
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
  // California's burned areas since 2020 (NIFC WFIGS, live): outlines over the fuel
  const [burnAreas, setBurnAreas] = useState<BurnArea[] | null>(null);
  /** B: the burned areas shown or hidden. */
  const [burnOn, setBurnOn] = useState(true);
  /** The fire clicked: its fuel histograms in the panel. Esc or a click outside every fire clears it. */
  const [burnPick, setBurnPick] = useState<{ f: BurnArea; key: string; name: string; year: number | null; acres: number | null } | null>(null);
  const burnPickRef = useRef(burnPick);
  burnPickRef.current = burnPick;
  const [burnHists, setBurnHists] = useState<YearHist[] | null>(null);
  useEffect(() => {
    const ac = new AbortController();
    loadCaBurnAreas(ac.signal).then(setBurnAreas).catch((e) => { if (!ac.signal.aborted) setError(`perimeters: ${String(e)}`); });
    return () => ac.abort();
  }, []);
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
  /** The season view is on: Space plays its days instead of the forecast hours. */
  const seasonOnRef = useRef(false);
  const toggleSeasonPlayRef = useRef<() => void>(() => {});
  const goesOnRef = useRef(false);
  const toggleGoesPlayRef = useRef<() => void>(() => {});
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
      // Space held: the imagery; P (side by side) is off; [ ] step the embeddings' year
      // const pairKey = e.key === "p" || e.key === "P";
      const space = e.code === "Space" || e.key === " ";
      const pairKey = false;
      const yearKey = e.key === "[" ? -1 : e.key === "]" ? 1 : 0;
      const seasonKey = !FUEL_ONLY && (e.key === "s" || e.key === "S");
      const full = e.key === "f" || e.key === "F";
      const labels = e.key === "l" || e.key === "L";
      const pick = e.key === "q" || e.key === "Q";
      const ground = e.key === "g" || e.key === "G";
      const burn = e.key === "b" || e.key === "B";
      if (e.key === "Escape" && burnPickRef.current) { setBurnPick(null); return; }
      if (e.key === "Escape" && boxPickRef.current !== null) { backOutRef.current(); return; }
      if (!space && !pairKey && !yearKey && !seasonKey && !full && !labels && !pick && !ground && !burn) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const el = e.target as HTMLElement | null;
      const tag = el?.tagName;
      // typing goes to text fields and open dropdowns; a button, box or slider last clicked gives the key to the map
      const typing = tag === "TEXTAREA" || tag === "SELECT" || el?.isContentEditable ||
        (tag === "INPUT" && !["checkbox", "range", "radio", "button"].includes((el as HTMLInputElement).type));
      if (typing) return;
      if (tag === "BUTTON" || tag === "INPUT") el?.blur();
      e.preventDefault();
      // Space held: the imagery in place of the fuel (the season's and the weather's play keep their buttons)
      // if (space && goesOnRef.current) toggleGoesPlayRef.current();
      // else if (space && seasonOnRef.current) toggleSeasonPlayRef.current();
      // else if (space) setIsPlaying((p) => !p);
      if (space) { if (!e.repeat) setHolding(true); }
      else if (pairKey) { if (!e.repeat) setPair((v) => !v); }
      // [ and ]: the embeddings a year back or forward
      else if (yearKey) stepEmbYearRef.current(yearKey);
      // B: the burned areas on or off
      else if (burn) { if (!e.repeat) setBurnOn((v) => !v); }
      // S: the 2025 season on or off
      else if (seasonKey) { if (!e.repeat) setSeasonOnRef.current((v) => !v); }
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
    // letting go of Space (or leaving the window while holding it) brings the fuel back
    const onUp = (e: KeyboardEvent) => { if (e.code === "Space" || e.key === " ") setHolding(false); };
    const onBlur = () => setHolding(false);
    // while Space is held the wheel steps the imagery year instead of zooming (caught before the map sees it)
    let wheelAcc = 0, lastStep = 0;
    const onWheel = (e: WheelEvent) => {
      if (!holdingRef.current) return;
      e.preventDefault();
      e.stopPropagation();
      wheelAcc += e.deltaY;
      const now = performance.now();
      if (Math.abs(wheelAcc) >= 40 && now - lastStep > 140) {
        stepImgYear(wheelAcc > 0 ? 1 : -1);
        wheelAcc = 0;
        lastStep = now;
      }
    };
    // the browser's real full screen (F): the maps take the new size
    const onFull = () => { mapRef.current?.getMap().resize(); leftRef.current?.getMap().resize(); };
    window.addEventListener("keydown", onKey, { capture: true });
    window.addEventListener("keyup", onUp, { capture: true });
    window.addEventListener("blur", onBlur);
    window.addEventListener("wheel", onWheel, { capture: true, passive: false });
    document.addEventListener("fullscreenchange", onFull);
    return () => {
      window.removeEventListener("keydown", onKey, { capture: true });
      window.removeEventListener("keyup", onUp, { capture: true });
      window.removeEventListener("blur", onBlur);
      window.removeEventListener("wheel", onWheel, { capture: true });
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
  const [ignScale, setIgnScale] = useState<IgnScale>("logit");
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

  const [fuelOpacity, setFuelOpacity] = useState(FUEL_ONLY ? 1 : 0.9);
  const [groundRamp, setGroundRamp] = useState<GroundRamp>("viridis");
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
  // the load starts over only when the view settles (moveend), not on every zoom step while moving: the first
  // query takes ~30 s and each restart cancels it
  const buildingsZoom = !FUEL_ONLY && zoom >= BUILDINGS_MIN_ZOOM;
  useEffect(() => {
    if (!view || !buildingsZoom) return;
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
  }, [view, buildingsZoom]);

  // ---- each building's outline from the burnable ground near it (building-exposure.ts): the highest P(fuel)
  // within the reach, nearer ground counting more; dark gray under the hide line. Recomputed a moment after the
  // buildings, the ground tiles, the reach or the ramp change.
  const [buildingReach, setBuildingReach] = useState(30);
  const [bColors, setBColors] = useState<{ colors: Uint8Array[]; key: number } | null>(null);
  useEffect(() => {
    if (!bTable || !buildingsApi || !aefLayers) return;
    const t = window.setTimeout(() => {
      const tiles = aefLayers.fineTiles();
      const stripe = rampStripe(rampStops(groundRamp));
      const colors = bTable.batches.map((b) =>
        exposureColors(buildingExposure(buildingsApi.ringsOf(b), tiles, buildingReach), pHide, pMax, stripe, BUILDING_GRAY, ignScale === "logit"),
      );
      setBColors((c) => ({ colors, key: (c?.key ?? 0) + 1 }));
    }, 250);
    return () => clearTimeout(t);
  }, [bTable, buildingsApi, aefLayers, aefVersion, buildingReach, pHide, pMax, groundRamp, ignScale]);

  // ---- the fire season (season.ts): the year's California starts and burn perimeters, read from NIFC when the
  // view opens, over the fuel of the year before. A start whose 30 m circle holds fuel above the hide line lights
  // the ground in that circle; the burns light the ground inside them; the rest of the fuel stays dim.
  const [seasonOn, setSeasonOn] = useState(false);
  setSeasonOnRef.current = setSeasonOn;
  const seasonDays = daysIn(SEASON_YEAR);
  const [seasonDay, setSeasonDay] = useState(seasonDays - 1);
  const [seasonPlaying, setSeasonPlaying] = useState(false);
  const [seasonReach, setSeasonReach] = useState(30);
  const [seasonData, setSeasonData] = useState<{ starts: SeasonStarts; perims: SeasonPerimeter[] } | null>(null);
  const [seasonLoading, setSeasonLoading] = useState(false);
  const [seasonHover, setSeasonHover] = useState<{ i: number; x: number; y: number } | null>(null);
  useEffect(() => {
    seasonOnRef.current = seasonOn;
    if (!seasonOn) setSeasonPlaying(false);
  }, [seasonOn]);
  useEffect(() => {
    if (!seasonOn || seasonData) return;
    const ac = new AbortController();
    setSeasonLoading(true);
    Promise.all([loadSeasonStarts(SEASON_YEAR, ac.signal), loadSeasonPerimeters(SEASON_YEAR, ac.signal)])
      .then(([starts, perims]) => setSeasonData({ starts, perims }))
      .catch((e) => { if ((e as Error).name !== "AbortError") setError(`fire season: ${String(e)}`); })
      .finally(() => setSeasonLoading(false));
    return () => ac.abort();
  }, [seasonOn, seasonData]);
  // the season's fuel is the year before it; otherwise the newest year
  // ---- one fire from orbit (goes.ts): GOES-West's fire pixels over the fire, hour by hour, read from NOAA's bucket
  const [goesOn, setGoesOn] = useState(false);
  const [goesDoc, setGoesDoc] = useState<GoesDoc | null>(null);
  const [goesPerim, setGoesPerim] = useState<[number, number][][][]>([]);
  const goesPixelsRef = useRef<FirePixel[]>([]);
  const [goesRead, setGoesRead] = useState(0);
  const [goesHour, setGoesHour] = useState(0);
  const [goesPlaying, setGoesPlaying] = useState(false);
  useEffect(() => {
    goesOnRef.current = goesOn;
    if (!goesOn) setGoesPlaying(false);
    if (goesOn) setSeasonOn(false);
  }, [goesOn]);
  useEffect(() => {
    if (seasonOn) setGoesOn(false);
  }, [seasonOn]);
  // the reference file once, then the perimeter and the scans (read once per session; turning the view off and on
  // again keeps what was read)
  useEffect(() => {
    if (!goesOn || goesDoc) return;
    let gone = false;
    loadGoesDoc(`${import.meta.env.BASE_URL}goes/${GOES_FIRE}.json`)
      .then((d) => { if (!gone) setGoesDoc(d); })
      .catch((e) => setError(`GOES: ${String(e)}`));
    return () => { gone = true; };
  }, [goesOn, goesDoc]);
  useEffect(() => {
    if (!goesOn || !goesDoc) return;
    const [w, s, e, n] = goesDoc.fire.bbox;
    mapRef.current?.getMap().fitBounds([[w, s], [e, n]], { padding: 60, duration: 900 });
  }, [goesOn, goesDoc]);
  const goesStarted = useRef(false);
  useEffect(() => {
    if (!goesDoc || goesStarted.current) return;
    goesStarted.current = true;
    loadGoesPerimeter(goesDoc.fire.objectid).then(setGoesPerim).catch((e) => console.warn("GOES perimeter:", e));
    // the scans stream in; the counter re-renders at most every 300 ms
    let timer = 0;
    let read = 0;
    readGoesFire(goesDoc, (_i, px) => {
      read++;
      if (px.length) goesPixelsRef.current.push(...px);
      if (!timer) timer = window.setTimeout(() => { timer = 0; setGoesRead(read); }, 300);
    })
      .then(() => setGoesRead(read))
      .catch((e) => setError(`GOES: ${String(e)}`));
  }, [goesDoc]);
  const goesT0 = goesDoc ? Math.floor(goesDoc.scans[0]!.t / 3600e3) * 3600e3 : 0;
  const goesHours = goesDoc ? Math.floor((goesDoc.scans[goesDoc.scans.length - 1]!.t - goesT0) / 3600e3) + 1 : 1;
  const goesBins = useMemo(() => byHour(goesPixelsRef.current, goesT0), [goesRead, goesT0]); // eslint-disable-line react-hooks/exhaustive-deps
  const goesNow = goesBins.get(goesHour) ?? [];
  const goesSeen = useMemo(() => {
    const m = new Map<string, FirePixel>();
    for (const [h, px] of goesBins) if (h <= goesHour) for (const p of px) m.set(p.key, p);
    return [...m.values()];
  }, [goesBins, goesHour]);
  const toggleGoesPlay = useCallback(() => {
    if (!goesPlaying && goesHour >= goesHours - 1) setGoesHour(0);
    setGoesPlaying(!goesPlaying);
  }, [goesPlaying, goesHour, goesHours]);
  toggleGoesPlayRef.current = toggleGoesPlay;
  useEffect(() => {
    if (!goesPlaying) return;
    const t = window.setInterval(() => setGoesHour((h) => Math.min(h + 1, goesHours - 1)), 150);
    return () => clearInterval(t);
  }, [goesPlaying, goesHours]);
  useEffect(() => {
    if (goesPlaying && goesHour >= goesHours - 1) setGoesPlaying(false);
  }, [goesPlaying, goesHour, goesHours]);
  // the ground inside the final perimeter that GOES has seen burning by this hour lights up
  useEffect(() => {
    if (!aefLayers || !goesOn || !view) return;
    const seenKey = goesSeen.map((p) => p.key).sort().join(",");
    let changed = false;
    for (const t of aefLayers.fineTiles()) {
      const [w, s, e, n] = t.bbox;
      if (e < view[0] || w > view[2] || n < view[1] || s > view[3]) continue;
      const fp = goesSeen.filter((p) => p.polygon.some(([x, y]) => x >= w - 0.05 && x <= e + 0.05 && y >= s - 0.05 && y <= n + 0.05));
      const key = `goes|${goesPerim.length}|${fp.length ? seenKey : ""}`;
      if (t.seasonKey === key) continue;
      changed = aefLayers.setSeasonMask(t, key, goesMask(t, fp.map((p) => p.polygon), goesPerim)) || changed;
    }
    if (changed) aefLayers.seasonChanged();
    // aefVersion: new tiles need their mask
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aefLayers, aefVersion, goesOn, goesSeen, goesPerim, view]);

  // ---- GOES boxes (goesboxes.ts): every fire GOES-West saw over California in the window, as boxes; clicking one
  // flies to it and lights the fuel inside it and in a ring around it, the rest dim
  const [boxesOn, setBoxesOn] = useState(!FUEL_ONLY);
  const [boxesDoc, setBoxesDoc] = useState<GoesDoc | null>(null);
  const boxPixelsRef = useRef<FirePixel[]>([]);
  const [boxesRead, setBoxesRead] = useState(0);
  const [boxPick, setBoxPick] = useState<number | null>(null);
  const [boxHover, setBoxHover] = useState<{ ev: GoesEvent; x: number; y: number } | null>(null);
  boxPickRef.current = boxPick;
  useEffect(() => {
    if (boxesOn) { setSeasonOn(false); setGoesOn(false); }
    else setBoxPick(null);
  }, [boxesOn]);
  useEffect(() => {
    if (seasonOn || goesOn) setBoxesOn(false);
  }, [seasonOn, goesOn]);
  const boxesStarted = useRef(false);
  useEffect(() => {
    if (!boxesOn || boxesStarted.current) return;
    boxesStarted.current = true;
    loadGoesDoc(`${import.meta.env.BASE_URL}goes/${BOXES_DOC}.json`)
      .then((d) => {
        setBoxesDoc(d);
        // the scans stream in; the events are joined again at most every 500 ms
        let timer = 0;
        let read = 0;
        return readGoesFire(d, (_i, px) => {
          read++;
          if (px.length) boxPixelsRef.current.push(...px);
          if (!timer) timer = window.setTimeout(() => { timer = 0; setBoxesRead(read); }, 500);
        }, undefined, 12, false).then(() => setBoxesRead(read));
      })
      .catch((e) => setError(`GOES boxes: ${String(e)}`));
  }, [boxesOn]);
  const boxEvents = useMemo(() => goesEvents(boxPixelsRef.current), [boxesRead]); // eslint-disable-line react-hooks/exhaustive-deps
  const boxPicked = boxPick === null ? null : boxEvents.find((ev) => ev.id === boxPick) ?? null;
  // once every scan is read, the opening fire is picked and plays (the biggest event over its spot)
  const autoPicked = useRef(false);
  // fly to a box: the box and its ring fill the view, never zoomed out past where the fuel draws
  const flyToBox = useCallback((ev: GoesEvent) => {
    const m = mapRef.current?.getMap();
    if (!m) return;
    setBoxPick(ev.id);
    setBoxHover(null);
    reachRef.current?.dispose();
    const r = new FireReach(ev.px, ev.bbox);
    reachRef.current = r;
    setReach(r);
    setBoxHour(0);
    setBoxPlaying(false);
    autoStartRef.current = true;
    // play once the fuel is read under a quarter of the grid, or after 25 s at the latest
    window.setTimeout(() => { if (autoStartRef.current && reachRef.current === r) { autoStartRef.current = false; setBoxPlaying(true); } }, 25000);
    // the whole fire in view, down to zoom 10 (its area is all that is read)
    const [w, s, e, n] = ringBox(ev.bbox);
    const cam = m.cameraForBounds([[w, s], [e, n]], { padding: 40 });
    const zoom = Math.max(MOSAIC_MIN_ZOOM - 1, Math.min(14, cam?.zoom ?? MOSAIC_MIN_ZOOM));
    m.flyTo({ center: [(w + e) / 2, (s + n) / 2], zoom, duration: 1800, essential: true });
  }, []);
  useEffect(() => {
    if (autoPicked.current || !boxesDoc || boxesRead < boxesDoc.scans.length || boxPick !== null) return;
    autoPicked.current = true;
    const ev = boxEvents
      .filter((e) => START_FIRE.lon >= e.bbox[0] && START_FIRE.lon <= e.bbox[2] && START_FIRE.lat >= e.bbox[1] && START_FIRE.lat <= e.bbox[3])
      .sort((a, b) => b.pixels - a.pixels)[0];
    if (ev) flyToBox(ev);
  }, [boxesDoc, boxesRead, boxEvents, boxPick, flyToBox]);
  backOutRef.current = () => {
    setBoxPick(null);
    setBoxPlaying(false);
    reachRef.current?.dispose();
    reachRef.current = null;
    setReach(null);
    setReachLines([]);
    mapRef.current?.getMap().flyTo({ center: [CA_VIEW.longitude, CA_VIEW.latitude], zoom: CA_VIEW.zoom, duration: 1500, essential: true });
  };
  // ---- the picked fire hour by hour (reach.ts): its burning pixels, and the fuel lit as far as the fire could reach
  // from them (slope and fuel), more the longer a pixel burns; the rest of the fuel gray
  const reachRef = useRef<FireReach | null>(null);
  /** Play once the fuel under the fire has been read (the flight over, the tiles in). */
  const autoStartRef = useRef(false);
  const [reach, setReach] = useState<FireReach | null>(null);
  const [boxHour, setBoxHour] = useState(0);
  const [boxPlaying, setBoxPlaying] = useState(false);
  const [reachLines, setReachLines] = useState<{ level: number; rings: [number, number][][] }[]>([]);
  /** The hour whose reach is drawn (the masks and lines), and how long the last one took. */
  const [reachDone, setReachDone] = useState<{ hour: number; ms: number } | null>(null);
  // the fuel under the grid, again as new 10 m tiles come in (at most every 1.5 s)
  useEffect(() => {
    if (!reach || !aefLayers) return;
    const t = window.setTimeout(() => {
      const share = reach.setFuel(aefLayers.fineTiles());
      if (autoStartRef.current && share > 0.25) { autoStartRef.current = false; setBoxPlaying(true); }
    }, 1500);
    return () => clearTimeout(t);
  }, [reach, aefLayers, aefVersion]);
  // an hour's reach: computed in the worker, then laid on the tiles
  useEffect(() => {
    if (!reach) return;
    let gone = false;
    void reach.run(boxHour).then(({ lines, ms }) => {
      if (gone) return;
      setReachLines(lines);
      setReachDone({ hour: boxHour, ms });
    });
    return () => { gone = true; };
  }, [reach, boxHour]);
  useEffect(() => {
    if (!aefLayers || !boxesOn) return;
    const key = reach && reachDone ? `reach|${boxPick}|${reachDone.hour}` : "reach|none";
    let changed = false;
    for (const t of aefLayers.fineTiles()) {
      if (t.seasonKey === key) continue;
      changed = aefLayers.setSeasonMask(t, key, reach && reachDone ? reach.mask(t) : null) || changed;
    }
    if (changed) aefLayers.seasonChanged();
    // aefVersion: new tiles need their mask
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aefLayers, aefVersion, boxesOn, reach, reachDone]);
  // playing: the next hour once this one is drawn, about 3 hours a second
  useEffect(() => {
    if (!boxPlaying || !reach || !reachDone || reachDone.hour !== boxHour) return;
    if (boxHour >= reach.hours - 1) { setBoxPlaying(false); return; }
    const t = window.setTimeout(() => setBoxHour((h) => h + 1), 330);
    return () => clearTimeout(t);
  }, [boxPlaying, reach, reachDone, boxHour]);
  const toggleBoxPlay = useCallback(() => {
    if (!reach) return;
    if (!boxPlaying && boxHour >= reach.hours - 1) setBoxHour(0);
    setBoxPlaying(!boxPlaying);
  }, [reach, boxPlaying, boxHour]);
  const boxNow = reach?.byHour.get(boxHour) ?? [];
  // this hour's burning pixels as one outline: edges shared by two touching pixels drop out
  const boxNowEdges = useMemo(() => fireOutline(boxNow), [boxNow]);

  // ---- the years on screen: the embeddings the fuel is read from (the newest of the fuel model's three years) and
  // the Sentinel-2 mosaic's. The season view puts the embeddings on the year before its fires; the imagery follows
  // the embeddings' year until it is picked itself
  const [embYear, setEmbYear] = useState(AEF_YEAR);
  // the selected fire's histograms: the chunks in view, the year on screen first, then outward
  const burnHist = useMemo(() => (aefLayers && poolRef.current ? new BurnHist(poolRef.current) : null), [aefLayers]);
  useEffect(() => {
    setBurnHists(null);
    if (!burnPick || !burnHist || !view || zoom < MOSAIC_MIN_ZOOM) return;
    const ac = new AbortController();
    void burnHist.run(burnPick.f, burnPick.key, view, embYear, setBurnHists, ac.signal);
    return () => ac.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [burnPick, burnHist, view, embYear]);
  stepEmbYearRef.current = (d: number) =>
    setEmbYear((y) => EMB_YEARS[Math.max(0, Math.min(EMB_YEARS.length - 1, EMB_YEARS.indexOf(y) + d))]!);
  useEffect(() => {
    setEmbYear(boxesOn ? BOXES_YEAR - 1 : seasonOn || goesOn ? SEASON_YEAR - 1 : AEF_YEAR);
  }, [seasonOn, goesOn, boxesOn]);
  useEffect(() => {
    aefLayers?.setFuelYear(embYear);
  }, [aefLayers, embYear]);
  const [imgYear, setImgYear] = useState(Math.min(Math.max(AEF_YEAR, S2_YEARS[0]!), S2_YEARS[S2_YEARS.length - 1]!));
  // [ ] and the wheel while holding: a year back or forward, stopping at the ends
  stepImgYearRef.current = (d: number) =>
    setImgYear((y) => S2_YEARS[Math.max(0, Math.min(S2_YEARS.length - 1, S2_YEARS.indexOf(y) + d))]!);
  useEffect(() => {
    setImgYear(Math.min(Math.max(embYear, S2_YEARS[0]!), S2_YEARS[S2_YEARS.length - 1]!));
  }, [embYear]);
  /** The imagery kept on in place of the fuel (the panel's box), as holding Space does for a moment. */
  const [imgOn, setImgOn] = useState(false);
  const showImg = holding || imgOn;
  const [s2Version, setS2Version] = useState(0);
  const s2 = useMemo(() => new S2Imagery(() => setS2Version((v) => v + 1)), []);
  // the year's footprints under the view, looked for as the view settles
  useEffect(() => {
    if (view) s2.discover(imgYear, view, zoom);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s2, view, imgYear]);
  // ---- a clicked fire: its whole NIFC record for the panel
  const [firePick, setFirePick] = useState<FirePick | null>(null);
  const [fireRec, setFireRec] = useState<Record<string, unknown> | null>(null);
  useEffect(() => {
    setFireRec(null);
    if (!firePick) return;
    const ac = new AbortController();
    fireRecord(firePick, ac.signal)
      .then(setFireRec)
      .catch((e) => { if ((e as Error).name !== "AbortError") setError(`fire record: ${String(e)}`); });
    return () => ac.abort();
  }, [firePick]);
  // play: a day every 140 ms (the year in about 50 s); from the last day it starts over at Jan 1
  const toggleSeasonPlay = useCallback(() => {
    if (!seasonPlaying && seasonDay >= seasonDays - 1) setSeasonDay(0);
    setSeasonPlaying(!seasonPlaying);
  }, [seasonPlaying, seasonDay, seasonDays]);
  toggleSeasonPlayRef.current = toggleSeasonPlay;
  useEffect(() => {
    if (!seasonPlaying) return;
    const t = window.setInterval(() => setSeasonDay((d) => Math.min(d + 1, seasonDays - 1)), 140);
    return () => clearInterval(t);
  }, [seasonPlaying, seasonDays]);
  useEffect(() => {
    if (seasonPlaying && seasonDay >= seasonDays - 1) setSeasonPlaying(false);
  }, [seasonPlaying, seasonDay, seasonDays]);
  /** How far into the starts (sorted by day) the day reaches. */
  const seasonCount = useMemo(() => {
    if (!seasonData) return 0;
    const d = seasonData.starts.day;
    let a = 0, b = seasonData.starts.n;
    while (a < b) {
      const m = (a + b) >> 1;
      if (d[m]! <= seasonDay) a = m + 1;
      else b = m;
    }
    return a;
  }, [seasonData, seasonDay]);
  // each start's fuel within the reach, from the 10 m tiles on hand: the ones in view read again as tiles land,
  // the rest keep their last reading (NaN until read)
  const seasonFuelRef = useRef<{ key: string; fuel: Float32Array } | null>(null);
  const seasonJoin = useMemo(() => {
    if (!seasonOn || !seasonData || !aefLayers) return null;
    const st = seasonData.starts;
    const key = `${aefLayers.year}|${seasonReach}`;
    if (!seasonFuelRef.current || seasonFuelRef.current.key !== key || seasonFuelRef.current.fuel.length !== st.n) {
      seasonFuelRef.current = { key, fuel: new Float32Array(st.n).fill(NaN) };
    }
    const all = seasonFuelRef.current.fuel;
    const tiles = zoom >= MOSAIC_MIN_ZOOM ? aefLayers.fineTiles() : [];
    const shown = new Int32Array(seasonCount);
    const fuel = new Float32Array(seasonCount);
    let inView = 0, found = 0, unread = 0;
    for (let i = 0; i < seasonCount; i++) {
      shown[i] = i;
      const lon = st.lon[i]!, lat = st.lat[i]!;
      const vis = !!view && lon >= view[0] && lon <= view[2] && lat >= view[1] && lat <= view[3];
      if (vis && tiles.length) {
        const f = fuelNear(lon, lat, seasonReach, tiles);
        if (!Number.isNaN(f)) all[i] = f;
      }
      fuel[i] = all[i]!;
      if (vis) {
        inView++;
        if (Number.isNaN(fuel[i]!)) unread++;
        else if (fuel[i]! >= pHide) found++;
      }
    }
    return { shown, fuel, inView, found, unread };
    // aefVersion: tiles landed
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seasonOn, seasonData, aefLayers, aefVersion, view, zoom, seasonCount, seasonReach, pHide]);
  const seasonPerims = useMemo(
    () => (seasonData ? seasonData.perims.filter((p) => p.day <= seasonDay) : []),
    [seasonData, seasonDay],
  );
  // each tile's mask: the circles of the starts that found fuel and the burns that touch it; redrawn only when the
  // set touching the tile changes
  useEffect(() => {
    if (!aefLayers || !seasonOn || !seasonData || !seasonJoin || !view) return;
    const st = seasonData.starts;
    const padLat = seasonReach / 111_320;
    let changed = false;
    for (const t of aefLayers.fineTiles()) {
      const [w, s, e, n] = t.bbox;
      if (e < view[0] || w > view[2] || n < view[1] || s > view[3]) continue;
      const padLon = padLat / Math.cos((((s + n) / 2) * Math.PI) / 180);
      const lit: number[] = [];
      for (let k = 0; k < seasonJoin.shown.length; k++) {
        const f = seasonJoin.fuel[k]!;
        if (!(f >= pHide)) continue;
        const i = seasonJoin.shown[k]!;
        const lon = st.lon[i]!, lat = st.lat[i]!;
        if (lon < w - padLon || lon > e + padLon || lat < s - padLat || lat > n + padLat) continue;
        lit.push(i);
      }
      const per: number[] = [];
      seasonPerims.forEach((p, j) => {
        if (p.bbox[2] < w || p.bbox[0] > e || p.bbox[3] < s || p.bbox[1] > n) return;
        per.push(j);
      });
      const key = `${seasonReach}|${lit.join(",")}|${per.join(",")}`;
      if (t.seasonKey === key) continue;
      const mask = seasonMask(t, lit.map((i) => [st.lon[i]!, st.lat[i]!]), seasonReach, per.map((j) => seasonPerims[j]!));
      changed = aefLayers.setSeasonMask(t, key, mask) || changed;
    }
    if (changed) aefLayers.seasonChanged();
  }, [aefLayers, seasonOn, seasonData, seasonJoin, seasonPerims, seasonReach, pHide, view]);

  const buildLayers = useCallback((hourFrac: number) => {
    const out: unknown[] = [];
    // the imagery bare: over the basemap's streets and labels (no beforeId puts it on top), nothing drawn over it
    if (view) out.push(...s2.layers({ year: imgYear, view, visible: showImg, prefix: "" }));
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
          opacity: boxesOn ? 1 : fuelOpacity,
          ramp: groundRamp,
          visible: fuel !== "off" && fuel !== "counties" && !showImg,
          // the GOES fires view: every fuel pixel at full strength, no dimming or gray around the fire
          seasonDim: (seasonOn && seasonData) || goesOn ? 0.18 : 0,
          seasonGray: false,
          // every fuel pixel at full strength (the GOES view and the fuel alone)
          solid: boxesOn || FUEL_ONLY,
          // a picked fire: its whole area read, so it can be seen whole from zoom 10 (nothing outside it is read)
          only: boxesOn && boxPicked && reach ? { bbox: reach.bbox(), minZoom: MOSAIC_MIN_ZOOM - 1 } : undefined,
          beforeId: BEFORE_ID,
        }),
      );
    }
    // the burned areas over the fuel; the imagery stays bare
    if (burnAreas) out.push(burnAreaLayer({ areas: burnAreas, visible: burnOn && !showImg, beforeId: BEFORE_ID }));
    if (burnPick) out.push(selectedBurnLayer({ area: burnPick.f, visible: burnOn && !showImg, beforeId: BEFORE_ID }));
    if (!FUEL_ONLY && buildingsApi && bTable && zoom >= BUILDINGS_MIN_ZOOM && !showImg) {
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
    if (seasonOn && seasonData && seasonJoin && !showImg) {
      out.push(
        ...seasonLayers({
          starts: seasonData.starts,
          shown: seasonJoin.shown,
          fuel: seasonJoin.fuel,
          perims: seasonPerims,
          day: seasonDay,
          reachM: seasonReach,
          pHide,
          beforeId: BEFORE_ID,
          onHover: (i, x, y) => setSeasonHover(i < 0 || !pickRef.current ? null : { i, x, y }),
          onClick: setFirePick,
        }),
      );
    }
    if (goesOn && goesDoc && !showImg) {
      out.push(...goesLayers({ now: goesNow, seen: goesSeen, perimeter: goesPerim, beforeId: BEFORE_ID, trigger: [goesHour, goesRead] }));
    }
    if (boxesOn && !showImg && boxPicked) {
      // the reach lines: off (the fuel's glow carries the reach on its own)
      // out.push(new PathLayer<[number, number][]>({
      //   id: "reach-lines",
      //   data: reachLines.flatMap((l) => l.rings),
      //   getPath: (d: [number, number][]) => d,
      //   getColor: [255, 255, 255, 110],
      //   getWidth: 1,
      //   widthUnits: "pixels",
      //   beforeId: BEFORE_ID,
      // } as never));
      out.push(new PolygonLayer<FirePixel>({
        id: "reach-fire-fill",
        data: boxNow,
        getPolygon: (d: FirePixel) => d.polygon,
        filled: true,
        stroked: false,
        getFillColor: [255, 150, 30, 90],
        beforeId: BEFORE_ID,
      } as never));
      out.push(new PathLayer<[number, number][]>({
        id: "reach-fire-now",
        data: boxNowEdges,
        getPath: (d: [number, number][]) => d,
        // orange: apart from viridis (no orange in it) and from the gray fuel
        getColor: [255, 150, 30, 255],
        getWidth: 2,
        widthUnits: "pixels",
        beforeId: BEFORE_ID,
      } as never));
    }
    if (boxesOn && !showImg && !boxPicked) {
      out.push(...goesBoxLayers({
        events: boxEvents,
        picked: boxPick,
        beforeId: BEFORE_ID,
        onHover: (ev, x, y) => setBoxHover(ev && pickRef.current ? { ev, x, y } : null),
        onClick: flyToBox,
      }));
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
    buildingsApi, bTable, bColors, seasonOn, seasonData, seasonJoin, seasonPerims, seasonDay, seasonReach,
    s2, s2Version, imgYear, showImg, view, burnAreas, burnOn, burnPick,
    goesOn, goesDoc, goesNow, goesSeen, goesPerim, goesHour, goesRead,
    boxesOn, boxEvents, boxPick, flyToBox, boxPicked, reachLines, boxNowEdges, boxNow, reach,
  ]);
  buildLayersRef.current = buildLayers;
  const deckLayers = useMemo(() => buildLayers(hourFracRef.current), [buildLayers]);
  // the pair's left map: the imagery bare, over the basemap's streets and labels, nothing drawn over it
  // (the fires over the imagery: not for now; seasonLayers takes a prefix for this map when they come back)
  const leftLayers = useMemo(() => {
    if (!pair || !view) return [];
    return s2.layers({ year: imgYear, view, visible: true, prefix: "L-" });
  }, [pair, view, s2, s2Version, imgYear]);
  // the two maps move together: each one's move set on the other (the flag keeps the echo from bouncing back)
  const syncing = useRef(false);
  const follow = (from: import("maplibre-gl").Map, to: MapRef | null) => {
    if (!pair || syncing.current || !to) return;
    syncing.current = true;
    to.getMap().jumpTo({ center: from.getCenter(), zoom: from.getZoom() });
    syncing.current = false;
  };
  // a new size for both maps when the pair opens or closes
  useEffect(() => {
    const r = requestAnimationFrame(() => {
      mapRef.current?.getMap().resize();
      const l = leftRef.current?.getMap();
      const m = mapRef.current?.getMap();
      if (l && m) l.jumpTo({ center: m.getCenter(), zoom: m.getZoom() });
    });
    return () => cancelAnimationFrame(r);
  }, [pair]);

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
  // x, y come from the main map (its own corner), which sits on the right half while the pair is open
  const mapLeft = pair ? (mapRef.current?.getContainer().getBoundingClientRect().left ?? 0) : 0;
  const tooltipStyle = (x: number, y: number): React.CSSProperties => ({
    position: "absolute",
    left: x + mapLeft + 14,
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
      {pair && mapStyle ? (
        <div style={{ position: "absolute", top: 0, bottom: 0, left: 0, width: "50%", borderRight: "1px solid rgba(255,255,255,0.25)" }}>
          <MaplibreMap
            ref={leftRef}
            initialViewState={mapRef.current ? { longitude: mapRef.current.getCenter().lng, latitude: mapRef.current.getCenter().lat, zoom: mapRef.current.getZoom() } : START_VIEW}
            mapStyle={mapStyle}
            attributionControl={false}
            maxPitch={0}
            dragRotate={false}
            pitchWithRotate={false}
            touchPitch={false}
            style={{ width: "100%", height: "100%" }}
            onLoad={(e) => hideBasemapBuildings(e.target)}
            onMove={(e) => follow(e.target, mapRef.current)}
          >
            <DeckGlOverlay layers={leftLayers as never} interleaved />
          </MaplibreMap>
          <div className="pair-tag" style={{ position: "absolute", top: 12, right: 12 }}>Sentinel-2 {imgYear}</div>
        </div>
      ) : null}
      {mapStyle ? (
      <div style={{ position: "absolute", top: 0, bottom: 0, right: 0, left: pair ? "50%" : 0 }}>
      <MaplibreMap
        ref={mapRef}
        style={{ width: "100%", height: "100%" }}
        initialViewState={START_VIEW}
        hash
        mapStyle={mapStyle}
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
        onMove={(e) => { setZoom(e.viewState.zoom); follow(e.target, leftRef.current); }}
        onMoveEnd={(e) => readView(e.target)}
        onClick={(e) => {
          if (!burnOn || !burnAreas) return;
          const hit = burnAreasAt(burnAreas, e.lngLat.lng, e.lngLat.lat)[0];
          setBurnPick(hit ? { f: hit.f, key: hit.f.properties.attr_IrwinID ?? `${hit.name}:${hit.year}`, name: titleCase(hit.name), year: hit.year, acres: hit.acres } : null);
        }}
        onMouseMove={(e) => { if (pickRef.current) setHover({ lon: e.lngLat.lng, lat: e.lngLat.lat, x: e.point.x, y: e.point.y }); }}
        onMouseOut={() => setHover(null)}
      >
        <DeckGlOverlay layers={deckLayers as never} interleaved onDeviceInitialized={setDevice} overlayRef={overlayRef} />
      </MaplibreMap>
      {pair || showImg ? (
        <div className="pair-tag" style={{ position: "absolute", top: 12, right: 12 }}>
          {showImg ? `Sentinel-2 ${imgYear}` : `Fuel from the ${embYear} embeddings`}
        </div>
      ) : null}
      </div>
      ) : null}
      {boxesOn ? (
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
          {boxPicked ? (
            <>
              {boxLine(boxPicked)}
              <br />
              <span style={{ color: "#aaa", fontSize: 12 }}>
                {reach ? pacificHour(reach.t0 + boxHour * 3600e3) : ""} · {boxNow.length} GOES pixel{boxNow.length === 1 ? "" : "s"} burning · Esc to go back out
              </span>
            </>
          ) : (
            <>
              {boxesDoc && boxesRead < boxesDoc.scans.length && !autoPicked.current
                ? `${START_FIRE.name}: reading GOES-West's scans from NOAA, ${boxesRead} of ${boxesDoc.scans.length}`
                : `GOES-West fires over California, July 22 to Aug 2, 2024: ${boxEvents.length} boxes`}
              <br />
              <span style={{ color: "#aaa", fontSize: 12 }}>
                {boxesDoc && boxesRead < boxesDoc.scans.length && !autoPicked.current ? "it plays once they are in" : "Click a box to fly to it"}
              </span>
            </>
          )}
        </div>
      ) : fuel === "lit" && !WEATHER_ON && zoom < MOSAIC_MIN_ZOOM ? (
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
        fuelOnly={FUEL_ONLY}
        buildingReach={buildingReach}
        onBuildingReachChange={setBuildingReach}
        years={{
          emb: embYear,
          embOptions: EMB_YEARS,
          onEmbChange: setEmbYear,
          img: imgYear,
          imgOptions: S2_YEARS,
          onImgChange: setImgYear,
          imgOn,
          onImgOnChange: setImgOn,
          holding,
          pair,
          onPairChange: setPair,
        }}
        fireHour={reach && boxPicked ? {
          playing: boxPlaying,
          onPlayToggle: toggleBoxPlay,
          hour: boxHour,
          hours: reach.hours,
          onHourChange: (h) => { setBoxPlaying(false); setBoxHour(h); },
          t0: reach.t0,
          burning: boxNow.length,
          cellM: reach.cellM,
          ms: reachDone?.ms ?? 0,
          waiting: autoStartRef.current,
        } : null}
        fire={{ pick: firePick, record: fireRec, onClear: () => setFirePick(null) }}
        burnHist={burnPick ? (
          <FireHist
            name={burnPick.name}
            fireYear={burnPick.year}
            acres={burnPick.acres}
            hists={burnHists}
            embYear={embYear}
            zoomOk={zoom >= MOSAIC_MIN_ZOOM}
            minZoom={MOSAIC_MIN_ZOOM}
          />
        ) : null}
        season={{
          year: SEASON_YEAR,
          on: seasonOn,
          onChange: setSeasonOn,
          loading: seasonLoading,
          day: seasonDay,
          days: seasonDays,
          onDayChange: (d) => { setSeasonPlaying(false); setSeasonDay(d); },
          playing: seasonPlaying,
          onPlayToggle: toggleSeasonPlay,
          reach: seasonReach,
          onReachChange: setSeasonReach,
          inView: seasonJoin?.inView ?? 0,
          found: seasonJoin?.found ?? 0,
          unread: seasonJoin?.unread ?? 0,
          total: seasonCount,
          burns: seasonPerims.length,
        }}
        goes={{
          on: goesOn,
          onChange: setGoesOn,
          name: goesDoc?.fire.name ?? "Green",
          acres: goesDoc?.fire.acres ?? 0,
          cause: goesDoc?.fire.cause ?? "",
          scans: goesDoc?.scans.length ?? 0,
          read: goesRead,
          hour: goesHour,
          hours: goesHours,
          t0: goesT0,
          onHourChange: (h) => { setGoesPlaying(false); setGoesHour(h); },
          playing: goesPlaying,
          onPlayToggle: toggleGoesPlay,
          now: goesNow.length,
          nowPower: goesNow.reduce((a, p) => a + (Number.isNaN(p.power) ? 0 : p.power), 0),
          seen: goesSeen.length,
        }}
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
      {showImg ? null : boxesOn && boxHover ? (
        <div style={tooltipStyle(boxHover.x, boxHover.y)}>
          <div style={{ fontWeight: 600 }}>{boxLine(boxHover.ev)}</div>
          <div style={{ color: "#aaa" }}>click to fly to it</div>
        </div>
      ) : seasonOn && seasonData && seasonHover ? (
        <div style={tooltipStyle(seasonHover.x, seasonHover.y)}>
          <div style={{ fontWeight: 600 }}>{seasonData.starts.name[seasonHover.i] || "unnamed start"}</div>
          <div>
            {new Date(Date.UTC(SEASON_YEAR, 0, 1 + seasonData.starts.day[seasonHover.i]!)).toISOString().slice(0, 10)}
            {" · "}{["cause unknown", "natural", "human"][seasonData.starts.cause[seasonHover.i]!]}
            {seasonData.starts.acres[seasonHover.i]! > 0 ? ` · ${seasonData.starts.acres[seasonHover.i]!.toLocaleString(undefined, { maximumFractionDigits: 1 })} ac` : ""}
          </div>
          <div style={{ color: "#aaa" }}>
            {(() => {
              const f = seasonFuelRef.current?.fuel[seasonHover.i] ?? NaN;
              return Number.isNaN(f) ? "fuel not read yet" : `most burnable ground within ${seasonReach} m: ${Math.round(f * 100)}%`;
            })()}
          </div>
        </div>
      ) : fuel === "lit" && !WEATHER_ON && hover && hoverInfo ? (
        // the weather off: the fuel head's reading alone
        <div style={tooltipStyle(hover.x, hover.y)}>
          <div style={{ fontWeight: 600 }}>
            {hoverInfo.ground && Number.isFinite(hoverInfo.ground.fuel)
              ? `chance of fuel ${Math.round(hoverInfo.ground.fuel * 100)}%`
              : "no fuel reading here (zoom in past " + MOSAIC_MIN_ZOOM + ")"}
          </div>
          {burnOn && burnAreas
            ? burnAreasAt(burnAreas, hover.lon, hover.lat).map((b, i) => (
                <div key={i} style={{ color: "#00e5ff" }}>
                  {titleCase(b.name)} fire · {b.year ?? "year unknown"}
                  {b.acres != null ? ` · ${Math.round(b.acres).toLocaleString()} acres` : ""}
                </div>
              ))
            : null}
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
