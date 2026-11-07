// the HRRR field fold (the 3 km weather drawn itself) is commented out below; its imports with it
// import { COLORMAP_INDEX } from "@developmentseed/deck.gl-raster/gpu-modules";
// import colormapsPngUrl from "@developmentseed/deck.gl-raster/gpu-modules/colormaps.png";
import { useState } from "react";
// HEADS and HOLD: the afterglow help, commented out with the weather
// import { HEADS } from "../aef/head.js";
// import { HOLD } from "../aef/fuel.js";
import { type Teacher } from "../aef/head.js";
import type { FuelValue } from "../aef/layers.js";
import volumeJson from "../../../models/fuel_volume.json";
import { volMeters, type VolumeHead } from "../fuel/volume.js";
import { MOSAIC_MIN_ZOOM } from "../aef/mosaic.js";
import { GROUND_RAMPS, rampGradient, rampStops, type GroundRamp } from "../aef/ramp.js";
import { type S2Comp, S2_COMPS } from "../s2.js";
import type { WindMode } from "../ffwi/compute.js";
import {
  dateFromInitTimeIdx,
  HRRR_LEAD_TIME_COUNT,
  HRRR_LEAD_TIME_HOURS,
  INIT_TIME_ORIGIN,
  initTimeIdxFromDate,
  isoDateString,
} from "../hrrr/metadata.js";
import { type RasterVariable } from "../hrrr/variables.js";
// import { RASTER_VARIABLES, windLabel } from "../hrrr/variables.js";
import { HOURS_PER_DAY, RUN_HOUR_UTC, type DayState } from "../hrrr/week.js";
import { A, Check, Field, Fold, Help, Key, Seg, Slider } from "./controls.js";
// import { Ramp } from "./controls.js";
import { ViewHist } from "./ViewHist.js";
import type { ViewHist as ViewHistData } from "../fuel/viewHist.js";
import "./panel.css";

// const COLORMAP_ROW_COUNT = Object.keys(COLORMAP_INDEX).length;

/** Protan-safe, luminance-monotone ramps with a bright hot end (the HRRR field). */
export const COLORMAP_CHOICES = [
  { id: "solar", label: "Solar" },
  { id: "inferno", label: "Inferno" },
  { id: "magma", label: "Magma" },
  { id: "thermal", label: "Thermal" },
  { id: "hot", label: "Hot" },
  { id: "viridis", label: "Viridis" },
  { id: "cividis", label: "Cividis" },
  { id: "ylorbr", label: "YlOrBr" },
  { id: "coolwarm", label: "Cool-warm (diverging)" },
] as const;
export type ColormapChoiceId = (typeof COLORMAP_CHOICES)[number]["id"];

/** What the land cover layer draws: the ground lit by the hour's FFWI, the classes themselves, or nothing. */
export type FuelChoice = "ignition" | "counties" | "lit" | "cover" | "off";
/** How the fuel is drawn: the ramp's colors, or the Sentinel-2 imagery showing through on the fuel (or the reverse). */
export type FuelPaint = "colors" | "imagery" | "tint";

export type CountyRow = { id: string; name: string; state: string; share: number; rate: number; starts: number; bbox: [number, number, number, number] };
/** How the ignition head meets the ramp. */
export type IgnScale = "probability" | "logit" | "rank";
/** The time axis: one 48 h forecast run, or a week of daily 12Z runs stitched hour by hour. */
export type TimeMode = "run" | "week";
/** The longest window: each day is about 430 MB over the wire for the whole country, 90 MB of bytes in memory. */
export const MAX_WEEK_DAYS = 7;

export type Status = {
  backend: "icechunk" | "zarr-v3" | null;
  /** False: the HRRR store is not opened at all (App.tsx WEATHER_ON). */
  weatherOn: boolean;
  tilesLoaded: number;
  tilesTotal: number;
  aefIndexReady: boolean;
  aefReading: number;
  aefDone: number;
  /** Week mode: shard-days in hand and asked for. */
  weekDone: number;
  weekWanted: number;
  /** Week mode: days whose shards under the view are all in hand, of the days asked for. */
  weekDaysIn: number;
  weekDaysTotal: number;
  error: string | null;
};

/** What is still loading, for the panel bar and the map pill. */
export function loadingSummary(s: Status): { fraction: number; label: string } | null {
  const parts: string[] = [];
  let done = 0;
  let total = 0;
  if (s.weatherOn && !s.backend) {
    parts.push("opening the HRRR store");
    total += 1;
  }
  total += s.tilesTotal;
  done += Math.min(s.tilesLoaded, s.tilesTotal);
  if (s.tilesLoaded < s.tilesTotal) parts.push(`forecast ${s.tilesLoaded} / ${s.tilesTotal}`);
  if (!s.aefIndexReady) {
    total += 1;
    parts.push("the embedding index");
  }
  if (s.aefReading > 0) {
    total += s.aefReading + s.aefDone;
    done += s.aefDone;
    parts.push(`embeddings ${s.aefDone} / ${s.aefReading + s.aefDone}`);
  }
  if (s.weekWanted > s.weekDone) {
    total += s.weekWanted;
    done += s.weekDone;
    parts.push(s.weekDaysTotal > 0 ? `weather: day ${Math.min(s.weekDaysIn + 1, s.weekDaysTotal)} of ${s.weekDaysTotal} under the view (${s.weekDone} / ${s.weekWanted} shards)` : `weather ${s.weekDone} / ${s.weekWanted} shards`);
  }
  if (parts.length === 0) return null;
  return { fraction: total > 0 ? done / total : 0, label: `Loading ${parts.join(" · ")}` };
}

export type ControlPanelProps = {
  leadTimeIdx: number;
  initTimeIdx: number;
  initTimeCount: number;
  isPlaying: boolean;
  frameDurationMs: number;
  onInitTimeIdxChange: (idx: number) => void;
  onLeadTimeIdxChange: (idx: number) => void;
  onPlayPauseToggle: () => void;
  onFrameDurationMsChange: (ms: number) => void;
  timeMode: TimeMode;
  onTimeModeChange: (m: TimeMode) => void;
  /** Week mode: the UTC date of day 0, the number of days, the hour on screen (0 .. days x 24 - 1). */
  weekStart: Date;
  onWeekStartChange: (d: Date) => void;
  weekDays: number;
  onWeekDaysChange: (n: number) => void;
  hourIdx: number;
  onHourIdxChange: (h: number) => void;
  weekDepth: number;
  weekHourLabel: string;
  /** Week mode: each day's state under the view (in hand, loading, waiting, not in the store). */
  weekDayStates: DayState[];

  fuel: FuelChoice;
  onFuelChange: (f: FuelChoice) => void;
  /** Counties mode: the counties in view ranked by the hour's rate; the ground sums' progress (files done, files). */
  countyRows: CountyRow[];
  countyProgress: [number, number] | null;
  onCountyPick: (bbox: [number, number, number, number]) => void;
  pHide: number;
  onPHideChange: (v: number) => void;
  pMax: number;
  onPMaxChange: (v: number) => void;
  ignScale: IgnScale;
  onIgnScaleChange: (v: IgnScale) => void;
  zLo: number;
  onZLoChange: (v: number) => void;
  zHi: number;
  onZHiChange: (v: number) => void;
  qLo: number;
  onQLoChange: (v: number) => void;
  qHi: number;
  onQHiChange: (v: number) => void;
  /** The logit ramp the rank scale is using this hour, when known. */
  rankRamp: [number, number] | null;
  useWeather: boolean;
  onUseWeatherChange: (v: boolean) => void;
  standingAlpha: number;
  onStandingAlphaChange: (v: number) => void;
  wxGate: number;
  fieldAlpha: number;
  onFieldAlphaChange: (v: number) => void;
  onWxGateChange: (v: number) => void;
  /** The weather term under the cursor, to read the gate against. */
  wxNow: number | null;


  teacher: Teacher;
  onTeacherChange: (t: Teacher) => void;
  groundRamp: GroundRamp;
  onGroundRampChange: (r: GroundRamp) => void;
  fuelOpacity: number;
  onFuelOpacityChange: (v: number) => void;
  glowHours: number;
  onGlowHoursChange: (v: number) => void;
  grain: number;
  onGrainChange: (v: number) => void;
  light: boolean;
  onLightChange: (v: boolean) => void;
  relief: boolean;
  onReliefChange: (v: boolean) => void;
  contourStep: number;
  onContourStepChange: (v: number) => void;
  zoom: number;

  windMode: WindMode;
  onWindModeChange: (m: WindMode) => void;
  colormapId: ColormapChoiceId;
  onColormapChange: (id: ColormapChoiceId) => void;
  rescaleMax: number;
  onRescaleMaxChange: (v: number) => void;
  hideBelow: number;
  onHideBelowChange: (v: number) => void;
  showRaster: boolean;
  onShowRasterChange: (v: boolean) => void;
  rasterVariable: RasterVariable;
  onRasterVariableChange: (v: RasterVariable) => void;
  rasterOpacity: number;
  onRasterOpacityChange: (v: number) => void;
  tauHours: number;
  onTauHoursChange: (v: number) => void;

  status: Status;
  /** False: no weather; the ground is the fuel head's P(fuel) alone and the Time panel is off. */
  weatherOn: boolean;
  /** The fuel alone: the season, fire and building rows hidden. */
  fuelOnly?: boolean;
  /** How the fuel is drawn (I): the ramp's colors, or the imagery through the fuel or the non-fuel. */
  paint: FuelPaint;
  onPaintChange: (v: FuelPaint) => void;
  /** The painted imagery's colors (C). */
  comp: S2Comp;
  onCompChange: (v: S2Comp) => void;
  /** The fuel's colors stretched to the view, and the P(fuel) at the ramp's bottom, middle and top. */
  stretch: boolean;
  onStretchChange: (on: boolean) => void;
  stretchMarks: [number, number, number] | null;
  /** The fuel drawn in view, binned by its color (fuel/viewHist.ts); null before enough is in. */
  viewHist: ViewHistData | null;
  /** What the fuel's colors stand for (V): the volume head's vegetation volume, or the fuel head's P(fuel). */
  fuelValue: FuelValue;
  onFuelValueChange: (v: FuelValue) => void;
  /** The fuel only inside the burned areas (M). */
  burnMask: boolean;
  onBurnMaskChange: (on: boolean) => void;
  /** The burned areas' outlines shown (B). */
  burnOn: boolean;
  onBurnOnChange: (on: boolean) => void;
  season: SeasonPanel;
  goes: GoesPanel;
  years: YearsPanel;
  fire: FirePanel;
  /** The selected fire's fuel histograms, under the years box. */
  burnHist?: React.ReactNode;
  /** The picked GOES fire hour by hour (null when none is picked). */
  fireHour: FireHourPanel | null;
};

export type FireHourPanel = {
  playing: boolean;
  onPlayToggle: () => void;
  hour: number;
  hours: number;
  onHourChange: (h: number) => void;
  t0: number;
  burning: number;
  cellM: number;
  ms: number;
  /** Waiting for the fuel under the fire before playing. */
  waiting: boolean;
};

/** The years on screen: the embeddings the fuel is read from, and the Sentinel-2 mosaic's. */
export type YearsPanel = {
  emb: number;
  embOptions: number[];
  onEmbChange: (y: number) => void;
  img: number;
  imgOptions: number[];
  onImgChange: (y: number) => void;
  imgOn: boolean;
  onImgOnChange: (on: boolean) => void;
  holding: boolean;
  pair: boolean;
  onPairChange: (on: boolean) => void;
};

/** A clicked fire: which one, and its NIFC record once fetched. */
export type FirePanel = {
  pick: { kind: "start" | "burn"; oid: number } | null;
  record: Record<string, unknown> | null;
  onClear: () => void;
};

const pacificDay = new Intl.DateTimeFormat("en-US", { timeZone: "America/Los_Angeles", year: "numeric", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" });

/** The record's fields worth reading, in order, each with its label and how to show it. */
function fireRows(r: Record<string, unknown>): [string, string][] {
  const str = (k: string) => (r[k] == null || r[k] === "" ? null : String(r[k]));
  const num = (k: string) => (typeof r[k] === "number" && Number.isFinite(r[k]) ? (r[k] as number) : null);
  const when = (k: string) => (num(k) != null ? pacificDay.format(new Date(num(k)!)) : null);
  const acres = num("GISAcres") ?? num("FinalAcres") ?? num("IncidentSize") ?? num("CalculatedAcres");
  const cause = [str("FireCause"), str("FireCauseGeneral"), str("FireCauseSpecific")].filter((v, i, a) => v && a.indexOf(v) === i).join(", ");
  const place = [str("POOCity"), str("POOCounty") ? `${str("POOCounty")} County` : null].filter(Boolean).join(", ");
  const land = [str("POOLandownerKind"), str("POOLandownerCategory")].filter((v, i, a) => v && a.indexOf(v) === i).join(", ");
  const fuel = [str("PredominantFuelGroup"), str("PrimaryFuelModel"), str("SecondaryFuelModel")].filter(Boolean).join(" · ");
  const cost = num("EstimatedCostToDate") ?? num("EstimatedFinalCost");
  const rows: [string, string | null][] = [
    ["Discovered", when("FireDiscoveryDateTime")],
    ["Contained", when("ContainmentDateTime")],
    ["Out", when("FireOutDateTime")],
    ["Size", acres != null ? `${Math.round(acres).toLocaleString()} acres${num("GISAcres") != null ? " (mapped outline)" : ""}` : null],
    ["Cause", cause || null],
    ["Where", place || null],
    ["Land", land || null],
    ["Protecting agency", str("POOProtectingAgency")],
    ["Fuel (NIFC record)", fuel || null],
    ["Fire behavior", str("FireBehaviorGeneral")],
    ["Personnel", num("TotalIncidentPersonnel") != null ? num("TotalIncidentPersonnel")!.toLocaleString() : null],
    ["Cost to date", cost != null ? `$${Math.round(cost).toLocaleString()}` : null],
    ["Outline mapped", when("PolygonDateTime")],
    ["Map method", str("MapMethod")],
    ["Description", str("IncidentShortDescription")],
    ["Fire ID", str("UniqueFireIdentifier")],
  ];
  return rows.filter((x): x is [string, string] => x[1] != null);
}

/** One fire as GOES-West saw it (App.tsx, goes.ts). */
export type GoesPanel = {
  on: boolean;
  onChange: (on: boolean) => void;
  name: string;
  acres: number;
  cause: string;
  /** Scans referenced and read so far. */
  scans: number;
  read: number;
  /** Hour on the slider (0 = t0, ms UTC) and the hours in all. */
  hour: number;
  hours: number;
  t0: number;
  onHourChange: (h: number) => void;
  playing: boolean;
  onPlayToggle: () => void;
  /** Fire pixels this hour, their summed power (MW), and footprints seen so far. */
  now: number;
  nowPower: number;
  seen: number;
};

// the GOES fold's clock, with the fold commented out
// eslint-disable-next-line @typescript-eslint/no-unused-vars
export const pacific = new Intl.DateTimeFormat("en-US", { timeZone: "America/Los_Angeles", month: "short", day: "numeric", hour: "numeric", timeZoneName: "short" });

/** The fire season view (App.tsx, season.ts). */
export type SeasonPanel = {
  year: number;
  on: boolean;
  onChange: (on: boolean) => void;
  loading: boolean;
  /** Day of the season (0 = Jan 1) and the season's length. */
  day: number;
  days: number;
  onDayChange: (d: number) => void;
  playing: boolean;
  onPlayToggle: () => void;
  reach: number;
  onReachChange: (m: number) => void;
  /** Starts so far in view, how many found fuel, how many not read yet; all starts so far; burns so far. */
  inView: number;
  found: number;
  unread: number;
  total: number;
  burns: number;
};

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const dayLabel = (year: number, day: number) => {
  const d = new Date(Date.UTC(year, 0, 1 + day));
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}, ${year}`;
};

// const FIELD_OPTIONS = [
//   { id: "nothing", label: "nothing" },
//   ...Object.values(RASTER_VARIABLES).map((v) => ({ id: v.id as string, label: v.id === "ffwi" ? "FFWI" : v.label.toLowerCase() })),
// ];

const VOLUME = volumeJson as VolumeHead;
/** A place on the volume byte's scale (0..1) as meters of cover x height. */
const meters = (s: number) => {
  const m = volMeters(VOLUME, s);
  return m >= 10 ? `${Math.round(m)} m` : m >= 1 ? `${m.toFixed(1)} m` : `${m.toFixed(2)} m`;
};

export function ControlPanel(p: ControlPanelProps) {
  const hours = HRRR_LEAD_TIME_HOURS[p.leadTimeIdx] ?? 0;
  const validUTC = dateFromInitTimeIdx(p.initTimeIdx);
  validUTC.setUTCHours(validUTC.getUTCHours() + hours);
  const validStr = `${isoDateString(validUTC)} ${String(validUTC.getUTCHours()).padStart(2, "0")}:00Z`;
  const s = p.status;
  const loading = loadingSummary(s);
  const [min, setMin] = useState(false);
  const reader = p.zoom >= MOSAIC_MIN_ZOOM ? "10 m mosaic" : `nothing (the fuel reads from zoom ${MOSAIC_MIN_ZOOM} in)`;
  const weekEnd = new Date(p.weekStart.getTime());
  weekEnd.setUTCDate(weekEnd.getUTCDate() + p.weekDays - 1);
  /** The latest day whose 12Z run is in the store (the latest init, or the day before it when it is an earlier run). */
  const latestDay = (() => {
    if (p.initTimeCount === 0) return null;
    const t = dateFromInitTimeIdx(p.initTimeCount - 1);
    const d = new Date(Date.UTC(t.getUTCFullYear(), t.getUTCMonth(), t.getUTCDate()));
    if (t.getUTCHours() < RUN_HOUR_UTC) d.setUTCDate(d.getUTCDate() - 1);
    return d;
  })();
  const onLatestDay = latestDay !== null && weekEnd.getTime() === latestDay.getTime();
  /** The ramp's two ends and what they are, for the collapsed card's legend. */
  const legendEnds: [string, string, string] =
    p.fuel === "lit" && !p.weatherOn && p.fuelValue === "volume"
      ? p.stretchMarks
        ? [meters(p.stretchMarks[0]), meters(p.stretchMarks[2]), `vegetation volume, stretched, middle ${meters(p.stretchMarks[1])}`]
        : ["0 m", "50 m", "vegetation volume (cover x height)"]
      : p.fuel === "lit" && !p.weatherOn && p.stretchMarks
      ? [`${Math.round(p.stretchMarks[0] * 100)}%`, `${Math.round(p.stretchMarks[2] * 100)}%`, `stretched, middle ${Math.round(p.stretchMarks[1] * 100)}%`]
      : p.fuel === "lit" && !p.weatherOn
      ? [`${Math.round(p.pHide * 100)}%`, `${Math.round(p.pMax * 100)}%`, p.ignScale === "logit" ? "chance of fuel (log-odds)" : "chance of fuel"]
      : p.fuel === "counties"
      ? ["0%", "100%", "ignition risk by county"]
      : p.ignScale === "probability"
      ? [`${Math.round(p.pHide * 100)}%`, `${(p.pMax * 100).toFixed(p.pMax >= 0.99 ? 1 : 0)}%`, "ignition risk"]
      : p.ignScale === "logit"
        ? [p.zLo.toFixed(1), p.zHi.toFixed(1), "ignition logit"]
        : [`p${Math.round(p.qLo * 100)}`, `p${(p.qHi * 100).toFixed(p.qHi >= 0.99 ? 1 : 0)}`, "rank in view"];
  const timeTag = p.timeMode === "week" ? p.weekHourLabel : `+${hours} h · ${validStr}`;
  // the HRRR field fold is commented out below
  // const rv = RASTER_VARIABLES[p.rasterVariable];
  // const field = p.showRaster ? p.rasterVariable : "nothing";
  // const fieldOptions = FIELD_OPTIONS.map((o) => (o.id === "wind" ? { ...o, label: windLabel(p.windMode).replace(/^wind: /, "wind ") } : o));
  // const setField = (id: string) => {
  //   if (id === "nothing") p.onShowRasterChange(false);
  //   else {
  //     p.onShowRasterChange(true);
  //     p.onRasterVariableChange(id as RasterVariable);
  //   }
  // };
  // const fieldRampLabel =
  //   p.rasterVariable === "ffwi"
  //     ? `FFWI${p.tauHours > 0 ? ` (${p.tauHours} h fuel lag)` : ""} ${p.hideBelow} to ${p.rescaleMax}`
  //     : `${rv.format(rv.min)} to ${rv.format(rv.max)}${rv.reversed ? " (reversed: dry is bright)" : ""}`;

  return (
    <div className={`pn${min ? " min" : ""}`}>
      <div className="pn-title">
        <h1>Fuel from the embeddings</h1>
        <button type="button" className="pn-min" aria-label={min ? "Expand panel" : "Collapse panel"} onClick={() => setMin((v) => !v)}>
          {min ? "▸" : "▾"}
        </button>
      </div>
      {min && p.weatherOn ? <p className="pn-stamp">{timeTag}</p> : null}
      {min ? <p className="pn-stamp">Embeddings {p.years.emb}{p.fuelOnly ? (p.paint === "tint" ? ` · Sentinel-2 ${p.years.img}` : "") : ` · Sentinel-2 ${p.years.img}${p.years.holding || p.years.imgOn ? " (shown)" : ""}`}</p> : null}
      {min && (p.fuel === "ignition" || p.fuel === "counties" || (p.fuel === "lit" && !p.weatherOn)) ? (
        <div className="pn-legend">
          <div className="ramp" role="img" aria-label={`${p.groundRamp} ramp`} style={{ backgroundImage: rampGradient(rampStops(p.groundRamp)) }} />
          <div className="pn-legend-ends">
            <span>{legendEnds[0]}</span>
            <span className="muted">{legendEnds[2]}</span>
            <span>{legendEnds[1]}</span>
          </div>
        </div>
      ) : null}

      {/* the ignition-and-weather intro, while the weather is off:
      <p>
        Every pixel's <b>AlphaEarth embedding</b> goes through a small head fit on 22,000 real wildfire ignitions (WFIGS, 2019
        to 2026) against points 20 to 30 km away at other hours, together with the hour's weather from NOAA's HRRR
        forecast (Fosberg index, temperature, humidity, wind). The map plays the head's ignition risk
        hour by hour over a window of up to a week of daily forecasts: the ground in gray all the time, in color where
        the weather lifts it. This is a risk outlook, not a detection: nothing here shows an actual fire, and the colors
        mean a higher or lower chance of a start, never that one has happened.
      </p>
      */}
      <p>
        Each 10 m pixel's <b>AlphaEarth embeddings</b> go through a small model fit on LANDFIRE across California,
        {p.fuelValue === "volume"
          ? " giving how much vegetation stands there each year (cover x height): it drops where a fire burned and comes back as the ground regrows."
          : " giving the chance the ground is burnable fuel."}
        {p.fuelOnly ? null : " Fires are NIFC's records: where each one started and the outline it burned."}
        {/* the imagery is off for now: <b>Sentinel-2</b> shows what the ground looked like. */}
      </p>

      <div className="years">
        <div className="row">
          <label className="yr">
            <span>Embeddings</span>
            <select value={p.years.emb} onChange={(e) => { p.years.onEmbChange(Number(e.target.value)); e.currentTarget.blur(); }} aria-label="Embeddings year">
              {p.years.embOptions.map((y) => <option key={y} value={y}>{y}</option>)}
            </select>
          </label>
          <span className="with-key"><Key k="[" /><Key k="]" /></span>
          <span className="muted small">{p.fuelValue === "volume" ? <>the volume model reads {p.years.emb} alone</> : <>the fuel model reads {p.years.emb - 2} to {p.years.emb}</>}</span>
        </div>
        {/* the imagery is off for now
        <div className="row">
          <label className="yr">
            <span>Sentinel-2</span>
            <select value={p.years.img} onChange={(e) => { p.years.onImgChange(Number(e.target.value)); e.currentTarget.blur(); }} aria-label="Imagery year">
              {p.years.imgOptions.map((y) => <option key={y} value={y}>{y}</option>)}
            </select>
          </label>
          <span className="muted small">Earth Genome yearly mosaic</span>
        </div>
        <div className="row">
          <Check checked={p.years.imgOn} onChange={p.years.onImgOnChange} label="Imagery in place of the fuel" />
          <Check checked={p.years.pair} onChange={p.years.onPairChange} label="Side by side" />
        </div>
        <p className="muted small">
          Hold <b>Space</b> to see the imagery (the wheel steps its year while held), <b>[</b> <b>]</b> a year back or
          forward, <b>P</b> for the two side by side. Imagery from zoom 9 in, fuel from zoom {MOSAIC_MIN_ZOOM} in.
        */}
        <p className="muted small">Fuel from zoom {MOSAIC_MIN_ZOOM} in. Click inside a fire's outline to pick it out; Esc clears it.</p>
      </div>
      {p.burnHist}
      {p.viewHist && p.fuel === "lit" && !p.weatherOn && p.paint !== "imagery" ? (
        <ViewHist
          hist={p.viewHist}
          gradient={rampGradient(rampStops(p.groundRamp))}
          fmt={p.fuelValue === "volume" ? meters : (v) => `${Math.round(v * 100)}%`}
          what={p.fuelValue === "volume" ? "vegetation volume" : "chance of fuel"}
          year={p.years.emb}
        />
      ) : null}
      <p className="muted small" style={{ marginTop: 6 }}>
        Data: <A href="https://source.coop/tge-labs/aef">AlphaEarth Foundations</A> (tge-labs on Source Cooperative) ·{" "}
        <A href="https://www.earthgenome.org/">Earth Genome</A> Sentinel-2 mosaics (CC BY 4.0) ·{" "}
        {p.fuelOnly ? null : <><A href="https://data-nifc.opendata.arcgis.com/">NIFC WFIGS</A> fire starts and outlines ·{" "}</>}
        {/* the weather sources, while the weather is off:
        <A href="https://dynamical.org/catalog/noaa-hrrr-forecast-48-hour/">NOAA HRRR 48 h</A> via{" "}
        <A href="https://dynamical.org">dynamical.org</A> ({s.backend === "icechunk" ? "Icechunk" : s.backend === "zarr-v3" ? "Zarr v3" : "opening"}) ·{" "}
        <A href="https://www.frames.gov/catalog/27085">Fosberg 1978</A> ·{" "} */}
        <A href="https://github.com/kentstephen/aef-hls-s1-fire-knn-hrrr">code on GitHub</A>
      </p>
      <p className="muted small" style={{ marginTop: 4 }}>
        Keys: {p.fuelOnly ? null : <><b>S</b> the season · </>}hold <b>Space</b> for Sentinel-2 · <b>P</b> side by side ·{" "}
        <b>F</b> full screen · <b>L</b> map labels off and on · <b>Q</b> tooltip off and on · <b>Esc</b> clears a fire.
      </p>

      {loading ? (
        <div className="field" style={{ marginTop: 8 }}>
          <div className="bar"><span style={{ width: `${Math.round(loading.fraction * 100)}%` }} /></div>
          <p className="muted small">{loading.label}. Everything is read straight from the cloud stores; nothing is precomputed but the model weights.</p>
        </div>
      ) : null}

      {p.fireHour ? (
        <Fold k="firehour" label="The fire, hour by hour" defaultOpen tag={pacific.format(new Date(p.fireHour.t0 + p.fireHour.hour * 3600e3))}>
          <div className="row">
            <button type="button" className="play" aria-label={p.fireHour.playing ? "Pause" : "Play"} title={p.fireHour.playing ? "Pause" : "Play"}
              onClick={(e) => { p.fireHour!.onPlayToggle(); e.currentTarget.blur(); }}>
              {p.fireHour.playing ? "❚❚" : "▶"}
            </button>
            <span>{pacific.format(new Date(p.fireHour.t0 + p.fireHour.hour * 3600e3))}</span>
            <Help label="The fire, hour by hour">
              GOES-West's fire pixels, hour by hour (orange outline: burning this hour). Around them the fuel lights up as
              far as a fire could reach from them: faster through fuel (ground that is not fuel spreads at a tenth of the
              rate) and uphill (Rothermel's slope factor), the way fire models trace arrival lines. A pixel that keeps
              burning reaches farther; one that stops fades over a few hours. It
              is relative reach, not hours or a forecast: there is no wind or fuel moisture in it yet. The rest of the
              fuel is gray. Elevation: Mapterhorn (Copernicus 30 m).
            </Help>
          </div>
          <Field label="Hour">
            <Slider min={0} max={Math.max(0, p.fireHour.hours - 1)} step={1} value={p.fireHour.hour} onChange={p.fireHour.onHourChange} label="Hour" />
          </Field>
          <p className="muted small">
            {p.fireHour.waiting ? "Reading the fuel under the fire before playing. " : ""}
            {p.fireHour.burning} GOES pixel{p.fireHour.burning === 1 ? "" : "s"} burning this hour. Reach on {p.fireHour.cellM} m cells
            {p.fireHour.ms ? `, ${Math.round(p.fireHour.ms)} ms an hour` : ""}.
          </p>
        </Fold>
      ) : null}

      <Fold k="fuel" label="The ground" defaultOpen tag={p.fuel === "ignition" ? "ignition risk" : p.fuel === "counties" ? "counties" : p.fuel === "lit" ? (p.weatherOn ? "lit by FFWI" : p.fuelValue === "volume" ? "volume" : "P(fuel)") : p.fuel === "cover" ? "land cover" : "off"}>
        <div className="row">
          <Seg
            value={p.fuel}
            options={[
              // lit: the fuel head's P(fuel) x the hour's FFWI (the fuel head in place of the land cover weights)
              { id: "lit", label: p.weatherOn ? "Lit by FFWI" : "Fuel" },
              // the old ignition head (Risk) and its county sums: not offered; a risk model from the fuel head is to come
              // { id: "ignition", label: "Risk" },
              // { id: "counties", label: "Counties" },
              // the classes themselves and off: not offered; G toggles them
              // { id: "cover", label: "Land cover" },
              // { id: "off", label: "Off" },
            ]}
            onChange={p.onFuelChange}
          />
          <Help label="The ground">
            Every pixel's AlphaEarth embeddings (2023, 2024 and 2025) go through the fuel head, a small network fit on
            fuel and non-fuel ground across California, to a chance that the ground is burnable fuel. Drawn from the 10 m
            mosaic, zoom {MOSAIC_MIN_ZOOM} in; nothing draws zoomed out yet.
          </Help>
        </div>
        {p.fuel === "lit" && !p.weatherOn ? (
          <>
            <div className="row">
              <Seg
                value={p.fuelValue}
                options={[{ id: "volume", label: "Volume" }, { id: "pfuel", label: "P(fuel)" }]}
                onChange={(v) => p.onFuelValueChange(v as FuelValue)}
                k="V"
              />
              <Help label="Volume">
                Volume: how much vegetation stands on each 10 m pixel that year, its leading lifeform's cover times its
                height (LANDFIRE EVC and EVH), learned from the year's embedding alone. About 10 m in closed forest, 0.5 m
                in chaparral, 0.2 m in grass. It drops where a fire burned and climbs back as the ground regrows. Hover
                for the lifeform, canopy and the volume two years before. P(fuel): the first fuel head's chance the
                ground is burnable at all, which barely moves after a fire. <kbd>V</kbd> switches.
              </Help>
            </div>
            <div className="row">
              <Seg
                value={p.paint}
                options={[{ id: "colors", label: "Colors" }, { id: "imagery", label: "Imagery" }]}
                onChange={(v) => p.onPaintChange(v as FuelPaint)}
                k="I"
              />
              <Help label="Imagery">
                Imagery: the Sentinel-2 yearly mosaic (Earth Genome) drawn with the fuel head's chance of fuel as its
                opacity: faint below the hide line, in full from the ramp top up. Hold Space for the whole picture in
                true color, or P to see it beside this one. <kbd>I</kbd> switches colors and imagery, <kbd>C</kbd>
                steps the imagery's colors.
              </Help>
            </div>
            {p.paint !== "colors" ? (
              <div className="row">
                <Seg value={p.comp} options={S2_COMPS.map((c) => ({ id: c.id, label: c.label }))} onChange={(v) => p.onCompChange(v as S2Comp)} k="C" />
              </div>
            ) : null}
            {p.paint !== "imagery" ? (
              <div className="row">
                <Seg value={p.groundRamp} options={GROUND_RAMPS.map((r) => ({ id: r.id, label: r.label }))} onChange={p.onGroundRampChange} />
              </div>
            ) : null}
            {p.paint !== "imagery" ? (
              <div className="row">
                <Check checked={p.stretch} onChange={p.onStretchChange} label="Stretch to the view" />
                <Help label="Stretch">
                  The colors spread over the values in view, half way to each part of the ramp holding as many pixels:
                  the foothills at 95 to 99% get more of the ramp instead of one flat top color, and close values stay
                  close. A color then means a
                  place among this view's values, not a fixed chance; the legend gives the chance at the ramp's ends and
                  middle. Off: the ramp runs from the hide line to the ramp top.
                </Help>
              </div>
            ) : null}
            <div className="row">
              <Check checked={p.burnOn} onChange={p.onBurnOnChange} label="Burned-area outlines" k="B" />
            </div>
            <div className="row">
              <Check checked={p.burnMask} onChange={p.onBurnMaskChange} label="Only inside burned areas" k="M" />
              <Help label="Burned areas only">
                The fuel drawn only inside the outlines of California's fires of 100 acres and up since 2020 (NIFC
                WFIGS), the rest left to the basemap: every burn's volume year by year, with nothing around it.
                <kbd>M</kbd> switches.
              </Help>
            </div>
            {p.paint !== "imagery" && p.stretch ? null : (
            <div className="row">
              <Seg
                value={p.ignScale === "logit" ? "logit" : "probability"}
                options={[{ id: "probability", label: "Straight" }, { id: "logit", label: "Log-odds" }]}
                onChange={(v) => p.onIgnScaleChange(v as IgnScale)}
              />
              <Help label="Scale">
                Straight: the ramp runs evenly from the hide line to the ramp top, so 90% and 99% both sit near the top.
                Log-odds: each step of the ramp is the same change in odds, so the top end (90%, 96%, 99%) spreads out.
              </Help>
            </div>
            )}
            {p.paint !== "imagery" ? (
              <>
                <div className="ramp" role="img" aria-label={`${p.groundRamp} ramp`} style={{ backgroundImage: rampGradient(rampStops(p.groundRamp)) }} />
                <p className="muted small">
                  {p.fuelValue === "volume"
                    ? p.stretch && p.stretchMarks
                      ? `Vegetation volume (cover x height), stretched to the view: ${meters(p.stretchMarks[0])} at the dark end, ${meters(p.stretchMarks[1])} in the middle, ${meters(p.stretchMarks[2])} at the top. No weather.`
                      : "Vegetation volume (cover x height) on a log scale, 0 to 50 m. No weather."
                    : p.stretch && p.stretchMarks
                    ? `The fuel head's chance the ground is burnable fuel, stretched to the view: ${Math.round(p.stretchMarks[0] * 100)}% at the dark end, ${Math.round(p.stretchMarks[1] * 100)}% in the middle, ${Math.round(p.stretchMarks[2] * 100)}% at the top. No weather.`
                    : `The fuel head's chance the ground is burnable fuel, ${Math.round(p.pHide * 100)}% (invisible) to ${Math.round(p.pMax * 100)}% (ramp top). No weather.`}
                </p>
              </>
            ) : (
              <p className="muted small">
                Sentinel-2 {p.years.img}, {S2_COMPS.find((c) => c.id === p.comp)!.bands}: {S2_COMPS.find((c) => c.id === p.comp)!.about} Its
                opacity is the fuel head's chance the ground is fuel: faint below {Math.round(p.pHide * 100)}%, full from{" "}
                {Math.round(p.pMax * 100)}%.
                {p.years.img !== p.years.emb ? ` The imagery starts in ${p.years.img}, the nearest year to the embeddings' ${p.years.emb}.` : ""}
              </p>
            )}
            <Field label={`Hide below ${Math.round(p.pHide * 100)}%`}>
              <Slider min={0} max={0.99} step={0.01} value={p.pHide} onChange={p.onPHideChange} label="Hide below" onReset={() => p.onPHideChange(p.fuelOnly ? 0 : 0.5)} />
            </Field>
            {p.paint !== "imagery" && p.stretch ? null : (
              <Field label={`Ramp top ${Math.round(p.pMax * 100)}%`}>
                <Slider min={0.05} max={1} step={0.01} value={p.pMax} onChange={p.onPMaxChange} label="Ramp top" onReset={() => p.onPMaxChange(1)} />
              </Field>
            )}
          </>
        ) : null}
        <p className="muted small">Reading: {reader}{s.aefReading > 0 ? ` · ${s.aefReading} tiles in flight` : ""}</p>
        {p.fuel === "counties" ? (
          <div className="counties">
            <p className="muted small">
              The pixel picture by county: the share of the county's 2.6 km overview pixels at elevated risk this hour
              (the weather over the gate and the model score over the hide line), 0 clear to all of it at the ramp top.
              A share of area, not a probability: 100% means the whole county is above the line, not that anything is burning. Starts today is the sum of the calibrated rate over
              the county's land.
              {p.countyProgress ? ` Ground summed for ${p.countyProgress[0]} of ${p.countyProgress[1]} files.` : ""}
            </p>
            <div className="rank-box">
              {p.countyRows.length > 0 ? (
                <table className="rank">
                  <thead>
                    <tr>
                      <th>county in view</th>
                      <th>ground at risk</th>
                      <th>starts today</th>
                    </tr>
                  </thead>
                  <tbody>
                    {p.countyRows.map((r) => (
                      <tr key={r.id} onClick={() => p.onCountyPick(r.bbox)}>
                        <td>
                          {r.name}, {r.state}
                        </td>
                        <td>{Math.round(r.share * 100)}%</td>
                        <td>{r.starts.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="muted small rank-empty">No county in view has its ground and weather yet.</p>
              )}
            </div>
          </div>
        ) : null}
        {p.fuel === "ignition" || p.fuel === "counties" ? (
          <>
            <div className="row">
              <Seg value={p.groundRamp} options={GROUND_RAMPS.map((r) => ({ id: r.id, label: r.label }))} onChange={p.onGroundRampChange} />
              <Help label="Ramp">
                Viridis: purple, blue, teal, green, yellow. Inferno: violet, crimson, orange, pale yellow. Magma: violet, magenta, salmon, cream. Solar: brown-red,
                orange, yellow. Fire (colorcet): deep red, orange, yellow, white. All five rise in brightness the whole way, so they read without the hues; none sets red
                against green. The fiery two start on a dark color that still shows against the dark map.
              </Help>
            </div>
            <div className="ramp" role="img" aria-label={`${p.groundRamp} ramp`} style={{ backgroundImage: rampGradient(rampStops(p.groundRamp)) }} />
            <div className="row">
              <Seg
                value={p.ignScale}
                options={[
                  { id: "probability", label: "Probability" },
                  { id: "logit", label: "Logit" },
                  { id: "rank", label: "Rank in view" },
                ]}
                onChange={p.onIgnScaleChange}
              />
              <Help label="Scale">
                The head gives each pixel a logit: the ground's part (fixed) plus the hour's weather. Probability maps its
                sigmoid onto the ramp, which saturates on a bad day (everything near 100%). Logit maps the sum itself, no
                ceiling. Rank sets the ramp each hour from the pixels on screen, so the brightest fraction of the view is
                always the ramp top whatever the weather: where does the risk land among its surroundings.
              </Help>
            </div>
            {p.ignScale === "probability" ? (
              <>
                <p className="muted small">Ignition risk (model score), {(p.pHide * 100).toFixed(p.pHide >= 0.99 ? 1 : 0)}% (invisible) to {(p.pMax * 100).toFixed(p.pMax >= 0.99 ? 1 : 0)}% (ramp top).</p>
                <Field label={`Hide below ${(p.pHide * 100).toFixed(p.pHide >= 0.99 ? 1 : 0)}%`}>
                  <Slider min={0} max={0.99} step={0.01} value={p.pHide} onChange={p.onPHideChange} label="Hide below" onReset={() => p.onPHideChange(0.5)} />
                </Field>
                <Field label={`Ramp top ${(p.pMax * 100).toFixed(p.pMax >= 0.99 ? 1 : 0)}%`}>
                  <Slider min={0.5} max={0.999} step={0.001} value={p.pMax} onChange={p.onPMaxChange} label="Ramp top" onReset={() => p.onPMaxChange(0.95)} />
                </Field>
              </>
            ) : p.ignScale === "logit" ? (
              <>
                <p className="muted small">Logit {p.zLo.toFixed(1)} (invisible) to {p.zHi.toFixed(1)} (ramp top); 0 is even odds against a control point, each +1 about e times the odds.</p>
                <Field label={`Hide below ${p.zLo.toFixed(1)}`}>
                  <Slider min={-6} max={8} step={0.1} value={p.zLo} onChange={p.onZLoChange} label="Hide below (logit)" onReset={() => p.onZLoChange(0)} />
                </Field>
                <Field label={`Ramp top ${p.zHi.toFixed(1)}`}>
                  <Slider min={-4} max={12} step={0.1} value={p.zHi} onChange={p.onZHiChange} label="Ramp top (logit)" onReset={() => p.onZHiChange(4)} />
                </Field>
              </>
            ) : (
              <>
                <p className="muted small">
                  The ramp follows the view each hour: the {Math.round(p.qLo * 100)}th percentile of the pixels on screen is the hide line, the{" "}
                  {(p.qHi * 100).toFixed(p.qHi >= 0.99 ? 1 : 0)}th the ramp top{p.rankRamp ? ` (now logit ${p.rankRamp[0].toFixed(1)} to ${p.rankRamp[1].toFixed(1)})` : ""}.
                </p>
                <Field label={`Hide below the ${Math.round(p.qLo * 100)}th percentile`}>
                  <Slider min={0} max={0.95} step={0.05} value={p.qLo} onChange={p.onQLoChange} label="Hide percentile" onReset={() => p.onQLoChange(0.5)} />
                </Field>
                <Field label={`Ramp top at the ${(p.qHi * 100).toFixed(p.qHi >= 0.99 ? 1 : 0)}th percentile`}>
                  <Slider min={0.5} max={0.999} step={0.001} value={p.qHi} onChange={p.onQHiChange} label="Top percentile" onReset={() => p.onQHiChange(0.99)} />
                </Field>
              </>
            )}
            <Field
              label={
                <span className="row">
                  <span>{p.standingAlpha === 0 ? "Standing picture: off (hidden below the line)" : `Standing picture at ${Math.round(p.standingAlpha * 100)}% opacity`}</span>
                  <Help label="Standing picture">
                    The ground's own probability, faint all the time, so the where is always on screen; where the hour's weather term
                    crosses the gate the pixel goes opaque and its color takes the full hour (ground plus weather). At 0 the picture is
                    only what crosses the hide line.
                  </Help>
                </span>
              }
            >
              <Slider min={0} max={0.6} step={0.05} value={p.standingAlpha} onChange={p.onStandingAlphaChange} label="Standing opacity" onReset={() => p.onStandingAlphaChange(0.25)} />
            </Field>
            {p.standingAlpha > 0 ? (
              <Field label={`Weather gate: opaque from a weather term of ${p.wxGate >= 0 ? "+" : ""}${p.wxGate.toFixed(1)}${p.wxNow != null ? ` (under the cursor now ${p.wxNow >= 0 ? "+" : ""}${p.wxNow.toFixed(1)})` : ""}`}>
                <Slider min={-2} max={4} step={0.1} value={p.wxGate} onChange={p.onWxGateChange} label="Weather gate" onReset={() => p.onWxGateChange(1.0)} />
              </Field>
            ) : null}
            {p.useWeather ? (
              <Field
                label={
                  <span className="row">
                    <span>{p.fieldAlpha === 0 ? "Weather field under the ground: off" : `Weather field under the ground at ${Math.round(p.fieldAlpha * 100)}%`}</span>
                    <Help label="Weather field">
                      The hour's weather term drawn in gray under everything, ground or not: dark where calm, pale as it
                      nears the gate, so the weather's currents move across the dead space. Where it crosses the gate over
                      fuel, the ground's color sits on top; it is the same term, so the two line up exactly.
                    </Help>
                  </span>
                }
              >
                <Slider min={0} max={0.8} step={0.05} value={p.fieldAlpha} onChange={p.onFieldAlphaChange} label="Weather field opacity" onReset={() => p.onFieldAlphaChange(0.35)} />
              </Field>
            ) : null}
            <div className="row">
              <Check checked={!p.useWeather} onChange={(v) => p.onUseWeatherChange(!v)} label="Ground alone (leave the hour's weather out)" />
            </div>
            {/* burn perimeters: not drawn for now (TODO GeoArrow); see App.tsx */}
          </>
        ) : null}
        {/* the land cover modes (teacher, lit ramp), with the modes above
        {p.fuel !== "ignition" ? (
          <div className="row">
            <Seg value={p.teacher} options={[{ id: "nlcd", label: "NLCD (16 classes)" }, { id: "esri", label: "Esri (9 classes)" }]} onChange={p.onTeacherChange} />
          </div>
        ) : null}
        {p.fuel === "lit" ? (
          <>
            <div className="ramp" role="img" aria-label={`${p.groundRamp} ramp`} style={{ backgroundImage: rampGradient(rampStops(p.groundRamp)) }} />
            <p className="muted small">FFWI x fuel weight, {p.hideBelow} (invisible) to {p.rescaleMax} (ramp top){p.tauHours > 0 ? ` · ${p.tauHours} h fuel lag` : ""}</p>
          </>
        ) : null}
        */}
        <Field label={`Opacity ${Math.round(p.fuelOpacity * 100)}%`}>
          <Slider min={0.05} max={1} step={0.05} value={p.fuelOpacity} onChange={p.onFuelOpacityChange} label="Ground opacity" onReset={() => p.onFuelOpacityChange(0.9)} />
        </Field>
        {/* the afterglow (the past hours' weather held on the ground), while the weather is off
        <Field
          label={
            <span className="row">
              <span>{p.glowHours === 0 ? "Afterglow: off (this hour only)" : `Afterglow: looks back ${p.glowHours} h, the ground holding by land cover`}</span>
              <Help label="Afterglow">
                Off by default, and the data say off: at the same point, once this hour's weather is in, the past day's peak
                FFWI adds nothing to the chance of an ignition (its weight goes slightly negative), and rain in the past
                three days subtracts. The hour is what matters. Turned on, the ground holds the strongest recent hour by
                what it is, for the land cover's hold hours (a stand-in for fuel moisture lagging the air: grass a few
                hours, timber a day), and the gray weather field keeps 0.3 logit an hour.
                Hold hours ({p.teacher === "nlcd" ? "NLCD" : "Esri"}):{" "}
                {HEADS[p.teacher].classes.map((code, i) => `${HEADS[p.teacher].names[i]} ${HOLD[p.teacher][code] ?? 0}`).join(", ")}.
              </Help>
            </span>
          }
        >
          <Slider min={0} max={48} step={1} value={p.glowHours} onChange={p.onGlowHoursChange} label="Afterglow hours" onReset={() => p.onGlowHoursChange(0)} />
        </Field>
        */}
        {/* grain (the fill broken into cells): off, not offered
        <Field label={p.grain === 0 ? "Grain: off (flat fill)" : `Grain: ${p.grain} cells across a tile`}>
          <Slider min={0} max={128} step={8} value={p.grain} onChange={p.onGrainChange} label="Grain" onReset={() => p.onGrainChange(0)} />
        </Field>
        */}
        <div className="row">
          <Check checked={p.light} onChange={p.onLightChange} label="Draw as light (adds up over the map)" />
          {/* <Check checked={p.relief} onChange={p.onReliefChange} label="Relief under the ground" /> */}
        </div>
        {/* class legend with fuel weights, not shown while the classes themselves are not drawn
        <div className="legend">
          {head.classes.map((code, i) => (
            <div key={code} className="legend-row">
              <span className="sw" style={{ background: head.colors[i] }} />
              <span>{head.names[i]}</span>
              <span className="muted">{(FUEL[p.teacher][code] ?? 0).toFixed(2)}</span>
            </div>
          ))}
        </div>
        <p className="muted small">Right column: the fuel weight each class carries (first cut, to be tuned).</p>
        */}
      </Fold>

      {p.fuelOnly ? null : <>
      <Fold k="fire" label="The fire" defaultOpen tag={p.fire.record ? String(p.fire.record.IncidentName ?? "") : p.fire.pick ? "reading" : "none picked"}>
        {!p.fire.pick ? (
          <p className="muted small">Click a fire start or inside a burn outline (the season view, below) for NIFC's record of it.</p>
        ) : !p.fire.record ? (
          <p className="muted small">Reading the record from NIFC...</p>
        ) : (
          <>
            <div className="row">
              <b>{String(p.fire.record.IncidentName ?? "Unnamed")} Fire</b>
              <span className="muted small">{p.fire.pick.kind === "burn" ? "burn outline" : "start"}</span>
              <button type="button" className="small" onClick={p.fire.onClear} aria-label="Clear the fire">Clear</button>
            </div>
            <table className="rec">
              <tbody>
                {fireRows(p.fire.record).map(([k, v]) => (
                  <tr key={k}><th>{k}</th><td>{v}</td></tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </Fold>

      {/* the GOES-West one-fire view: off (2 to 4 km pixels cover a whole burn; goes.ts and its state stay)
      <Fold k="goes" label="One fire from orbit" defaultOpen tag={p.goes.on ? pacific.format(new Date(p.goes.t0 + p.goes.hour * 3600e3)) : "off"}>
        <div className="row">
          <Seg value={p.goes.on ? "on" : "off"} options={[{ id: "on", label: "Show" }, { id: "off", label: "Off" }]} onChange={(v) => p.goes.onChange(v === "on")} />
          <Help label="One fire from orbit">
            The {p.goes.name} Fire as GOES-West saw it: the satellite's fire product (ABI fire / hot spot
            characterization, every 5 minutes), read from NOAA's bucket as the view opens. Each box is one satellite
            pixel, about 3 to 4 km across here, that held a fire that hour; brighter blue for more fire radiative
            power. Faint outlines: pixels seen burning earlier. The fuel is the year before the fire, dim, lit inside
            the final burn outline where the satellite has already seen fire. Pixels are placed as if the ground had
            no height, so a fire high on a ridge can sit about a km off.
          </Help>
        </div>
        {p.goes.on ? (
          <>
            <div className="row">
              <button type="button" className="play" aria-label={p.goes.playing ? "Pause" : "Play"} title={p.goes.playing ? "Pause (Space)" : "Play (Space)"}
                onClick={(e) => { p.goes.onPlayToggle(); e.currentTarget.blur(); }}>
                {p.goes.playing ? "❚❚" : "▶"}
              </button>
              <span>{pacific.format(new Date(p.goes.t0 + p.goes.hour * 3600e3))}</span>
            </div>
            <Field label="Hour">
              <Slider min={0} max={Math.max(0, p.goes.hours - 1)} step={1} value={p.goes.hour} onChange={p.goes.onHourChange} label="Hour" />
            </Field>
            <p className="muted small">
              {p.goes.name} Fire, {p.goes.acres.toLocaleString()} acres{p.goes.cause ? `, ${p.goes.cause.toLowerCase()} cause` : ""}.
              {" "}This hour: {p.goes.now} fire pixel{p.goes.now === 1 ? "" : "s"}{p.goes.nowPower > 0 ? `, ${Math.round(p.goes.nowPower)} MW` : ""}.
              {" "}Seen burning so far: {p.goes.seen}.
              {p.goes.read < p.goes.scans ? ` Reading scans: ${p.goes.read.toLocaleString()} of ${p.goes.scans.toLocaleString()}.` : ""}
            </p>
          </>
        ) : null}
      </Fold>
      */}

      <Fold k="season" label={`The ${p.season.year} season`} defaultOpen tag={p.season.on ? dayLabel(p.season.year, p.season.day) : "off"}>
        <div className="row">
          <Seg value={p.season.on ? "on" : "off"} options={[{ id: "on", label: "Show" }, { id: "off", label: "Off" }]} onChange={(v) => p.season.onChange(v === "on")} />
          <span className="muted small">S</span>
          <Help label="The season">
            Every California wildfire start of {p.season.year} (NIFC WFIGS points of origin) appears on the day it was
            reported, with the burn outlines of the larger fires. The ground under them is the fuel of {p.season.year - 1},
            the year before the fires, so the map cannot see their scars. A start's circle turns light blue when there is
            fuel above the hide line within {p.season.reach} m of it, and the fuel inside the circle lights up; inside a burn
            the fuel lights up too. Gray: no fuel that close. Everything else stays dim. Fuel shows from zoom {MOSAIC_MIN_ZOOM} in.
          </Help>
        </div>
        {p.season.on ? (
          <>
            <div className="row">
              <button type="button" className="play" aria-label={p.season.playing ? "Pause" : "Play"} title={p.season.playing ? "Pause" : "Play"}
                onClick={(e) => { p.season.onPlayToggle(); e.currentTarget.blur(); }}>
                {p.season.playing ? "❚❚" : "▶"}
              </button>
              <span>{dayLabel(p.season.year, p.season.day)}</span>
            </div>
            <Field label="Day of the season">
              <Slider min={0} max={p.season.days - 1} step={1} value={p.season.day} onChange={p.season.onDayChange} label="Day of the season" onReset={() => p.season.onDayChange(p.season.days - 1)} />
            </Field>
            <Field label={`Reach ${p.season.reach} m${p.season.reach === 30 ? " (about 100 ft)" : ""}`}>
              <Slider min={10} max={150} step={5} value={p.season.reach} onChange={p.season.onReachChange} label="Reach" onReset={() => p.season.onReachChange(30)} />
            </Field>
            {p.season.loading ? (
              <p className="muted small">Reading the {p.season.year} starts and burns from NIFC...</p>
            ) : (
              <p className="muted small">
                Through {dayLabel(p.season.year, p.season.day)}: {p.season.total.toLocaleString()} starts in California,{" "}
                {p.season.burns.toLocaleString()} burn outlines. In view: {p.season.inView.toLocaleString()} starts
                {p.season.inView - p.season.unread > 0
                  ? `, ${p.season.found.toLocaleString()} of the ${(p.season.inView - p.season.unread).toLocaleString()} read (${Math.round((100 * p.season.found) / (p.season.inView - p.season.unread))}%) with fuel within ${p.season.reach} m`
                  : ""}
                {p.season.unread > 0 ? `; ${p.season.unread.toLocaleString()} not read yet${p.zoom < MOSAIC_MIN_ZOOM ? ` (zoom in past ${MOSAIC_MIN_ZOOM})` : ""}` : ""}.
              </p>
            )}
          </>
        ) : null}
      </Fold>
      </>}

      {/* the time controls only with the weather on (App.tsx WEATHER_ON) */}
      {p.weatherOn ? (
      <Fold k="time" label="Time" defaultOpen tag={timeTag}>
        <div className="row">
          {/* the single 48 h run mode is not offered; the code path stays (timeMode "run") */}
          <Help label="Time">
            The window: one 12Z HRRR run per day, its first 24 hours each, stitched into one hourly film, up to a week,
            fetched as you watch: only the shards under the view, the day on screen first, then the days after it. Play
            runs over the days in hand and wraps there until the rest arrive; the strip under the slider shows which days
            are in. About 18 MB per shard per day over the wire, kept as 4 MB of bytes in memory and not written to disk:
            a region is 25 to 100 MB, a whole week over the whole country about 3 GB over the wire and 640 MB in memory.
          </Help>
        </div>
        {p.timeMode === "week" ? (
          <>
            <Field label={`Window (UTC): ${p.weekDays} day${p.weekDays === 1 ? "" : "s"}, up to ${MAX_WEEK_DAYS} (about ${Math.round(p.weekDays * 0.43 * 10) / 10} GB over the wire for the whole country, 25 MB a day for a region)`}>
              <div className="row" style={{ flexWrap: "nowrap", alignItems: "center", gap: 6 }}>
                <input
                  type="date"
                  aria-label="First day"
                  min={isoDateString(INIT_TIME_ORIGIN)}
                  max={p.initTimeCount > 0 ? isoDateString(dateFromInitTimeIdx(p.initTimeCount - 1)) : undefined}
                  value={isoDateString(p.weekStart)}
                  disabled={p.initTimeCount === 0}
                  onChange={(e) => {
                    if (!e.target.value) return;
                    const start = new Date(`${e.target.value}T00:00:00Z`);
                    // keep the end day where it is when it still fits; else keep the length
                    const days = Math.round((weekEnd.getTime() - start.getTime()) / 864e5) + 1;
                    p.onWeekStartChange(start);
                    if (days >= 1 && days <= MAX_WEEK_DAYS) p.onWeekDaysChange(days);
                  }}
                />
                <span className="muted small">to</span>
                <input
                  type="date"
                  aria-label="Last day"
                  min={isoDateString(p.weekStart)}
                  max={isoDateString(new Date(Math.min(p.weekStart.getTime() + (MAX_WEEK_DAYS - 1) * 864e5, p.initTimeCount > 0 ? dateFromInitTimeIdx(p.initTimeCount - 1).getTime() : Infinity)))}
                  value={isoDateString(weekEnd)}
                  disabled={p.initTimeCount === 0}
                  onChange={(e) => {
                    if (!e.target.value) return;
                    const end = new Date(`${e.target.value}T00:00:00Z`);
                    const days = Math.round((end.getTime() - p.weekStart.getTime()) / 864e5) + 1;
                    p.onWeekDaysChange(Math.max(1, Math.min(MAX_WEEK_DAYS, days)));
                  }}
                />
                {latestDay !== null && !onLatestDay ? (
                  <a
                    href="#"
                    className="small"
                    onClick={(e) => {
                      e.preventDefault();
                      // the window ends on the latest day in the store; the playhead goes to that day's first hour
                      const start = new Date(latestDay.getTime());
                      start.setUTCDate(start.getUTCDate() - (p.weekDays - 1));
                      p.onWeekStartChange(start);
                      p.onHourIdxChange((p.weekDays - 1) * HOURS_PER_DAY);
                    }}
                  >
                    Current forecast
                  </a>
                ) : latestDay !== null ? (
                  <span className="muted small">latest in the store</span>
                ) : null}
              </div>
            </Field>
            <div className="days" role="list" aria-label="Days of the window in hand">
              {Array.from({ length: p.weekDays }, (_, d) => {
                const st = p.weekDayStates[d] ?? "waiting";
                const date = new Date(p.weekStart.getTime());
                date.setUTCDate(date.getUTCDate() + d);
                const on = Math.floor(p.hourIdx / HOURS_PER_DAY) === d;
                const word = st === "in" ? "in hand" : st === "loading" ? "loading" : st === "none" ? "not in the store" : "waiting";
                return (
                  <button
                    key={d}
                    type="button"
                    role="listitem"
                    className={`day ${st}${on ? " on" : ""}`}
                    title={`${isoDateString(date)}: ${word}`}
                    aria-label={`Day ${d + 1}, ${isoDateString(date)}, ${word}`}
                    onClick={(e) => { p.onHourIdxChange(d * HOURS_PER_DAY); e.currentTarget.blur(); }}
                  >
                    {date.getUTCDate()}
                  </button>
                );
              })}
            </div>
            <Field label={<span>Day {Math.floor(p.hourIdx / HOURS_PER_DAY) + 1} · {p.weekHourLabel} (+{p.hourIdx % HOURS_PER_DAY} h into that day's run)</span>}>
              <div className="row" style={{ flexWrap: "nowrap" }}>
                <button type="button" className="play" aria-label={p.isPlaying ? "Pause" : "Play"} title={p.isPlaying ? "Pause (Space)" : "Play (Space)"} onClick={(e) => { p.onPlayPauseToggle(); e.currentTarget.blur(); }}>
                  {p.isPlaying ? "❚❚" : "▶"}
                </button>
                <Slider min={0} max={Math.max(0, p.weekDepth - 1)} step={1} value={p.hourIdx} onChange={p.onHourIdxChange} label="Week hour" onReset={() => p.onHourIdxChange(0)} />
              </div>
            </Field>
            <Field label={`Seconds per hour: ${(p.frameDurationMs / 1000).toFixed(2)} (the window in ${Math.round((p.frameDurationMs * p.weekDepth) / 1000)} s)`}>
              <Slider min={40} max={1000} step={20} value={p.frameDurationMs} onChange={p.onFrameDurationMsChange} label="Frame duration" onReset={() => p.onFrameDurationMsChange(200)} />
            </Field>
          </>
        ) : (
          <>
        <Field label="Forecast run (UTC)">
          <div className="row">
            <input
              type="date"
              min={isoDateString(INIT_TIME_ORIGIN)}
              max={p.initTimeCount > 0 ? isoDateString(dateFromInitTimeIdx(p.initTimeCount - 1)) : undefined}
              value={isoDateString(dateFromInitTimeIdx(p.initTimeIdx))}
              disabled={p.initTimeCount === 0}
              onChange={(e) => p.onInitTimeIdxChange(initTimeIdxFromDate(new Date(`${e.target.value}T12:00:00Z`), Math.max(0, p.initTimeCount - 1)))}
            />
            {p.initTimeCount > 0 && p.initTimeIdx < p.initTimeCount - 1 ? (
              <a href="#" className="small" onClick={(e) => { e.preventDefault(); p.onInitTimeIdxChange(p.initTimeCount - 1); }}>Jump to latest forecast</a>
            ) : (
              <span className="muted small">Latest forecast run</span>
            )}
          </div>
        </Field>
        <Field
          label={
            <span className="row">
              <span>+<span style={{ display: "inline-block", minWidth: "2ch", textAlign: "right" }}>{hours}</span> h · {validStr}</span>
              <Help label="Forecast hour">49 hourly steps, +0 to +48 h, playing on open. Space toggles play. The run opens on the Southern Plains outbreak (2025-03-14, 12Z run; the wind peaks that afternoon from Texas to Kansas); pick any date, or jump to the latest run.</Help>
            </span>
          }
        >
          <div className="row" style={{ flexWrap: "nowrap" }}>
            <button type="button" className="play" aria-label={p.isPlaying ? "Pause" : "Play"} title={p.isPlaying ? "Pause (Space)" : "Play (Space)"} onClick={(e) => { p.onPlayPauseToggle(); e.currentTarget.blur(); }}>
              {p.isPlaying ? "❚❚" : "▶"}
            </button>
            <Slider min={0} max={HRRR_LEAD_TIME_COUNT - 1} step={1} value={p.leadTimeIdx} onChange={p.onLeadTimeIdxChange} label="Forecast hour" onReset={() => p.onLeadTimeIdxChange(0)} />
          </div>
        </Field>
        <Field label={`Seconds per forecast hour: ${(p.frameDurationMs / 1000).toFixed(2)}`}>
          <Slider min={80} max={1000} step={20} value={p.frameDurationMs} onChange={p.onFrameDurationMsChange} label="Frame duration" onReset={() => p.onFrameDurationMsChange(200)} />
        </Field>
          </>
        )}
      </Fold>
      ) : null}

      {/* the 3 km weather field drawn itself (FFWI contours, the field, its ramp, the hide line, the fuel lag, the wind
          mode): not shown for now; the picture is the ground. The state behind it is still in App.tsx.
      <Fold k="field" label="The weather" defaultOpen tag={fieldOptions.find((o) => o.id === field)?.label}>
        <Field label={p.contourStep === 0 ? "Contours: off" : `Contours of the FFWI field every ${p.contourStep}, from the hide line up`}>
          <Slider min={0} max={50} step={5} value={p.contourStep} onChange={p.onContourStepChange} label="Contour step" onReset={() => p.onContourStepChange(0)} />
        </Field>
        <div className="row">
          <span className="muted small">Draw the 3 km field itself (puts the ground off; one picture at a time):</span>
          <Help label="FFWI">
            Fosberg 1978. Temperature and humidity set how dry fine fuel is; wind does the escalating; bone-dry fuel in a 30 mph
            wind = 100. Sustained 10 m wind is the standard input. 3 km HRRR pixels, reprojected on the GPU.
          </Help>
        </div>
        <Seg value={field} options={fieldOptions} onChange={setField} />
        {p.showRaster ? (
          <>
            <div className="row" style={{ flexWrap: "nowrap" }}>
              <select value={p.colormapId} onChange={(e) => p.onColormapChange(e.target.value as ColormapChoiceId)} style={{ flex: "0 0 auto" }}>
                {COLORMAP_CHOICES.map((c) => (<option key={c.id} value={c.id}>{c.label}</option>))}
              </select>
              <Ramp spriteUrl={colormapsPngUrl} rowCount={COLORMAP_ROW_COUNT} rowIndex={COLORMAP_INDEX[p.colormapId]!} label={fieldRampLabel} />
            </div>
            <p className="muted small">{fieldRampLabel}</p>
            <Field label={`Opacity ${Math.round(p.rasterOpacity * 100)}%`}>
              <Slider min={0} max={1} step={0.01} value={p.rasterOpacity} onChange={p.onRasterOpacityChange} label="Field opacity" onReset={() => p.onRasterOpacityChange(0.7)} />
            </Field>
          </>
        ) : null}
        <Field label={p.hideBelow > 0 ? `Hide below ${p.hideBelow} (the ground and the field both; the map shows through)` : "Showing every pixel"}>
          <Slider min={0} max={80} step={1} value={p.hideBelow} onChange={p.onHideBelowChange} label="Hide below" onReset={() => p.onHideBelowChange(20)} />
        </Field>
        <Field label={`Ramp top (fully pushing): ${p.rescaleMax}`}>
          <Slider min={40} max={200} step={5} value={p.rescaleMax} onChange={p.onRescaleMaxChange} label="Rescale max" onReset={() => p.onRescaleMaxChange(100)} />
        </Field>
        <Field
          label={
            <span className="row">
              <span>
                {p.tauHours === 0
                  ? "Fuel moisture lag: 0 h (Fosberg as published, fuel dries and wets instantly)"
                  : `Fuel moisture lag: ${p.tauHours} h (memory reaches back ${Math.min(p.leadTimeIdx, 3 * p.tauHours)} of ~${3 * p.tauHours} h at this frame)`}
              </span>
              <Help label="Timelag">
                Fosberg uses the equilibrium moisture the air implies right now. Real dead fuel lags behind it: the standard NFDRS
                form is m(t) = m_eq(t) + (m(t-1) - m_eq(t)) x exp(-1/tau), with tau about 1 h for fine fuels and 10 h for the
                10-hour class. With tau set, the moisture term carries that memory (seeded at the run's +0 h) and wind stays
                instantaneous. The ground, the field and the tooltip all use the lagged value.
              </Help>
            </span>
          }
        >
          <Slider min={0} max={24} step={1} value={p.tauHours} onChange={p.onTauHoursChange} label="Fuel moisture lag" onReset={() => p.onTauHoursChange(0)} />
        </Field>
        <select value={p.windMode} onChange={(e) => p.onWindModeChange(e.target.value as WindMode)}>
          <option value="sustained">wind: 10 m sustained (standard FFWI)</option>
          <option value="gust">wind: gusts (higher; not comparable to FFWI tables)</option>
        </select>
        <Check checked={p.showRaster} onChange={p.onShowRasterChange} label="Show the field" />
      </Fold>
      */}

      {s.error ? <p className="warn small" style={{ marginTop: 8 }}>{s.error}</p> : null}

      <div className="pn-foot">
        <span>Built with <A href="https://github.com/developmentseed/deck.gl-raster">deck.gl-raster</A> by <A href="https://developmentseed.org">Development Seed</A></span>
        <span>© <A href="https://carto.com/attributions">CARTO</A>, © <A href="https://www.openstreetmap.org/copyright">OpenStreetMap</A> contributors</span>
      </div>
    </div>
  );
}
