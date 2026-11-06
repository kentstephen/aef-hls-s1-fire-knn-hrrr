import type { PickingInfo } from "@deck.gl/core";
import { BitmapLayer, ScatterplotLayer } from "@deck.gl/layers";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MapRef } from "react-map-gl/maplibre";
import { Map as MaplibreMap } from "react-map-gl/maplibre";
import { AefPool } from "../aef/pool.js";
import { DeckGlOverlay } from "../lib/deckgl-overlay.js";
import { type BurnArea, burnAreaLayer, loadCaBurnAreas } from "../perimeters.js";
import { CONT_RADIUS_M, EMB_YEAR, type PointFactors, RiskFactors } from "./factors.js";
import { FireHistory } from "./history.js";
import { BUILDINGS_MIN_ZOOM, KIND_COLORS, type OvertureFeature, overtureLayers, riskMapStyle } from "./overture.js";
import { FACE_SLOPE, SLOPE_FULL, TIME_FULL, viridisCss, VIRIDIS_GRADIENT } from "./score.js";
import "./risk.css";

/**
 * Where fuel is ready to burn, and why: the factors under the map read on demand for the area in view (fuel from the
 * embeddings, fuel continuity, slope and facing, fire history), put together as one value per 10 m pixel (score.ts)
 * and drawn in viridis. Clicking a spot lists the parts behind its value. Burned areas (2020 on) frame the map, B
 * toggles them. An exploration, not a forecast.
 *
 * The basemap is Overture's (overture.ts), colored by what each feature is; the risk under a feature
 * is on its hover and its card. CARTO sits under the picture (seen only where it is not drawn yet), its labels on top.
 */

/** Near Cohasset, on the Park Fire's west edge. */
const START_VIEW = { longitude: -121.82, latitude: 39.98, zoom: 13 };
/** Below this the view holds too many chunks to read on demand. */
const MIN_ZOOM = 12;
const SETTLE_MS = 400;
/** Opaque, so CARTO under it never shows through where the picture is drawn. */
const OPACITY = 1;
const NOW_YEAR = new Date().getFullYear();

const COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
const compass = (a: number) => (Number.isNaN(a) ? "flat" : COMPASS[Math.round(a / 45) % 8]!);
const pct = (v: number | null) => (v === null ? "no data" : `${Math.round(v * 100)}%`);

/** A feature under the pointer: what it is, its name, and every field it carries. */
type Inspect = { title: string; name: string | null; fields: [string, string][] };

const THEME: Record<string, string> = { "ov-water": "water", "ov-roads": "road or rail", "ov-infrastructure": "infrastructure", "ov-land-use": "land use", "ov-buildings": "building" };

function describe(info: PickingInfo): Inspect | null {
  const o = info.object as unknown;
  const id = info.layer?.id ?? "";
  const theme = Object.keys(THEME).find((k) => id.startsWith(k));
  if (!o || !theme) return null;
  const raw: Record<string, unknown> = (o as OvertureFeature).properties;
  const fields: [string, string][] = [];
  for (const [k, v] of Object.entries(raw)) {
    if (v == null || v === "" || k === "geometry" || k === "cx" || k === "cy" || k.startsWith("__")) continue;
    const text = typeof v === "string" ? v : typeof v === "number" ? String(Math.round(v * 100) / 100) : JSON.stringify(v);
    fields.push([k, text.length > 140 ? `${text.slice(0, 140)}…` : text]);
  }
  const kind = [raw.subtype, raw.class].filter((v) => v != null && v !== "").join(" · ");
  return { title: `${THEME[theme]}${kind ? `: ${kind}` : ""}`, name: (raw["@name"] as string | undefined) ?? null, fields };
}

export default function Risk() {
  const mapRef = useRef<MapRef>(null);
  const pool = useMemo(() => new AefPool(), []);
  const [areas, setAreas] = useState<BurnArea[] | null>(null);
  const [burnOn, setBurnOn] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const factorsRef = useRef<RiskFactors | null>(null);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [zoom, setZoom] = useState(START_VIEW.zoom);
  const [picked, setPicked] = useState<{ lon: number; lat: number } | null>(null);
  const [pickedF, setPickedF] = useState<PointFactors | null>(null);
  /** The feature clicked, and the one under the pointer. */
  const [pickedFeature, setPickedFeature] = useState<Inspect | null>(null);
  const [hover, setHover] = useState<{ d: Inspect; risk: number | null; x: number; y: number } | null>(null);
  /** Bumped after each chunk read, so the picked spot's list fills in as its neighbors arrive. */
  const [readVersion, setReadVersion] = useState(0);
  /** Bumped whenever a chunk's picture changes. */
  const [drawVersion, setDrawVersion] = useState(0);
  /** CARTO trimmed to water and labels, and the layers deck.gl's go under. */
  const [style, setStyle] = useState<Awaited<ReturnType<typeof riskMapStyle>> | null>(null);
  useEffect(() => { void riskMapStyle().then(setStyle); }, []);

  // ---- burned areas, 2020 on: the framing, and the recent half of the fire history
  useEffect(() => {
    const ac = new AbortController();
    loadCaBurnAreas(ac.signal)
      .then((a) => {
        setAreas(a);
        factorsRef.current = new RiskFactors(pool, new FireHistory(a), () => setDrawVersion((v) => v + 1));
        readView();
      })
      .catch((e) => { if (!ac.signal.aborted) setError(`burned areas: ${String(e)}`); });
    return () => ac.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pool]);

  // ---- the factors for the chunks in view, read once the map settles; a new view aborts the last read
  const readAc = useRef<AbortController | null>(null);
  const settle = useRef(0);
  const readView = useCallback(() => {
    const map = mapRef.current?.getMap();
    const f = factorsRef.current;
    if (!map || !f) return;
    readAc.current?.abort();
    setZoom(map.getZoom());
    if (map.getZoom() < MIN_ZOOM) { setProgress(null); return; }
    const b = map.getBounds();
    const ac = (readAc.current = new AbortController());
    void f.load([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()], (done, total) => {
      if (ac.signal.aborted) return;
      setProgress({ done, total });
      setReadVersion((v) => v + 1);
    }, ac.signal);
  }, []);
  const onMove = useCallback(() => {
    setZoom(mapRef.current?.getMap().getZoom() ?? 0);
    window.clearTimeout(settle.current);
    settle.current = window.setTimeout(readView, SETTLE_MS);
  }, [readView]);

  useEffect(() => {
    if (!picked || !factorsRef.current) return setPickedF(null);
    setPickedF(factorsRef.current.at(picked.lon, picked.lat));
  }, [picked, readVersion, drawVersion]);

  // ---- keys: B burned areas, F full screen, Esc clears the spot; the map follows any size change
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      if (e.key === "b" || e.key === "B") setBurnOn((v) => !v);
      else if (e.key === "Escape") { setPicked(null); setPickedFeature(null); }
      else if (e.key === "f" || e.key === "F") {
        if (document.fullscreenElement) void document.exitFullscreen();
        else void document.documentElement.requestFullscreen();
      }
    };
    window.addEventListener("keydown", onKey);
    const onSize = () => mapRef.current?.getMap().resize();
    window.addEventListener("resize", onSize);
    document.addEventListener("fullscreenchange", onSize);
    return () => { window.removeEventListener("keydown", onKey); window.removeEventListener("resize", onSize); document.removeEventListener("fullscreenchange", onSize); };
  }, []);

  const layers = useMemo(() => [
    ...[...(factorsRef.current?.chunks.values() ?? [])].filter((c) => c.image).map((c) =>
      new BitmapLayer({
        id: `risk-${c.cy}-${c.cx}`,
        image: c.image,
        bounds: c.bounds,
        opacity: OPACITY,
        textureParameters: { minFilter: "nearest", magFilter: "nearest" },
        beforeId: style?.groundBefore,
      } as never)),
    ...overtureLayers({ beforeId: style?.linesBefore, zoom }),
    areas ? burnAreaLayer({ areas, visible: burnOn, beforeId: style?.linesBefore }) : null,
    picked
      ? new ScatterplotLayer({ id: "risk-spot", data: [picked], getPosition: (d: { lon: number; lat: number }) => [d.lon, d.lat], getRadius: 5, radiusUnits: "pixels", getFillColor: [255, 255, 255, 255] })
      : null,
    // eslint-disable-next-line react-hooks/exhaustive-deps
  ], [areas, burnOn, picked, drawVersion, style, zoom >= MIN_ZOOM, zoom >= BUILDINGS_MIN_ZOOM]);

  const f = pickedF;
  return (
    <div className="risk">
      {style ? <MaplibreMap
        ref={mapRef}
        initialViewState={START_VIEW}
        mapStyle={style.style}
        attributionControl={{ compact: true }}
        maxPitch={0}
        dragRotate={false}
        pitchWithRotate={false}
        touchPitch={false}
        style={{ width: "100%", height: "100%" }}
        onMove={onMove}
        onLoad={readView}
        hash
      >
        <DeckGlOverlay
          layers={layers as never}
          interleaved
          onClick={(i: PickingInfo) => {
            if (i.coordinate) setPicked({ lon: i.coordinate[0]!, lat: i.coordinate[1]! });
            setPickedFeature(describe(i));
          }}
          onHover={(i: PickingInfo) => {
            const d = describe(i);
            if (!d || !i.coordinate) return setHover(null);
            const r = factorsRef.current?.at(i.coordinate[0]!, i.coordinate[1]!)?.parts?.risk;
            setHover({ d, risk: r == null ? null : r, x: i.x, y: i.y });
          }}
          getCursor={({ isHovering }: { isHovering: boolean }) => (isHovering ? "pointer" : "grab")}
        />
      </MaplibreMap> : null}
      {hover ? (
        <div className="risk-tip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
          <b>{hover.d.title}</b>
          {hover.d.name ? <div>{hover.d.name}</div> : null}
          <div>{hover.risk == null ? "risk not read here" : `risk ${Math.round(hover.risk * 100)} of 100`}</div>
        </div>
      ) : null}
      <div className="risk-card">
        <div className="risk-title">Where fuel is ready to burn</div>
        <div className="risk-sub">An exploration, not a forecast. Not yet tested against past fires.</div>
        <div className="risk-legend" style={{ background: VIRIDIS_GRADIENT }} />
        <div className="risk-legend-ends"><span>lower</span><span>higher</span></div>
        <div className="risk-kinds">
          {KIND_COLORS.map((k) => (
            <span key={k.name}><i style={{ background: `rgb(${k.rgb.join(",")})` }} />{k.name}</span>
          ))}
        </div>
        <div className="risk-tools">
          <button className={burnOn ? "on" : ""} onClick={() => setBurnOn((v) => !v)} title="Burned areas, 2020 on (B)">Burned areas</button>
        </div>
        <div className="risk-sub risk-status">
          {error ??
            (!areas
              ? "loading burned areas"
              : zoom < MIN_ZOOM
                ? `zoom in to ${MIN_ZOOM} to read the factors`
                : progress
                  ? progress.done >= progress.total
                    ? `${progress.total} chunks read`
                    : `chunks read ${progress.done} of ${progress.total}`
                  : "")}
        </div>
        {progress && progress.done < progress.total ? <div className="risk-bar"><div style={{ width: `${(100 * progress.done) / progress.total}%` }} /></div> : null}
        {pickedFeature ? (
          <div className="risk-feature">
            <div className="risk-feature-title">{pickedFeature.title}</div>
            {pickedFeature.name ? <div>{pickedFeature.name}</div> : null}
            <div>{f?.parts ? `risk here ${Math.round(f.parts.risk * 100)} of 100` : "risk not read here"}</div>
            <table>
              <tbody>
                {pickedFeature.fields.map(([k, v]) => <tr key={k}><td>{k}</td><td>{v}</td></tr>)}
              </tbody>
            </table>
          </div>
        ) : null}
        {picked ? (
          <div className="risk-spot">
            {!f ? (
              <div className="risk-sub">this spot is not read yet</div>
            ) : (
              <SpotCard f={f} />
            )}
          </div>
        ) : (
          <div className="risk-sub">click a spot to see why</div>
        )}
        <div className="risk-foot">
          Value = fuel here × the average of four parts, each counted the same. Fuel: AlphaEarth embeddings
          ({EMB_YEAR}), fuel model. Terrain: Mapterhorn. Fire history: NIFC, 100 acres and up, from about 1900.
        </div>
      </div>
    </div>
  );
}

/** The spot's value and the four parts behind it, highest first; then the context that is not in the value. */
function SpotCard({ f }: { f: PointFactors }) {
  const p = f.parts;
  if (!p) return <div className="risk-sub">no embedding here</div>;
  const rows = [
    { name: `Fuel within ${CONT_RADIUS_M} m`, v: p.continuity, text: pct(f.near) },
    { name: "Time since last fire", v: p.time, text: f.last ? `${NOW_YEAR - f.last} yr (${f.last}, ${f.fires[0]?.name ?? "?"})` : "none on record" },
    { name: "Slope", v: p.slope, text: Number.isNaN(f.slope) ? "no data" : `${Math.round(f.slope)}°` },
    { name: "Facing", v: p.facing, text: compass(f.aspect) },
  ].sort((a, b) => b.v - a.v);
  return (
    <>
      <div className="risk-value">
        <span className="risk-swatch" style={{ background: viridisCss(p.risk) }} />
        <b>{Math.round(p.risk * 100)}</b> of 100
      </div>
      <div className="risk-sub">fuel here {pct(f.fuel)} × average of:</div>
      <table>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name}>
              <td>{r.name}</td>
              <td className="risk-part"><div style={{ width: `${Math.round(r.v * 100)}%`, background: viridisCss(r.v) }} /></td>
              <td>{r.text}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="risk-note">
        Slope counts fully at {SLOPE_FULL}° and up, south-west facing most (in step with the slope up to {FACE_SLOPE}°), time
        since a fire fully at {TIME_FULL} years.
      </div>
      <table className="risk-context">
        <tbody>
          <tr>
            <td>Connected fuel patch</td>
            <td>{f.patchKm2 === null ? "no data" : f.patchKm2 === 0 ? "none here" : `${f.patchOpen ? "at least " : ""}${f.patchKm2 < 10 ? f.patchKm2.toFixed(1) : Math.round(f.patchKm2)} km²`}</td>
          </tr>
          <tr><td>Years burned on record</td><td>{f.count}{f.fires.length ? ` (${[...new Set(f.fires.map((x) => x.year))].join(", ")})` : ""}</td></tr>
        </tbody>
      </table>
    </>
  );
}
