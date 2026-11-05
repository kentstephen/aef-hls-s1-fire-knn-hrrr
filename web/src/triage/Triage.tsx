import { ScatterplotLayer } from "@deck.gl/layers";
import type { PickingInfo } from "@deck.gl/core";
import HighTable, { type CellContentProps, type OrderBy, type Selection } from "hightable";
import "hightable/src/HighTable.css";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MapRef } from "react-map-gl/maplibre";
import { Map as MaplibreMap } from "react-map-gl/maplibre";
import { type FirePixel, type GoesDoc, loadGoesDoc, readGoesFire } from "../goes.js";
import { goesEvents, ringBox } from "../goesboxes.js";
import { AefPool } from "../aef/pool.js";
import { DeckGlOverlay } from "../lib/deckgl-overlay.js";
import { eventFuel } from "./fuel.js";
import { COLUMNS, TriageFrame } from "./frame.js";
import { triageRow, type TriageRow } from "./rows.js";
import "./triage.css";

/**
 * GOES triage: map left, table right. Every fire GOES-West saw over California in the replay week fills the table
 * as its scans are read, ranked by the GOES-only score; each event is one mark on the map. Clicking a row or a mark
 * picks it in both and flies the map to it. Once every scan is read, the embeddings are read event by event in the
 * table's order (top first), filling fuel under and fuel around.
 */

const DOC = "ca-2024-07-22";
const WEEK = "Jul 22 to Aug 2, 2024";
const MAP_STYLE_URL = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
const CA_VIEW = { longitude: -119.4, latitude: 37.3, zoom: 5.6 };
/** Events are joined again from all pixels read so far at most this often while scans stream in. */
const REJOIN_MS = 1500;

/** Marks: one amber for all for now (encoding to be decided); the picked one is white. */
const MARK: [number, number, number] = [255, 176, 40];
const PICKED: [number, number, number] = [255, 255, 255];
const radius = (r: TriageRow) => Math.min(18, 3 + 2.2 * Math.sqrt(r.areaKm2 / 4));

/** Pixel centers inside California's Census outline only (web/public/ca.json, scripts/ca_outline.py), as
 * scripts/goes_events.py keeps them: the read box reaches into Mexico, Nevada and Arizona. */
type Ring = [number, number][];
function inside(rings: Ring[], x: number, y: number): boolean {
  let n = 0;
  for (const r of rings)
    for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
      const [xi, yi] = r[i]!, [xj, yj] = r[j]!;
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) n++;
    }
  return n % 2 === 1;
}
const center = (p: FirePixel): [number, number] => [p.polygon.reduce((a, q) => a + q[0], 0) / 4, p.polygon.reduce((a, q) => a + q[1], 0) / 4];

/** The embeddings of the year before the fires. */
const EMB_YEAR = 2023;
/** Events whose embeddings are read at the same time (each reads up to MAX_CHUNKS chunks of 3 years). */
const FUEL_AT_ONCE = 2;

const DEFAULT_ORDER: OrderBy = [{ column: "GOES score", direction: "descending" }];

export default function Triage() {
  const mapRef = useRef<MapRef>(null);
  const frame = useMemo(() => new TriageFrame(), []);
  const [rows, setRows] = useState<TriageRow[]>([]);
  const [doc, setDoc] = useState<GoesDoc | null>(null);
  const [read, setRead] = useState(0);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [orderBy, setOrderBy] = useState<OrderBy>(DEFAULT_ORDER);
  const [picked, setPicked] = useState<string | null>(null);
  /** The row under the pointer in the table: its mark lights up on the map. */
  const [rowHover, setRowHover] = useState<string | null>(null);
  const [hover, setHover] = useState<{ r: TriageRow; x: number; y: number } | null>(null);
  const [cell, setCell] = useState({ colIndex: 1, rowIndex: 1 });
  const [fuelDone, setFuelDone] = useState(0);
  const [fuelReading, setFuelReading] = useState<string[]>([]);

  // ---- GOES: the week's scans stream in; events and rows are rebuilt from every pixel read so far
  const started = useRef(false);
  useEffect(() => {
    if (started.current) return;
    started.current = true;
    const pixels: FirePixel[] = [];
    let n = 0, timer = 0;
    const rebuild = () => {
      timer = 0;
      const next = goesEvents(pixels).map(triageRow);
      frame.set(next);
      setRows(next);
      setRead(n);
    };
    Promise.all([
      loadGoesDoc(`${import.meta.env.BASE_URL}goes/${DOC}.json`),
      fetch(`${import.meta.env.BASE_URL}ca.json`).then((r) => r.json() as Promise<Ring[]>),
    ])
      .then(([d, ca]) => {
        setDoc(d);
        return readGoesFire(d, (_i, px) => {
          n++;
          for (const p of px) if (inside(ca, ...center(p))) pixels.push(p);
          if (!timer) timer = window.setTimeout(rebuild, REJOIN_MS);
        });
      })
      .then(() => { window.clearTimeout(timer); rebuild(); setDone(true); })
      .catch((e) => setError(`GOES: ${String(e)}`));
  }, [frame]);

  // ---- embeddings: once the events are final, read each one's fuel, next in the table's current order
  const poolRef = useRef<AefPool | null>(null);
  const orderRef = useRef(orderBy);
  orderRef.current = orderBy;
  useEffect(() => {
    if (!done) return;
    const pool = (poolRef.current ??= new AefPool());
    const ac = new AbortController();
    const refresh = () => {
      frame.set([...frame.rows]);
      setRows(frame.rows);
      setFuelDone(frame.rows.filter((r) => r.fuelState === "done" || r.fuelState === "failed").length);
      setFuelReading(frame.rows.filter((r) => r.fuelState === "reading").map((r) => r.key));
    };
    const next = (): TriageRow | undefined => {
      for (let i = 0; i < frame.rows.length; i++) {
        const r = frame.rows[frame.index(i, orderRef.current)]!;
        if (r.fuelState === "waiting") return r;
      }
    };
    const worker = async () => {
      for (let r = next(); r && !ac.signal.aborted; r = next()) {
        r.fuelState = "reading";
        refresh();
        try {
          const f = await eventFuel(pool, r, EMB_YEAR, ac.signal);
          r.fuelUnder = f.under;
          r.fuelAround = f.around;
          r.fuelChunks = [f.chunks, f.chunksTouched];
          r.fuelState = "done";
        } catch (e) {
          // stopped (page effect torn down): this event goes back in the queue
          if ((e as Error).name === "AbortError") { r.fuelState = "waiting"; return; }
          console.warn("embeddings:", e);
          r.fuelState = "failed";
        }
        refresh();
      }
    };
    void Promise.all(Array.from({ length: FUEL_AT_ONCE }, worker));
    return () => ac.abort();
  }, [done, frame]);

  // ---- picking: a row or a mark; the map flies to it, the table selects and scrolls to it
  const pickedIdx = picked === null ? -1 : rows.findIndex((r) => r.key === picked);
  const pickedRow = pickedIdx >= 0 ? rows[pickedIdx]! : null;
  const selection = useMemo<Selection>(() => (pickedIdx >= 0 ? { ranges: [{ start: pickedIdx, end: pickedIdx + 1 }], anchor: pickedIdx } : { ranges: [] }), [pickedIdx]);

  const flyTo = useCallback((r: TriageRow) => {
    const m = mapRef.current?.getMap();
    if (!m) return;
    const [w, s, e, n] = ringBox(r.ev.bbox);
    const cam = m.cameraForBounds([[w, s], [e, n]], { padding: 80 });
    m.flyTo({ center: [r.lon, r.lat], zoom: Math.min(11, cam?.zoom ?? 10), duration: 1400, essential: true });
  }, []);

  const pickRow = useCallback((r: TriageRow, scroll: boolean) => {
    setPicked(r.key);
    flyTo(r);
    if (scroll) {
      const i = frame.rows.indexOf(r);
      // aria row index: 1-based, the header is row 1
      if (i >= 0) setCell((c) => ({ colIndex: c.colIndex, rowIndex: frame.position(i, orderBy) + 2 }));
    }
  }, [flyTo, frame, orderBy]);

  // clicking the picked row again lets it go
  const onMouseDownCell = useCallback((_e: unknown, _col: number, row: number) => {
    const r = frame.rows[row];
    if (!r) return;
    if (r.key === picked) setPicked(null);
    else pickRow(r, false);
  }, [frame, pickRow, picked]);
  // the row header checkbox: ticked picks, unticked lets go
  const onSelectionChange = useCallback((s: Selection) => {
    // one row at a time: a newly ticked row wins over the one already picked
    const ticked = s.ranges.flatMap((g) => Array.from({ length: g.end - g.start }, (_, k) => g.start + k));
    const i = ticked.find((k) => k !== pickedIdx) ?? ticked[0];
    const r = i === undefined ? undefined : frame.rows[i];
    if (r) pickRow(r, false);
    else setPicked(null);
  }, [frame, pickRow, pickedIdx]);

  const onOrderByChange = useCallback((o: OrderBy) => {
    const c = o[0];
    setOrderBy((prev) => (c && c.column !== prev[0]?.column ? [{ column: c.column, direction: "descending" }] : o));
  }, []);

  const renderCellContent = useCallback(({ cell: c }: CellContentProps) => (c?.value as string) ?? "", []);
  const columnConfiguration = useMemo(
    () => Object.fromEntries(COLUMNS.map((c) => [c.name, { headerComponent: <span title={c.title}>{c.name}</span> }])),
    [],
  );

  const onTableHover = useCallback((e: React.MouseEvent) => {
    const tr = (e.target as HTMLElement).closest("tr");
    const n = tr?.querySelector("[data-rownumber]")?.getAttribute("data-rownumber");
    setRowHover(n == null ? null : frame.rows[Number(n)]?.key ?? null);
  }, [frame]);

  // ---- marks: every event at its center; the picked one white, the one under the pointer in the table bright
  // with a white ring, both drawn last
  const hoverRow = rowHover === null ? null : rows.find((r) => r.key === rowHover) ?? null;
  const layers = useMemo(() => {
    const top = [hoverRow, pickedRow].filter((r): r is TriageRow => !!r);
    const data = top.length ? [...rows.filter((r) => !top.includes(r)), ...top] : rows;
    return [
      new ScatterplotLayer<TriageRow>({
        id: "triage-marks",
        data,
        getPosition: (r) => [r.lon, r.lat],
        getRadius: (r) => radius(r) + (r === pickedRow ? 2 : r === hoverRow ? 4 : 0),
        radiusUnits: "pixels",
        getFillColor: (r) => (r === pickedRow ? [...PICKED, 235] : r === hoverRow ? [...MARK, 255] : [...MARK, 170]),
        stroked: true,
        getLineColor: (r) => (r === hoverRow ? [...PICKED, 255] : [0, 0, 0, 0]),
        getLineWidth: (r) => (r === hoverRow ? 2.5 : 0),
        lineWidthUnits: "pixels",
        pickable: true,
        updateTriggers: { getFillColor: [picked, rowHover], getRadius: [picked, rowHover], getLineColor: rowHover, getLineWidth: rowHover },
        onHover: (i: PickingInfo<TriageRow>) => setHover(i.object ? { r: i.object, x: i.x, y: i.y } : null),
        onClick: (i: PickingInfo<TriageRow>) => { if (i.object) pickRow(i.object, true); },
      }),
    ];
  }, [rows, pickedRow, picked, hoverRow, rowHover, pickRow]);

  // ---- full screen (F), and the map follows any size change of its pane
  const mapPane = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") return setPicked(null);
      if (e.target instanceof HTMLElement && e.target.closest(".triage-side")) return;
      if (e.key === "f" || e.key === "F") {
        if (document.fullscreenElement) void document.exitFullscreen();
        else void document.documentElement.requestFullscreen();
      }
    };
    window.addEventListener("keydown", onKey);
    const ro = new ResizeObserver(() => mapRef.current?.getMap().resize());
    if (mapPane.current) ro.observe(mapPane.current);
    const onFs = () => mapRef.current?.getMap().resize();
    document.addEventListener("fullscreenchange", onFs);
    return () => { window.removeEventListener("keydown", onKey); ro.disconnect(); document.removeEventListener("fullscreenchange", onFs); };
  }, []);

  const total = doc?.scans.length ?? 0;
  return (
    <div className="triage">
      <div className="triage-map" ref={mapPane}>
        <MaplibreMap
          ref={mapRef}
          initialViewState={CA_VIEW}
          mapStyle={MAP_STYLE_URL}
          attributionControl={{ compact: true }}
          maxPitch={0}
          dragRotate={false}
          pitchWithRotate={false}
          touchPitch={false}
          style={{ width: "100%", height: "100%" }}
        >
          <DeckGlOverlay layers={layers as never} interleaved onClick={(i: PickingInfo) => { if (!i.object) setPicked(null); }} />
        </MaplibreMap>
        {hover ? (
          <div className="triage-tip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
            <b>{Math.round(hover.r.score * 100)}</b> GOES score · {COLUMNS.find((c) => c.name === "first seen")!.text(hover.r)}
            <br />
            {hover.r.scans} scans · {hover.r.pixels} pixels · {COLUMNS.find((c) => c.name === "peak MW")!.text(hover.r) || "no"} MW peak
            <br />
            fuel under {COLUMNS.find((c) => c.name === "fuel under")!.text(hover.r)} · around {COLUMNS.find((c) => c.name === "fuel around")!.text(hover.r)}
          </div>
        ) : null}
      </div>
      <div className="triage-side" onMouseOver={onTableHover} onMouseLeave={() => setRowHover(null)}>
        <div className="triage-head">
          <div className="triage-title">
            GOES-West fires over California
            {pickedRow ? <button className="triage-clear" onClick={() => setPicked(null)} title="Unselect (Esc)">clear selection ×</button> : null}
          </div>
          <div className="triage-sub">
            {WEEK} · {rows.length.toLocaleString()} events ·{" "}
            {done ? "all scans read" : total ? `scans read ${read.toLocaleString()} of ${total.toLocaleString()}` : "loading scan list"}
          </div>
          {!done && total ? <div className="triage-bar"><div style={{ width: `${(100 * read) / total}%` }} /></div> : null}
          <div className="triage-sub">
            Embeddings ({EMB_YEAR}):{" "}
            {!done
              ? "start once every GOES scan is read"
              : fuelDone >= rows.length
                ? `all ${rows.length.toLocaleString()} events read`
                : `${fuelDone.toLocaleString()} of ${rows.length.toLocaleString()} events read, ${fuelReading.length} reading now`}
          </div>
          <div className="triage-bar fuel"><div style={{ width: `${done && rows.length ? (100 * fuelDone) / rows.length : 0}%` }} /></div>
          {error ? <div className="triage-error">{error}</div> : null}
        </div>
        <HighTable
          data={frame}
          className="triage-table"
          cacheKey="triage-v1"
          orderBy={orderBy}
          onOrderByChange={onOrderByChange}
          selection={selection}
          onSelectionChange={onSelectionChange}
          onMouseDownCell={onMouseDownCell}
          cellPosition={cell}
          onCellPositionChange={setCell}
          focus={false}
          renderCellContent={renderCellContent}
          columnConfiguration={columnConfiguration}
        />
      </div>
    </div>
  );
}
