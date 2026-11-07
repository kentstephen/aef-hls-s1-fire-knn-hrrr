import { Key, Seg, Slider } from "./controls.js";

/**
 * The replay's card: the fire's clock (Pacific time, where it burned), play and the hour, the speed, what each look
 * means, and how much of GOES is read.
 */

export const REPLAY_SPEEDS = [2, 4, 8, 16];

type Props = {
  label: string;
  year: number;
  before: number;
  after: number;
  /** Hours from the start; the start in ms. */
  hour: number;
  /** The clock's first hour (before discovery: negative). */
  minHour: number;
  hours: number;
  t0: number;
  playing: boolean;
  onPlay: () => void;
  onHour: (h: number) => void;
  speed: number;
  onSpeed: (s: number) => void;
  scansRead: number;
  scans: number;
  done: boolean;
  /** 10 m tiles in (each two years of embeddings, run through the volume head on the GPU). */
  tiles: number;
  /** The hour's mean Fosberg index over the fire (HRRR analysis), and the line where the fuel connects. */
  ffwi: number | null;
  gate: number;
  /** GOES drawn (burning, the after year), or the fuel and the weather alone. */
  goes: boolean;
  /** The basemap and the ones to switch to. */
  basemap: string;
  basemaps: { id: string; label: string }[];
  onBasemap: (id: string) => void;
  /** The fuel ramp as a CSS gradient, for the legend (dim before, full after). */
  gradient: string;
};

const pacific = new Intl.DateTimeFormat("en-US", { timeZone: "America/Los_Angeles", weekday: "short", month: "short", day: "numeric", hour: "numeric", timeZoneName: "short" });

export function ReplayCard(p: Props) {
  const max = Math.max(1, p.hours);
  const day = Math.floor(p.hour / 24) + 1;
  const when = p.hour < 0 ? `${Math.ceil(-p.hour)} h before it was reported` : `day ${day}, hour ${Math.floor(p.hour)} of ${p.hours}`;
  return (
    <div className="replay">
      <div className="rp-title">{p.label}, {p.year}</div>
      <div className="rp-clock">{pacific.format(new Date(p.t0 + p.hour * 3600e3))}</div>
      <div className="muted small">{when}</div>
      <div className="muted small">
        {p.ffwi === null ? "Reading the fire weather (HRRR)..." : <>Fosberg fire weather over the fire: <b className="rp-ffwi">{Math.round(p.ffwi)}</b></>}
      </div>
      <div className="row rp-play">
        <button type="button" className="rp-btn" onClick={(e) => { p.onPlay(); e.currentTarget.blur(); }}>{p.playing ? "Pause" : "Play"}</button>
        <Key k="Space" />
        <Slider min={p.minHour} max={max} step={0.25} value={Math.min(p.hour, max)} onChange={p.onHour} label="Hour" onReset={() => p.onHour(p.minHour)} />
      </div>
      <div className="row">
        <span className="muted small">hours a second</span>
        <Seg value={String(p.speed)} options={REPLAY_SPEEDS.map((s) => ({ id: String(s), label: String(s) }))} onChange={(v) => p.onSpeed(Number(v))} />
      </div>
      <div className="row">
        <span className="muted small">map</span>
        <Seg value={p.basemap} options={p.basemaps} onChange={p.onBasemap} />
      </div>
      <ul className="rp-key">
        <li><span className="sw" style={{ backgroundImage: p.gradient, opacity: 0.5 }} />{p.before} vegetation inside the outline, partly lit</li>
        <li><span className="sw" style={{ backgroundImage: p.gradient }} />in full where the hour's Fosberg is {p.gate} and up</li>
        <li><span className="sw" style={{ background: "linear-gradient(to right, #141414, #a1a1a1)" }} />Fosberg around the fire: dark calm, pale toward {p.gate}</li>
        {p.goes ? (
          <>
            <li><span className="sw" style={{ background: "linear-gradient(to right, #8c5b1e, #ffa838, #fffae6)" }} />burning: GOES-West saw it, brighter where more stood</li>
            <li><span className="sw" style={{ backgroundImage: p.gradient }} />{p.after} vegetation, after it burned</li>
          </>
        ) : null}
      </ul>
      <p className="muted small">
        Weather: the HRRR analysis (what it was, not a forecast), hour by hour, blended between hours.{" "}
        {p.goes
          ? <>Where each 10 m pixel burned and when: GOES-West's first scan that saw its 2 x 3 km cell burning, blended between cell centers, inside the final outline. {p.done ? `${p.scans.toLocaleString()} scans read.` : `Reading GOES: ${p.scansRead.toLocaleString()} of ${p.scans.toLocaleString()} scans.`}{" "}</>
          : null}
        {p.tiles.toLocaleString()} tiles of embeddings in.
      </p>
    </div>
  );
}
