import { fromLat, fromLon, toLat } from "../aef/mercator.js";
import { rampStripe } from "../aef/ramp.js";

/**
 * The view's histogram: the fuel pixels the map draws, counted on an even grid over the screen and binned by
 * their place on the color ramp, so each bar is one of the ramp's colors and its height how much of the view
 * is drawn in it. The values come from the 10 m tiles' fuel planes on the CPU (the bytes the GPU textures are
 * made from), and each byte's place on the ramp is worked out the way render.ts does it, so the bars carry the
 * map's own colors.
 */

/** Grid cells across the view. */
export const GRID_W = 480;
/** Bars across the ramp. */
export const BINS = 64;

/** An even grid over the view: lon even across, Web Mercator y even down, as the screen is. Cell centers. */
export type ViewGrid = { gw: number; gh: number; lons: Float64Array; lats: Float64Array };

export function viewGrid(view: [number, number, number, number]): ViewGrid {
  const [w, s, e, n] = view;
  const x0 = fromLon(w), x1 = fromLon(e), yS = fromLat(s), yN = fromLat(n);
  const gw = GRID_W;
  const gh = Math.max(1, Math.min(2 * GRID_W, Math.round((gw * (yN - yS)) / Math.max(x1 - x0, 1e-9))));
  const lons = new Float64Array(gw);
  const lats = new Float64Array(gh);
  for (let c = 0; c < gw; c++) lons[c] = w + ((c + 0.5) / gw) * (e - w);
  for (let r = 0; r < gh; r++) lats[r] = toLat(yN - ((r + 0.5) / gh) * (yN - yS));
  return { gw, gh, lons, lats };
}

/** How a fuel byte is placed on the ramp, as the fuel layer draws it (render.ts, the fuel alone). */
export type Placing = {
  pHide: number;
  pMax: number;
  /** The log-odds scale (render.ts scale 1), else straight. */
  logit: boolean;
  /** The view's stretch (each byte's place, 0..255), when it is drawn stretched. */
  stretch: Uint8Array | null;
};

/** The byte's place on the ramp (0..1), or null where the map draws nothing (no embedding, under the hide line). */
export function placeOf(b: number, p: Placing): number | null {
  if (b < 1) return null;
  const fw = (b - 1) / 254;
  if (fw < p.pHide) return null;
  if (p.stretch) return p.stretch[b]! / 255;
  if (p.logit) {
    const c = (v: number) => Math.min(0.996, Math.max(0.002, v));
    const lg = (v: number) => Math.log(c(v) / (1 - c(v)));
    const lo = lg(p.pHide), hi = lg(p.pMax);
    return Math.min(1, Math.max(0, (lg(fw) - lo) / Math.max(hi - lo, 0.01)));
  }
  return Math.min(1, Math.max(0, (fw - p.pHide) / Math.max(p.pMax - p.pHide, 0.0001)));
}

export type ViewBin = { share: number; color: string; lo: number | null; hi: number | null };
export type ViewHist = {
  bins: ViewBin[];
  /** The scaled value (byte fraction, 0..1) at the ramp's 0, 1/4, 1/2, 3/4 and end; null where no byte lands. */
  ticks: (number | null)[];
  /** Grid cells counted (drawn), and those on ground the map holds but does not draw (hide line, outside the mask). */
  drawn: number;
  hidden: number;
  /** Grid cells with no 10 m tile in yet. */
  unread: number;
  cells: number;
};

/**
 * counts: grid cells per fuel byte (index 0 = no embedding); off: cells with an embedding left out by the mask.
 * Null when too little of the view is in.
 */
export function binView(counts: Float64Array, off: number, unread: number, cells: number, p: Placing, stops: readonly string[]): ViewHist | null {
  const stripe = rampStripe(stops);
  const hex = (t: number) => {
    const o = Math.round(t * 255) * 4;
    return "#" + [stripe[o]!, stripe[o + 1]!, stripe[o + 2]!].map((v) => v.toString(16).padStart(2, "0")).join("");
  };
  const n = new Float64Array(BINS);
  const lo: (number | null)[] = new Array(BINS).fill(null);
  const hi: (number | null)[] = new Array(BINS).fill(null);
  const place: (number | null)[] = new Array(256).fill(null);
  let drawn = 0, hidden = off;
  for (let b = 1; b < 256; b++) {
    const c = counts[b]!;
    const t = placeOf(b, p);
    place[b] = t;
    if (t === null) { hidden += c; continue; }
    if (c === 0) continue;
    const k = Math.min(BINS - 1, Math.floor(t * BINS));
    n[k]! += c;
    drawn += c;
    const v = (b - 1) / 254;
    lo[k] = lo[k] === null ? v : Math.min(lo[k]!, v);
    hi[k] = hi[k] === null ? v : Math.max(hi[k]!, v);
  }
  if (drawn < 200) return null;
  const bins = Array.from({ length: BINS }, (_, k) => ({ share: n[k]! / drawn, color: hex((k + 0.5) / BINS), lo: lo[k] ?? null, hi: hi[k] ?? null }));
  // the value at each mark: the first byte the map places at or past it
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => {
    for (let b = 1; b < 256; b++) if (place[b] !== null && place[b]! >= t - 1e-6) return (b - 1) / 254;
    return null;
  });
  return { bins, ticks, drawn, hidden, unread, cells };
}
