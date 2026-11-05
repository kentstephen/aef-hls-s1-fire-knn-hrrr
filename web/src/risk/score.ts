import { hexToRgb, VIRIDIS_STOPS } from "../aef/ramp.js";

/**
 * The risk value: an exploration, not a forecast, and not yet tested against fires. Four parts, each 0 to 1, count
 * the same; the fuel at the spot scales their average, so ground that cannot burn stays low whatever is around it.
 *
 *   continuity  share of fuel within 500 m
 *   slope       steeper is higher, full at 30 degrees and up (fire runs faster uphill)
 *   facing      south-west facing is highest, north-east lowest (drier afternoon sun), counted in step with the
 *               slope up to FACE_SLOPE: on flat ground it stays at the middle (a tree's crown in the elevation
 *               data would otherwise light its sunny side)
 *   time        years since the last fire on record, full at 20 years and up, and where none is on record
 *               (recently burned ground has less left to burn)
 *
 *   risk = fuel x (continuity + slope + facing + time) / 4
 */

export const SLOPE_FULL = 30;
export const TIME_FULL = 20;
const FACE_MOST = 225;
/** Slope (degrees) from which facing counts fully. */
export const FACE_SLOPE = 10;

export type Parts = { fuel: number; continuity: number; slope: number; facing: number; time: number; risk: number };

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

export function parts(fuel: number, near: number, slopeDeg: number, aspectDeg: number, lastYear: number, nowYear: number): Parts {
  const continuity = clamp01(near);
  const slope = Number.isNaN(slopeDeg) ? 0 : clamp01(slopeDeg / SLOPE_FULL);
  const face = Number.isNaN(aspectDeg) ? 0.5 : (1 + Math.cos(((aspectDeg - FACE_MOST) * Math.PI) / 180)) / 2;
  const facing = 0.5 + (face - 0.5) * (Number.isNaN(slopeDeg) ? 0 : clamp01(slopeDeg / FACE_SLOPE));
  const time = lastYear ? clamp01((nowYear - lastYear) / TIME_FULL) : 1;
  return { fuel, continuity, slope, facing, time, risk: fuel * ((continuity + slope + facing + time) / 4) };
}

/** Viridis as 256 RGB entries (matplotlib's own range, dark purple to yellow). */
export const VIRIDIS: Uint8Array = (() => {
  const stops = VIRIDIS_STOPS.slice(5, 47).map(hexToRgb);
  const out = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (stops.length - 1);
    const a = Math.floor(t), b = Math.min(stops.length - 1, a + 1), f = t - a;
    for (let k = 0; k < 3; k++) out[i * 3 + k] = Math.round(stops[a]![k]! * (1 - f) + stops[b]![k]! * f);
  }
  return out;
})();

export const viridisCss = (v: number) => {
  const i = Math.round(clamp01(v) * 255) * 3;
  return `rgb(${VIRIDIS[i]}, ${VIRIDIS[i + 1]}, ${VIRIDIS[i + 2]})`;
};
export const VIRIDIS_GRADIENT = `linear-gradient(to right, ${Array.from({ length: 11 }, (_, i) => viridisCss(i / 10)).join(", ")})`;
