import { HEADS, type Teacher } from "./head.js";

/**
 * The ignition head's logit in real units (models/ignition_calibration.json, docs/ignition-calibration.md):
 * the head was fit on ignitions against nearby control hours, so its 50% is "as fire-like as a typical
 * control hour", not an hourly chance. A population sample of real pixel-hours (2019 to 2026, 152,333
 * known-cause WFIGS ignitions) gives the offset that turns the logit into ignitions per km2 per hour;
 * the ordering is the head's, the level is the country's. Per class, the ratio of observed to expected
 * ignitions corrects what local controls canceled (evergreen under 2.9x, crops and dense built over 4x);
 * the map's picture does not use it, the tooltip's number does.
 */
export const OFFSET = -14.83940123564505;
/** ln(1000 km2 x 24 h): per km2 per hour to per 1000 km2 per day. */
const TO_1000KM2_DAY = Math.log(1000 * 24);
/** The CONUS mean, known-cause ignitions per 1000 km2 per day. */
export const CONUS_MEAN = 0.006803004687535892;

const RATIO_NLCD: Record<number, number> = {
  11: 0.633, 12: 0.616, 21: 1.094, 22: 0.783, 23: 0.279, 24: 0.242, 31: 1.088, 41: 0.81, 42: 2.889, 43: 0.938,
  52: 1.359, 71: 0.805, 81: 0.629, 82: 0.254, 90: 0.92, 95: 0.814,
};

/** Ignitions per 1000 km2 per day for an ignition logit (ground plus weather), with the class's correction. */
export function ignitionRate(logit: number, teacher: Teacher, classIndex: number): number {
  const code = HEADS[teacher].classes[classIndex];
  const ratio = teacher === "nlcd" && code !== undefined ? (RATIO_NLCD[code] ?? 1) : 1;
  // sigmoid(z + c) with c about -15 is exp(z + c) to every digit shown
  return Math.exp(logit + OFFSET + TO_1000KM2_DAY) * ratio;
}

/** "0.12 per 1000 km2 per day, 18x the CONUS mean" (or a fraction of it). */
export function rateLabel(rate: number): string {
  const x = rate / CONUS_MEAN;
  const mult = x >= 10 ? `${Math.round(x)}x` : x >= 1 ? `${x.toFixed(1)}x` : `${x.toFixed(2)}x`;
  const r = rate >= 1 ? rate.toFixed(1) : rate >= 0.01 ? rate.toFixed(2) : rate.toExponential(1);
  return `${r} per 1000 km² per day, ${mult} the CONUS mean`;
}
