/**
 * Fosberg Fire Weather Index (Fosberg 1978), the single source of truth.
 *
 * Used by (1) the per-tile CPU pass that fills the tile cache and the GPU
 * textures, and (2) the county join and tooltip. `scripts/verify_ffwi.py`
 * and `scripts/ffwi_check.mjs` are the parity checks; keep all three in sync.
 * `gpu/lagged-ffwi.ts` carries a GLSL copy of `ffwiFromMoisture` (eta and the
 * 0.3002 anchor only; EMC stays CPU-side); keep that in sync too.
 *
 * Inputs: temperature degC, relative humidity %, wind speed m/s (sustained
 * 10 m wind `sqrt(u^2+v^2)` is the standard input; gust is a separate,
 * clearly labelled option). Internally degF and mph.
 */

const C_TO_F = (c: number) => c * 1.8 + 32;
export const MS_TO_MPH = 2.2369363;

const clamp = (v: number, lo: number, hi: number) =>
  v < lo ? lo : v > hi ? hi : v;

/** Equilibrium moisture content (%), piecewise in RH (%) and T (degF). */
export function equilibriumMoisture(rhPct: number, tF: number): number {
  const h = clamp(rhPct, 0, 100);
  let m: number;
  if (h < 10) {
    m = 0.03229 + 0.281073 * h - 0.000578 * h * tF;
  } else if (h < 50) {
    m = 2.22749 + 0.160107 * h - 0.014784 * tF;
  } else {
    m = 21.0606 + 0.005565 * h * h - 0.00035 * h * tF - 0.483199 * h;
  }
  return clamp(m, 0, 30);
}

/** Fosberg's moisture damping eta(m) times the wind term, anchored so that
 *  zero moisture + 30 mph wind = 100. `m` in %, `windMph` in mph. */
export function ffwiFromMoisture(m: number, windMph: number): number {
  const mr = m / 30;
  const eta = 1 - 2 * mr + 1.5 * mr * mr - 0.5 * mr * mr * mr;
  return (eta * Math.sqrt(1 + windMph * windMph)) / 0.3002;
}

/**
 * FFWI for one sample. Returns NaN when any input is not finite (HRRR fill).
 * Values above 100 are possible and left untruncated.
 */
export function ffwi(tC: number, rhPct: number, windMs: number): number {
  if (!Number.isFinite(tC) || !Number.isFinite(rhPct) || !Number.isFinite(windMs)) {
    return NaN;
  }
  return ffwiFromMoisture(equilibriumMoisture(rhPct, C_TO_F(tC)), windMs * MS_TO_MPH);
}

export type WindMode = "sustained" | "gust";

/** Fixed-point scales for the per-tile blocks (Uint16). 65535 = NaN sentinel. */
export const M_SCALE = 1000; // moisture %, 0..30 -> 0..30000
export const WIND_SCALE = 100; // mph, 0..655
export const U16_NAN = 65535;

/**
 * Per-tile pass over the five bands: equilibrium moisture (Uint16 x1000),
 * wind mph (Uint16 x100, sustained or gust), plus Uint8 RH % and degC+100
 * for the tooltip. FFWI itself is computed on demand from moisture and wind
 * (`ffwiFromMoisture`) so that a fuel-moisture timelag can be applied to the
 * moisture term later (`lagMoisture`) without refetching anything.
 */
export function fosbergBlock(
  n: number,
  t: Float32Array,
  rh: Float32Array,
  u: Float32Array,
  v: Float32Array,
  gust: Float32Array,
  windMode: WindMode,
  outM: Uint16Array,
  outWind: Uint16Array,
  outRh: Uint8Array,
  outTc: Uint8Array,
): void {
  for (let i = 0; i < n; i++) {
    const ui = u[i]!;
    const vi = v[i]!;
    const w = windMode === "gust" ? gust[i]! : Math.sqrt(ui * ui + vi * vi);
    const tc = t[i]!;
    const h = rh[i]!;
    if (Number.isFinite(tc) && Number.isFinite(h) && Number.isFinite(w)) {
      outM[i] = Math.round(equilibriumMoisture(h, C_TO_F(tc)) * M_SCALE);
      outWind[i] = Math.min(U16_NAN - 1, Math.round(w * MS_TO_MPH * WIND_SCALE));
    } else {
      outM[i] = U16_NAN;
      outWind[i] = U16_NAN;
    }
    outRh[i] = Number.isFinite(h) ? Math.min(255, Math.max(0, Math.round(h))) : 0;
    outTc[i] = Number.isFinite(tc) ? Math.min(255, Math.max(0, Math.round(tc + 100))) : 0;
  }
}

/**
 * Fuel-moisture timelag: the standard exponential response of dead fuel
 * toward equilibrium (NFDRS 1-h / 10-h / 100-h fuel model form),
 *   m(t) = m_eq(t) + (m(t-1) - m_eq(t)) * exp(-1 / tau),
 * seeded with m(0) = m_eq(0) at the run's first hour, so the memory only
 * reaches back to +0 h. tau in hours; tau = 0 returns m_eq unchanged
 * (Fosberg as published: fuel equilibrates instantly). Blocks are lead-major
 * (idx = lead * perLead + pixel). Wind is not lagged; it has no memory.
 */
export function lagMoisture(mEq: Uint16Array, leads: number, perLead: number, tauHours: number, out: Uint16Array): void {
  if (!(tauHours > 0)) {
    out.set(mEq);
    return;
  }
  const a = Math.exp(-1 / tauHours);
  out.set(mEq.subarray(0, perLead), 0);
  for (let lead = 1; lead < leads; lead++) {
    const o = lead * perLead;
    const po = o - perLead;
    for (let i = 0; i < perLead; i++) {
      const me = mEq[o + i]!;
      const prev = out[po + i]!;
      out[o + i] = me === U16_NAN || prev === U16_NAN ? U16_NAN : Math.round(me + (prev - me) * a);
    }
  }
}
