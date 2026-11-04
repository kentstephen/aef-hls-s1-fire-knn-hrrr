/**
 * The fuel head (scripts/export_fuel_head.py): five 128 -> 64 -> 11 networks, averaged. Input per pixel:
 * the newest year's AEF, dequantized sign(q) * (q / 127.5)^2 (not unit-normalized), and the population
 * standard deviation of the dequantized AEF over the head's years; both standardized. Output: P(not fuel)
 * is the softmax summed over the non-fuel kinds. The map draws P(fuel).
 *
 * A score byte per pixel: 0 = no embedding, else 1 + round(P(fuel) * 254).
 */
export type FuelHead = {
  name: string;
  years: number[];
  n_in: number;
  hidden: number;
  n_out: number;
  kinds: string[];
  nf_cols: number[];
  mean: number[];
  scale: number[];
  nets: { w1: number[]; b1: number[]; w2: number[]; b2: number[] }[];
  /** P(not fuel) cutoffs keyed by the share of fuel hidden above them ("0.01" = 1 %). */
  p_notfuel_cut: Record<string, number>;
};

export const NODATA = -128;

/**
 * The weights in one array, as the GPU reads them: mean (n_in), 1 / scale (n_in), then per net
 * w1 [hidden, n_in], b1, w2 [n_out, hidden], b2.
 */
export function packWeights(h: FuelHead): Float32Array {
  const per = h.hidden * h.n_in + h.hidden + h.n_out * h.hidden + h.n_out;
  const out = new Float32Array(2 * h.n_in + h.nets.length * per);
  out.set(h.mean, 0);
  out.set(h.scale.map((s) => 1 / s), h.n_in);
  h.nets.forEach((n, i) => {
    let o = 2 * h.n_in + i * per;
    out.set(n.w1, o);
    o += n.w1.length;
    out.set(n.b1, o);
    o += n.b1.length;
    out.set(n.w2, o);
    o += n.w2.length;
    out.set(n.b2, o);
  });
  return out;
}

const deq = (q: number) => {
  const f = q / 127.5;
  return f * Math.abs(f);
};

/** The head on the CPU, for when WebGPU is missing. years: one band-major int8 [64, n] chunk per head year, oldest first. */
export function scoreCpu(h: FuelHead, w: Float32Array, years: Int8Array[], n: number): Uint8Array {
  const out = new Uint8Array(n);
  const ny = years.length;
  const B = h.n_in / 2;
  const per = h.hidden * h.n_in + h.hidden + h.n_out * h.hidden + h.n_out;
  const x = new Float32Array(h.n_in);
  const hid = new Float32Array(h.hidden);
  const z = new Float32Array(h.n_out);
  const nf = new Set(h.nf_cols);
  const v = new Float32Array(ny);
  for (let p = 0; p < n; p++) {
    const missing = new Array(ny).fill(0);
    for (let b = 0; b < B; b++) {
      let m = 0;
      for (let y = 0; y < ny; y++) {
        const q = years[y]![b * n + p]!;
        if (q === NODATA) missing[y]++;
        v[y] = deq(q);
        m += v[y]!;
      }
      m /= ny;
      let s = 0;
      for (let y = 0; y < ny; y++) s += (v[y]! - m) ** 2;
      x[b] = (v[ny - 1]! - w[b]!) * w[h.n_in + b]!;
      x[B + b] = (Math.sqrt(s / ny) - w[B + b]!) * w[h.n_in + B + b]!;
    }
    if (missing.some((c) => c === B)) continue;
    let pnf = 0;
    for (let k = 0; k < h.nets.length; k++) {
      const o1 = 2 * h.n_in + k * per;
      const ob1 = o1 + h.hidden * h.n_in;
      const o2 = ob1 + h.hidden;
      const ob2 = o2 + h.n_out * h.hidden;
      for (let j = 0; j < h.hidden; j++) {
        let s = w[ob1 + j]!;
        const r = o1 + j * h.n_in;
        for (let i = 0; i < h.n_in; i++) s += w[r + i]! * x[i]!;
        hid[j] = s > 0 ? s : 0;
      }
      let zmax = -Infinity;
      for (let c = 0; c < h.n_out; c++) {
        let s = w[ob2 + c]!;
        const r = o2 + c * h.hidden;
        for (let j = 0; j < h.hidden; j++) s += w[r + j]! * hid[j]!;
        z[c] = s;
        if (s > zmax) zmax = s;
      }
      let den = 0;
      let num = 0;
      for (let c = 0; c < h.n_out; c++) {
        const e = Math.exp(z[c]! - zmax);
        den += e;
        if (nf.has(c)) num += e;
      }
      pnf += num / den;
    }
    const pf = Math.min(1, Math.max(0, 1 - pnf / h.nets.length));
    out[p] = 1 + Math.round(pf * 254);
  }
  return out;
}

/** P(fuel) from a score byte, or null for no embedding. */
export const pFuel = (byte: number): number | null => (byte === 0 ? null : (byte - 1) / 254);
