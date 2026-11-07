/**
 * The fuel volume head (scripts/fit_volume.py, export_volume_head.py): one 64 -> 64 -> out network on a single
 * year's embedding, dequantized sign(q) * (q / 127.5)^2 and standardized. Outputs, in order: reg (vol, vc, vh, cc,
 * ch, scaled), the 11 fuel kinds' logits, the 4 lifeforms' logits (none, tree, shrub, herb).
 *
 * vol is log(0.02 + cover x height in m), scaled 0..1 between vol_lo and vol_hi: LANDFIRE's leading lifeform's
 * cover times its height, a proxy for how much stands on the pixel to burn.
 *
 * Per pixel the scorer packs one u32: byte 0 the volume (0 = no embedding, else 1 + round(vol * 254)), byte 1
 * canopy cover (round(cc * 255)), byte 2 the fuel kind (low 4 bits) and lifeform (bits 4 to 5), byte 3 the
 * leading lifeform's height in quarter meters (to 63.75 m).
 */
export type VolumeHead = {
  name: string;
  n_in: number;
  hidden: number;
  n_out: number;
  reg: string[];
  kinds: string[];
  life: string[];
  vol_lo: number;
  vol_hi: number;
  mean: number[];
  scale: number[];
  w1: number[];
  b1: number[];
  w2: number[];
  b2: number[];
  probe_q: number[];
  probe_out: number[];
};

export const NODATA = -128;

/** The scaled volume (0..1) back to meters (cover x height). */
export function volMeters(h: VolumeHead, s: number): number {
  return Math.max(0, Math.exp(s * (h.vol_hi - h.vol_lo) + h.vol_lo) - 0.02);
}

/** A volume byte (1..255) back to meters; NaN for 0 (no embedding). */
export function volByteMeters(h: VolumeHead, b: number): number {
  return b > 0 ? volMeters(h, (b - 1) / 254) : NaN;
}

export type VolumeAt = { vol: number; cc: number; kind: string; life: string; height: number };

export function unpack(h: VolumeHead, u: number): VolumeAt | null {
  const vb = u & 255;
  if (vb === 0) return null;
  const kl = (u >>> 16) & 255;
  return {
    vol: volByteMeters(h, vb),
    cc: ((u >>> 8) & 255) / 255,
    kind: h.kinds[kl & 15] ?? "?",
    life: h.life[(kl >>> 4) & 3] ?? "?",
    height: ((u >>> 24) & 255) / 4,
  };
}

/** The weights in one array, as the GPU reads them: mean, 1 / scale, w1, b1, w2, b2. */
export function packVolume(h: VolumeHead): Float32Array {
  const out = new Float32Array(2 * h.n_in + h.w1.length + h.b1.length + h.w2.length + h.b2.length);
  let o = 0;
  for (const a of [h.mean, h.scale.map((s) => 1 / s), h.w1, h.b1, h.w2, h.b2]) {
    out.set(a, o);
    o += a.length;
  }
  return out;
}

const deq = (q: number) => {
  const f = q / 127.5;
  return f * Math.abs(f);
};

/** The raw outputs for one pixel of a band-major chunk (for the probe check and the CPU path). */
function forward(h: VolumeHead, w: Float32Array, e: Int8Array, n: number, p: number, x: Float32Array, hid: Float32Array, z: Float32Array): boolean {
  const NI = h.n_in;
  let miss = 0;
  for (let b = 0; b < NI; b++) {
    const q = e[b * n + p]!;
    if (q === NODATA) miss++;
    x[b] = (deq(q) - w[b]!) * w[NI + b]!;
  }
  if (miss === NI) return false;
  const o1 = 2 * NI, ob1 = o1 + h.hidden * NI, o2 = ob1 + h.hidden, ob2 = o2 + h.n_out * h.hidden;
  for (let j = 0; j < h.hidden; j++) {
    let s = w[ob1 + j]!;
    for (let i = 0; i < NI; i++) s += w[o1 + j * NI + i]! * x[i]!;
    hid[j] = Math.max(s, 0);
  }
  for (let c = 0; c < h.n_out; c++) {
    let s = w[ob2 + c]!;
    for (let j = 0; j < h.hidden; j++) s += w[o2 + c * h.hidden + j]! * hid[j]!;
    z[c] = s;
  }
  return true;
}

function pack(h: VolumeHead, z: Float32Array): number {
  const nr = h.reg.length, nk = h.kinds.length;
  const clamp = (v: number) => Math.min(1, Math.max(0, v));
  let k = 0;
  for (let c = 1; c < nk; c++) if (z[nr + c]! > z[nr + k]!) k = c;
  let l = 0;
  for (let c = 1; c < h.life.length; c++) if (z[nr + nk + c]! > z[nr + nk + l]!) l = c;
  const vb = 1 + Math.round(clamp(z[0]!) * 254);
  const cc = Math.round(clamp(z[3]!) * 255);
  const ht = Math.round(Math.min(63.75, Math.max(0, z[2]! * 40)) * 4);
  return (vb | (cc << 8) | ((k | (l << 4)) << 16) | (ht << 24)) >>> 0;
}

/** The head on the CPU, for when WebGPU is missing: e is one year, band-major int8 [64, n]. */
export function volumeCpu(h: VolumeHead, w: Float32Array, e: Int8Array, n: number): Uint32Array {
  const out = new Uint32Array(n);
  const x = new Float32Array(h.n_in), hid = new Float32Array(h.hidden), z = new Float32Array(h.n_out);
  for (let p = 0; p < n; p++) if (forward(h, w, e, n, p, x, hid, z)) out[p] = pack(h, z);
  return out;
}

/** The largest gap between this code's outputs and the export's own on its probe embeddings. */
export function probeGap(h: VolumeHead, w: Float32Array): number {
  const n = h.probe_q.length / h.n_in;
  // probe_q is pixel-major [n, 64]; the forward pass reads band-major
  const e = new Int8Array(h.probe_q.length);
  for (let p = 0; p < n; p++) for (let b = 0; b < h.n_in; b++) e[b * n + p] = h.probe_q[p * h.n_in + b]!;
  const x = new Float32Array(h.n_in), hid = new Float32Array(h.hidden), z = new Float32Array(h.n_out);
  let worst = 0;
  for (let p = 0; p < n; p++) {
    forward(h, w, e, n, p, x, hid, z);
    for (let c = 0; c < h.n_out; c++) worst = Math.max(worst, Math.abs(z[c]! - h.probe_out[p * h.n_out + c]!));
  }
  return worst;
}
