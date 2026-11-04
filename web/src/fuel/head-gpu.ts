/// <reference types="@webgpu/types" />
import type { FuelHead } from "./head";

/**
 * The fuel head as a WebGPU compute pass, one thread per pixel, run once per chunk when it loads (the
 * map then only looks the score up). The chunk's years go up as raw int8 packed in u32 words; each
 * thread dequantizes, takes the newest year and the spread over the years, standardizes, and runs the
 * five networks. Same math and byte as scoreCpu in head.ts.
 */
export type GpuScorer = (years: Int8Array[], n: number) => Promise<Uint8Array>;

function wgsl(h: FuelHead, ny: number): string {
  const B = h.n_in / 2;
  const per = h.hidden * h.n_in + h.hidden + h.n_out * h.hidden + h.n_out;
  const nf = h.nf_cols;
  return /* wgsl */ `
struct Params { n: u32 }
@group(0) @binding(0) var<storage, read> aef: array<u32>;
@group(0) @binding(1) var<storage, read> wts: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<u32>;
@group(0) @binding(3) var<uniform> prm: Params;

const NY: u32 = ${ny}u;
const NB: u32 = ${B}u;
const NI: u32 = ${h.n_in}u;
const NH: u32 = ${h.hidden}u;
const NO: u32 = ${h.n_out}u;
const NNET: u32 = ${h.nets.length}u;
const PER: u32 = ${per}u;

fn q8(k: u32) -> i32 {
  return extractBits(bitcast<i32>(aef[k >> 2u]), (k & 3u) * 8u, 8u);
}
fn deq(q: i32) -> f32 {
  let f = f32(q) / 127.5;
  return f * abs(f);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  let n = prm.n;
  if (p >= n) { return; }
  var x: array<f32, ${h.n_in}>;
  var miss: array<u32, ${ny}>;
  var v: array<f32, ${ny}>;
  for (var b = 0u; b < NB; b++) {
    var m = 0.0;
    for (var y = 0u; y < NY; y++) {
      let q = q8(y * NB * n + b * n + p);
      if (q == -128) { miss[y] += 1u; }
      v[y] = deq(q);
      m += v[y];
    }
    m /= f32(NY);
    var s = 0.0;
    for (var y = 0u; y < NY; y++) { s += (v[y] - m) * (v[y] - m); }
    x[b] = (v[NY - 1u] - wts[b]) * wts[NI + b];
    x[NB + b] = (sqrt(s / f32(NY)) - wts[NB + b]) * wts[NI + NB + b];
  }
  for (var y = 0u; y < NY; y++) {
    if (miss[y] == NB) { outp[p] = 0u; return; }
  }
  var pnf = 0.0;
  var hid: array<f32, ${h.hidden}>;
  var z: array<f32, ${h.n_out}>;
  for (var k = 0u; k < NNET; k++) {
    let o1 = 2u * NI + k * PER;
    let ob1 = o1 + NH * NI;
    let o2 = ob1 + NH;
    let ob2 = o2 + NO * NH;
    for (var j = 0u; j < NH; j++) {
      var s = wts[ob1 + j];
      let r = o1 + j * NI;
      for (var i = 0u; i < NI; i++) { s += wts[r + i] * x[i]; }
      hid[j] = max(s, 0.0);
    }
    var zmax = -1e30;
    for (var c = 0u; c < NO; c++) {
      var s = wts[ob2 + c];
      let r = o2 + c * NH;
      for (var j = 0u; j < NH; j++) { s += wts[r + j] * hid[j]; }
      z[c] = s;
      zmax = max(zmax, s);
    }
    var den = 0.0;
    var num = 0.0;
    for (var c = 0u; c < NO; c++) {
      let e = exp(z[c] - zmax);
      den += e;
      if (${nf.map((c) => `c == ${c}u`).join(" || ")}) { num += e; }
    }
    pnf += num / den;
  }
  let pf = clamp(1.0 - pnf / f32(NNET), 0.0, 1.0);
  outp[p] = 1u + u32(round(pf * 254.0));
}
`;
}

export async function makeGpuScorer(h: FuelHead, w: Float32Array, maxN: number): Promise<GpuScorer | null> {
  const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
  if (!gpu) return null;
  const adapter = await gpu.requestAdapter();
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  const ny = h.years.length;
  const module = device.createShaderModule({ code: wgsl(h, ny) });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === "error");
  if (errs.length) throw new Error(`fuel head shader: ${errs.map((m) => m.message).join("; ")}`);
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });

  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const aefBuf = device.createBuffer({ size: ny * 64 * maxN, usage: S });
  const wBuf = device.createBuffer({ size: w.byteLength, usage: S });
  device.queue.writeBuffer(wBuf, 0, w as Float32Array<ArrayBuffer>);
  const outBuf = device.createBuffer({ size: 4 * maxN, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: 4 * maxN, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const prmBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aefBuf } },
      { binding: 1, resource: { buffer: wBuf } },
      { binding: 2, resource: { buffer: outBuf } },
      { binding: 3, resource: { buffer: prmBuf } },
    ],
  });

  // one chunk at a time per worker: the buffers are shared
  let queue: Promise<unknown> = Promise.resolve();
  return (years, n) => {
    const run = async () => {
      if (n > maxN) throw new Error(`chunk of ${n} px is over the scorer's ${maxN}`);
      years.forEach((y, i) => device.queue.writeBuffer(aefBuf, i * 64 * n, y.buffer, y.byteOffset, y.byteLength));
      device.queue.writeBuffer(prmBuf, 0, new Uint32Array([n, 0, 0, 0]));
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(n / 64));
      pass.end();
      enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, 4 * n);
      device.queue.submit([enc.finish()]);
      await readBuf.mapAsync(GPUMapMode.READ, 0, 4 * n);
      const u = new Uint32Array(readBuf.getMappedRange(0, 4 * n));
      const out = new Uint8Array(n);
      for (let i = 0; i < n; i++) out[i] = u[i]!;
      readBuf.unmap();
      return out;
    };
    const r = queue.then(run, run);
    queue = r.catch(() => {});
    return r;
  };
}
