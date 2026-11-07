import { PolygonLayer } from "@deck.gl/layers";

/**
 * One fire as GOES-West saw it: the fire pixels of ABI-L2-FDCC (every 5 minutes) over the fire's box, read from
 * NOAA's bucket in the browser. The byte ranges come from scripts/goes_fire_refs.py (VirtualiZarr references,
 * web/public/goes/<fire>.json); the pixels themselves are fetched and decoded here, nothing is stored.
 *
 * The scans are split into eras of one file form each (chunks, codecs, grid); each era is decoded with its own.
 */

type Codec = { name: string; configuration?: Record<string, unknown> };
type VarLayout = { shape: number[]; chunks: number[]; dtype: string; fill: string; codecs: Codec[] };
export type Era = {
  form: string;
  first: string;
  last: string;
  box: [number, number, number, number];
  grid: { nx: number; ny: number; x0: number; dx: number; y0: number; dy: number };
  projection: { semi_major_axis: number; semi_minor_axis: number; perspective_point_height: number; longitude_of_projection_origin: number };
  vars: { Mask: VarLayout; Power: VarLayout };
};
type Ref = [cy: number, cx: number, offset: number, length: number];
type ScanRef = { t: number; era: number; key: string; refs: { Mask: Ref[]; Power: Ref[] } };
export type GoesDoc = {
  fire: { name: string; objectid: number; acres: number; cause: string; discovered: number; bbox: [number, number, number, number] };
  product: string;
  origin: string;
  fire_mask: number[];
  eras: Era[];
  scans: ScanRef[];
};

/** A fire pixel of one scan: its footprint on the ground and its fire radiative power (MW, NaN if not computed). */
export type FirePixel = { t: number; key: string; polygon: [number, number][]; power: number; mask: number };

export async function loadGoesDoc(url: string): Promise<GoesDoc> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return (await r.json()) as GoesDoc;
}

/** The fire's burn perimeter from NIFC (the same record the references were built for). */
export async function loadGoesPerimeter(objectid: number, signal?: AbortSignal): Promise<[number, number][][][]> {
  const u = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters/FeatureServer/0/query";
  const r = await fetch(`${u}?${new URLSearchParams({ where: `OBJECTID = ${objectid}`, outSR: "4326", f: "geojson", maxAllowableOffset: "0.00005" })}`, { signal });
  const j = await r.json();
  const g = j.features?.[0]?.geometry;
  if (!g) return [];
  return g.type === "Polygon" ? [g.coordinates] : g.coordinates;
}

// ---- decoding one chunk: the era's codecs in reverse (zlib, then shuffle, then bytes)

async function inflate(b: Uint8Array): Promise<Uint8Array> {
  const s = new Blob([b as BlobPart]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(s).arrayBuffer());
}

function unshuffle(b: Uint8Array, size: number): Uint8Array {
  const n = b.length / size;
  const out = new Uint8Array(b.length);
  for (let k = 0; k < size; k++) for (let i = 0; i < n; i++) out[i * size + k] = b[k * n + i]!;
  return out;
}

async function decode(raw: Uint8Array, v: VarLayout): Promise<Int16Array | Float32Array | Uint8Array> {
  let b = raw;
  let little = true;
  for (const c of [...v.codecs].reverse()) {
    if (c.name === "numcodecs.zlib" || c.name === "zlib" || c.name === "gzip") b = await inflate(b);
    else if (c.name === "numcodecs.shuffle") b = unshuffle(b, Number(c.configuration?.elementsize ?? 1));
    else if (c.name === "bytes") little = (c.configuration?.endian ?? "little") === "little";
    else throw new Error(`GOES: codec ${c.name} not handled (new file form?)`);
  }
  if (!little) throw new Error("GOES: big-endian chunks not handled");
  const buf = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  // zarr's data type as VirtualiZarr writes it, e.g. "Int16(endianness='little')"
  const dt = (v.dtype.match(/^[A-Za-z]+\d*/)?.[0] ?? v.dtype).toLowerCase();
  if (dt === "int16") return new Int16Array(buf);
  if (dt === "float32") return new Float32Array(buf);
  if (dt === "uint8") return new Uint8Array(buf);
  throw new Error(`GOES: dtype ${v.dtype} not handled`);
}

async function fetchRange(url: string, off: number, len: number, signal?: AbortSignal): Promise<Uint8Array> {
  for (let attempt = 0; ; attempt++) {
    try {
      const r = await fetch(url, { headers: { Range: `bytes=${off}-${off + len - 1}` }, signal });
      if (r.status !== 206 && r.status !== 200) throw new Error(`${r.status}`);
      const b = new Uint8Array(await r.arrayBuffer());
      return r.status === 200 ? b.subarray(off, off + len) : b;
    } catch (e) {
      if ((e as Error).name === "AbortError" || attempt >= 3) throw e;
      await new Promise((ok) => setTimeout(ok, 500 * (attempt + 1)));
    }
  }
}

/** A scan's chunks, nearby byte ranges merged into one request (gaps under 32 KB), decoded. */
async function fetchChunks<T extends Int16Array | Float32Array = Int16Array>(url: string, refs: Ref[], v: VarLayout, signal?: AbortSignal): Promise<[number, number, T][]> {
  const sorted = [...refs].sort((a, b) => a[2] - b[2]);
  const groups: Ref[][] = [];
  for (const r of sorted) {
    const g = groups[groups.length - 1];
    const last = g?.[g.length - 1];
    if (g && last && r[2] - (last[2] + last[3]) < 32768) g.push(r);
    else groups.push([r]);
  }
  const out: [number, number, T][] = [];
  for (const g of groups) {
    const off = g[0]![2];
    const end = Math.max(...g.map((r) => r[2] + r[3]));
    const b = await fetchRange(url, off, end - off, signal);
    for (const [cy, cx, o, len] of g) out.push([cy, cx, (await decode(b.subarray(o - off, o - off + len), v)) as T]);
  }
  return out;
}

// ---- navigation: GOES fixed grid scan angles to lon/lat (GOES-R PUG)

function xyToLonLat(x: number, y: number, p: Era["projection"]): [number, number] | null {
  const req = p.semi_major_axis, rpol = p.semi_minor_axis, H = p.perspective_point_height + p.semi_major_axis;
  const lon0 = (p.longitude_of_projection_origin * Math.PI) / 180;
  const a = Math.sin(x) ** 2 + Math.cos(x) ** 2 * (Math.cos(y) ** 2 + ((req * req) / (rpol * rpol)) * Math.sin(y) ** 2);
  const b = -2 * H * Math.cos(x) * Math.cos(y);
  const c = H * H - req * req;
  const disc = b * b - 4 * a * c;
  if (disc < 0) return null;
  const rs = (-b - Math.sqrt(disc)) / (2 * a);
  const sx = rs * Math.cos(x) * Math.cos(y), sy = -rs * Math.sin(x), sz = rs * Math.cos(x) * Math.sin(y);
  const lat = Math.atan(((req * req) / (rpol * rpol)) * (sz / Math.sqrt((H - sx) ** 2 + sy * sy)));
  const lon = lon0 - Math.atan(sy / (H - sx));
  return [(lon * 180) / Math.PI, (lat * 180) / Math.PI];
}

/** A ground point's fractional GOES row and column in an era's grid (GOES-R PUG; a cell's center is a whole number). */
export function lonLatToRowCol(e: Era, lon: number, lat: number): [number, number] {
  const p = e.projection, g = e.grid;
  const req = p.semi_major_axis, rpol = p.semi_minor_axis, H = p.perspective_point_height + p.semi_major_axis;
  const lon0 = (p.longitude_of_projection_origin * Math.PI) / 180;
  const e2 = 1 - (rpol * rpol) / (req * req);
  const phi = (lat * Math.PI) / 180, lam = (lon * Math.PI) / 180 - lon0;
  const latc = Math.atan(((rpol * rpol) / (req * req)) * Math.tan(phi));
  const rc = rpol / Math.sqrt(1 - e2 * Math.cos(latc) ** 2);
  const sx = H - rc * Math.cos(latc) * Math.cos(lam);
  const sy = -rc * Math.cos(latc) * Math.sin(lam);
  const sz = rc * Math.sin(latc);
  const x = Math.asin(-sy / Math.sqrt(sx * sx + sy * sy + sz * sz));
  const y = Math.atan(sz / sx);
  return [(y - g.y0) / g.dy, (x - g.x0) / g.dx];
}

/**
 * The reference file cut to one fire: the scans from `from` to `to` (ms), each scan's chunks only where they cover
 * the lon / lat box, and each era's box of rows and columns set to it. Reads a few KB a scan instead of the state.
 */
export function cutDoc(doc: GoesDoc, bbox: [number, number, number, number], from: number, to: number): GoesDoc {
  const [w, s, e, n] = bbox;
  const eras = doc.eras.map((era) => {
    const rc = [[w, s], [w, n], [e, s], [e, n]].map(([x, y]) => lonLatToRowCol(era, x!, y!));
    const r0 = Math.floor(Math.min(...rc.map((v) => v[0]))) - 1, r1 = Math.ceil(Math.max(...rc.map((v) => v[0]))) + 1;
    const c0 = Math.floor(Math.min(...rc.map((v) => v[1]))) - 1, c1 = Math.ceil(Math.max(...rc.map((v) => v[1]))) + 1;
    return { ...era, box: [r0, r1, c0, c1] as [number, number, number, number] };
  });
  const keep = (era: Era, v: VarLayout, refs: Ref[]) => {
    const [chY, chX] = v.chunks as [number, number];
    const [r0, r1, c0, c1] = era.box;
    return refs.filter(([cy, cx]) => cy * chY <= r1 && (cy + 1) * chY > r0 && cx * chX <= c1 && (cx + 1) * chX > c0);
  };
  const scans = doc.scans
    .filter((sc) => sc.t >= from && sc.t <= to)
    .map((sc) => {
      const era = eras[sc.era]!;
      return { ...sc, refs: { Mask: keep(era, era.vars.Mask, sc.refs.Mask), Power: keep(era, era.vars.Power, sc.refs.Power) } };
    });
  return { ...doc, eras, scans };
}

/** A pixel's footprint: its four corners on the ground (no terrain: a fire high on a ridge can sit about a km off). */
function footprint(e: Era, row: number, col: number): [number, number][] | null {
  const g = e.grid;
  const pts: [number, number][] = [];
  for (const [dr, dc] of [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]] as const) {
    const p = xyToLonLat(g.x0 + g.dx * (col + dc), g.y0 + g.dy * (row + dr), e.projection);
    if (!p) return null;
    pts.push(p);
  }
  return pts;
}

/**
 * Read every scan's chunks over the fire's box and hand back its fire pixels as they arrive (a few scans in
 * flight; onScan is called for each scan read, in any order). Resolves when all are read.
 */
export async function readGoesFire(
  doc: GoesDoc,
  onScan: (i: number, pixels: FirePixel[]) => void,
  signal?: AbortSignal,
  inFlight = 12,
  /** false: Mask only (the boxes view needs no fire power), each scan's chunks fetched in as few requests as fit. */
  withPower = true,
): Promise<void> {
  const fire = new Set(doc.fire_mask);
  let next = 0;
  const one = async (i: number) => {
    const s = doc.scans[i]!;
    const e = doc.eras[s.era]!;
    const url = `${doc.origin}/${s.key}`;
    const [r0, r1, c0, c1] = e.box;
    const [chY, chX] = e.vars.Mask.chunks as [number, number];
    const power = new Map<string, Float32Array>();
    if (!withPower) {
      const out: FirePixel[] = [];
      for (const [cy, cx, m] of await fetchChunks(url, s.refs.Mask, e.vars.Mask, signal)) {
        for (let y = 0; y < chY; y++) {
          const row = cy * chY + y;
          if (row < r0 || row > r1) continue;
          for (let x = 0; x < chX; x++) {
            const col = cx * chX + x;
            if (col < c0 || col > c1) continue;
            const v = m[y * chX + x]!;
            if (!fire.has(v)) continue;
            const poly = footprint(e, row, col);
            if (poly) out.push({ t: s.t, key: `${s.era}:${row}:${col}`, polygon: poly, power: NaN, mask: v });
          }
        }
      }
      onScan(i, out);
      return;
    }
    // nearby chunks in one request each, as for the mask alone
    for (const [cy, cx, p] of await fetchChunks<Float32Array>(url, s.refs.Power, e.vars.Power, signal)) power.set(`${cy}.${cx}`, p);
    const [pY, pX] = e.vars.Power.chunks as [number, number];
    const out: FirePixel[] = [];
    for (const [cy, cx, m] of await fetchChunks(url, s.refs.Mask, e.vars.Mask, signal)) {
      for (let y = 0; y < chY; y++) {
        const row = cy * chY + y;
        if (row < r0 || row > r1) continue;
        for (let x = 0; x < chX; x++) {
          const col = cx * chX + x;
          if (col < c0 || col > c1) continue;
          const v = m[y * chX + x]!;
          if (!fire.has(v)) continue;
          const poly = footprint(e, row, col);
          if (!poly) continue;
          const pc = power.get(`${Math.floor(row / pY)}.${Math.floor(col / pX)}`);
          const pw = pc ? pc[(row % pY) * pX + (col % pX)]! : NaN;
          out.push({ t: s.t, key: `${s.era}:${row}:${col}`, polygon: poly, power: pw > 0 ? pw : NaN, mask: v });
        }
      }
    }
    onScan(i, out);
  };
  const workers = Array.from({ length: inFlight }, async () => {
    while (next < doc.scans.length) {
      if (signal?.aborted) return;
      const i = next++;
      try {
        await one(i);
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        console.warn(`GOES scan ${doc.scans[i]!.key}:`, e);
        onScan(i, []);
      }
    }
  });
  await Promise.all(workers);
}

/** Fire pixels per hour: each footprint once per hour, with the hour's highest power. */
export function byHour(pixels: FirePixel[], t0: number): Map<number, FirePixel[]> {
  const out = new Map<number, Map<string, FirePixel>>();
  for (const p of pixels) {
    const h = Math.floor((p.t - t0) / 3600e3);
    let m = out.get(h);
    if (!m) out.set(h, (m = new Map()));
    const o = m.get(p.key);
    if (!o || (p.power > o.power || (Number.isNaN(o.power) && !Number.isNaN(p.power)))) m.set(p.key, p);
  }
  return new Map([...out].map(([h, m]) => [h, [...m.values()]]));
}

/** Fire power on a blue lightness ramp (dim blue at 10 MW, near white at 1000 MW and up): apart from the fuel's colors. */
function powerColor(p: number): [number, number, number] {
  const t = Number.isNaN(p) ? 0.15 : Math.min(1, Math.max(0, (Math.log10(p) - 1) / 2));
  return [Math.round(70 + 170 * t), Math.round(140 + 105 * t), 255];
}

export function goesLayers(o: { now: FirePixel[]; seen: FirePixel[]; perimeter: [number, number][][][]; beforeId?: string; trigger: unknown }) {
  return [
    new PolygonLayer<{ polygon: [number, number][][] }>({
      id: "goes-perimeter",
      data: o.perimeter.map((polygon) => ({ polygon })),
      getPolygon: (d: { polygon: [number, number][][] }) => d.polygon,
      filled: false,
      stroked: true,
      getLineColor: [236, 236, 236, 200],
      lineWidthUnits: "pixels",
      getLineWidth: 1.5,
      beforeId: o.beforeId,
    } as never),
    // every footprint seen so far: a faint outline
    new PolygonLayer<FirePixel>({
      id: "goes-seen",
      data: o.seen,
      getPolygon: (d: FirePixel) => d.polygon,
      filled: false,
      stroked: true,
      getLineColor: [120, 170, 230, 110],
      lineWidthUnits: "pixels",
      getLineWidth: 1,
      updateTriggers: { getPolygon: o.trigger },
      beforeId: o.beforeId,
    } as never),
    // this hour's fire pixels, filled by their power
    new PolygonLayer<FirePixel>({
      id: "goes-now",
      data: o.now,
      getPolygon: (d: FirePixel) => d.polygon,
      filled: true,
      stroked: true,
      getFillColor: (d: FirePixel) => [...powerColor(d.power), 70],
      getLineColor: (d: FirePixel) => [...powerColor(d.power), 255],
      lineWidthUnits: "pixels",
      getLineWidth: 2,
      updateTriggers: { getFillColor: o.trigger, getLineColor: o.trigger },
      beforeId: o.beforeId,
    } as never),
  ];
}
