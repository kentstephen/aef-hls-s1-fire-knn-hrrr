import { ScatterplotLayer } from "@deck.gl/layers";

/**
 * The real ignitions over the model: every WFIGS wildfire (ignition/export_fires.py, fires.bin), each
 * lit on the map at its discovery hour. A fire flares when its hour comes (a wide ring that shrinks
 * over the first hours) and then stays as a point that fades over two days, so a run's window shows
 * the fires that started in it landing on the ground the head has lit. Natural fires are rings,
 * human and unknown ones filled points; size sets the point's radius. Orange, which viridis never
 * reaches, so the fires read against the ground for every kind of color vision.
 */
export type Fires = {
  n: number;
  lon: Float32Array;
  lat: Float32Array;
  /** Hours since 2018-01-01T00Z of discovery. */
  hour: Uint32Array;
  /** 0 unknown, 1 natural, 2 human. */
  cause: Uint8Array;
  /** 0 unknown, 1 under 1 acre, 2 under 10, 3 under 100, 4 under 1000, 5 larger. */
  size: Uint8Array;
};

export const FIRES_EPOCH_MS = Date.UTC(2018, 0, 1);

export async function loadFires(baseUrl: string): Promise<Fires> {
  const r = await fetch(`${baseUrl}fires.bin`);
  if (!r.ok) throw new Error(`fires.bin: ${r.status}`);
  const buf = await r.arrayBuffer();
  const n = new DataView(buf).getUint32(0, true);
  let o = 4;
  const lon = new Float32Array(buf, o, n); o += 4 * n;
  const lat = new Float32Array(buf, o, n); o += 4 * n;
  const hour = new Uint32Array(buf, o, n); o += 4 * n;
  const cause = new Uint8Array(buf, o, n); o += n;
  const size = new Uint8Array(buf, o, n);
  return { n, lon, lat, hour, cause, size };
}

export type Fire = { i: number; lon: number; lat: number; hour: number; cause: number; size: number };

/** How many hours before the run's +0 a fire still shows (fading), and how long after discovery it stays. */
const BEFORE_H = 24;
const KEEP_H = 48;

/** The fires whose discovery falls in the run's window: BEFORE_H before +0 to +48 h. The arrays are sorted by hour. */
export function firesInRun(f: Fires, initTime: Date): Fire[] {
  const h0 = Math.floor((initTime.getTime() - FIRES_EPOCH_MS) / 3600e3);
  const lo = h0 - BEFORE_H;
  const hi = h0 + 48;
  // binary search the first hour >= lo
  let a = 0;
  let b = f.n;
  while (a < b) {
    const m = (a + b) >> 1;
    if (f.hour[m]! < lo) a = m + 1;
    else b = m;
  }
  const out: Fire[] = [];
  for (let i = a; i < f.n && f.hour[i]! <= hi; i++) {
    out.push({ i, lon: f.lon[i]!, lat: f.lat[i]!, hour: f.hour[i]!, cause: f.cause[i]!, size: f.size[i]! });
  }
  return out;
}

const ORANGE: [number, number, number] = [255, 150, 40];

export type FireLayerOpts = {
  fires: Fire[];
  initTime: Date;
  leadHours: number;
  visible: boolean;
  beforeId?: string;
};

/** Age in hours of a fire at the run's lead (negative = not yet discovered). */
function ageAt(d: Fire, initTime: Date, leadHours: number): number {
  const now = (initTime.getTime() - FIRES_EPOCH_MS) / 3600e3 + leadHours;
  return now - d.hour;
}

export function fireLayers(o: FireLayerOpts) {
  const { fires, initTime, leadHours } = o;
  const alpha = (d: Fire) => {
    const age = ageAt(d, initTime, leadHours);
    if (age < 0 || age > KEEP_H) return 0;
    return Math.round(255 * Math.max(0.3, Math.exp(-age / 24)));
  };
  const radius = (d: Fire) => {
    const age = ageAt(d, initTime, leadHours);
    if (age < 0) return 0;
    const base = 1.5 + 0.9 * d.size;
    return base * (1 + 2.0 * Math.exp(-age / 2));
  };
  const common = {
    data: fires,
    getPosition: (d: Fire) => [d.lon, d.lat] as [number, number],
    radiusUnits: "pixels" as const,
    getRadius: radius,
    radiusMinPixels: 0,
    lineWidthUnits: "pixels" as const,
    pickable: true,
    visible: o.visible,
    beforeId: o.beforeId,
    updateTriggers: { getRadius: [initTime.getTime(), leadHours], getFillColor: [initTime.getTime(), leadHours], getLineColor: [initTime.getTime(), leadHours] },
  };
  return [
    new ScatterplotLayer<Fire>({
      ...common,
      id: "fires-human",
      data: fires.filter((d) => d.cause !== 1),
      filled: true,
      stroked: false,
      getFillColor: (d: Fire) => [...ORANGE, alpha(d)],
    } as never),
    new ScatterplotLayer<Fire>({
      ...common,
      id: "fires-natural",
      data: fires.filter((d) => d.cause === 1),
      filled: false,
      stroked: true,
      getLineWidth: 1.5,
      getLineColor: (d: Fire) => [...ORANGE, alpha(d)],
    } as never),
  ];
}
