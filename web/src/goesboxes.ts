import { PolygonLayer } from "@deck.gl/layers";
import type { PickingInfo } from "@deck.gl/core";
import type { AefTile } from "./aef/layers.js";
import { RES, X0, Y0 } from "./aef/mosaic.js";
import type { FirePixel } from "./goes.js";

/**
 * GOES-West's fires over a window as boxes: every fire pixel read over California (goes.ts, readGoesFire on a
 * web/public/goes/ca-<date>.json reference file), joined into events the way scripts/goes_events.py does it:
 * pixels that touch (the same or a neighboring GOES pixel) on the same or the next UTC day, chained. Each event's
 * box is the outer edge of all its pixels' footprints.
 */

export type GoesEvent = {
  id: number;
  /** West, south, east, north of all its pixels' footprints. */
  bbox: [number, number, number, number];
  t0: number;
  t1: number;
  scans: number;
  pixels: number;
  /** Every fire pixel of every scan. */
  px: FirePixel[];
};

const DAY = 86400e3;

/** Pixels (from every scan read so far) into events, biggest first. */
export function goesEvents(pixels: FirePixel[]): GoesEvent[] {
  // bins of (pixel, UTC day); a bin joins its 3 x 3 neighbors on the same and the previous day
  type Bin = { key: string; era: string; row: number; col: number; day: number; px: FirePixel[] };
  const bins = new Map<string, Bin>();
  for (const p of pixels) {
    const [era, row, col] = p.key.split(":");
    const day = Math.floor(p.t / DAY);
    const k = `${era}:${row}:${col}:${day}`;
    let b = bins.get(k);
    if (!b) bins.set(k, (b = { key: k, era: era!, row: Number(row), col: Number(col), day, px: [] }));
    b.px.push(p);
  }
  const list = [...bins.values()];
  const idx = new Map(list.map((b, i) => [b.key, i]));
  const parent = list.map((_, i) => i);
  const find = (i: number): number => {
    while (parent[i] !== i) i = parent[i] = parent[parent[i]!]!;
    return i;
  };
  list.forEach((b, i) => {
    for (const dd of [0, -1])
      for (let dr = -1; dr <= 1; dr++)
        for (let dc = -1; dc <= 1; dc++) {
          const j = idx.get(`${b.era}:${b.row + dr}:${b.col + dc}:${b.day + dd}`);
          if (j !== undefined) parent[find(i)] = find(j);
        }
  });
  const groups = new Map<number, Bin[]>();
  list.forEach((b, i) => {
    const r = find(i);
    const g = groups.get(r);
    if (g) g.push(b);
    else groups.set(r, [b]);
  });
  const out: GoesEvent[] = [];
  for (const g of groups.values()) {
    let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity, t0 = Infinity, t1 = -Infinity;
    const scans = new Set<number>();
    const px = new Set<string>();
    const all: FirePixel[] = [];
    for (const b of g) {
      all.push(...b.px);
      for (const p of b.px) {
        for (const [x, y] of p.polygon) {
          if (x < w) w = x;
          if (x > e) e = x;
          if (y < s) s = y;
          if (y > n) n = y;
        }
        if (p.t < t0) t0 = p.t;
        if (p.t > t1) t1 = p.t;
        scans.add(p.t);
        px.add(p.key);
      }
    }
    out.push({ id: 0, bbox: [w, s, e, n], t0, t1, scans: scans.size, pixels: px.size, px: all });
  }
  // ids by first time, so an event keeps its id as more scans come in (mostly)
  out.sort((a, b) => a.t0 - b.t0 || a.bbox[0] - b.bbox[0]);
  out.forEach((ev, i) => (ev.id = i));
  return out;
}

/** The ring around a box: 1.5 GOES pixels out, as in the fire test (about 3 km each side over California). */
export const RING_KM = 3;

export function ringBox(b: [number, number, number, number]): [number, number, number, number] {
  const dLat = RING_KM / 111.32;
  const dLon = dLat / Math.cos((((b[1] + b[3]) / 2) * Math.PI) / 180);
  return [b[0] - dLon, b[1] - dLat, b[2] + dLon, b[3] + dLat];
}

const rect = ([w, s, e, n]: [number, number, number, number]): [number, number][] => [[w, s], [e, s], [e, n], [w, n], [w, s]];

/** One color for every box: a light blue, apart from the fuel ramp's colors; the picked one drawn wider. */
const BOX: [number, number, number] = [130, 200, 255];

export function goesBoxLayers(o: {
  events: GoesEvent[];
  picked: number | null;
  beforeId?: string;
  onHover?: (ev: GoesEvent | null, x: number, y: number) => void;
  onClick?: (ev: GoesEvent) => void;
}) {
  return [
    new PolygonLayer<GoesEvent>({
      id: "goes-boxes",
      data: o.events,
      getPolygon: (d: GoesEvent) => rect(d.bbox),
      filled: true,
      stroked: true,
      // the fill is nearly clear: it is there so a click inside the box picks it
      getFillColor: [...BOX, 8],
      getLineColor: (d: GoesEvent) => [...BOX, d.id === o.picked ? 255 : 200],
      getLineWidth: (d: GoesEvent) => (d.id === o.picked ? 2.5 : 1.25),
      lineWidthUnits: "pixels",
      pickable: true,
      updateTriggers: { getLineColor: o.picked, getLineWidth: o.picked },
      onHover: (info: PickingInfo<GoesEvent>) => o.onHover?.(info.object ?? null, info.x, info.y),
      onClick: (info: PickingInfo<GoesEvent>) => { if (info.object) o.onClick?.(info.object); },
      beforeId: o.beforeId,
    } as never),
  ];
}

/** The fuel lit inside the picked box (full) and its ring (part way); the rest stays dim. */
export function boxMask(t: AefTile, box: [number, number, number, number]): Uint8Array | null {
  const ring = ringBox(box);
  const [tw, ts, te, tn] = t.bbox;
  if (ring[2] < tw || ring[0] > te || ring[3] < ts || ring[1] > tn) return null;
  // the tile's own pixel grid (the mosaic's), as season.ts draws its masks
  const px0 = Math.round((tw - X0) / RES);
  const py0 = Math.round((Y0 - tn) / RES);
  const out = new Uint8Array(t.width * t.height);
  const fill = (b: [number, number, number, number], v: number) => {
    const c0 = Math.max(0, Math.floor((b[0] - X0) / RES - px0)), c1 = Math.min(t.width, Math.ceil((b[2] - X0) / RES - px0));
    const r0 = Math.max(0, Math.floor((Y0 - b[3]) / RES - py0)), r1 = Math.min(t.height, Math.ceil((Y0 - b[1]) / RES - py0));
    if (c1 <= c0) return;
    for (let r = r0; r < r1; r++) out.fill(v, r * t.width + c0, r * t.width + c1);
  };
  fill(ring, 130);
  fill(box, 255);
  return out;
}

/**
 * The outline of a set of GOES pixels, touching ones merged: each pixel's four edges, keyed by the grid corners
 * they join; an edge two pixels share appears twice and is dropped. Returns the remaining edges as segments.
 */
export function fireOutline(pixels: FirePixel[]): [number, number][][] {
  const edges = new Map<string, { n: number; seg: [number, number][] }>();
  for (const p of pixels) {
    const [era, rs, cs] = p.key.split(":");
    const r = Number(rs), c = Number(cs);
    // footprint corners in goes.ts order: (r-.5,c-.5) (r-.5,c+.5) (r+.5,c+.5) (r+.5,c-.5) -> corner ids
    const ids = [`${r}:${c}`, `${r}:${c + 1}`, `${r + 1}:${c + 1}`, `${r + 1}:${c}`];
    for (let k = 0; k < 4; k++) {
      const a = ids[k]!, b = ids[(k + 1) % 4]!;
      const key = `${era}|${a < b ? a + "|" + b : b + "|" + a}`;
      const e = edges.get(key);
      if (e) e.n++;
      else edges.set(key, { n: 1, seg: [p.polygon[k]!, p.polygon[(k + 1) % 4]!] });
    }
  }
  return [...edges.values()].filter((e) => e.n === 1).map((e) => e.seg);
}
