import type { Classified } from "./head.js";

/** A few classify workers, the least busy first; an ask carries its own abort. */
type W = { w: Worker; busy: number; waiting: Map<number, { ok: (c: Classified) => void; no: (e: Error) => void }> };

const aborted = () => Object.assign(new Error("aborted"), { name: "AbortError" });

export class AefPool {
  private ws: W[];
  private nextId = 0;

  constructor(size = Math.max(2, Math.min(4, (navigator.hardwareConcurrency || 4) - 2))) {
    this.ws = Array.from({ length: size }, () => {
      const w = new Worker(new URL("./aef.worker.ts", import.meta.url), { type: "module" });
      const ent: W = { w, busy: 0, waiting: new Map() };
      w.onmessage = (ev: MessageEvent<{ id: number; error?: { name: string; message: string }; nlcd?: Uint8Array; esri?: Uint8Array; cover?: Uint8Array; ign?: Uint8Array; fuel?: Uint8Array; vx?: Uint32Array; v0?: Uint8Array; ms?: Classified["ms"] }>) => {
        const { id, error, nlcd, esri, cover, ign, fuel, vx, v0, ms } = ev.data;
        const p = ent.waiting.get(id);
        if (!p) return;
        ent.waiting.delete(id);
        ent.busy--;
        if (error) p.no(Object.assign(new Error(error.message), { name: error.name }));
        else p.ok({ nlcd: nlcd!, esri: esri!, cover: cover!, ign: ign!, fuel: fuel ?? new Uint8Array(nlcd!.length), vx, v0, ms });
      };
      return ent;
    });
  }

  /** `slot`: the worker to use (a chunk always goes to the same one, which keeps its reads); else the least busy. */
  private ask(msg: Record<string, unknown>, transfer: ArrayBuffer[], signal?: AbortSignal, slot?: number): Promise<Classified> {
    if (signal?.aborted) return Promise.reject(aborted());
    const ent = slot === undefined ? this.ws.reduce((a, b) => (b.busy < a.busy ? b : a)) : this.ws[slot % this.ws.length]!;
    const id = ++this.nextId;
    ent.busy++;
    return new Promise<Classified>((ok, no) => {
      ent.waiting.set(id, { ok, no });
      ent.w.postMessage({ ...msg, id }, transfer);
      signal?.addEventListener(
        "abort",
        () => {
          if (!ent.waiting.has(id)) return;
          ent.waiting.delete(id);
          ent.busy--;
          ent.w.postMessage({ type: "abort", id });
          no(aborted());
        },
        { once: true },
      );
    });
  }

  /** Band-major int8 [64, n]; the buffer is handed to the worker. */
  classify(data: Int8Array, n: number, signal?: AbortSignal): Promise<Classified> {
    return this.ask({ type: "classify", data, n }, [data.buffer as ArrayBuffer], signal);
  }

  /** A mosaic chunk read and classified in the worker; fuelYearIdx: the fuel head's years (oldest first), else its own. */
  /** lean: no land cover heads and nothing kept for reuse (aef.worker.ts), for a view that draws the volume alone. */
  zarr(timeIdx: number, py0: number, px0: number, h: number, w: number, signal?: AbortSignal, fuelYearIdx?: number[], value?: "pfuel" | "volume", lean = false): Promise<Classified> {
    // a small hash of the chunk's place, so neighbors spread over the workers
    const slot = Math.abs(Math.imul(py0 >> 8, 73856093) ^ Math.imul(px0 >> 8, 19349663));
    return this.ask({ type: "zarr", timeIdx, py0, px0, h, w, fuelYearIdx, value, lean }, [], signal, slot);
  }
}
