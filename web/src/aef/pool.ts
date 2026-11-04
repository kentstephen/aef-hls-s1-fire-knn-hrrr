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
      w.onmessage = (ev: MessageEvent<{ id: number; error?: { name: string; message: string }; nlcd?: Uint8Array; esri?: Uint8Array; cover?: Uint8Array; ign?: Uint8Array; fuel?: Uint8Array }>) => {
        const { id, error, nlcd, esri, cover, ign, fuel } = ev.data;
        const p = ent.waiting.get(id);
        if (!p) return;
        ent.waiting.delete(id);
        ent.busy--;
        if (error) p.no(Object.assign(new Error(error.message), { name: error.name }));
        else p.ok({ nlcd: nlcd!, esri: esri!, cover: cover!, ign: ign!, fuel: fuel ?? new Uint8Array(nlcd!.length) });
      };
      return ent;
    });
  }

  private ask(msg: Record<string, unknown>, transfer: ArrayBuffer[], signal?: AbortSignal): Promise<Classified> {
    if (signal?.aborted) return Promise.reject(aborted());
    const ent = this.ws.reduce((a, b) => (b.busy < a.busy ? b : a));
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

  /** A mosaic chunk read and classified in the worker. */
  zarr(timeIdx: number, py0: number, px0: number, h: number, w: number, signal?: AbortSignal): Promise<Classified> {
    return this.ask({ type: "zarr", timeIdx, py0, px0, h, w }, [], signal);
  }
}
