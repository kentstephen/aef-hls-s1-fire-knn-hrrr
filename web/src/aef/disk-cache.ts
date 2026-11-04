import type * as zarr from "zarrita";

/**
 * Bytes kept on disk between visits, in the browser's IndexedDB (the page and the chunk worker share it). Nothing to
 * look after:
 * - it trims itself, least recently used first, to CAP (or half of what the browser lets this site keep, whichever is
 *   less), and drops what has expired;
 * - the storage is the browser's best-effort kind (never asked to persist), so the browser clears it on its own when
 *   the disk runs short;
 * - what it keeps refreshes itself: a Zarr chunk is kept under a fingerprint of its array's zarr.json (read from the
 *   network every visit), so a store that changes reads afresh and its old chunks age out; everything has an expiry.
 */

const DB = "aef-fuel-cache";
const BYTES = "bytes";
const META = "meta";
const GB = 1024 ** 3;
const DAY = 864e5;
/** The most kept, before the browser's own share is known. */
const CAP = 16 * GB;
/** How long a Zarr chunk is kept without its array changing. */
export const ZARR_TTL = 90 * DAY;
/** A trim after this much written. */
const TRIM_EVERY = 256 * 1024 ** 2;
/** A hit marks its entry used again only when the last mark is older than this (a write a read is not worth). */
const TOUCH_AFTER = 6 * 36e5;

type Meta = { k: string; n: number; used: number; exp: number };

let dbp: Promise<IDBDatabase | null> | null = null;
let written = 0;
let trimming: Promise<void> | null = null;
const stats = { hits: 0, misses: 0, hitBytes: 0, putBytes: 0 };

const req = <T>(r: IDBRequest<T>) =>
  new Promise<T>((ok, no) => {
    r.onsuccess = () => ok(r.result);
    r.onerror = () => no(r.error);
  });

function open(): Promise<IDBDatabase | null> {
  if (dbp) return dbp;
  dbp = new Promise((ok) => {
    try {
      const r = indexedDB.open(DB, 1);
      r.onupgradeneeded = () => {
        const db = r.result;
        db.createObjectStore(BYTES);
        db.createObjectStore(META, { keyPath: "k" }).createIndex("used", "used");
      };
      r.onsuccess = () => {
        const db = r.result;
        // another tab's newer version: let it have the database, start over on the next ask
        db.onversionchange = () => {
          db.close();
          dbp = null;
        };
        ok(db);
      };
      // no IndexedDB (a private window, blocked storage): everything from the network, as before
      r.onerror = () => ok(null);
      r.onblocked = () => ok(null);
    } catch {
      ok(null);
    }
  });
  // the first trim once the map has started (expired entries, a cap lowered since)
  setTimeout(() => void trim(), 15000);
  return dbp;
}

/**
 * The bytes kept under `key`, unless missing or expired. The asks of one moment are read in one transaction and all
 * answered at once, so the misses reach the network together.
 */
export function cacheGet(key: string): Promise<Uint8Array | undefined> {
  return new Promise((resolve) => {
    if (!pending.length) setTimeout(flushGets, 0);
    pending.push({ key, resolve });
  });
}

let pending: { key: string; resolve: (b: Uint8Array | undefined) => void }[] = [];

async function flushGets() {
  const asks = pending;
  pending = [];
  const db = await open();
  const out: (Uint8Array | undefined)[] = asks.map(() => undefined);
  if (db) {
    try {
      const tx = db.transaction([BYTES, META], "readonly");
      const ms = tx.objectStore(META), bs = tx.objectStore(BYTES);
      const now = Date.now(), touch: Meta[] = [];
      asks.forEach((a, i) => {
        const mr = ms.get(a.key) as IDBRequest<Meta | undefined>;
        mr.onsuccess = () => {
          const m = mr.result;
          if (!m || m.exp < now) return;
          const br = bs.get(a.key) as IDBRequest<ArrayBuffer | undefined>;
          br.onsuccess = () => {
            if (!br.result) return;
            out[i] = new Uint8Array(br.result);
            if (now - m.used > TOUCH_AFTER) touch.push({ ...m, used: now });
          };
        };
      });
      await new Promise((ok) => {
        tx.oncomplete = tx.onerror = tx.onabort = ok;
      });
      if (touch.length) {
        const w = db.transaction(META, "readwrite").objectStore(META);
        for (const m of touch) w.put(m);
      }
    } catch {
      // read from the network
    }
  }
  asks.forEach((a, i) => {
    const b = out[i];
    if (b) {
      stats.hits++;
      stats.hitBytes += b.byteLength;
    } else stats.misses++;
    a.resolve(b);
  });
}

/** Keeps `bytes` under `key` for `ttl` ms. Never waited on: a failed write only means a later read from the network. */
export function cachePut(key: string, bytes: Uint8Array, ttl: number) {
  void (async () => {
    const db = await open();
    if (!db) return;
    try {
      // its own buffer: a view's whole buffer would be cloned otherwise
      const buf = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength ? bytes.buffer : bytes.slice().buffer;
      const now = Date.now();
      const tx = db.transaction([BYTES, META], "readwrite");
      tx.objectStore(BYTES).put(buf, key);
      tx.objectStore(META).put({ k: key, n: buf.byteLength, used: now, exp: now + ttl } satisfies Meta);
      stats.putBytes += buf.byteLength;
      written += buf.byteLength;
      if (written >= TRIM_EVERY) {
        written = 0;
        void trim();
      }
    } catch {
      // over the browser's quota: make room for the next one
      void trim();
    }
  })();
}

/** Drops the expired, then the least recently used until under the cap. */
export function trim(): Promise<void> {
  if (trimming) return trimming;
  trimming = (async () => {
    const db = await open();
    if (!db) return;
    let cap = CAP;
    try {
      const q = (await navigator.storage?.estimate?.())?.quota;
      if (q) cap = Math.min(cap, q / 2);
    } catch {
      // no estimate: the cap as it is
    }
    // every entry's size and use, oldest use first (the sizes only, never the bytes)
    const all: Meta[] = await req(db.transaction(META, "readonly").objectStore(META).index("used").getAll());
    const now = Date.now();
    let total = 0;
    for (const m of all) total += m.n;
    const drop: string[] = [];
    for (const m of all) {
      if (m.exp < now || total > cap) {
        drop.push(m.k);
        total -= m.n;
      }
    }
    for (let i = 0; i < drop.length; i += 500) {
      const tx = db.transaction([BYTES, META], "readwrite");
      for (const k of drop.slice(i, i + 500)) {
        tx.objectStore(BYTES).delete(k);
        tx.objectStore(META).delete(k);
      }
      await new Promise((ok) => {
        tx.oncomplete = tx.onerror = tx.onabort = ok;
      });
    }
  })()
    .catch(() => undefined)
    .finally(() => {
      trimming = null;
    });
  return trimming;
}

/** What this page (or worker) read from the cache and wrote to it, and what the cache holds. */
export async function cacheStats() {
  const db = await open();
  let entries = 0,
    bytes = 0;
  if (db) {
    const all: Meta[] = await req(db.transaction(META, "readonly").objectStore(META).getAll());
    entries = all.length;
    for (const m of all) bytes += m.n;
  }
  return { ...stats, entries, bytes };
}

/** A short fingerprint of text (FNV-1a), for keys. */
export function fingerprint(s: string) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 0x01000193);
  return (h >>> 0).toString(36) + s.length.toString(36);
}

type Range = { offset: number; length: number } | { suffixLength: number };
const dirOf = (key: string) => key.slice(0, key.lastIndexOf("/"));

/**
 * A Zarr store whose chunks are kept on disk (cacheGet). Every zarr.json still comes from the network, and its
 * fingerprint goes into the keys of the chunks under it (its array's), so a changed array is read afresh.
 */
export class DiskCachedStore implements zarr.AsyncReadable {
  private meta = new Map<string, string>();

  constructor(
    private url: string,
    private inner: zarr.AsyncReadable,
  ) {}

  /** The fingerprint of the nearest zarr.json above `key`: its array's. */
  private gen(key: string) {
    for (let d = dirOf(key); ; d = dirOf(d)) {
      const g = this.meta.get(d);
      if (g) return g;
      if (!d) return null;
    }
  }

  keyOf(key: string, range?: Range) {
    const g = this.gen(key);
    if (!g) return null;
    const r = !range ? "" : "suffixLength" in range ? `:-${range.suffixLength}` : `:${range.offset}+${range.length}`;
    return `zarr|${g}|${this.url}${key}${r}`;
  }

  async get(key: zarr.AbsolutePath, opts?: { signal?: AbortSignal }) {
    if (key.endsWith("zarr.json")) {
      const b = await this.inner.get(key, opts);
      if (b) this.meta.set(dirOf(key), fingerprint(new TextDecoder().decode(b)));
      return b;
    }
    return this.cached(this.keyOf(key), () => this.inner.get(key, opts));
  }

  async getRange(key: zarr.AbsolutePath, range: Range, opts?: { signal?: AbortSignal }) {
    return this.cached(this.keyOf(key, range), () => this.inner.getRange!(key, range as never, opts));
  }

  private async cached(k: string | null, read: () => Promise<Uint8Array | undefined>) {
    if (k) {
      const have = await cacheGet(k);
      if (have) return have;
    }
    const b = await read();
    if (k && b) cachePut(k, b, ZARR_TTL);
    return b;
  }
}
