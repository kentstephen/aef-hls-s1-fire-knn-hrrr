import * as zarr from "zarrita";
import { fosbergBlock, type WindMode } from "../ffwi/compute.js";
import { FFWI_BANDS, HRRR_GRID, HRRR_LEAD_TIME_COUNT, HRRR_SHARD } from "./metadata.js";
import type { HrrrArrays } from "./store.js";
import { FfwiTileCache, type FfwiTile } from "./tile-cache.js";

/**
 * Fetch the shards the cell grid needs, independent of any raster layer.
 * Same fetch + FFWI pass as getTileData, minus the GPU texture. Fills the
 * shared FfwiTileCache; in-flight shards are tracked so the raster layer
 * (when on) and this channel never double-fetch.
 */
export class SideChannel {
  private inflight = new Set<string>();
  private ctrl: AbortController | null = null;

  constructor(
    private arrays: HrrrArrays,
    private cache: FfwiTileCache,
  ) {}

  abort(): void {
    this.ctrl?.abort();
    this.ctrl = null;
    this.inflight.clear();
  }

  /** Ensure shards covering the given HRRR pixels are loaded. */
  async ensure(
    hrrrX: Int16Array,
    hrrrY: Int16Array,
    initTimeIdx: number,
    windMode: WindMode,
    concurrency = 4,
  ): Promise<void> {
    if (!this.ctrl) this.ctrl = new AbortController();
    const signal = this.ctrl.signal;
    const wanted = new Map<string, { row: number; col: number }>();
    for (let i = 0; i < hrrrX.length; i++) {
      const x = hrrrX[i]!;
      if (x < 0) continue;
      const key = FfwiTileCache.keyFor(x, hrrrY[i]!);
      if (this.cache.get(key) || this.inflight.has(key) || wanted.has(key)) continue;
      wanted.set(key, {
        row: Math.floor(hrrrY[i]! / HRRR_SHARD.height),
        col: Math.floor(x / HRRR_SHARD.width),
      });
    }
    const queue = [...wanted.values()];
    let next = 0;
    const worker = async () => {
      while (!signal.aborted) {
        const job = queue[next++];
        if (!job) return;
        const key = `${job.row},${job.col}`;
        this.inflight.add(key);
        try {
          await this.fetchShard(job.row, job.col, initTimeIdx, windMode, signal);
        } catch (e) {
          if (!signal.aborted) console.warn("side channel shard failed", key, e);
        } finally {
          this.inflight.delete(key);
        }
      }
    };
    await Promise.all(Array.from({ length: concurrency }, worker));
  }

  private async fetchShard(
    row: number,
    col: number,
    initTimeIdx: number,
    windMode: WindMode,
    signal: AbortSignal,
  ): Promise<void> {
    const rowStart = row * HRRR_SHARD.height;
    const colStart = col * HRRR_SHARD.width;
    const rowEnd = Math.min(rowStart + HRRR_SHARD.height, HRRR_GRID.height);
    const colEnd = Math.min(colStart + HRRR_SHARD.width, HRRR_GRID.width);
    const height = rowEnd - rowStart;
    const width = colEnd - colStart;
    const n = HRRR_LEAD_TIME_COUNT * height * width;
    const slices = [initTimeIdx, null, zarr.slice(rowStart, rowEnd), zarr.slice(colStart, colEnd)];
    const chunks = await Promise.all(
      FFWI_BANDS.map((b) => zarr.get(this.arrays[b], slices, { signal })),
    );
    if (signal.aborted) return;
    const mEq = new Uint16Array(n);
    const wind = new Uint16Array(n);
    const rh = new Uint8Array(n);
    const tc = new Uint8Array(n);
    fosbergBlock(
      n,
      chunks[0]!.data as Float32Array,
      chunks[1]!.data as Float32Array,
      chunks[2]!.data as Float32Array,
      chunks[3]!.data as Float32Array,
      chunks[4]!.data as Float32Array,
      windMode,
      mEq,
      wind,
      rh,
      tc,
    );
    const tile: FfwiTile = { key: `${row},${col}`, rowStart, colStart, width, height, mEq, mLag: mEq, wind, rh, tc };
    this.cache.set(tile);
  }
}
