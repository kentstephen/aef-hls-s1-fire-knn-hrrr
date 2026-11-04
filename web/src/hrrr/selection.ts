import type * as zarr from "zarrita";

/**
 * Pin init_time, keep the full lead_time dim. One spatial shard
 * `(1, 49, 265, 300)` already holds all 49 leads, so one fetch per tile
 * delivers the whole animation.
 */
export function buildSelection(
  initTimeIdx: number,
): Record<string, number | zarr.Slice | null> {
  return { init_time: initTimeIdx, lead_time: null };
}
