import * as zarr from "zarrita";
import { DiskCachedStore } from "./disk-cache.js";

/**
 * The AlphaEarth Zarr mosaic on Source Cooperative: one level at 10 m, no overviews, 256 px inner
 * chunks of all 64 bands in 4096 px shards. Read from the S3 host behind the gateway (the gateway
 * throttles many requests at once), with the browser's HTTP cache off (a suffix range answered with
 * a whole shard was coming back short from it) and the chunks kept on disk instead (disk-cache.ts,
 * keyed by the gateway URL).
 */
export const ZARR_URL = "https://data.source.coop/tge-labs/aef-mosaic";
export const fromS3 = (url: string) =>
  url.replace("https://data.source.coop/", "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/");
export const VARIABLE = "embeddings";
export const YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];
/** The embedding year the map reads: the year before the fires it is judged on (2025 perimeters). */
export const AEF_YEAR = 2025;
export const YEAR_IDX = YEARS.indexOf(AEF_YEAR);
/** Below this the mosaic would read too much (every zoom step out is four times the chunks). */
export const MOSAIC_MIN_ZOOM = 12;
/** The grid: EPSG:4326, pixel registered. */
export const RES = 8.983111749910169e-5;
export const X0 = -180;
export const Y0 = 83.68570533713473;
export const CHUNK = 256;

export type AefArray = zarr.Array<"int8", zarr.Readable>;

export async function openAefMosaic(): Promise<{ arr: AefArray; attrs: unknown }> {
  const store = new DiskCachedStore(ZARR_URL, new zarr.FetchStore(fromS3(ZARR_URL), { overrides: { cache: "no-store" } }));
  const root = await zarr.open.v3(store, { kind: "group" });
  const arr = await zarr.open.v3(root.resolve(VARIABLE), { kind: "array" });
  if (!arr.is("int8")) throw new Error(`Expected int8 embeddings, got ${arr.dtype}`);
  return { arr, attrs: root.attrs };
}

export const pixelOf = (lon: number, lat: number) => [Math.floor((lon - X0) / RES), Math.floor((Y0 - lat) / RES)] as const;
