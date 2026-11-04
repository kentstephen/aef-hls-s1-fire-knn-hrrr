import { IcechunkStore } from "icechunk-js";
import * as zarr from "zarrita";
import {
  FFWI_BANDS,
  HRRR_ICECHUNK_URL,
  HRRR_ZARR_FALLBACK_URL,
  type FfwiBand,
} from "./metadata.js";

export type HrrrArrays = Record<FfwiBand, zarr.Array<"float32", zarr.Readable>>;

export type HrrrSource = {
  arrays: HrrrArrays;
  /** Which backend actually opened, for the UI footer. */
  backend: "icechunk" | "zarr-v3";
  initTimeCount: number;
};

function checkFloat32(
  band: string,
  a: zarr.Array<zarr.DataType, zarr.Readable>,
): zarr.Array<"float32", zarr.Readable> {
  if (!a.is("float32")) {
    throw new Error(`Expected ${band} float32, got ${a.dtype}`);
  }
  return a;
}

async function openIcechunk(): Promise<HrrrArrays> {
  const store = await IcechunkStore.open(HRRR_ICECHUNK_URL, {
    withRangeCoalescing: zarr.withRangeCoalescing,
  });
  const out = {} as HrrrArrays;
  await Promise.all(
    FFWI_BANDS.map(async (band) => {
      const a = await zarr.open(store.resolve(`/${band}`), { kind: "array" });
      out[band] = checkFloat32(band, a);
    }),
  );
  return out;
}

async function openZarrV3(): Promise<HrrrArrays> {
  const store = await zarr.withConsolidatedMetadata(
    new zarr.FetchStore(HRRR_ZARR_FALLBACK_URL),
    { format: "v3" },
  );
  const root = await zarr.open.v3(store, { kind: "group" });
  const out = {} as HrrrArrays;
  await Promise.all(
    FFWI_BANDS.map(async (band) => {
      const a = await zarr.open.v3(root.resolve(band), { kind: "array" });
      out[band] = checkFloat32(band, a);
    }),
  );
  return out;
}

/** Open the five FFWI input arrays. Icechunk first, Zarr v3 mirror second. */
export async function openHrrr(): Promise<HrrrSource> {
  try {
    const arrays = await openIcechunk();
    return {
      arrays,
      backend: "icechunk",
      initTimeCount: arrays.temperature_2m.shape[0]!,
    };
  } catch (err) {
    console.warn("icechunk open failed, falling back to Zarr v3 mirror", err);
    const arrays = await openZarrV3();
    return {
      arrays,
      backend: "zarr-v3",
      initTimeCount: arrays.temperature_2m.shape[0]!,
    };
  }
}
