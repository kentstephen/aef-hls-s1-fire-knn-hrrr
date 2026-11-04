/** The byte range the week keeps the weather term over (logit); 255 = missing. Shared by the week and its worker. */
const TERM_LO = -6;
const TERM_HI = 6;
export const termByte = (z: number) => (z === z ? Math.max(0, Math.min(254, Math.round(((z - TERM_LO) / (TERM_HI - TERM_LO)) * 254))) : 255);
export const termLogit = (b: number) => (b === 255 ? NaN : TERM_LO + (b / 254) * (TERM_HI - TERM_LO));
/** termLogit for every byte, for the hot loops. */
export const TERM_LUT = Float32Array.from({ length: 256 }, (_, b) => termLogit(b));
