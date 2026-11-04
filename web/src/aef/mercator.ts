/** Web Mercator tile math for the coarse reader (EPSG:3857, 256 px tiles). */
export const E = 20037508.342789244;
export const TILE = 256;

export function tileRes(z: number): number {
  return (2 * E) / (TILE * 2 ** z);
}

/** [minX, minY, maxX, maxY] in meters. */
export function tileBounds(z: number, x: number, y: number): [number, number, number, number] {
  const r = tileRes(z) * TILE;
  const minX = -E + x * r;
  const maxY = E - y * r;
  return [minX, maxY - r, minX + r, maxY];
}

export function toLon(x: number): number {
  return (x / E) * 180;
}
export function toLat(y: number): number {
  return (Math.atan(Math.sinh((y / E) * Math.PI)) * 180) / Math.PI;
}
export function fromLon(lon: number): number {
  return (lon / 180) * E;
}
export function fromLat(lat: number): number {
  return (Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360)) / Math.PI) * E;
}
