import { createEventTarget, type DataFrame, type DataFrameEvents, type OrderBy } from "hightable";
import type { TriageRow } from "./rows.js";

/**
 * The table's rows as a HighTable DataFrame: every column sorts on its own (one at a time). The rows are replaced as
 * scans come in (set), which tells the table to redraw. Empty values (fuel not read yet) sort last either way.
 */

const pacific = (t: number) =>
  new Date(t).toLocaleString("en-US", { timeZone: "America/Los_Angeles", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
const num = (v: number | null, d = 0) => (v === null || Number.isNaN(v) ? "" : v.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }));

export type Column = { name: string; value: (r: TriageRow) => number | null; text: (r: TriageRow) => string; title: string };

export const COLUMNS: Column[] = [
  { name: "GOES score", value: (r) => r.score, text: (r) => num(r.score * 100, 0), title: "Chance it is a recorded wildfire, from GOES alone (first scan: power, strength, pixels), in percent" },
  { name: "fuel under", value: (r) => r.fuelUnder, text: (r) => (r.fuelUnder === null ? (r.fuelState === "done" ? "no data" : r.fuelState) : num(r.fuelUnder * 100)), title: "Fuel under the hot GOES pixels: mean chance of fuel of every 10 m pixel (fuel head on the embeddings), in percent" },
  { name: "fuel around", value: (r) => r.fuelAround, text: (r) => (r.fuelAround === null ? (r.fuelState === "done" ? "no data" : r.fuelState) : num(r.fuelAround * 100)), title: "Fuel in the ring 1.5 GOES pixels out, same reading, in percent" },
  { name: "first seen", value: (r) => r.t0, text: (r) => pacific(r.t0), title: "First GOES-West scan with a fire pixel here (California time, PDT in this week)" },
  { name: "scans", value: (r) => r.scans, text: (r) => num(r.scans), title: "Scans with a fire pixel since first seen (one scan every 5 minutes)" },
  { name: "hours", value: (r) => (r.t1 - r.t0) / 3600e3, text: (r) => num((r.t1 - r.t0) / 3600e3, 1), title: "First to last scan with a fire pixel" },
  { name: "pixels", value: (r) => r.pixels, text: (r) => num(r.pixels), title: "Distinct GOES pixels that saw fire" },
  { name: "area km²", value: (r) => r.areaKm2, text: (r) => num(r.areaKm2, 0), title: "Ground area of those pixels" },
  { name: "peak MW", value: (r) => (Number.isNaN(r.peakMw) ? null : r.peakMw), text: (r) => num(r.peakMw), title: "Highest fire radiative power of any pixel" },
];
const BY_NAME = new Map(COLUMNS.map((c) => [c.name, c]));

export class TriageFrame implements DataFrame {
  rows: TriageRow[] = [];
  columnDescriptors = COLUMNS.map((c) => ({ name: c.name, sortable: true }));
  exclusiveSort = true;
  eventTarget = createEventTarget<DataFrameEvents>();
  private sorted = new Map<string, number[]>();

  get numRows() {
    return this.rows.length;
  }

  set(rows: TriageRow[]) {
    const grew = rows.length !== this.rows.length;
    this.rows = rows;
    this.sorted.clear();
    if (grew) this.eventTarget.dispatchEvent(new CustomEvent("numrowschange") as never);
    this.eventTarget.dispatchEvent(new CustomEvent("update") as never);
  }

  /** Data index of the row at a position in the given order. */
  index(row: number, orderBy?: OrderBy): number {
    const o = orderBy?.[0];
    if (!o) return row;
    const k = `${o.column}|${o.direction}`;
    let idx = this.sorted.get(k);
    if (!idx) {
      const c = BY_NAME.get(o.column);
      if (!c) throw new Error(`no column ${o.column}`);
      const sign = o.direction === "descending" ? -1 : 1;
      const v = this.rows.map(c.value);
      idx = this.rows.map((_, i) => i).sort((a, b) => {
        const va = v[a], vb = v[b];
        if (va === null) return vb === null ? a - b : 1;
        if (vb === null) return -1;
        return sign * (va - vb) || a - b;
      });
      this.sorted.set(k, idx);
    }
    return idx[row]!;
  }

  /** Position of a data index in the given order (for scrolling to a picked row). */
  position(dataIndex: number, orderBy?: OrderBy): number {
    if (!orderBy?.[0]) return dataIndex;
    this.index(0, orderBy);
    const o = orderBy[0];
    return this.sorted.get(`${o.column}|${o.direction}`)!.indexOf(dataIndex);
  }

  getRowNumber({ row, orderBy }: { row: number; orderBy?: OrderBy }) {
    if (row < 0 || row >= this.rows.length) return undefined;
    return { value: this.index(row, orderBy) };
  }

  getCell({ row, column, orderBy }: { row: number; column: string; orderBy?: OrderBy }) {
    if (row < 0 || row >= this.rows.length) return undefined;
    const r = this.rows[this.index(row, orderBy)]!;
    return { value: BY_NAME.get(column)!.text(r) };
  }
}
