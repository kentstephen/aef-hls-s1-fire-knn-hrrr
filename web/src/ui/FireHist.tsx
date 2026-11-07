import type { YearHist } from "../fuel/burnHist.js";

/**
 * The selected fire's fuel by year: one bar per embeddings year, left to right, its height the average chance of
 * fuel over the outline's pixels in view. A bar grows from the chunks read so far, drawn faint with the count of
 * chunks in until its year is complete.
 */

const W = 340;
const H = 150;
const LEFT = 44;
const TOP = 16;
const BOTTOM = 30;
const BAR = "#00e5ff";

type Props = {
  name: string;
  fireYear: number | null;
  acres: number | null;
  hists: YearHist[] | null;
  embYear: number;
  zoomOk: boolean;
  minZoom: number;
  /** The bars' value: vegetation volume in meters, or the chance of fuel. */
  volume?: boolean;
};

const fmt = (m: number) => (m >= 10 ? `${Math.round(m)} m` : m >= 1 ? `${m.toFixed(1)} m` : `${m.toFixed(2)} m`);

export function FireHist(p: Props) {
  const hists = p.hists ?? [];
  const total = hists[0]?.total ?? 0;
  const plotH = H - TOP - BOTTOM;
  const colW = (W - LEFT) / Math.max(1, hists.length);
  const barW = Math.min(30, colW - 10);
  // volume: the axis runs to the tallest bar (rounded up); P(fuel): 0 to 100 %
  const top = p.volume ? Math.max(0.5, ...hists.map((h) => h.mean ?? 0)) * 1.1 : 1;
  const y = (v: number) => TOP + plotH * (1 - v / top);
  const what = p.volume ? "Average vegetation volume (cover x height)" : "Average chance of fuel";
  const label = (v: number) => (p.volume ? fmt(v) : `${Math.round(v * 100)}%`);
  return (
    <div className="fire-hist">
      <div className="fh-title">
        {p.name} fire · {p.fireYear ?? "year unknown"}
        {p.acres != null ? ` · ${Math.round(p.acres).toLocaleString()} acres` : ""}
      </div>
      <p className="muted small">
        {what} inside the outline, in view{total ? ` (${total} chunk${total === 1 ? "" : "s"} of 2.5 km)` : ""}.
        Esc or a click outside a fire clears it.
      </p>
      {!p.zoomOk ? (
        <p className="muted small">Zoom in past {p.minZoom} to read it.</p>
      ) : total === 0 && p.hists ? (
        <p className="muted small">None of the outline is in view.</p>
      ) : (
        <svg width={W} height={H} role="img" aria-label={`${what} by year`}>
          {[0, 0.5, 1].map((f) => f * top).map((v) => (
            <g key={v}>
              <line x1={LEFT} x2={W} y1={y(v)} y2={y(v)} stroke="rgba(255,255,255,0.12)" />
              <text x={LEFT - 4} y={y(v) + 3} textAnchor="end" className="fh-axis">{label(v)}</text>
            </g>
          ))}
          {hists.map((h, i) => {
            const cx = LEFT + colW * (i + 0.5);
            const on = h.year === p.embYear;
            const complete = h.done === h.total;
            return (
              <g key={h.year}>
                {h.mean != null ? (
                  <>
                    <rect x={cx - barW / 2} width={barW} y={y(h.mean)} height={(plotH * h.mean) / top} rx={2} fill={BAR} opacity={!complete ? 0.25 : on ? 1 : 0.6}>
                      <title>{`${h.year}: ${what.toLowerCase()} ${p.volume ? fmt(h.mean) : `${(h.mean * 100).toFixed(1)}%`} over ${h.n.toLocaleString()} pixels${complete ? "" : ` (${h.done} of ${h.total} chunks so far)`}`}</title>
                    </rect>
                    <text x={cx} y={y(h.mean) - 4} textAnchor="middle" className={complete ? "fh-val" : "fh-axis"}>
                      {complete ? label(h.mean) : `${h.done}/${h.total}`}
                    </text>
                  </>
                ) : (
                  <text x={cx} y={y(0) - 4} textAnchor="middle" className="fh-axis">{h.done}/{h.total}</text>
                )}
                <text x={cx} y={H - BOTTOM + 14} textAnchor="middle" className={on ? "fh-year on" : "fh-year"}>{h.year}</text>
                {p.fireYear === h.year ? <text x={cx} y={H - BOTTOM + 26} textAnchor="middle" className="fh-tag">fire</text> : null}
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}
