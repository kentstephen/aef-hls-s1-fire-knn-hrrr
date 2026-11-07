import { BINS, type ViewHist as Hist } from "../fuel/viewHist.js";

/**
 * The view's histogram over the ramp strip: one column per stretch of the ramp, in that stretch's color, as tall
 * as the share of the view's drawn pixels drawn in it. The strip under it and the marks under the strip are the
 * same ramp, so each column stands over its own color.
 */

const H = 84;
const MARKS = [0, 25, 50, 75, 100];

type Props = {
  hist: Hist | null;
  /** The ramp as a CSS gradient. */
  gradient: string;
  /** A scaled value (0..1, the fuel byte's fraction) as text: meters of volume or a chance. */
  fmt: (v: number) => string;
  /** What the values are, for the hover text. */
  what: string;
  /** The embeddings year on screen. */
  year: number | string;
};

const pct = (v: number) => (v >= 0.1 ? `${Math.round(v * 100)}%` : `${(v * 100).toFixed(1)}%`);

export function ViewHist(p: Props) {
  const h = p.hist;
  const top = h ? Math.max(...h.bins.map((b) => b.share)) : 0;
  return (
    <div className="view-hist">
      {h ? (
        <>
          <div className="vh-head muted small">
            <span className="vh-title">{p.what[0]!.toUpperCase() + p.what.slice(1)} in view, {p.year}</span>
            <span>tallest {pct(top)}</span>
          </div>
          <svg viewBox={`0 0 ${BINS} ${H}`} preserveAspectRatio="none" width="100%" height={H} role="img" aria-label={`Histogram of the ${p.what} drawn in view`}>
            {h.bins.map((b, k) => {
              const bh = top > 0 ? (H * b.share) / top : 0;
              const range = b.lo === null ? "nothing in view" : b.lo === b.hi ? p.fmt(b.lo) : `${p.fmt(b.lo)} to ${p.fmt(b.hi!)}`;
              return (
                <g key={k}>
                  <rect x={k} y={H - bh} width={1} height={bh} fill={b.color} shapeRendering="crispEdges" />
                  <rect x={k} y={0} width={1} height={H} fill="transparent">
                    <title>{`${p.what} ${range}: ${pct(b.share)} of the drawn pixels`}</title>
                  </rect>
                </g>
              );
            })}
          </svg>
        </>
      ) : null}
      <div className="ramp" role="img" aria-label="ramp" style={{ backgroundImage: p.gradient }} />
      {h ? (
        <>
          <div className="vh-marks">
            {MARKS.map((m, i) => (
              <span key={m} style={{ left: `${m}%` }} className={i === 0 ? "first" : i === MARKS.length - 1 ? "last" : ""}>
                {h.ticks[i] != null ? p.fmt(h.ticks[i]!) : ""}
              </span>
            ))}
          </div>
          <p className="muted small">
            {h.drawn.toLocaleString()} of {h.cells.toLocaleString()} points on an even grid over the view
            {h.hidden > 0 ? `; ${pct(h.hidden / (h.drawn + h.hidden))} of the ground with an embedding not drawn (hide line or mask)` : ""}
            {h.unread > 0 ? `; ${pct(h.unread / h.cells)} not read yet` : ""}.
          </p>
        </>
      ) : null}
    </div>
  );
}
