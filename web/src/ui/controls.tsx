/** Plain controls for the glass panel: foldable section, help toggle, slider, check, segmented buttons. No UI library. */
import { useEffect, useState, type ReactNode } from "react";

const FOLD_KEY = "panel.fold.v1";

function readFolds(): Record<string, boolean> {
  try {
    return JSON.parse(localStorage.getItem(FOLD_KEY) || "{}") as Record<string, boolean>;
  } catch {
    return {};
  }
}

/** A section with a header button; open or closed state is kept per key in localStorage. Several can be open at once. */
export function Fold(props: { k: string; label: string; tag?: ReactNode; defaultOpen?: boolean; children: ReactNode }) {
  const [open, setOpen] = useState<boolean>(() => {
    const saved = readFolds()[props.k];
    return saved === undefined ? !!props.defaultOpen : saved;
  });
  useEffect(() => {
    try {
      const all = readFolds();
      all[props.k] = open;
      localStorage.setItem(FOLD_KEY, JSON.stringify(all));
    } catch {
      /* storage blocked: the layout just does not persist */
    }
  }, [props.k, open]);
  return (
    <div className={`fold${open ? " open" : ""}`}>
      <button type="button" className="fold-hd" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <span className="gl">▸</span>
        {props.label}
        {props.tag !== undefined ? <span className="tag">{props.tag}</span> : null}
      </button>
      {open ? <div className="fold-bd">{props.children}</div> : null}
    </div>
  );
}

/** A "?" that shows the explanation under the row it sits in. The text is always one click away, never hidden in a hover. */
export function Help(props: { children: ReactNode; label?: string }) {
  const [on, setOn] = useState(false);
  return (
    <>
      <button type="button" className={`help-btn${on ? " on" : ""}`} aria-label={props.label ?? "More information"} aria-expanded={on} onClick={() => setOn((v) => !v)}>
        ?
      </button>
      {on ? <div className="help-txt" style={{ flexBasis: "100%" }}>{props.children}</div> : null}
    </>
  );
}

export function Field(props: { label: ReactNode; children: ReactNode }) {
  return (
    <div className="field">
      <div className="lab">{props.label}</div>
      {props.children}
    </div>
  );
}

/** Native range input. Double-click resets. */
export function Slider(props: {
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  label: string;
  onReset?: () => void;
  disabled?: boolean;
}) {
  return (
    <input
      type="range"
      min={props.min}
      max={props.max}
      step={props.step}
      value={props.value}
      aria-label={props.label}
      disabled={props.disabled}
      onChange={(e) => props.onChange(Number(e.target.value))}
      onDoubleClick={props.onReset}
    />
  );
}

export function Check(props: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <label className="check">
      <input type="checkbox" checked={props.checked} onChange={(e) => props.onChange(e.target.checked)} />
      {props.label}
    </label>
  );
}

export function Seg<T extends string>(props: { value: T | null; options: { id: T; label: string }[]; onChange: (id: T) => void }) {
  return (
    <div className="seg" role="group">
      {props.options.map((o) => (
        <button key={o.id} type="button" className={props.value === o.id ? "on" : ""} onClick={() => props.onChange(o.id)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function A(props: { href: string; children: ReactNode }) {
  return (
    <a href={props.href} target="_blank" rel="noopener noreferrer">
      {props.children}
    </a>
  );
}

/** One row of the colormap sprite PNG, stretched to a strip. */
export function Ramp(props: { spriteUrl: string; rowCount: number; rowIndex: number; reversed?: boolean; label: string }) {
  const h = 12;
  return (
    <div
      className="ramp"
      role="img"
      aria-label={`Colormap: ${props.label}`}
      style={{
        backgroundImage: `url(${props.spriteUrl})`,
        backgroundSize: `100% ${props.rowCount * h}px`,
        backgroundPosition: `0 -${props.rowIndex * h}px`,
        transform: props.reversed ? "scaleX(-1)" : undefined,
      }}
    />
  );
}
