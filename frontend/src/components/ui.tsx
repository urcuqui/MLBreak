import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import type { Ratio } from "../types";

export const pct = (v: number | null | undefined, digits = 1) =>
  v == null ? "—" : `${(100 * v).toFixed(digits)}%`;

export function RatioView({ r, emphasize }: { r?: Ratio; emphasize?: boolean }) {
  if (!r) return <span className="muted">—</span>;
  return (
    <span title={r.definition}>
      <span className={emphasize ? "num strong" : "num"}>{r.value == null ? "undefined" : pct(r.value)}</span>{" "}
      <span className="muted small">({r.numerator}/{r.denominator})</span>
    </span>
  );
}

export function Panel({ title, children, actions, id, subtitle }:
  { title: string; children: ReactNode; actions?: ReactNode; id?: string; subtitle?: ReactNode }) {
  return (
    <section className="panel" id={id}>
      <header className="panel-head">
        <div>
          <h2>{title}</h2>
          {subtitle && <div className="muted small">{subtitle}</div>}
        </div>
        {actions && <div className="row gap">{actions}</div>}
      </header>
      <div className="panel-body">{children}</div>
    </section>
  );
}

export function Tile({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <div className="tile" title={hint}>
      <div className="tile-label">{label}</div>
      <div className="tile-value">{children}</div>
    </div>
  );
}

type Tone = "good" | "critical" | "warning" | "info" | "neutral";
const ICON: Record<Tone, string> = { good: "✓", critical: "✕", warning: "!", info: "i", neutral: "•" };

export function Badge({ tone, children }: { tone: Tone; children: ReactNode }) {
  return <span className={`badge badge-${tone}`}><span aria-hidden>{ICON[tone]}</span> {children}</span>;
}

export function StatusBadge({ status }: { status: string }) {
  return status === "completed" ? <Badge tone="good">completed</Badge> : <Badge tone="critical">{status}</Badge>;
}

export function EvalBadge({ evaluation }: { evaluation: string }) {
  const map: Record<string, [Tone, string]> = {
    clean: ["neutral", "baseline"],
    attack: ["critical", "attack"],
    adaptive_attack: ["warning", "adaptive attack"],
    replay: ["info", "defense replay"],
  };
  const [tone, label] = map[evaluation] ?? ["neutral", evaluation];
  return <Badge tone={tone}>{label}</Badge>;
}

export function ExpLink({ id }: { id: string | null | undefined }) {
  if (!id) return <span className="muted">—</span>;
  return <Link className="mono" to={`/experiments/${id}`}>{id}</Link>;
}

export function ErrorBox({ error }: { error: string | null }) {
  if (!error) return null;
  return <div className="error" role="alert">✕ {error}</div>;
}

export function VulnerableBanner() {
  return (
    <div className="vuln-banner" role="note">
      <strong>⚠ Intentionally vulnerable scenario.</strong> The undefended model in this lab is deliberately
      left exposed to test-time evasion for education and defensive research. Local use only.
    </div>
  );
}
