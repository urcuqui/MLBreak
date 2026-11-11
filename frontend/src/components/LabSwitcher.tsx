import { Link } from "react-router-dom";

export type LabId = "ml01" | "ml02";
const KEY = "mlbreak.lastLab";
const LABS: { id: LabId; label: string; title: string }[] = [
  { id: "ml01", label: "ML01 · Evasion", title: "OWASP ML01:2023 Input Manipulation Attack" },
  { id: "ml02", label: "ML02 · Poisoning", title: "OWASP ML02:2023 Data Poisoning Attack" },
];

export function lastLab(): LabId {
  try {
    const v = localStorage.getItem(KEY);
    if (v === "ml01" || v === "ml02") return v;
  } catch { /* storage unavailable */ }
  return "ml01";
}

/** Segmented control to switch laboratory, keeping the current step. Remembers the choice for the sidebar. */
export function LabSwitcher({ current, tab }: { current: LabId; tab: string }) {
  try { localStorage.setItem(KEY, current); } catch { /* ignore */ }
  return (
    <div className="lab-switch" role="group" aria-label="Laboratory">
      <span className="muted small">Laboratory</span>
      <div className="segmented">
        {LABS.map((l) => (
          <Link key={l.id} to={`/lab/${l.id}/${tab}`} title={l.title}
            className={l.id === current ? "seg active" : "seg"} aria-current={l.id === current ? "page" : undefined}>
            {l.label}
          </Link>
        ))}
      </div>
    </div>
  );
}
