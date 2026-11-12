import { Link } from "react-router-dom";

export type LabId = "ml01" | "ml02";
/** A lab path prefix under /lab/: a lab, or an ML01 scenario inside it. */
export type LabPath = "ml01" | "ml01/spam" | "ml02";
const KEY = "mlbreak.lastLab";
const ML01_KEY = "mlbreak.lastMl01Scenario";
const PATHS: LabPath[] = ["ml01", "ml01/spam", "ml02"];
const LABS: { id: LabId; label: string; title: string }[] = [
  { id: "ml01", label: "ML01 · Evasion", title: "OWASP ML01:2023 Input Manipulation Attack" },
  { id: "ml02", label: "ML02 · Poisoning", title: "OWASP ML02:2023 Data Poisoning Attack" },
];

function read(key: string, fallback: LabPath): LabPath {
  try {
    const v = localStorage.getItem(key) as LabPath | null;
    if (v && PATHS.includes(v)) return v;
  } catch { /* storage unavailable */ }
  return fallback;
}

function remember(path: LabPath) {
  try {
    localStorage.setItem(KEY, path);
    if (path.startsWith("ml01")) localStorage.setItem(ML01_KEY, path);
  } catch { /* ignore */ }
}

/** Last-used lab path (for the sidebar's Attack Lab / Defense Lab links). */
export function lastLab(): LabPath {
  return read(KEY, "ml01");
}

/** Lab path prefix of a pathname, e.g. "/lab/ml01/spam/attack" → "ml01/spam". */
export function labPathOf(pathname: string): LabPath | null {
  const m = /^\/lab\/(ml01\/spam|ml01|ml02)(?:\/|$)/.exec(pathname);
  return (m?.[1] as LabPath) ?? null;
}

/** Segmented control to switch laboratory, keeping the current step. ML01 returns to its last scenario. */
export function LabSwitcher({ current, tab, path }: { current: LabId; tab: string; path?: LabPath }) {
  remember(path ?? current);
  const target = (id: LabId) => (id === "ml01" ? read(ML01_KEY, "ml01") : id);
  return (
    <div className="lab-switch" role="group" aria-label="Laboratory">
      <span className="muted small">Laboratory</span>
      <div className="segmented">
        {LABS.map((l) => (
          <Link key={l.id} to={`/lab/${l.id === current ? (path ?? current) : target(l.id)}/${tab}`} title={l.title}
            className={l.id === current ? "seg active" : "seg"} aria-current={l.id === current ? "page" : undefined}>
            {l.label}
          </Link>
        ))}
      </div>
    </div>
  );
}

const SCENARIOS: { path: LabPath; label: string; title: string }[] = [
  { path: "ml01", label: "2D Classifier Evasion", title: "Bounded numerical perturbations of a two-moons classifier" },
  { path: "ml01/spam", label: "Spam Filter Evasion", title: "Character-level adversarial text against a TF-IDF spam filter" },
];

/** Scenario selector inside the ML01 Input Manipulation lab. */
export function ScenarioSwitcher({ current, tab }: { current: LabPath; tab: string }) {
  return (
    <div className="lab-switch" role="group" aria-label="ML01 scenario">
      <span className="muted small">ML01 scenario</span>
      <div className="segmented">
        {SCENARIOS.map((s) => (
          <Link key={s.path} to={`/lab/${s.path}/${tab}`} title={s.title}
            className={s.path === current ? "seg active" : "seg"} aria-current={s.path === current ? "page" : undefined}>
            {s.label}
          </Link>
        ))}
      </div>
    </div>
  );
}
