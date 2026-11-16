import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { useState } from "react";
import { evaluatorToken, setEvaluatorToken } from "./api";
import { labPathOf, lastLab } from "./components/LabSwitcher";
import { Dashboard } from "./pages/Dashboard";
import { Evidence } from "./pages/Evidence";
import { ExperimentDetail } from "./pages/ExperimentDetail";
import { Experiments } from "./pages/Experiments";
import { InversionWorkspace } from "./pages/InversionWorkspace";
import { MembershipWorkspace } from "./pages/MembershipWorkspace";
import { Owasp } from "./pages/Owasp";
import { PoisoningWorkspace } from "./pages/PoisoningWorkspace";
import { Regression } from "./pages/Regression";
import { SpamWorkspace } from "./pages/SpamWorkspace";
import { Workspace } from "./pages/Workspace";

function nav(lab: string) {
  return [
    { to: "/", label: "Dashboard", end: true },
    { to: "/owasp", label: "OWASP ML Top 10" },
    { to: "/experiments", label: "Experiments" },
    { to: `/lab/${lab}/attack`, label: "Attack Lab" },
    { to: `/lab/${lab}/defend`, label: "Defense Lab" },
    { to: "/evidence", label: "Evidence & Reports" },
    { to: "/regression", label: "Security Regression" },
  ];
}

/** Only needed when the server runs in evaluator-token mode (MLBREAK_EVALUATOR_TOKEN); empty by default. */
function EvaluatorAccess() {
  const [value, setValue] = useState(evaluatorToken());
  const [saved, setSaved] = useState(!!evaluatorToken());
  return (
    <details className="evaluator-access">
      <summary>Evaluator access{saved ? " · token set" : ""}</summary>
      <form onSubmit={(e) => { e.preventDefault(); setEvaluatorToken(value.trim()); setSaved(!!value.trim()); window.location.reload(); }}>
        <label className="small">Evaluator token (this tab only)
          <input type="password" autoComplete="off" value={value} onChange={(e) => setValue(e.target.value)} />
        </label>
        <button className="btn small" type="submit">Save</button>
      </form>
    </details>
  );
}

export function App() {
  // The lab in the URL wins (the switcher stores it during the same render); otherwise the last-used lab.
  const { pathname } = useLocation();
  const NAV = nav(labPathOf(pathname) ?? lastLab());
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">◆</span> MLBreak<span className="brand-sub">adversarial ML lab</span></div>
        <nav aria-label="Main">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => (isActive ? "nav active" : "nav")}>{n.label}</NavLink>
          ))}
        </nav>
        <div className="sidebar-foot small muted">
          <EvaluatorAccess />
          Local-first · no uploads<br />Educational use only
        </div>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/owasp" element={<Owasp />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/experiments/:id" element={<ExperimentDetail />} />
          <Route path="/lab/ml01" element={<Navigate to="/lab/ml01/baseline" replace />} />
          <Route path="/lab/ml01/spam" element={<Navigate to="/lab/ml01/spam/baseline" replace />} />
          <Route path="/lab/ml01/spam/:tab" element={<SpamWorkspace />} />
          <Route path="/lab/ml01/:tab" element={<Workspace />} />
          <Route path="/lab/ml02" element={<Navigate to="/lab/ml02/baseline" replace />} />
          <Route path="/lab/ml02/:tab" element={<PoisoningWorkspace />} />
          <Route path="/lab/ml03" element={<Navigate to="/lab/ml03/baseline" replace />} />
          <Route path="/lab/ml03/:tab" element={<InversionWorkspace />} />
          <Route path="/lab/ml04" element={<Navigate to="/lab/ml04/baseline" replace />} />
          <Route path="/lab/ml04/:tab" element={<MembershipWorkspace />} />
          <Route path="/evidence" element={<Evidence />} />
          <Route path="/regression" element={<Regression />} />
          <Route path="*" element={<div className="page"><h1>Not found</h1></div>} />
        </Routes>
      </main>
    </div>
  );
}
