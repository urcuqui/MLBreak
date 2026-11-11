import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { lastLab } from "./components/LabSwitcher";
import { Dashboard } from "./pages/Dashboard";
import { Evidence } from "./pages/Evidence";
import { ExperimentDetail } from "./pages/ExperimentDetail";
import { Experiments } from "./pages/Experiments";
import { Owasp } from "./pages/Owasp";
import { PoisoningWorkspace } from "./pages/PoisoningWorkspace";
import { Regression } from "./pages/Regression";
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

export function App() {
  // The lab in the URL wins (the switcher stores it during the same render); otherwise the last-used lab.
  const { pathname } = useLocation();
  const NAV = nav(/^\/lab\/(ml0[12])\b/.exec(pathname)?.[1] ?? lastLab());
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">◆</span> MLBreak<span className="brand-sub">adversarial ML lab</span></div>
        <nav aria-label="Main">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => (isActive ? "nav active" : "nav")}>{n.label}</NavLink>
          ))}
        </nav>
        <div className="sidebar-foot small muted">Local-first · no credentials · no uploads<br />Educational use only</div>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/owasp" element={<Owasp />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/experiments/:id" element={<ExperimentDetail />} />
          <Route path="/lab/ml01" element={<Navigate to="/lab/ml01/baseline" replace />} />
          <Route path="/lab/ml01/:tab" element={<Workspace />} />
          <Route path="/lab/ml02" element={<Navigate to="/lab/ml02/baseline" replace />} />
          <Route path="/lab/ml02/:tab" element={<PoisoningWorkspace />} />
          <Route path="/evidence" element={<Evidence />} />
          <Route path="/regression" element={<Regression />} />
          <Route path="*" element={<div className="page"><h1>Not found</h1></div>} />
        </Routes>
      </main>
    </div>
  );
}
