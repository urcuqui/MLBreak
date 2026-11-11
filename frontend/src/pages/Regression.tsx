import { useEffect, useState } from "react";
import { api } from "../api";
import { Badge, ErrorBox, ExpLink, Panel } from "../components/ui";
import type { PoisoningCase, RegressionReport, SecurityCase } from "../types";

const KIND_HELP: Record<string, string> = {
  security: "the defense must hold",
  utility: "the defense must not destroy clean performance",
  attack_canary: "the attack (or a known limitation) must still be observable",
};

export function Regression() {
  const [cases, setCases] = useState<(SecurityCase | PoisoningCase)[]>([]);
  const [report, setReport] = useState<RegressionReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.cases().then(setCases).catch((e) => setError(String(e.message ?? e)));
    api.latestRegression().then(setReport).catch(() => {});
  }, []);
  const run = async () => {
    setBusy(true); setError(null);
    try { setReport(await api.runRegression()); } catch (e) { setError(String((e as Error).message ?? e)); } finally { setBusy(false); }
  };
  const result = (id: string) => report?.cases.find((c) => c.id === id);
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <div className="eyebrow">Continuous verification</div><h1>Security Regression</h1>
          <p className="muted">Each case declares invariants checked against freshly executed experiments. The same runner backs
            <code> mlbreak verify</code> and the pytest suite.</p>
        </div>
        <button className="btn primary" onClick={run} disabled={busy}>{busy ? "Running…" : "Run regression suite"}</button>
      </div>
      <ErrorBox error={error} />
      {report && (
        <div className={report.ok ? "callout good" : "callout bad"}>
          {report.ok ? "✓ PASS" : "✕ FAIL"} — {report.passed}/{report.total} cases passed · {report.generated_at.slice(0, 19).replace("T", " ")} UTC
        </div>
      )}
      {cases.map((c) => {
        const r = result(c.id);
        return (
          <Panel key={c.id} title={`${c.id} — ${c.title}`}
            subtitle={"poison" in c
              ? <>{c.owasp_id} · label flipping {c.poison.strategy}{c.poison.strategy === "targeted"
                  ? ` (class ${c.poison.target_class}, ${c.poison.selection})` : ""} {Math.round(c.poison.fraction * 100)}% ·
                  defense {c.defense.kind} (k={c.defense.k}, {c.defense.action}) · defended evaluation: <strong>replay</strong></>
              : <>{c.owasp_id} · {c.attack.attack} {c.attack.norm} ε={c.attack.epsilon} · defense {c.defense.kind} ·
                  defended evaluation: <strong>{c.defended_evaluation}</strong></>}
            actions={r ? (r.passed ? <Badge tone="good">pass</Badge> : <Badge tone="critical">fail</Badge>) : <Badge tone="neutral">not run</Badge>}>
            <p className="muted small">{c.rationale}</p>
            <div className="table-wrap"><table className="table">
              <thead><tr><th>Invariant</th><th>Kind</th><th>Posture</th><th>Check</th><th>Observed</th><th>Result</th></tr></thead>
              <tbody>{c.invariants.map((inv) => {
                const o = r?.invariants.find((i) => i.id === inv.id);
                return (
                  <tr key={inv.id}>
                    <td title={inv.description}>{inv.id}<div className="muted small">{inv.description}</div></td>
                    <td title={KIND_HELP[inv.kind]}>{inv.kind}</td><td>{inv.posture}</td>
                    <td className="mono">{inv.metric} {inv.op} {inv.value}</td>
                    <td className="num">{o?.observed == null ? "—" : o.observed.toFixed(4)}</td>
                    <td>{o ? (o.passed ? <Badge tone="good">pass</Badge> : <Badge tone="critical">fail</Badge>) : "—"}
                      {o?.experiment_id && <div className="small"><ExpLink id={o.experiment_id} /></div>}</td>
                  </tr>);
              })}</tbody>
            </table></div>
          </Panel>
        );
      })}
    </div>
  );
}
