import { Fragment, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { Badge, ErrorBox, EvalBadge, ExpLink, Panel, StatusBadge, Tile, VulnerableBanner, pct } from "../components/ui";
import type { Dashboard as D } from "../types";

export function Dashboard() {
  const [d, setD] = useState<D | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api.dashboard().then(setD).catch((e) => setError(String(e.message ?? e))); }, []);
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <div className="eyebrow">Adversarial ML security lab</div>
          <h1>Dashboard</h1>
          <p className="muted">Counts below come from stored evidence on this machine. Nothing here is sample data.</p>
        </div>
        <Link className="btn primary" to="/lab/ml01/baseline">Open ML01 lab →</Link>
      </div>
      <VulnerableBanner />
      <ErrorBox error={error} />
      {d && (
        <>
          <div className="tiles">
            <Tile label="Stored experiments"><span className="num strong">{d.experiments_total}</span></Tile>
            <Tile label="Failed experiments"><span className="num">{d.experiments_failed}</span></Tile>
            <Tile label="OWASP ML Top 10 labs implemented">
              <span className="num strong">{d.owasp_implemented}</span><span className="muted"> / {d.owasp_total}</span>
            </Tile>
            <Tile label="Security regression">
              {d.regression_latest
                ? (d.regression_latest.ok
                  ? <Badge tone="good">{d.regression_latest.passed}/{d.regression_latest.total} passing</Badge>
                  : <Badge tone="critical">{d.regression_latest.failed} failing</Badge>)
                : <span className="muted">{d.regression_cases} cases · not run yet</span>}
            </Tile>
          </div>
          <div className="grid-2">
            <Panel title="Recent experiments" actions={<Link to="/experiments">All experiments</Link>}>
              {d.recent.length === 0 ? (
                <p className="muted">No experiments yet. Open the <Link to="/lab/ml01/baseline">ML01 lab</Link> and run a baseline.</p>
              ) : (
                <div className="table-wrap"><table className="table">
                  <thead><tr><th>Experiment</th><th>Type</th><th>Status</th><th>Defense</th><th>Attack success</th></tr></thead>
                  <tbody>{d.recent.map((e) => (
                    <tr key={e.experiment_id}>
                      <td><ExpLink id={e.experiment_id} /></td><td><EvalBadge evaluation={e.evaluation} /></td>
                      <td><StatusBadge status={e.status} /></td><td>{e.defense ?? <span className="muted">none</span>}</td>
                      <td>{e.attack_success_rate == null ? <span className="muted">—</span>
                        : <span className="num">{pct(e.attack_success_rate)} <span className="muted small">({e.attack_success})</span></span>}</td>
                    </tr>))}</tbody>
                </table></div>
              )}
            </Panel>
            <Panel title="Workflow">
              <ol className="steps">
                <li><Link to="/lab/ml01/baseline"><strong>Baseline</strong></Link> — evaluate the unmodified model on clean inputs.</li>
                <li><Link to="/lab/ml01/attack"><strong>Attack</strong></Link> — craft bounded perturbations (white-box or black-box).</li>
                <li><Link to="/lab/ml01/defend"><strong>Defend</strong></Link> — replay the perturbations against a control, then run an adaptive attack.</li>
                <li><Link to="/lab/ml01/evidence"><strong>Evidence</strong></Link> — export JSON/Markdown and verify exact replay.</li>
                <li><Link to="/regression"><strong>Regression</strong></Link> — check declared defensive invariants.</li>
              </ol>
              {Object.keys(d.experiments_by_evaluation).length > 0 && (
                <dl className="kv mt">{Object.entries(d.experiments_by_evaluation).map(([k, v]) => (
                  <Fragment key={k}><dt>{k.replace("_", " ")}</dt><dd className="num">{v}</dd></Fragment>))}</dl>
              )}
            </Panel>
          </div>
        </>
      )}
    </div>
  );
}
