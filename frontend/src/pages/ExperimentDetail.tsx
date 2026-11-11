import { Fragment, useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api, evidenceLinks } from "../api";
import { BoundaryPlot } from "../components/BoundaryPlot";
import { Badge, ErrorBox, EvalBadge, ExpLink, Panel, RatioView, StatusBadge, Tile } from "../components/ui";
import type { BoundaryResponse, DatasetResponse, ExperimentRecord, ReplayResult } from "../types";

export function ExperimentDetail() {
  const { id = "" } = useParams();
  const [rec, setRec] = useState<ExperimentRecord | null>(null);
  const [dataset, setDataset] = useState<DatasetResponse | null>(null);
  const [boundary, setBoundary] = useState<BoundaryResponse | null>(null);
  const [replay, setReplay] = useState<ReplayResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setRec(null); setReplay(null); setError(null);
    api.experiment(id).then((r) => {
      setRec(r);
      const ctx = { model_id: r.model.model_id, seed: r.seed };
      api.dataset(ctx).then(setDataset).catch(() => {});
      api.boundary(ctx, r.defense_config).then(setBoundary).catch(() => {});
    }).catch((e) => setError(String(e.message ?? e)));
  }, [id]);

  const doReplay = async () => {
    setBusy(true); setError(null);
    try { setReplay(await api.replay(id)); } catch (e) { setError(String((e as Error).message ?? e)); } finally { setBusy(false); }
  };

  if (!rec) return <div className="page"><ErrorBox error={error} />{!error && <p className="muted">Loading…</p>}</div>;
  const m = rec.metrics, links = evidenceLinks(rec.experiment_id);
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <div className="eyebrow">{rec.owasp.id} · {rec.owasp.name} · {rec.scenario_id}</div>
          <h1 className="mono">{rec.experiment_id}</h1>
          <div className="row gap wrap">
            <EvalBadge evaluation={rec.evaluation} /><StatusBadge status={rec.status} />
            {rec.intentionally_vulnerable ? <Badge tone="warning">intentionally vulnerable target</Badge> : <Badge tone="info">defended target</Badge>}
            {rec.parent_experiment_id && <span className="small">parent <ExpLink id={rec.parent_experiment_id} /></span>}
            {rec.replay_of && <span className="small">replay of <ExpLink id={rec.replay_of} /></span>}
          </div>
        </div>
        <div className="row gap wrap">
          <a className="btn" href={links.json} download>JSON</a>
          <a className="btn" href={links.md} download>Markdown</a>
          <a className="btn" href={links.csv} download>CSV</a>
          <button className="btn primary" disabled={busy || rec.status !== "completed"} onClick={doReplay}>{busy ? "Replaying…" : "Replay experiment"}</button>
        </div>
      </div>
      <ErrorBox error={error ?? (rec.error ? `Experiment failed: ${rec.error}` : null)} />
      {replay && (
        <div className={replay.identical ? "callout good" : "callout bad"}>
          {replay.identical ? "✓ Replay produced identical results" : "✕ Replay diverged"} — new experiment <ExpLink id={replay.replay_experiment_id} />.
          <ul className="small">{replay.checks.map((c) => <li key={c.check}>{c.match ? "✓" : "✕"} {c.check}</li>)}</ul>
        </div>
      )}
      <div className="tiles">
        <Tile label="Clean accuracy"><RatioView r={m.clean_accuracy} emphasize /></Tile>
        {m.coverage && <Tile label="Coverage"><RatioView r={m.coverage} /></Tile>}
        {m.adversarial_accuracy && <Tile label="Adversarial accuracy"><RatioView r={m.adversarial_accuracy} /></Tile>}
        {m.attack_success_rate && <Tile label="Attack success rate" hint={m.attack_success_rate.definition}><RatioView r={m.attack_success_rate} emphasize /></Tile>}
        {m.perturbation?.successful.n ? <Tile label="Mean ‖δ‖∞ (successful)"><span className="num">{m.perturbation.successful.linf_mean!.toFixed(4)}</span></Tile> : null}
      </div>
      <div className="grid-2">
        <Panel title="Decision boundary of the evaluated system" subtitle="✕ = evaded, ○ = did not evade, line = perturbation">
          <BoundaryPlot boundary={boundary} dataset={dataset} samples={rec.samples} showTrain={false} title={`exp-${rec.experiment_id}`} />
        </Panel>
        <div className="stack">
          <Panel title="Configuration">
            <dl className="kv">
              <dt>Model</dt><dd>{rec.model.model_id} · {rec.model.estimator}</dd>
              <dt>Model artifact</dt><dd className="mono hash">{rec.model.artifact_sha256}</dd>
              <dt>Target system artifact</dt><dd className="mono hash">{rec.target_system.model_artifact_sha256}</dd>
              <dt>Decision policy</dt><dd>{rec.target_system.decision_policy}</dd>
              <dt>Dataset split</dt><dd className="mono">{rec.dataset.split_id}</dd>
              <dt>Seed</dt><dd>{rec.seed}</dd>
              <dt>Attack</dt><dd className="mono small">{rec.attack_config ? JSON.stringify(rec.attack_config) : "—"}</dd>
              <dt>Defense</dt><dd className="mono small">{rec.defense_config ? JSON.stringify(rec.defense_config) : "none"}</dd>
              <dt>Duration</dt><dd>{rec.duration_ms.toFixed(0)} ms</dd>
            </dl>
          </Panel>
          {rec.attacker_capabilities && (
            <Panel title="Attacker capabilities">
              <dl className="kv">{Object.entries(rec.attacker_capabilities).map(([k, v]) => (
                <Fragment key={k}><dt>{k.replace(/_/g, " ")}</dt><dd>{String(v)}</dd></Fragment>))}</dl>
            </Panel>
          )}
          {m.comparison && (
            <Panel title="Undefended vs defended">
              <table className="table"><thead><tr><th>Metric</th><th>Undefended</th><th>Defended</th></tr></thead>
                <tbody>{Object.entries(m.comparison).map(([k, v]) => (
                  <tr key={k}><td>{k.replace(/_/g, " ")}</td><td><RatioView r={v.undefended} /></td><td><RatioView r={v.defended} /></td></tr>))}</tbody>
              </table>
              {m.notes && <ul className="notes">{m.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
            </Panel>
          )}
        </div>
      </div>
      <Panel title={`Per-sample results (${rec.samples.length})`}>
        <div className="table-wrap scroll-y"><table className="table small">
          <thead><tr><th>test #</th><th>y</th><th>x original</th><th>pred</th><th>conf</th><th>x adversarial</th><th>pred adv</th><th>conf adv</th><th>‖δ‖∞</th><th>‖δ‖₂</th><th>queries</th><th>outcome</th></tr></thead>
          <tbody>{rec.samples.map((s) => (
            <tr key={s.test_index}>
              <td className="num">{s.test_index}</td><td>{s.y_true}</td>
              <td className="mono">{s.x_original.map((v) => v.toFixed(3)).join(", ")}</td>
              <td>{s.pred_original}</td><td className="num">{s.confidence_original.toFixed(3)}</td>
              <td className="mono">{s.x_adversarial ? s.x_adversarial.map((v) => v.toFixed(3)).join(", ") : "—"}</td>
              <td>{s.pred_adversarial ?? "—"}</td><td className="num">{s.confidence_adversarial?.toFixed(3) ?? "—"}</td>
              <td className="num">{s.perturbation_linf?.toFixed(4) ?? "—"}</td><td className="num">{s.perturbation_l2?.toFixed(4) ?? "—"}</td>
              <td className="num">{s.queries ?? "—"}</td>
              <td>{s.attack_success == null ? (s.clean_correct ? "" : <span className="muted">clean miss</span>)
                : s.attack_success ? <Badge tone="critical">evaded</Badge> : <Badge tone="good">held</Badge>}</td>
            </tr>))}</tbody>
        </table></div>
        <p className="muted small">pred −1 = the system abstained (confidence rejection).</p>
      </Panel>
    </div>
  );
}
