import { Fragment } from "react";
import { IntegrityPanel } from "../components/lab";
import { evidenceLinks } from "../api";
import { PoisonPlot } from "../components/PoisonPlot";
import { Badge, ErrorBox, EvalBadge, ExpLink, Panel, StatusBadge } from "../components/ui";
import type { PoisoningRecord } from "../types";
import { AttackPlots, DefendPlots, DefenseResults, MetricsCompare, useExperimentBoundary } from "./PoisoningWorkspace";

/** Detail view of a stored ML02 poisoning experiment. Every chart is rebuilt from this record. */
export function PoisoningDetail({ rec }: { rec: PoisoningRecord }) {
  const links = evidenceLinks(rec.experiment_id);
  const domain = rec.dataset.domain as number[][] | undefined;
  const ok = rec.status === "completed";
  const clean = useExperimentBoundary(rec.experiment_id, "clean", ok && rec.mode === "baseline");
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <div className="eyebrow">{rec.owasp.id} · {rec.owasp.name} · {rec.scenario_id}</div>
          <h1 className="mono">{rec.experiment_id}</h1>
          <div className="row gap wrap">
            <EvalBadge evaluation={rec.evaluation} /><StatusBadge status={rec.status} />
            {rec.intentionally_vulnerable ? <Badge tone="warning">intentionally vulnerable pipeline</Badge>
              : <Badge tone="info">defended pipeline</Badge>}
            {rec.parent_experiment_id && <span className="small">parent <ExpLink id={rec.parent_experiment_id} /></span>}
            {rec.replay_of && <span className="small">replay of <ExpLink id={rec.replay_of} /></span>}
          </div>
        </div>
        <div className="row gap wrap">
          <a className="btn" href={links.json} download>JSON</a>
          <a className="btn" href={links.md} download>Markdown</a>
          <a className="btn" href={links.csv} download>CSV</a>
        </div>
      </div>
      <ErrorBox error={rec.error ? `Experiment failed: ${rec.error}` : null} />
      <section className="panel" aria-label="Integrity and replay">
        <header className="panel-head"><h2>Integrity &amp; replay</h2></header>
        <div className="panel-body"><IntegrityPanel id={rec.experiment_id} completed={ok} /></div>
      </section>
      {ok && rec.mode === "baseline" && (
        <Panel title="Original dataset & clean decision boundary">
          <PoisonPlot boundary={clean.boundary} rows={rec.training_samples} labelMode="true" highlight="none"
            domain={domain} title={`clean-${rec.experiment_id}`} />
        </Panel>
      )}
      {ok && rec.mode !== "baseline" && <AttackPlots rec={rec} domain={domain} />}
      {ok && rec.mode === "defend" && <DefendPlots rec={rec} domain={domain} />}
      <div className="grid-2">
        <Panel title="Model performance">{ok ? <MetricsCompare rec={rec} /> : <p className="muted">No results.</p>}</Panel>
        <div className="stack">
          {ok && rec.mode === "defend" && <Panel title="Defense results"><DefenseResults rec={rec} /></Panel>}
          {rec.attacker_capabilities && (
            <Panel title="Attacker capabilities">
              <dl className="kv">{Object.entries(rec.attacker_capabilities).map(([k, v]) => (
                <Fragment key={k}><dt>{k.replace(/_/g, " ")}</dt><dd>{String(v)}</dd></Fragment>))}</dl>
            </Panel>
          )}
          <Panel title="Configuration & provenance">
            <dl className="kv">
              <dt>Model</dt><dd>{rec.model.model_id} · {rec.model.estimator}</dd>
              {Object.entries(rec.models).filter(([, v]) => v).map(([k, v]) => (
                <Fragment key={k}><dt>{k} model</dt><dd className="mono hash">{v!.artifact_sha256}</dd></Fragment>))}
              <dt>Split</dt><dd className="mono">{rec.dataset.split_id}</dd>
              <dt>Seed</dt><dd>{rec.seed}</dd>
              <dt>Poisoning</dt><dd className="mono small">{rec.poison_config ? JSON.stringify(rec.poison_config) : "none"}</dd>
              {rec.poisoning?.selection_rule && <><dt>Selection</dt><dd className="small">{rec.poisoning.selection_rule}</dd></>}
              <dt>Defense</dt><dd className="mono small">{rec.defense_config ? JSON.stringify(rec.defense_config) : "none"}</dd>
              <dt>Duration</dt><dd>{rec.duration_ms.toFixed(0)} ms</dd>
            </dl>
          </Panel>
        </div>
      </div>
    </div>
  );
}
