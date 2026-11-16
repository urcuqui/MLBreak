import { Fragment } from "react";
import { evidenceLinks } from "../api";
import { InversionSummary, PrivacyUtility, TargetTable, TraceExplorer, policyLabel } from "../components/inversion";
import { IntegrityPanel } from "../components/lab";
import { Badge, ErrorBox, EvalBadge, ExpLink, Panel, RatioView, Tile } from "../components/ui";
import type { InversionRecord, Ratio } from "../types";

/** Detail view of a stored ML03 model-inversion experiment. */
export function InversionDetail({ rec }: { rec: InversionRecord }) {
  const links = evidenceLinks(rec.experiment_id);
  const done = rec.status === "completed";
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <div className="eyebrow">{rec.owasp.id} · {rec.owasp.name} · {rec.scenario_id}</div>
          <h1 className="mono">{rec.experiment_id}</h1>
          <div className="row gap wrap">
            <EvalBadge evaluation={rec.evaluation} />
            {done ? <Badge tone="good">completed</Badge> : <Badge tone="critical">{rec.status}</Badge>}
            <Badge tone="neutral">{policyLabel(rec.output_policy)}</Badge>
            <span className="small mono">{rec.model.model_id}</span>
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
        <div className="panel-body"><IntegrityPanel id={rec.experiment_id} completed={done} /></div>
      </section>
      {done && rec.evaluation === "clean" && (
        <Panel title="Baseline evaluation">
          <div className="tiles">
            <Tile label="Victim decision accuracy"><RatioView r={rec.metrics.victim_prediction_accuracy as Ratio} /></Tile>
            <Tile label="Without the attribute"><RatioView r={rec.metrics.accuracy_without_private_attribute as Ratio} /></Tile>
            <Tile label="Auxiliary-only inference"><RatioView r={rec.metrics.baseline_inference_accuracy as Ratio} /></Tile>
          </div>
        </Panel>
      )}
      {done && rec.evaluation !== "clean" && (
        <>
          <Panel title="Baseline vs inversion"><InversionSummary rec={rec} /></Panel>
          <Panel title="Privacy–utility of this output policy"><PrivacyUtility rows={[rec]} /></Panel>
          <Panel title="Attack trace" subtitle="Which information the adversary had at each step."><TraceExplorer rec={rec} /></Panel>
          <Panel title="Per-target results" subtitle="Attacker outputs next to the evaluation-only ground truth."><TargetTable rec={rec} /></Panel>
        </>
      )}
      {rec.attacker_capabilities && (
        <Panel title="Attacker capabilities">
          <dl className="kv">{Object.entries(rec.attacker_capabilities).map(([k, v]) => (
            <Fragment key={k}><dt>{k}</dt><dd>{String(v)}</dd></Fragment>))}</dl>
        </Panel>
      )}
    </div>
  );
}
