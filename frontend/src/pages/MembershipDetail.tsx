import { Fragment } from "react";
import { evidenceLinks } from "../api";
import { IntegrityPanel } from "../components/lab";
import { AttackerKnowledgePanel, CalibrationVsEvaluation, CandidateExplorer, ConfusionMatrices, DefenseComparison,
  MembershipSummary, OperationalPrecision, RocCurves, UtilityTiles, defenseLabel, mratio } from "../components/membership";
import { Badge, ErrorBox, EvalBadge, ExpLink, Panel, RatioView, Tile } from "../components/ui";
import type { MembershipRecord } from "../types";

/** Detail view of a stored ML04 membership-inference experiment (evaluator evidence). */
export function MembershipDetail({ rec }: { rec: MembershipRecord }) {
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
            <Badge tone="neutral">{defenseLabel(rec.defense_config)}</Badge>
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
      {done && (
        <Panel title="Victim utility" subtitle={rec.model.description}>
          <UtilityTiles rec={rec} />
          <div className="tiles"><Tile label="No-signal baseline"><RatioView r={mratio(rec, "no_signal_accuracy")} /></Tile></div>
        </Panel>
      )}
      {done && rec.evaluation !== "clean" && (
        <>
          <Panel title="Final evaluation" subtitle="Untouched candidates; thresholds from the attacker's calibration only."><MembershipSummary rec={rec} /></Panel>
          <div className="grid-2">
            <Panel title={rec.evaluation_contract === "placebo" ? "Placebo curves (no member among the candidates)" : "ROC curves (evaluation population)"}><RocCurves rec={rec} /></Panel>
            <Panel title="Confusion matrices"><ConfusionMatrices rec={rec} /></Panel>
          </div>
          <div className="grid-2">
            <Panel title="Calibration vs evaluation"><CalibrationVsEvaluation rec={rec} /></Panel>
            <Panel title="Operational precision"><OperationalPrecision rec={rec} /></Panel>
          </div>
          <Panel title="Privacy–utility of this configuration"><DefenseComparison rows={[rec]} /></Panel>
          <Panel title="Membership predictions" subtitle="Attacker view first; ground truth only in the evaluator view."><CandidateExplorer rec={rec} /></Panel>
          <Panel title="Attacker knowledge"><AttackerKnowledgePanel rec={rec} /></Panel>
        </>
      )}
      {rec.shadow && (
        <Panel title="Shadow models and calibration provenance">
          <dl className="kv">
            <dt>Shadow population</dt><dd className="mono hash">{rec.shadow.population_sha256}</dd>
            <dt>Shadow models</dt><dd>{rec.shadow.n_shadow_models} · {rec.shadow.models.map((m) => m.artifact_sha256.slice(0, 12)).join(", ")}</dd>
            <dt>Meta-classifier</dt><dd>{rec.shadow.meta_classifier.estimator} on {rec.shadow.meta_classifier.n_training_examples} shadow examples · features {rec.shadow.meta_classifier.features.join(", ")}</dd>
            {rec.calibration && <><dt>Calibration population</dt><dd className="mono hash">{rec.calibration.population_sha256}</dd>
              <dt>Reference models</dt><dd>{rec.calibration.reference_models.map((m) => m.artifact_sha256.slice(0, 12)).join(", ")}</dd></>}
            {rec.lineage && <Fragment><dt>Lineage</dt><dd>{rec.lineage.relation} <ExpLink id={rec.lineage.parent_experiment_id} /></dd></Fragment>}
          </dl>
        </Panel>
      )}
    </div>
  );
}
