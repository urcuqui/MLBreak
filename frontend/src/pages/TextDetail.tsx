import { IntegrityPanel } from "../components/lab";
import { evidenceLinks } from "../api";
import { Badge, ErrorBox, EvalBadge, ExpLink } from "../components/ui";
import type { TextRecord } from "../types";
import { TextDetailBody } from "./SpamWorkspace";

/** Detail view of a stored ML01 spam-filter evasion experiment. */
export function TextDetail({ rec }: { rec: TextRecord }) {
  const links = evidenceLinks(rec.experiment_id);
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <div className="eyebrow">{rec.owasp.id} · {rec.owasp.name} · {rec.scenario_id}</div>
          <h1 className="mono">{rec.experiment_id}</h1>
          <div className="row gap wrap">
            <EvalBadge evaluation={rec.evaluation} />
            {rec.status === "completed" ? <Badge tone="good">completed</Badge> : <Badge tone="critical">{rec.status}</Badge>}
            {rec.intentionally_vulnerable ? <Badge tone="warning">intentionally vulnerable target</Badge> : <Badge tone="info">defended target</Badge>}
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
        <div className="panel-body"><IntegrityPanel id={rec.experiment_id} completed={rec.status === "completed"} /></div>
      </section>
      <TextDetailBody rec={rec} />
    </div>
  );
}
