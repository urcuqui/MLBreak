import { useState } from "react";
import { api, evidenceLinks } from "../api";
import { Badge, ErrorBox, EvalBadge, ExpLink } from "../components/ui";
import type { ReplayResult, TextRecord } from "../types";
import { TextDetailBody } from "./SpamWorkspace";

/** Detail view of a stored ML01 spam-filter evasion experiment. */
export function TextDetail({ rec }: { rec: TextRecord }) {
  const [replay, setReplay] = useState<ReplayResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const links = evidenceLinks(rec.experiment_id);
  const doReplay = async () => {
    setBusy(true); setError(null);
    try { setReplay(await api.replay(rec.experiment_id)); } catch (e) { setError(String((e as Error).message ?? e)); }
    finally { setBusy(false); }
  };
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
      <TextDetailBody rec={rec} />
    </div>
  );
}
