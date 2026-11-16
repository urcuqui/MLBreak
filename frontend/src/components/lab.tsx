/** Shared building blocks of every laboratory workspace, so all labs use the same steps, terms and states.
 *
 * Terminology (used verbatim in every lab):
 *   Baseline → Attack → Defend → Evidence & replay
 *   "Run baseline", "Replay against defense" (non-adaptive), "Run adaptive attack" (new experiment).
 */
import { Fragment, useState } from "react";
import { Link, NavLink } from "react-router-dom";
import { api, evidenceLinks } from "../api";
import type { Ratio, ReplayResult, VerifyReport } from "../types";
import { Badge, ErrorBox, EvalBadge, ExpLink, RatioView, StatusBadge } from "./ui";

export type StepId = "baseline" | "attack" | "defend" | "evidence";
export const STEPS: { id: StepId; label: string }[] = [
  { id: "baseline", label: "1 · Baseline" },
  { id: "attack", label: "2 · Attack" },
  { id: "defend", label: "3 · Defend" },
  { id: "evidence", label: "4 · Evidence & replay" },
];

export type ProgressStatus = "done" | "pending" | "unsupported";
export interface ProgressItem { label: "baseline" | "attack" | "replay" | "adaptive"; status: ProgressStatus; title?: string }

const SYMBOL: Record<ProgressStatus, string> = { done: "✓", pending: "○", unsupported: "—" };
const WORD: Record<ProgressStatus, string> = { done: "done", pending: "not run", unsupported: "not available in this lab" };

/** Step tabs + one progress strip with the same labels in every lab. */
export function StepNav({ base, progress }: { base: string; progress: ProgressItem[] }) {
  return (
    <nav className="tabs" aria-label="Lab steps">
      {STEPS.map((s) => (
        <NavLink key={s.id} to={`/lab/${base}/${s.id}`} className={({ isActive }) => (isActive ? "tab active" : "tab")}>{s.label}</NavLink>
      ))}
      <ul className="pipeline muted small" aria-label="Progress in this session">
        {progress.map((p) => (
          <li key={p.label} className={`pipe ${p.status}`} title={p.title ?? `${p.label}: ${WORD[p.status]}`}>
            <span aria-hidden>{SYMBOL[p.status]}</span> {p.label}<span className="sr-only">: {WORD[p.status]}</span>
          </li>
        ))}
      </ul>
    </nav>
  );
}

/** Same wording for every action that needs an earlier step. */
export function RequiresStep({ base, step, what }: { base: string; step: StepId; what: string }) {
  const label = STEPS.find((s) => s.id === step)!.label.replace(/^\d · /, "");
  return <p className="muted small mt">Requires {what}: run it in the <Link to={`/lab/${base}/${step}`}>{label}</Link> step first.</p>;
}

interface EvidenceRecord { experiment_id: string; evaluation: string; status: string; metrics: { attack_success_rate?: Ratio } }

/** The Evidence & replay step, identical in every lab: export, verify integrity, replay. */
export function EvidenceTable({ rows, disabled }: { rows: [string, EvidenceRecord | undefined][]; disabled?: boolean }) {
  const [replays, setReplays] = useState<Record<string, ReplayResult>>({});
  const [verifies, setVerifies] = useState<Record<string, VerifyReport>>({});
  const [running, setRunning] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const act = async (id: string, kind: "verify" | "replay") => {
    setRunning(`${kind} ${id}`); setError(null);
    try {
      if (kind === "replay") { const r = await api.replay(id); setReplays((m) => ({ ...m, [id]: r })); }
      else { const v = await api.verify(id); setVerifies((m) => ({ ...m, [id]: v })); }
    } catch (e) { setError(String((e as Error).message ?? e)); } finally { setRunning(null); }
  };
  const off = disabled || running !== null;
  return (
    <>
      <ErrorBox error={error} />
      {running && <div className="busy" role="status"><span className="spinner" /> Running: {running}…</div>}
      <div className="table-wrap"><table className="table">
        <caption className="sr-only">Experiments of this session with export, integrity and replay actions</caption>
        <thead><tr><th>Step</th><th>Experiment</th><th>Type</th><th>Status</th><th>Attack success</th><th>Export</th>
          <th>Integrity</th><th>Exact replay</th></tr></thead>
        <tbody>{rows.map(([step, r]) => {
          if (!r) return <tr key={step}><td>{step}</td><td colSpan={7} className="muted">not run</td></tr>;
          const l = evidenceLinks(r.experiment_id), rp = replays[r.experiment_id], v = verifies[r.experiment_id];
          return (
            <tr key={step}>
              <td>{step}</td><td><ExpLink id={r.experiment_id} /></td><td><EvalBadge evaluation={r.evaluation} /></td>
              <td><StatusBadge status={r.status} /></td><td><RatioView r={r.metrics.attack_success_rate} /></td>
              <td><span className="row gap"><a href={l.json} download>JSON</a><a href={l.md} download>MD</a><a href={l.csv} download>CSV</a></span></td>
              <td><button className="btn small" disabled={off} onClick={() => act(r.experiment_id, "verify")}>Verify</button>{" "}
                {v && (v.ok ? <Badge tone="good">consistent</Badge> : <Badge tone="critical">failed</Badge>)}</td>
              <td><button className="btn small" disabled={off || r.status !== "completed"} onClick={() => act(r.experiment_id, "replay")}>Replay</button>{" "}
                {rp && (rp.identical ? <Badge tone="good">identical</Badge> : <Badge tone="critical">diverged</Badge>)}</td>
            </tr>);
        })}</tbody>
      </table></div>
      <p className="muted small mt">Verify recomputes every fingerprint, re-derives the metrics from the stored per-sample
        evidence and checks the lineage. It proves internal consistency, <strong>not authenticity</strong>. Replay re-executes
        the experiment and is refused if verification fails. Reports (MD) are rendered with a fresh integrity check.</p>
      {[...Object.values(verifies).filter((v) => !v.ok)].map((v) => <CheckList key={v.experiment_id} report={v} />)}
      {Object.values(replays).map((rp) => <ReplayChecks key={rp.replay_experiment_id} r={rp} />)}
    </>
  );
}

export function CheckList({ report }: { report: VerifyReport }) {
  return (
    <details className="mt" open={!report.ok}>
      <summary>Integrity checks for {report.experiment_id}: {report.ok ? "all passed" : "FAILED"}</summary>
      <div className="table-wrap"><table className="table small"><tbody>{report.checks.map((c) => (
        <tr key={c.check}><td>{c.check}</td>
          <td>{c.status === "pass" ? <Badge tone="good">pass</Badge> : c.status === "fail" ? <Badge tone="critical">fail</Badge>
            : <Badge tone="neutral">skipped</Badge>}</td><td className="small">{c.detail}</td></tr>))}</tbody></table></div>
      <p className="muted small">{report.guarantee}</p>
    </details>
  );
}

export function ReplayChecks({ r }: { r: ReplayResult }) {
  return (
    <details className="mt">
      <summary>Replay {r.original_experiment_id} → {r.replay_experiment_id}: {r.identical ? "identical" : "diverged"}</summary>
      <div className="table-wrap"><table className="table small"><tbody>{r.checks.map((c) => (
        <tr key={c.check}><td>{c.check}</td><td className="mono hash">{String(c.original)}</td>
          <td>{c.match ? <Badge tone="good">match</Badge> : <Badge tone="critical">differs</Badge>}</td></tr>))}</tbody></table></div>
    </details>
  );
}

/** Verify + replay for one stored experiment (experiment detail pages of every lab). */
export function IntegrityPanel({ id, completed }: { id: string; completed: boolean }) {
  const [verify, setVerify] = useState<VerifyReport | null>(null);
  const [replay, setReplay] = useState<ReplayResult | null>(null);
  const [running, setRunning] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const run = async (kind: "verify" | "replay") => {
    setRunning(kind); setError(null);
    try { if (kind === "verify") setVerify(await api.verify(id)); else setReplay(await api.replay(id)); }
    catch (e) { setError(String((e as Error).message ?? e)); } finally { setRunning(null); }
  };
  return (
    <Fragment>
      <div className="row gap wrap">
        <button className="btn" disabled={running !== null} onClick={() => run("verify")}>{running === "verify" ? "Verifying…" : "Verify integrity"}</button>
        <button className="btn primary" disabled={running !== null || !completed} onClick={() => run("replay")}>{running === "replay" ? "Replaying…" : "Replay experiment"}</button>
      </div>
      <ErrorBox error={error} />
      {verify && <div className={verify.ok ? "callout good mt" : "callout bad mt"}>{verify.ok ? "✓ Evidence is internally consistent" : "✕ Integrity check FAILED"}<CheckList report={verify} /></div>}
      {replay && <div className={replay.identical ? "callout good mt" : "callout bad mt"}>{replay.identical ? "✓ Replay produced identical results" : "✕ Replay diverged"} — new experiment <ExpLink id={replay.replay_experiment_id} />.
        <ReplayChecks r={replay} /></div>}
    </Fragment>
  );
}
