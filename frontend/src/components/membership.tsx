/** ML04 membership-inference views shared by the lab workspace and the experiment detail page.
 *
 * Every value comes from a stored evidence record. Attacker-side values (released output, scores, decisions) and
 * evaluator-only ground truth (membership) are labelled; the ground truth is hidden until the viewer explicitly
 * switches to the evaluator results view.
 */
import { Fragment, useMemo, useState } from "react";
import type { MembershipCandidate, MembershipDefense, MembershipRecord, OperatingPoint, OutputPolicy, Ratio } from "../types";
import { Plot } from "./Plot";
import { Badge, ExpLink, RatioView, Tile, pct } from "./ui";

export const ATTACKS = ["threshold", "shadow"] as const;
type Attack = (typeof ATTACKS)[number];
const ATTACK_LABEL: Record<Attack, string> = { threshold: "Threshold attack", shadow: "Shadow-model attack" };

export const mratio = (rec: MembershipRecord, key: string) => rec.metrics[key] as Ratio | undefined;
export const mval = (rec: MembershipRecord, key: string) => (rec.metrics[key] as { value: number | null } | undefined)?.value ?? null;
const fix = (v: number | null | undefined, d = 3) => (v == null ? "—" : v.toFixed(d));
const signed = (v: number | null | undefined, d = 3) => (v == null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(d)}`);

export function policyText(p: OutputPolicy | null | undefined): string {
  if (!p || p.kind === "full_probability") return "full probability";
  return p.kind === "label_only" ? "label only" : `quantised (${p.bins} buckets)`;
}

export function defenseLabel(d: MembershipDefense | null | undefined): string {
  if (!d) return "Undefended";
  if (d.kind === "regularization") return `Regularisation (leaf ≥ ${d.min_samples_leaf} / C = ${d.C})`;
  if (d.kind === "complexity_reduction") return `Complexity reduction (depth ${d.max_depth}, ${d.n_estimators} trees)`;
  return `Output restriction: ${policyText(d.policy)}`;
}

function Adv() { return <Badge tone="info">adversary</Badge>; }
function EvalOnly() { return <Badge tone="warning">evaluator only</Badge>; }

/** Victim utility: training vs held-out accuracy and the generalisation gap. */
export function UtilityTiles({ rec }: { rec: MembershipRecord }) {
  return (
    <div className="tiles">
      <Tile label="Training accuracy"><RatioView r={mratio(rec, "train_accuracy")} /></Tile>
      <Tile label="Held-out accuracy" hint="victim_test population, never attacked"><RatioView r={mratio(rec, "test_accuracy")} emphasize /></Tile>
      <Tile label="Generalisation gap" hint="training − held-out accuracy: the overfitting membership inference uses">
        <span className="num strong">{signed(mval(rec, "generalization_gap"))}</span></Tile>
      <Tile label="Released-output AUC (held-out)" hint="ranking utility of what consumers receive">
        <span className="num">{fix(mval(rec, "test_roc_auc"))}</span></Tile>
    </div>
  );
}

/** What the adversary knows and sees in this experiment (from the record's attacker_capabilities). */
export function AttackerKnowledgePanel({ rec }: { rec: MembershipRecord }) {
  const caps = rec.attacker_capabilities;
  if (!caps) return <p className="muted">Baseline: no attacker was simulated.</p>;
  return (
    <dl className="kv">{Object.entries(caps).filter(([k]) => k !== "recipe_assumed").map(([k, val]) => (
      <Fragment key={k}><dt>{k.replace(/_/g, " ")}</dt><dd>{String(val)}</dd></Fragment>))}
      <dt>output released</dt><dd>{rec.target_system.output_policy.releases}</dd>
    </dl>
  );
}

function lowFpr(rec: MembershipRecord, a: Attack) {
  const t = rec.metrics[`${a}_tpr_at_low_fpr`] as (Ratio & { status: string; target_fpr: number; reason?: string }) | undefined;
  const f = rec.metrics[`${a}_fpr_at_low_fpr`] as (Ratio & { wilson_95?: number[]; exceeds_target?: boolean }) | undefined;
  return { t, f };
}

/** Headline per attack: AUC, advantage, accuracy, TPR at the calibrated low-FPR point, leakage claim. */
export function MembershipSummary({ rec }: { rec: MembershipRecord }) {
  return (
    <>
      {ATTACKS.map((a) => {
        const lp = rec.metrics[`${a}_leakage_p_value`] as { value: number | null; leakage_claimed: boolean } | undefined;
        const { t, f } = lowFpr(rec, a);
        return (
          <div key={a} className="stack">
            <h3 className="small">{ATTACK_LABEL[a]}</h3>
            <div className="tiles">
              <Tile label="ROC AUC" hint="threshold-free; 0.5 = no membership signal"><span className="num strong">{fix(mval(rec, `${a}_roc_auc`))}</span></Tile>
              <Tile label="Membership advantage" hint="TPR − FPR at the balanced operating point (0 = no-signal baseline)">
                <span className="num strong">{signed(mval(rec, `${a}_membership_advantage`))}</span></Tile>
              <Tile label="Membership accuracy"><RatioView r={mratio(rec, `${a}_membership_accuracy`)} /></Tile>
              <Tile label={`TPR at FPR ≤ ${t?.target_fpr ?? 0.01}`} hint="threshold calibrated on the attacker's reference models only">
                {!t || t.value == null ? <span className="muted small">undefined — {t?.reason ?? "no calibrated threshold"}</span>
                  : <><span className="num">{pct(t.value)}</span>{" "}
                    <span className="muted small">eval FPR {pct(f?.value)}{t.status !== "calibrated" ? ` · ${t.status}` : ""}</span>
                    {f?.exceeds_target && <div><Badge tone="warning">FPR above target</Badge></div>}</>}
              </Tile>
              <Tile label="Leakage claim" hint="one-sided Mann–Whitney test of AUC > 0.5, alpha 0.05 / 2 attacks">
                {lp ? <>{lp.leakage_claimed ? <Badge tone="critical">claimed</Badge> : <Badge tone="good">not claimed</Badge>}
                  <div className="muted small">p = {lp.value == null ? "—" : lp.value.toPrecision(2)}</div></> : "—"}
              </Tile>
            </div>
          </div>);
      })}
      <p className="muted small">Thresholds were calibrated on the attacker's own reference models; the final candidates were
        used only for scoring. Precision depends on prevalence — see the operational-precision table.</p>
    </>
  );
}

/** Calibration (attacker reference models) next to final evaluation, so the two are never confused. */
export function CalibrationVsEvaluation({ rec }: { rec: MembershipRecord }) {
  if (!rec.calibration) return null;
  return (
    <div className="table-wrap" tabIndex={0} role="region" aria-label="Calibration versus evaluation">
      <table className="table small">
        <caption className="sr-only">Operating points: calibration rates on reference models and evaluation rates on the candidates</caption>
        <thead><tr><th>Attack</th><th>Operating point</th><th>Status</th><th>Calibration TPR / FPR</th><th>Evaluation TPR / FPR</th></tr></thead>
        <tbody>{ATTACKS.flatMap((a) => rec.calibration!.attacks[a].operating_points.map((p: OperatingPoint) => {
          const evalTpr = p.name === "balanced" ? mval(rec, `${a}_membership_recall`) : mval(rec, `${a}_tpr_at_low_fpr`);
          const evalFpr = p.name === "balanced" ? mval(rec, `${a}_false_positive_rate`) : mval(rec, `${a}_fpr_at_low_fpr`);
          return (
            <tr key={`${a}-${p.name}`}><td>{ATTACK_LABEL[a]}</td><td>{p.name === "balanced" ? "balanced (max TPR − FPR)" : `low FPR (≤ ${rec.calibration!.attacks[a].target_fpr})`}</td>
              <td>{p.status === "calibrated" ? <Badge tone="good">calibrated</Badge> : p.status === "approximate"
                ? <Badge tone="warning">approximate</Badge> : <Badge tone="neutral">undefined</Badge>}</td>
              <td className="mono">{pct(p.calibration_tpr)} / {pct(p.calibration_fpr)}</td>
              <td className="mono">{pct(evalTpr)} / {pct(evalFpr)}</td></tr>);
        }))}</tbody>
      </table>
      <p className="muted small mt">Calibration: {rec.calibration.n_reference_models} reference model(s) trained by the attacker on its
        own calibration population (known split). Evaluation: {rec.candidate_selection.n_members} members + {rec.candidate_selection.n_non_members} non-members
        never used for calibration.</p>
    </div>
  );
}

/** ROC curves of both attacks on the EVALUATION population, with the calibrated operating points marked. */
export function RocCurves({ rec }: { rec: MembershipRecord }) {
  const colors: Record<Attack, string> = { threshold: "#e5a50a", shadow: "#62a0ea" };
  const data: unknown[] = [{ type: "scatter", mode: "lines", x: [0, 1], y: [0, 1], name: "no signal",
    line: { dash: "dot", color: "#5b6b7a" }, hoverinfo: "skip" }];
  for (const a of ATTACKS) {
    const roc = rec.metrics[`${a}_roc_curve`] as { points: number[][] } | undefined;
    if (!roc) continue;
    data.push({ type: "scatter", mode: "lines", x: roc.points.map((p) => p[0]), y: roc.points.map((p) => p[1]),
      name: `${ATTACK_LABEL[a]} (AUC ${fix(mval(rec, `${a}_roc_auc`))})`, line: { color: colors[a], width: 2 },
      hovertemplate: "FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>" });
    const { t, f } = lowFpr(rec, a);
    const pts = [[mval(rec, `${a}_false_positive_rate`), mval(rec, `${a}_membership_recall`), "balanced"]];
    if (t?.value != null) pts.push([f?.value ?? null, t.value, "low FPR"]);
    data.push({ type: "scatter", mode: "markers", x: pts.map((p) => p[0]), y: pts.map((p) => p[1]), text: pts.map((p) => p[2]),
      name: `${ATTACK_LABEL[a]} operating points`, marker: { color: colors[a], size: 10, symbol: "diamond", line: { color: "#fff", width: 1 } },
      hovertemplate: "%{text}: FPR %{x:.3f}, TPR %{y:.3f}<extra></extra>" });
  }
  const n = rec.candidate_selection;
  return (
    <>
      <Plot height={340} data={data}
        ariaLabel={`ROC curves on the evaluation population (${n.n_members} members, ${n.n_non_members} non-members): ` +
          ATTACKS.map((a) => `${ATTACK_LABEL[a]} AUC ${fix(mval(rec, `${a}_roc_auc`))}`).join("; ")}
        layout={{ margin: { l: 56, r: 16, t: 16, b: 48 }, xaxis: { title: { text: "false positive rate (non-members called members)" }, range: [0, 1] },
          legend: { orientation: "v", x: 0.98, xanchor: "right", y: 0.04, yanchor: "bottom", bgcolor: "rgba(15,20,27,0.85)", font: { size: 11 } },
          yaxis: { title: { text: "true positive rate" }, range: [0, 1] } }} />
      <p className="muted small">Evaluation population: {n.n_members} training members + {n.n_non_members} disjoint non-members.
        The curve is descriptive; diamonds are the operating points chosen on calibration data, not on this curve.</p>
    </>
  );
}

export function ConfusionMatrices({ rec }: { rec: MembershipRecord }) {
  return (
    <div className="grid-2">
      {ATTACKS.map((a) => {
        const c = rec.metrics[`${a}_confusion`] as { tp: number; fp: number; tn: number; fn: number } | undefined;
        if (!c) return null;
        return (
          <div key={a} className="table-wrap">
            <table className="table small">
              <caption>{ATTACK_LABEL[a]} — balanced operating point</caption>
              <thead><tr><th scope="col">truth ↓ / called →</th><th scope="col">member</th><th scope="col">non-member</th></tr></thead>
              <tbody>
                <tr><th scope="row">member</th><td className="num">{c.tp} TP</td><td className="num">{c.fn} FN</td></tr>
                <tr><th scope="row">non-member</th><td className="num">{c.fp} FP</td><td className="num">{c.tn} TN</td></tr>
              </tbody>
            </table>
          </div>);
      })}
    </div>
  );
}

export function OperationalPrecision({ rec }: { rec: MembershipRecord }) {
  const th = rec.metrics.threshold_operational_precision as { values: { prevalence: number; precision: number | null; false_discovery_rate: number | null }[] } | undefined;
  const sh = rec.metrics.shadow_operational_precision as typeof th;
  if (!th || !sh) return null;
  return (
    <div className="table-wrap">
      <table className="table small">
        <caption className="sr-only">Projected precision and false discovery rate by membership prevalence</caption>
        <thead><tr><th>Prevalence of members</th><th>Threshold precision</th><th>Threshold FDR</th><th>Shadow precision</th><th>Shadow FDR</th></tr></thead>
        <tbody>{th.values.map((r, i) => (
          <tr key={r.prevalence}><td>{pct(r.prevalence, r.prevalence < 0.1 ? 1 : 0)}</td><td className="mono">{pct(r.precision)}</td>
            <td className="mono">{pct(r.false_discovery_rate)}</td><td className="mono">{pct(sh.values[i].precision)}</td>
            <td className="mono">{pct(sh.values[i].false_discovery_rate)}</td></tr>))}</tbody>
      </table>
      <p className="muted small mt">Projected from the measured TPR and FPR at the balanced operating point. When members are rare,
        most records called "member" are false discoveries even for a strong attack.</p>
    </div>
  );
}

/** Membership predictions for selected candidates; ground truth only after switching to the evaluator view. */
export function CandidateExplorer({ rec }: { rec: MembershipRecord }) {
  const [reveal, setReveal] = useState(false);
  const [k, setK] = useState(0);
  const cands = useMemo(() => rec.candidates.filter((c) => c.observation), [rec]);
  const c: MembershipCandidate | undefined = cands[Math.min(k, cands.length - 1)];
  if (!c || !c.observation) return <p className="muted">No attacker output in this experiment.</p>;
  const names = rec.dataset.features;
  return (
    <div className="stack">
      <div className="row gap wrap">
        <label className="inline-field">Candidate
          <select value={k} onChange={(e) => setK(parseInt(e.target.value, 10))} aria-label="Candidate to inspect">
            {cands.map((x, i) => <option key={x.candidate_id} value={i}>{x.candidate_id}</option>)}
          </select>
        </label>
        <label className="check"><input type="checkbox" checked={reveal} onChange={(e) => setReveal(e.target.checked)} />{" "}
          Evaluator view: reveal ground truth</label>
      </div>
      <ol className="trace-steps">
        <li><div className="row gap wrap"><strong>1 · Candidate record</strong><Adv /></div>
          <dl className="kv">{names.map((n, j) => <Fragment key={n}><dt>{n}</dt><dd className="mono">{c.observation!.features[j].toFixed(4)}</dd></Fragment>)}
            <dt>label</dt><dd className="mono">{c.observation.label ?? "unknown to the attacker"}</dd></dl></li>
        <li><div className="row gap wrap"><strong>2 · Victim's released output</strong><Adv /><span className="muted small">query #{c.observation.query_index}</span></div>
          <p className="mono">{c.observation.released_output} <span className="muted small">({policyText(rec.output_policy)})</span></p></li>
        <li><div className="row gap wrap"><strong>3 · Membership predictions</strong><Adv /></div>
          <div className="table-wrap"><table className="table small"><thead><tr><th>attack</th><th>score</th><th>balanced point</th><th>low-FPR point</th></tr></thead>
            <tbody>{ATTACKS.map((a) => { const d = c[a]!; return (
              <tr key={a}><td>{ATTACK_LABEL[a]}</td><td className="mono">{d.score.toPrecision(4)}</td>
                <td>{d.member_balanced ? <Badge tone="critical">member</Badge> : <Badge tone="neutral">non-member</Badge>}</td>
                <td>{d.member_low_fpr == null ? <span className="muted">undefined</span> : d.member_low_fpr ? <Badge tone="critical">member</Badge> : <Badge tone="neutral">non-member</Badge>}</td></tr>); })}</tbody></table></div></li>
        <li><div className="row gap wrap"><strong>4 · Ground truth</strong><EvalOnly /></div>
          {reveal ? <p>Designated group <strong>{c.truth.group}</strong> · in victim training set: <strong>{String(c.truth.in_training)}</strong> ·
            source {c.truth.population} #{c.truth.population_index} · threshold {c.threshold!.member_balanced === (c.truth.group === "member") ? "✓ correct" : "✕ wrong"} ·
            shadow {c.shadow!.member_balanced === (c.truth.group === "member") ? "✓ correct" : "✕ wrong"}</p>
            : <p className="muted"><span className="hidden-attr">🔒 hidden</span> The evaluation engine holds membership; it was never passed to the attack code.</p>}</li>
      </ol>
      <CandidateTable rec={rec} reveal={reveal} />
    </div>
  );
}

function CandidateTable({ rec, reveal }: { rec: MembershipRecord; reveal: boolean }) {
  return (
    <div className="table-wrap scroll-y" tabIndex={0} role="region" aria-label="Per-candidate membership predictions">
      <table className="table small">
        <caption className="sr-only">Attacker-side columns, then the evaluator-only ground truth</caption>
        <thead><tr><th>candidate</th><th>released</th><th>threshold score</th><th>threshold call</th><th>shadow score</th><th>shadow call</th>
          <th>truth (evaluator)</th></tr></thead>
        <tbody>{rec.candidates.filter((c) => c.observation).map((c) => (
          <tr key={c.candidate_id}><td className="mono">{c.candidate_id}</td><td className="mono">{c.observation!.released_output.toPrecision(3)}</td>
            <td className="mono">{c.threshold!.score.toPrecision(3)}</td><td>{c.threshold!.member_balanced ? "member" : "non-member"}</td>
            <td className="mono">{c.shadow!.score.toPrecision(3)}</td><td>{c.shadow!.member_balanced ? "member" : "non-member"}</td>
            <td>{reveal ? c.truth.group : "🔒"}</td></tr>))}</tbody>
      </table>
    </div>
  );
}

/** Defense comparison + privacy–utility trade-off: one row per record (undefended first). */
export function DefenseComparison({ rows }: { rows: MembershipRecord[] }) {
  const data = rows.filter((r) => r.status === "completed" && r.evaluation !== "clean");
  if (!data.length) return <p className="muted">Run the attack and at least one defense to compare.</p>;
  const label = (r: MembershipRecord) => defenseLabel(r.defense_config);
  const short = (r: MembershipRecord) => { const d = r.defense_config; return !d ? "undefended" : d.kind === "regularization"
    ? "regularisation" : d.kind === "complexity_reduction" ? "complexity" : d.policy.kind === "label_only" ? "label only" : `${d.policy.bins} buckets`; };
  const xs = data.map((r) => mval(r, "test_accuracy") ?? 0);
  const xr = [Math.min(...xs) - 0.015, Math.max(...xs) + 0.015];
  return (
    <>
      <div className="table-wrap" tabIndex={0} role="region" aria-label="Defense comparison">
        <table className="table small">
          <caption className="sr-only">Leakage and utility per configuration, same candidates, recalibrated attacks</caption>
          <thead><tr><th>Configuration</th><th>Experiment</th><th>Held-out acc.</th><th>Gap</th><th>Threshold AUC</th><th>Shadow AUC</th>
            <th>Advantage (thr / shadow)</th><th>TPR@low FPR (thr)</th><th>Leakage</th><th>Reduction vs undefended (paired p)</th></tr></thead>
          <tbody>{data.map((r) => {
            const claimed = (r.metrics.leakage_claimed as { value: number })?.value === 1;
            const red = r.metrics.threshold_defense_reduction_p_value as { value: number } | undefined;
            return (
              <tr key={r.experiment_id}><td>{label(r)}</td><td><ExpLink id={r.experiment_id} /></td>
                <td><RatioView r={mratio(r, "test_accuracy")} /></td><td className="mono">{signed(mval(r, "generalization_gap"))}</td>
                <td className="mono">{fix(mval(r, "threshold_roc_auc"))}</td><td className="mono">{fix(mval(r, "shadow_roc_auc"))}</td>
                <td className="mono">{signed(mval(r, "threshold_membership_advantage"))} / {signed(mval(r, "shadow_membership_advantage"))}</td>
                <td className="mono">{pct(mval(r, "threshold_tpr_at_low_fpr"))}</td>
                <td>{claimed ? <Badge tone="critical">claimed</Badge> : <Badge tone="good">not claimed</Badge>}</td>
                <td>{red ? <><RatioView r={mratio(r, "threshold_correctness_reduction")} /> <span className="muted small">p={red.value.toPrecision(2)}</span></> : "reference"}</td></tr>);
          })}</tbody>
        </table>
      </div>
      <Plot height={300} ariaLabel={`Privacy–utility trade-off: ${data.map((r) => `${label(r)}: AUC ${fix(mval(r, "threshold_roc_auc"))}, accuracy ${pct(mval(r, "test_accuracy"))}`).join("; ")}`}
        data={[{ type: "scatter", mode: "markers+text", x: data.map((r) => mval(r, "test_accuracy")), y: data.map((r) => mval(r, "threshold_roc_auc")),
          text: data.map(short), customdata: data.map(label), textposition: "top center", cliponaxis: false, marker: { size: 12, color: "#e5a50a" },
          hovertemplate: "%{customdata}<br>held-out accuracy %{x:.3f}<br>threshold AUC %{y:.3f}<extra></extra>" }]}
        layout={{ margin: { l: 56, r: 24, t: 40, b: 48 }, xaxis: { title: { text: "utility: held-out accuracy" }, tickformat: ".1%", range: xr },
          yaxis: { title: { text: "leakage: threshold-attack AUC" }, range: [0.45, 1] } }} />
      <p className="muted small">Each defended row is a RECALIBRATED attack (reference and shadow models retrained with the defended
        recipe and output policy) on the same candidates. None of these defenses is differential privacy.</p>
    </>
  );
}
