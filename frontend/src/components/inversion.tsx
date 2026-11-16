/** ML03 model-inversion views shared by the lab workspace and the experiment detail page.
 *
 * Every value shown comes from a stored record (or, for the target list, from the attacker-facing
 * /api/inversion/targets view, which never contains the private attribute). The attack trace labels each step
 * with WHO had that information: the adversary, or only the evaluation engine.
 */
import { useMemo, useState } from "react";
import type { InversionRecord, InversionTarget, MeanMetric, OutputPolicy, Ratio } from "../types";
import { Plot } from "./Plot";
import { Badge, ExpLink, RatioView, Tile, pct } from "./ui";

export const ratioOf = (rec: InversionRecord, key: string) => rec.metrics[key] as Ratio | undefined;
export const meanOf = (rec: InversionRecord, key: string) => rec.metrics[key] as MeanMetric | undefined;
const num = (m: MeanMetric | undefined, digits = 3) => (m?.value == null ? "—" : m.value.toFixed(digits));

export function policyLabel(p: OutputPolicy | null | undefined): string {
  if (!p || p.kind === "full_probability") return "Full probability";
  return p.kind === "label_only" ? "Label only" : `Quantised (${p.bins} buckets)`;
}

export const fmtOut = (v: number, p: OutputPolicy) => (p.kind === "label_only" ? String(v) : v.toFixed(p.kind === "full_probability" ? 6 : 4));

function Adv() { return <Badge tone="info">adversary</Badge>; }
function EvalOnly() { return <Badge tone="warning">evaluation only</Badge>; }

/** Baseline vs inversion headline tiles + the leakage test. */
export function InversionSummary({ rec }: { rec: InversionRecord }) {
  const lp = rec.metrics.leakage_p_value as { value: number | null; leakage_claimed: boolean; inversion_only_correct: number;
    baseline_only_correct: number } | undefined;
  const q = rec.metrics.queries as { mean: number | null; max: number; budget_per_target: number } | undefined;
  const adv = ratioOf(rec, "inference_advantage");
  return (
    <>
      <div className="tiles">
        <Tile label="Auxiliary-only accuracy" hint="No victim output, no query: public features + auxiliary population only">
          <RatioView r={ratioOf(rec, "baseline_inference_accuracy")} /></Tile>
        <Tile label="Inversion accuracy"><RatioView r={ratioOf(rec, "attribute_inference_accuracy")} emphasize /></Tile>
        <Tile label="Inference advantage" hint={adv?.definition}>
          <span className="num strong">{adv?.value == null ? "undefined" : `${adv.value >= 0 ? "+" : ""}${pct(adv.value)}`}</span>{" "}
          <span className="muted small">({adv?.numerator}/{adv?.denominator})</span></Tile>
        <Tile label="Leakage claim" hint="one-sided paired McNemar test, inversion vs auxiliary-only, alpha 0.05">
          {lp ? <>{lp.leakage_claimed ? <Badge tone="critical">claimed</Badge> : <Badge tone="good">not claimed</Badge>}
            <div className="muted small">p = {lp.value == null ? "—" : lp.value.toPrecision(2)} · {lp.inversion_only_correct} vs {lp.baseline_only_correct} discordant</div></> : "—"}
        </Tile>
      </div>
      <div className="tiles">
        <Tile label="Ambiguous reconstructions"><RatioView r={ratioOf(rec, "ambiguity_rate")} /></Tile>
        <Tile label="Accuracy when unambiguous"><RatioView r={ratioOf(rec, "unambiguous_accuracy")} /></Tile>
        <Tile label="Queries per target">{q ? <span className="num">{q.mean?.toFixed(2)} mean · {q.max} max <span className="muted small">(budget {q.budget_per_target})</span></span> : "—"}</Tile>
        <Tile label="Budget exhausted"><RatioView r={ratioOf(rec, "budget_exhaustion_rate")} /></Tile>
      </div>
      <p className="muted small">High inversion accuracy alone is <strong>not</strong> evidence of model-induced leakage: the
        attribute is partly predictable from public data. Only the advantage over the auxiliary-only attacker on the same
        targets, with its paired test, is attributed to the model's outputs.</p>
    </>
  );
}

/** Step-by-step trace of one target, marking which information the adversary had. */
export function TraceExplorer({ rec }: { rec: InversionRecord }) {
  const traced = useMemo(() => rec.targets.filter((t) => t.trace), [rec]);
  const [k, setK] = useState(0);
  const t: InversionTarget | undefined = traced[Math.min(k, traced.length - 1)];
  if (!t || !t.trace) return <p className="muted">No attack trace in this experiment.</p>;
  const tr = t.trace, pol = rec.output_policy, names = rec.dataset.public_features;
  return (
    <div className="stack">
      <label className="inline-field">Target
        <select value={k} onChange={(e) => setK(parseInt(e.target.value, 10))} aria-label="Target to inspect">
          {traced.map((x, i) => <option key={x.target_index} value={i}>#{x.target_index}{x.trace!.ambiguous ? " (ambiguous)" : ""}
            {x.inversion_correct ? " ✓" : " ✕"}</option>)}
        </select>
      </label>
      <ol className="trace-steps">
        <li><div className="row gap wrap"><strong>1 · Known public features</strong><Adv /></div>
          <dl className="kv">{names.map((n, j) => <FragmentKV key={n} k={n} v={tr.public_features[j].toFixed(4)} />)}
            <FragmentKV k={`${rec.dataset.private_attribute}`} v="hidden — unknown to the attacker" />
            <FragmentKV k="auxiliary prior P(s=1)" v={`${tr.auxiliary_p1.toFixed(3)} (attacker's own model on auxiliary data)`} /></dl></li>
        <li><div className="row gap wrap"><strong>2 · Observed output of the real record</strong><Adv /></div>
          <p className="mono">{fmtOut(tr.observed_output, pol)} <span className="muted small">({policyLabel(pol)}; not a query)</span></p></li>
        <li><div className="row gap wrap"><strong>3 · Candidate queries</strong><Adv /><span className="muted small">{tr.n_queries} of {tr.query_budget} allowed</span></div>
          <div className="table-wrap" tabIndex={0} role="region" aria-label={`Candidate queries for target ${t.target_index}`}>
            <table className="table small"><thead><tr><th>#</th><th>crafted record</th><th>response</th><th>|response − observed|</th><th>consistent?</th></tr></thead>
              <tbody>{tr.queries.map((q) => (
                <tr key={q.query_index}><td>{q.query_index}</td><td>public features + s = {q.candidate}</td>
                  <td className="mono">{fmtOut(q.response, pol)}</td><td className="mono">{q.distance.toPrecision(3)}</td>
                  <td>{q.distance <= (rec.attack_config?.tolerance ?? 0) ? <Badge tone="info">matches</Badge> : <Badge tone="neutral">excluded</Badge>}</td></tr>))}
                {tr.candidate_domain.filter((c) => !tr.queries.some((q) => q.candidate === c)).map((c) => (
                  <tr key={`u${c}`}><td>–</td><td>s = {c}</td><td colSpan={2} className="muted">not queried{tr.budget_exhausted ? " (budget exhausted)" : " (not needed)"}</td>
                    <td>{tr.consistent_candidates.includes(c) ? <Badge tone="neutral">cannot exclude</Badge> : "—"}</td></tr>))}</tbody></table></div></li>
        <li><div className="row gap wrap"><strong>4 · Reconstruction</strong><Adv /></div>
          <p>Consistent candidates: <span className="mono">{`{${tr.consistent_candidates.join(", ")}}`}</span> · rule <code>{tr.rule}</code>
            {tr.ambiguous && <> · <Badge tone="warning">ambiguous</Badge></>} → inferred <strong className="mono">{tr.inferred ?? "abstain"}</strong></p></li>
        <li><div className="row gap wrap"><strong>5 · Ground-truth comparison</strong><EvalOnly /></div>
          <p>True attribute <strong className="mono">{t.ground_truth.private_attribute}</strong> →{" "}
            {t.inversion_correct ? <Badge tone="critical">reconstructed</Badge> : <Badge tone="good">not reconstructed</Badge>}
            {" "}· auxiliary-only guess {t.baseline_inferred} ({t.baseline_correct ? "correct" : "wrong"})
            {t.undefended_inversion_correct != null && <> · against full probability: {t.undefended_inversion_correct ? "reconstructed" : "not reconstructed"}</>}</p>
          <p className="muted small">The evaluation engine holds this value; it was never passed to the attack code.</p></li>
      </ol>
    </div>
  );
}

function FragmentKV({ k, v }: { k: string; v: string }) {
  return <><dt>{k}</dt><dd className="mono">{v}</dd></>;
}

/** All targets: attacker outputs next to evaluation-only ground truth. */
export function TargetTable({ rec, limit = 400 }: { rec: InversionRecord; limit?: number }) {
  const pol = rec.output_policy;
  return (
    <div className="table-wrap scroll-y" tabIndex={0} role="region" aria-label="Per-target reconstruction results">
      <table className="table small">
        <caption className="sr-only">Per-target results: attacker-side columns, then evaluation-only columns</caption>
        <thead><tr><th>target</th><th>observed</th><th>queries</th><th>rule</th><th>inferred</th>
          <th>aux. guess</th><th>true (eval only)</th><th>inversion</th><th>aux. only</th></tr></thead>
        <tbody>{rec.targets.slice(0, limit).map((t) => (
          <tr key={t.target_index}><td>{t.target_index}</td>
            <td className="mono">{t.trace ? fmtOut(t.trace.observed_output, pol) : "—"}</td>
            <td className="mono">{t.trace ? t.trace.queries.map((q) => `s=${q.candidate}→${fmtOut(q.response, pol)}`).join("; ") || "–" : "—"}</td>
            <td>{t.trace?.rule ?? "—"}</td><td className="mono">{t.trace ? t.trace.inferred ?? "abstain" : "—"}</td>
            <td className="mono">{t.baseline_inferred}</td><td className="mono">{t.ground_truth.private_attribute}</td>
            <td>{t.inversion_correct == null ? "—" : t.inversion_correct ? "✓ correct" : "✕ wrong"}</td>
            <td>{t.baseline_correct ? "✓ correct" : "✕ wrong"}</td></tr>))}</tbody>
      </table>
    </div>
  );
}

/** Privacy (inference advantage) vs utility (released ranking AUC, Brier, decisions), one row per output policy. */
export function PrivacyUtility({ rows }: { rows: InversionRecord[] }) {
  const data = rows.filter((r) => r.status === "completed");
  if (!data.length) return <p className="muted">Run the attack and at least one output policy to compare.</p>;
  const pts = data.map((r) => ({ name: policyLabel(r.output_policy), x: meanOf(r, "released_ranking_auc")?.value ?? null,
    y: ratioOf(r, "inference_advantage")?.value ?? null }));
  return (
    <>
      <div className="table-wrap" tabIndex={0} role="region" aria-label="Privacy–utility comparison">
        <table className="table small">
          <caption className="sr-only">Privacy and utility per output policy, same targets</caption>
          <thead><tr><th>Output policy</th><th>Experiment</th><th>Inversion acc.</th><th>Advantage vs aux.</th><th>Leakage</th>
            <th>Ambiguous</th><th>Reduction vs full (paired p)</th><th>Decision acc.</th><th>Brier ↓</th><th>Ranking AUC</th><th>Score distortion</th></tr></thead>
          <tbody>{data.map((r) => {
            const lp = r.metrics.leakage_p_value as { leakage_claimed: boolean; value: number } | undefined;
            const red = r.metrics.defense_reduction_p_value as { value: number } | undefined;
            return (
              <tr key={r.experiment_id}><td>{policyLabel(r.output_policy)}</td><td><ExpLink id={r.experiment_id} /></td>
                <td><RatioView r={ratioOf(r, "attribute_inference_accuracy")} /></td>
                <td><RatioView r={ratioOf(r, "inference_advantage")} /></td>
                <td>{lp ? (lp.leakage_claimed ? <Badge tone="critical">claimed</Badge> : <Badge tone="good">not claimed</Badge>) : "—"}</td>
                <td><RatioView r={ratioOf(r, "ambiguity_rate")} /></td>
                <td>{red ? <><RatioView r={ratioOf(r, "defense_accuracy_reduction")} /> <span className="muted small">p={red.value.toPrecision(2)}</span></> : "reference"}</td>
                <td><RatioView r={ratioOf(r, "consumer_decision_accuracy")} /></td>
                <td className="mono">{num(meanOf(r, "released_brier_score"))}</td>
                <td className="mono">{num(meanOf(r, "released_ranking_auc"))}</td>
                <td className="mono">{num(meanOf(r, "score_distortion"))}</td></tr>);
          })}</tbody>
        </table>
      </div>
      <Plot height={300} ariaLabel={`Privacy–utility trade-off: ${pts.map((p) => `${p.name}: advantage ${pct(p.y)}, AUC ${p.x?.toFixed(3)}`).join("; ")}`}
        data={[{ type: "scatter", mode: "markers+text", x: pts.map((p) => p.x), y: pts.map((p) => p.y), text: pts.map((p) => p.name),
          textposition: "top center", marker: { size: 12, color: "#e5a50a" }, hovertemplate: "%{text}<br>AUC %{x:.3f}<br>advantage %{y:.1%}<extra></extra>" }]}
        layout={{ margin: { l: 56, r: 24, t: 40, b: 44 }, xaxis: { title: { text: "utility: ranking AUC of released output" } },
          yaxis: { title: { text: "leakage: inference advantage" }, tickformat: ".0%", rangemode: "tozero" } }} />
      <p className="muted small">Each row is a NEW adaptive reconstruction against that policy, on the same targets as the
        full-probability attack. Output restriction is not a formal privacy guarantee.</p>
    </>
  );
}
