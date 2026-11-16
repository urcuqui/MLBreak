import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api";
import { InversionSummary, PrivacyUtility, TargetTable, TraceExplorer, fmtOut, meanOf, policyLabel, ratioOf } from "../components/inversion";
import { LabSwitcher } from "../components/LabSwitcher";
import { EvidenceTable, RequiresStep, StepNav, type StepId } from "../components/lab";
import { ErrorBox, ExpLink, Panel, RatioView, StatusBadge, Tile, VulnerableBanner } from "../components/ui";
import type { InversionAttackConfig, InversionModelId, InversionRecord, InversionTargetsView, RestrictivePolicy,
  TargetSelection } from "../types";

interface Ctx { model_id: InversionModelId; seed: number; max_targets: number }
interface Session { baseline?: string; attack?: string; policies: Record<string, string> }
const SESSION_KEY = "mlbreak.ml03.session";
const DEFAULT_ATTACK: InversionAttackConfig = { attack: "candidate_matching", query_budget: 2, tie_policy: "auxiliary_prior", tolerance: 1e-9 };
const policyKey = (p: RestrictivePolicy) => (p.kind === "label_only" ? "label_only" : `quantized_${p.bins}`);

function loadSession(): { ctx: Ctx; ids: Session } {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (raw) { const s = JSON.parse(raw); if (s?.ids?.policies) return s; }
  } catch { /* storage unavailable */ }
  return { ctx: { model_id: "logreg", seed: 7, max_targets: 200 }, ids: { policies: {} } };
}

export function InversionWorkspace() {
  const { tab = "baseline" } = useParams<{ tab: StepId }>();
  const initial = useMemo(loadSession, []);
  const [ctx, setCtx] = useState<Ctx>(initial.ctx);
  const [ids, setIds] = useState<Session>(initial.ids);
  const [records, setRecords] = useState<Record<string, InversionRecord>>({});
  const [attackCfg, setAttackCfg] = useState<InversionAttackConfig>(DEFAULT_ATTACK);
  const [policy, setPolicy] = useState<RestrictivePolicy>({ kind: "label_only" });
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const targets: TargetSelection = { max_targets: ctx.max_targets, order: "first" };

  useEffect(() => {
    try { localStorage.setItem(SESSION_KEY, JSON.stringify({ ctx, ids })); } catch { /* ignore */ }
  }, [ctx, ids]);

  useEffect(() => {
    [ids.baseline, ids.attack, ...Object.values(ids.policies)].forEach((id) => {
      if (id && !records[id]) {
        api.anyExperiment(id).then((r) => {
          if ((r as InversionRecord).family === "inversion") setRecords((m) => ({ ...m, [id]: r as InversionRecord }));
        }).catch(() => {});
      }
    });
  }, [ids]); // eslint-disable-line react-hooks/exhaustive-deps

  const run = useCallback(async (what: string, fn: () => Promise<InversionRecord>, store: (r: InversionRecord) => void) => {
    setBusy(what); setError(null);
    try {
      const rec = await fn();
      setRecords((m) => ({ ...m, [rec.experiment_id]: rec }));
      store(rec);
      if (rec.status !== "completed") setError(`Experiment ${rec.experiment_id} failed: ${rec.error}`);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(null);
    }
  }, []);

  const changeCtx = (c: Ctx) => { setCtx(c); setIds({ policies: {} }); setRecords({}); };
  const baseline = ids.baseline ? records[ids.baseline] : undefined;
  const attack = ids.attack ? records[ids.attack] : undefined;
  const defended = Object.entries(ids.policies).map(([k, id]) => [k, records[id]] as const)
    .filter(([, r]) => r) as [string, InversionRecord][];
  const latest = ids.policies[policyKey(policy)] ? records[ids.policies[policyKey(policy)]] : undefined;

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <div className="eyebrow">OWASP ML03:2023 · Model Inversion Attack</div>
          <h1>Private Attribute Inversion</h1>
          <p className="muted">
            A fictional account-risk service scores synthetic records that include a confidential binary attribute.
            Knowing a target's public features and the service's output for that record, the attacker queries
            candidate records to reconstruct the attribute. Every result is compared with an attacker who has only
            auxiliary data, on the same held-out targets.
          </p>
        </div>
        <div className="ctx-controls">
          <LabSwitcher current="ml03" tab={tab} />
          <label>Victim model
            <select value={ctx.model_id} onChange={(e) => changeCtx({ ...ctx, model_id: e.target.value as InversionModelId })}>
              <option value="logreg">Logistic regression (uses the attribute)</option>
              <option value="logreg-public-only">Negative control (ignores the attribute)</option>
            </select>
          </label>
          <label>Seed
            <input type="number" min={0} max={100000} value={ctx.seed}
              onChange={(e) => { const v = parseInt(e.target.value, 10); if (v >= 0 && v <= 100000) changeCtx({ ...ctx, seed: v }); }} />
          </label>
          <label>Targets
            <input type="number" min={1} max={400} value={ctx.max_targets}
              onChange={(e) => { const v = parseInt(e.target.value, 10); if (v >= 1 && v <= 400) changeCtx({ ...ctx, max_targets: v }); }} />
          </label>
        </div>
      </div>
      <VulnerableBanner>
        The undefended service in this lab releases full-precision scores computed on a confidential attribute, so the
        attribute can be reconstructed. Synthetic data only — no real person or record. For education and defensive
        research. Local use only.
      </VulnerableBanner>

      <StepNav base="ml03" progress={[
        { label: "baseline", status: ids.baseline ? "done" : "pending", title: "victim utility + auxiliary-only attacker" },
        { label: "attack", status: ids.attack ? "done" : "pending", title: "inversion against full probability" },
        { label: "replay", status: "unsupported", title: "replay: not applicable — an output policy changes what the attacker observes, so every defense evaluation is a NEW adaptive inversion" },
        { label: "adaptive", status: defended.length ? "done" : "pending", title: "adaptive inversion against an output policy" },
      ]} />

      <ErrorBox error={error} />
      {busy && <div className="busy" role="status"><span className="spinner" /> Running: {busy}…</div>}

      {tab === "baseline" && (
        <>
        <div className="grid-2">
            <Panel title="Baseline evaluation" subtitle="Victim utility, attribute relevance and the auxiliary-only attacker (no queries)."
              actions={<button className="btn primary" disabled={!!busy}
                onClick={() => run("baseline", () => api.inversionBaseline(ctx.model_id, ctx.seed, targets),
                  (r) => setIds((s) => ({ ...s, baseline: r.experiment_id })))}>Run baseline</button>}>
              {baseline ? <BaselineResults rec={baseline} /> : <p className="muted">Evaluate {ctx.model_id} (seed {ctx.seed}) on the
                400 held-out records and run the auxiliary-only attacker on the first {ctx.max_targets} targets.</p>}
            </Panel>
            {baseline && (
              <Panel title="Provenance">
                <dl className="kv">
                  <dt>Victim SHA-256</dt><dd className="mono hash">{baseline.model.artifact_sha256}</dd>
                  <dt>Victim features</dt><dd className="mono">{baseline.model.features.join(", ")}</dd>
                  <dt>Attacker's auxiliary model</dt><dd className="mono hash">{baseline.auxiliary_model?.artifact_sha256}</dd>
                  <dt>Split</dt><dd className="mono">{baseline.dataset.split_id}</dd>
                  <dt>Train / aux / eval SHA-256</dt><dd className="mono hash">{baseline.dataset.train_sha256} / {baseline.dataset.auxiliary_sha256} / {baseline.dataset.test_sha256}</dd>
                </dl>
              </Panel>
            )}
        </div>
        <Panel title="Attacker's view of the targets" subtitle="Before evaluation: public features and the observed output only.">
          <TargetsPreview ctx={ctx} />
        </Panel>
        </>
      )}

      {tab === "attack" && (
        <>
          <div className="grid-2">
            <Panel title="Attack configuration" subtitle="Black-box candidate matching against the full-probability service.">
              <div className="form">
                <label>Query budget per target
                  <input type="number" min={1} max={16} value={attackCfg.query_budget}
                    onChange={(e) => { const v = parseInt(e.target.value, 10); if (v >= 1 && v <= 16) setAttackCfg({ ...attackCfg, query_budget: v }); }} />
                </label>
                <label>Ambiguous targets
                  <select value={attackCfg.tie_policy} onChange={(e) => setAttackCfg({ ...attackCfg, tie_policy: e.target.value as InversionAttackConfig["tie_policy"] })}>
                    <option value="auxiliary_prior">fall back to the auxiliary prior</option>
                    <option value="abstain">abstain (counted as not reconstructed)</option>
                  </select>
                </label>
              </div>
              <div className="row gap wrap mt">
                <button className="btn danger" disabled={!!busy}
                  onClick={() => run("model inversion", () => api.inversionAttack(ctx.model_id, ctx.seed, targets, attackCfg),
                    (r) => setIds((s) => ({ ...s, attack: r.experiment_id, policies: {} })))}>Run inversion attack</button>
              </div>
              <p className="muted small mt">The attacker submits the target's public features with s = 0 and s = 1 (the candidate
                its auxiliary prior prefers first), keeps the candidates whose response matches the observed output and stops
                when one remains. It never sees the attribute, the label, the model's parameters or the training records.</p>
            </Panel>
            <Panel title="Baseline vs inversion" subtitle={attack ? <ExpLink id={attack.experiment_id} /> : "not run"}>
              {attack ? <InversionSummary rec={attack} /> : <p className="muted">No inversion experiment yet.</p>}
            </Panel>
          </div>
          {attack?.status === "completed" && (
            <>
              <Panel title="Attack trace" subtitle="Which information the adversary had at each step."><TraceExplorer rec={attack} /></Panel>
              <Panel title="Per-target results" subtitle="Attacker outputs next to the evaluation-only ground truth."><TargetTable rec={attack} /></Panel>
            </>
          )}
        </>
      )}

      {tab === "defend" && (
        <>
          <div className="grid-2">
            <Panel title="Defense configuration" subtitle="Output policy of the prediction service.">
              <fieldset className="form">
                <legend className="sr-only">Output policy</legend>
                <label className="check"><input type="radio" name="policy" checked={policy.kind === "label_only"}
                  onChange={() => setPolicy({ kind: "label_only" })} /> Label only (decision 0/1)</label>
                <label className="check"><input type="radio" name="policy" checked={policy.kind === "quantized_probability"}
                  onChange={() => setPolicy({ kind: "quantized_probability", bins: 4 })} /> Quantised probability</label>
                {policy.kind === "quantized_probability" && (
                  <label>Buckets
                    <input type="number" min={2} max={1000} value={policy.bins}
                      onChange={(e) => { const v = parseInt(e.target.value, 10); if (v >= 2 && v <= 1000) setPolicy({ kind: "quantized_probability", bins: v }); }} />
                  </label>)}
              </fieldset>
              <div className="row gap wrap mt">
                <button className="btn good" disabled={!!busy || !attack || attack.status !== "completed"}
                  onClick={() => run("adaptive inversion", () => api.inversionDefend(attack!.experiment_id, policy),
                    (r) => setIds((s) => ({ ...s, policies: { ...s.policies, [policyKey(policy)]: r.experiment_id } })))}>
                  Run adaptive attack
                </button>
              </div>
              {!attack && <RequiresStep base="ml03" step="attack" what="an inversion attack" />}
              <div className="callout mt">
                <strong>Output restriction is not a formal privacy guarantee.</strong> The policy is applied to the observed
                output AND to every candidate query, and a NEW reconstruction runs against it with the same targets and
                budget as {attack ? <ExpLink id={attack.experiment_id} /> : "the attack"}. It only removes the information
                this attack uses; no differential-privacy mechanism is implemented.
              </div>
            </Panel>
            <Panel title="Defense comparison" subtitle={latest ? <ExpLink id={latest.experiment_id} /> : "not run for this policy"}>
              {latest ? <InversionSummary rec={latest} /> : <p className="muted">No adaptive inversion against {policyLabel(policy)} yet.</p>}
            </Panel>
          </div>
          <Panel title="Privacy–utility trade-off" subtitle="Full probability vs every output policy run in this session (same targets).">
            <PrivacyUtility rows={[...(attack ? [attack] : []), ...defended.map(([, r]) => r)]} />
          </Panel>
          {latest?.status === "completed" && (
            <>
              <Panel title={`Attack trace — ${policyLabel(latest.output_policy)}`}><TraceExplorer rec={latest} /></Panel>
              <Panel title="Per-target results" subtitle="Attacker outputs next to the evaluation-only ground truth."><TargetTable rec={latest} /></Panel>
            </>
          )}
        </>
      )}

      {tab === "evidence" && (
        <Panel title="Session evidence" subtitle="JSON (mlbreak.inversion/v1), Markdown report and CSV (one row per target) for every experiment.">
          <EvidenceTable disabled={!!busy} headline={{ label: "Attribute inference", key: "attribute_inference_accuracy" }}
            rows={[["baseline", baseline], ["attack (inversion)", attack],
              ...defended.map(([, r]) => [`adaptive · ${policyLabel(r.output_policy)}`, r] as [string, InversionRecord])]} />
          <p className="muted small mt">There is no non-adaptive replay in ML03: an output policy changes the observation itself,
            so each policy is evaluated with a new adaptive inversion (linked to its parent attack).</p>
        </Panel>
      )}
    </div>
  );
}

function BaselineResults({ rec }: { rec: InversionRecord }) {
  const ll = meanOf(rec, "log_loss_with_private_attribute"), llp = meanOf(rec, "log_loss_without_private_attribute");
  return (
    <>
      <div className="tiles">
        <Tile label="Victim decision accuracy"><RatioView r={ratioOf(rec, "victim_prediction_accuracy")} emphasize /></Tile>
        <Tile label="Without the attribute" hint="reference model trained on public features only">
          <RatioView r={ratioOf(rec, "accuracy_without_private_attribute")} /></Tile>
        <Tile label="Log-loss with / without">
          <span className="num">{ll?.value?.toFixed(3)} / {llp?.value?.toFixed(3)}</span></Tile>
        <Tile label="Auxiliary-only inference"><RatioView r={ratioOf(rec, "baseline_inference_accuracy")} emphasize /></Tile>
      </div>
      <p className="muted small">The attribute is relevant to the prediction (lower log-loss with it, weight{" "}
        {rec.model.private_attribute_weight?.toFixed(2) ?? "0 — negative control"}), which is why outputs can leak it. The
        auxiliary-only accuracy is the reference: an attacker reaches it without ever querying the service.</p>
      <div className="small muted">Experiment <ExpLink id={rec.experiment_id} /> · <StatusBadge status={rec.status} /></div>
    </>
  );
}

function TargetsPreview({ ctx }: { ctx: Ctx }) {
  const [view, setView] = useState<InversionTargetsView | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    setView(null); setErr(null);
    api.inversionTargets(ctx.model_id, ctx.seed, { kind: "full_probability" }, { max_targets: Math.min(ctx.max_targets, 25), order: "first" })
      .then((v) => { if (live) setView(v); }).catch((e) => { if (live) setErr(String(e.message ?? e)); });
    return () => { live = false; };
  }, [ctx.model_id, ctx.seed, ctx.max_targets]);
  if (err) return <ErrorBox error={err} />;
  if (!view) return <p className="muted">Loading…</p>;
  return (
    <>
      <div className="table-wrap scroll-y" tabIndex={0} role="region" aria-label="Attacker's view of the first targets">
        <table className="table small">
          <caption className="sr-only">Public features, hidden private attribute and observed output per target</caption>
          <thead><tr><th>target</th>{view.public_features.map((f) => <th key={f}>{f}</th>)}<th>{view.private_attribute.name}</th><th>observed output</th></tr></thead>
          <tbody>{view.targets.map((t) => (
            <tr key={t.target_index}><td>{t.target_index}</td>
              {t.public_features.map((v, j) => <td key={j} className="mono">{v.toFixed(3)}</td>)}
              <td><span className="hidden-attr" title="unknown to the attacker">🔒 hidden</span></td>
              <td className="mono">{fmtOut(t.observed_output, view.output_policy)}</td></tr>))}</tbody>
        </table>
      </div>
      <p className="muted small mt">Served by the attacker-facing endpoint: it never returns the private attribute, the label or
        the pre-policy score. The attribute values stay hidden until the evaluation engine scores a finished experiment.</p>
    </>
  );
}
