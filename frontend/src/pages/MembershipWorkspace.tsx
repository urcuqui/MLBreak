import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api";
import { LabSwitcher } from "../components/LabSwitcher";
import { EvidenceTable, RequiresStep, StepNav, type StepId } from "../components/lab";
import { AttackerKnowledgePanel, CalibrationVsEvaluation, CandidateExplorer, ConfusionMatrices, DefenseComparison,
  MembershipSummary, OperationalPrecision, RocCurves, UtilityTiles, defenseLabel, mratio } from "../components/membership";
import { ErrorBox, ExpLink, Panel, RatioView, StatusBadge, Tile, VulnerableBanner } from "../components/ui";
import type { CandidateSelection, MembershipAttackConfig, MembershipDefense, MembershipModelId, MembershipRecord } from "../types";

interface Ctx { model_id: MembershipModelId; seed: number; n_members: number; n_non_members: number }
interface Session { baseline?: string; attack?: string; defenses: Record<string, string> }
const SESSION_KEY = "mlbreak.ml04.session";
const DEFAULT_ATTACK: MembershipAttackConfig = {
  label_known: true, threshold_signal: "true_label_confidence", threshold_mode: "global", min_class_calibration: 50,
  target_fpr: 0.01, n_shadow_models: 8, n_reference_models: 2, query_budget: 1200,
};
const DEFENSES: Record<string, MembershipDefense> = {
  regularization: { kind: "regularization", min_samples_leaf: 25, C: 0.01 },
  complexity_reduction: { kind: "complexity_reduction", max_depth: 4, n_estimators: 30 },
  quantized: { kind: "output_restriction", policy: { kind: "quantized_probability", bins: 4 } },
  label_only: { kind: "output_restriction", policy: { kind: "label_only" } },
};
const defenseKey = (d: MembershipDefense) => (d.kind === "output_restriction"
  ? (d.policy.kind === "label_only" ? "label_only" : `quantized_${d.policy.bins}`)
  : d.kind === "regularization" ? `regularization_${d.min_samples_leaf}_${d.C}` : `complexity_${d.max_depth}_${d.n_estimators}`);

function loadSession(): { ctx: Ctx; ids: Session } {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (raw) { const s = JSON.parse(raw); if (s?.ids?.defenses && s?.ctx?.model_id) return s; }
  } catch { /* storage unavailable */ }
  return { ctx: { model_id: "random-forest-overfit", seed: 7, n_members: 250, n_non_members: 250 }, ids: { defenses: {} } };
}

export function MembershipWorkspace() {
  const { tab = "baseline" } = useParams<{ tab: StepId }>();
  const initial = useMemo(loadSession, []);
  const [ctx, setCtx] = useState<Ctx>(initial.ctx);
  const [ids, setIds] = useState<Session>(initial.ids);
  const [records, setRecords] = useState<Record<string, MembershipRecord>>({});
  const [attackCfg, setAttackCfg] = useState<MembershipAttackConfig>(DEFAULT_ATTACK);
  const [defense, setDefense] = useState<MembershipDefense>(DEFENSES.regularization);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const candidates: CandidateSelection = { n_members: ctx.n_members, n_non_members: ctx.n_non_members, operational_prevalences: [0.5, 0.1, 0.01] };

  useEffect(() => {
    try { localStorage.setItem(SESSION_KEY, JSON.stringify({ ctx, ids })); } catch { /* ignore */ }
  }, [ctx, ids]);

  useEffect(() => {
    [ids.baseline, ids.attack, ...Object.values(ids.defenses)].forEach((id) => {
      if (id && !records[id]) {
        api.anyExperiment(id).then((r) => {
          if ((r as MembershipRecord).family === "membership") setRecords((m) => ({ ...m, [id]: r as MembershipRecord }));
        }).catch(() => {});
      }
    });
  }, [ids]); // eslint-disable-line react-hooks/exhaustive-deps

  const run = useCallback(async (what: string, fn: () => Promise<MembershipRecord>, store: (r: MembershipRecord) => void) => {
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

  const changeCtx = (c: Ctx) => { setCtx(c); setIds({ defenses: {} }); setRecords({}); };
  const baseline = ids.baseline ? records[ids.baseline] : undefined;
  const attack = ids.attack ? records[ids.attack] : undefined;
  const defended = Object.values(ids.defenses).map((id) => records[id]).filter(Boolean) as MembershipRecord[];
  const latest = ids.defenses[defenseKey(defense)] ? records[ids.defenses[defenseKey(defense)]] : undefined;
  const num = (v: string, lo: number, hi: number, f: (n: number) => void) => { const n = parseInt(v, 10); if (n >= lo && n <= hi) f(n); };

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <div className="eyebrow">OWASP ML04:2023 · Membership Inference Attack</div>
          <h1>Training Set Membership Inference</h1>
          <p className="muted">
            A fictional transaction-risk classifier releases prediction scores. Given a candidate record and the service's
            output for it, the attacker decides whether the record was in the training set — with a threshold calibrated on
            its own reference models and with shadow models trained on its own auxiliary data. Results are measured on
            untouched members and non-members whose ground truth only the evaluator holds.
          </p>
        </div>
        <div className="ctx-controls">
          <LabSwitcher current="ml04" tab={tab} />
          <label>Victim model
            <select value={ctx.model_id} onChange={(e) => changeCtx({ ...ctx, model_id: e.target.value as MembershipModelId })}>
              <option value="random-forest-overfit">Random forest (intentionally overfit)</option>
              <option value="logreg-l2">Logistic regression (L2, lower capacity)</option>
              <option value="random-forest-decoy">Negative control (no candidate is a member)</option>
            </select>
          </label>
          <label>Seed
            <input type="number" min={0} max={100000} value={ctx.seed}
              onChange={(e) => num(e.target.value, 0, 100000, (v) => changeCtx({ ...ctx, seed: v }))} />
          </label>
          <label>Members / non-members
            <span className="row gap">
              <input type="number" min={1} max={600} value={ctx.n_members} aria-label="Members"
                onChange={(e) => num(e.target.value, 1, 600, (v) => changeCtx({ ...ctx, n_members: v }))} />
              <input type="number" min={1} max={600} value={ctx.n_non_members} aria-label="Non-members"
                onChange={(e) => num(e.target.value, 1, 600, (v) => changeCtx({ ...ctx, n_non_members: v }))} />
            </span>
          </label>
        </div>
      </div>
      <VulnerableBanner>
        The undefended forest memorises its training records, so its outputs reveal who was in the training set. Synthetic
        data only — no real person or record. For education and defensive research. Local use only. Experiment records are
        evaluator evidence: they contain the membership ground truth.
      </VulnerableBanner>

      <StepNav base="ml04" progress={[
        { label: "baseline", status: ids.baseline ? "done" : "pending", title: "victim utility and the no-signal baseline" },
        { label: "attack", status: ids.attack ? "done" : "pending", title: "threshold + shadow-model membership inference" },
        { label: "replay", status: "unsupported", title: "replay: not applicable — every defense is evaluated by a RECALIBRATED attack" },
        { label: "adaptive", status: defended.length ? "done" : "pending", title: "recalibrated attack against a defense" },
      ]} />

      <ErrorBox error={error} />
      {busy && <div className="busy" role="status"><span className="spinner" /> Running: {busy}…</div>}

      {tab === "baseline" && (
        <div className="grid-2">
          <Panel title="Baseline evaluation" subtitle="Training vs held-out performance; no queries."
            actions={<button className="btn primary" disabled={!!busy}
              onClick={() => run("baseline", () => api.membershipBaseline(ctx.model_id, ctx.seed, candidates),
                (r) => setIds((s) => ({ ...s, baseline: r.experiment_id })))}>Run baseline</button>}>
            {baseline ? <BaselineResults rec={baseline} /> : <p className="muted">Train {ctx.model_id} (seed {ctx.seed}) and measure it
              on its 600 training records and 600 held-out records.</p>}
          </Panel>
          {baseline && (
            <Panel title="Victim configuration and populations">
              <dl className="kv">
                <dt>Model</dt><dd>{baseline.model.description}</dd>
                <dt>Recipe</dt><dd className="mono">{JSON.stringify(baseline.model.hyperparameters)}</dd>
                <dt>Victim SHA-256</dt><dd className="mono hash">{baseline.model.artifact_sha256}</dd>
                <dt>Trained on</dt><dd className="mono">{baseline.model.training_population}</dd>
                <dt>Populations</dt><dd className="mono">{Object.entries(baseline.dataset.population_sizes).map(([k, n]) => `${k}=${n}`).join(", ")}</dd>
                <dt>Disjoint</dt><dd>{baseline.dataset.disjointness.disjoint ? "yes — no record shared or duplicated (row-level check)" : "NO"}</dd>
              </dl>
            </Panel>
          )}
        </div>
      )}

      {tab === "attack" && (
        <>
          <div className="grid-2">
            <Panel title="Attack configuration" subtitle="Both attacks run on the same candidates and calibration data.">
              <div className="form">
                <label className="check"><input type="checkbox" checked={attackCfg.label_known}
                  onChange={(e) => setAttackCfg({ ...attackCfg, label_known: e.target.checked,
                    threshold_signal: e.target.checked ? "true_label_confidence" : "max_confidence" })} /> Attacker knows each candidate's label</label>
                <label>Threshold signal
                  <select value={attackCfg.threshold_signal} onChange={(e) => setAttackCfg({ ...attackCfg, threshold_signal: e.target.value as MembershipAttackConfig["threshold_signal"] })}>
                    <option value="true_label_confidence" disabled={!attackCfg.label_known}>true-label confidence</option>
                    <option value="negative_log_likelihood" disabled={!attackCfg.label_known}>negative log-likelihood (loss)</option>
                    <option value="max_confidence">maximum confidence (label unknown)</option>
                  </select>
                </label>
                <label>Thresholds
                  <select value={attackCfg.threshold_mode} onChange={(e) => setAttackCfg({ ...attackCfg, threshold_mode: e.target.value as MembershipAttackConfig["threshold_mode"] })}>
                    <option value="global">global</option><option value="class_conditional">class-conditional (if enough calibration data)</option>
                  </select>
                </label>
                <label>Target FPR (low-FPR operating point)
                  <select value={attackCfg.target_fpr} onChange={(e) => setAttackCfg({ ...attackCfg, target_fpr: parseFloat(e.target.value) })}>
                    {[0.001, 0.01, 0.05, 0.1].map((f) => <option key={f} value={f}>{f * 100}%</option>)}
                  </select>
                </label>
                <label>Shadow models
                  <input type="number" min={1} max={16} value={attackCfg.n_shadow_models}
                    onChange={(e) => num(e.target.value, 1, 16, (v) => setAttackCfg({ ...attackCfg, n_shadow_models: v }))} />
                </label>
                <label>Reference models (threshold calibration)
                  <input type="number" min={1} max={4} value={attackCfg.n_reference_models}
                    onChange={(e) => num(e.target.value, 1, 4, (v) => setAttackCfg({ ...attackCfg, n_reference_models: v }))} />
                </label>
              </div>
              <div className="row gap wrap mt">
                <button className="btn danger" disabled={!!busy}
                  onClick={() => run("membership inference", () => api.membershipAttack(ctx.model_id, ctx.seed, candidates, attackCfg),
                    (r) => setIds((s) => ({ ...s, attack: r.experiment_id, defenses: {} })))}>Run membership inference</button>
              </div>
              <p className="muted small mt">One query per candidate ({ctx.n_members + ctx.n_non_members} of {attackCfg.query_budget} allowed).
                Reference and shadow models are trained locally on the attacker's own populations; they never query the victim and
                never see a victim membership flag.</p>
            </Panel>
            <Panel title="Attacker knowledge and output" subtitle={attack ? <ExpLink id={attack.experiment_id} /> : "not run"}>
              {attack ? <AttackerKnowledgePanel rec={attack} /> : <p className="muted">No membership-inference experiment yet.</p>}
            </Panel>
          </div>
          {attack?.status === "completed" && (
            <>
              <Panel title="Final evaluation" subtitle="Untouched candidates; thresholds from calibration only."><MembershipSummary rec={attack} /></Panel>
              <div className="grid-2">
                <Panel title="ROC curves (evaluation population)"><RocCurves rec={attack} /></Panel>
                <Panel title="Confusion matrices"><ConfusionMatrices rec={attack} /></Panel>
              </div>
              <div className="grid-2">
                <Panel title="Calibration vs evaluation" subtitle="TPR/FPR operating points"><CalibrationVsEvaluation rec={attack} /></Panel>
                <Panel title="Operational precision" subtitle="When members are rare"><OperationalPrecision rec={attack} /></Panel>
              </div>
              <Panel title="Membership predictions" subtitle="Attacker view first; ground truth only in the evaluator view."><CandidateExplorer rec={attack} /></Panel>
            </>
          )}
        </>
      )}

      {tab === "defend" && (
        <>
          <div className="grid-2">
            <Panel title="Defense configuration" subtitle="The attack is recalibrated against every defense.">
              <fieldset className="form">
                <legend className="sr-only">Defense</legend>
                {Object.entries(DEFENSES).map(([k, d]) => (
                  <label key={k} className="check"><input type="radio" name="ml04-defense" checked={defense.kind === d.kind &&
                    (d.kind !== "output_restriction" || (defense.kind === "output_restriction" && defense.policy.kind === d.policy.kind))}
                    onChange={() => setDefense(d)} /> {defenseLabel(d)}</label>))}
                {defense.kind === "regularization" && (
                  <label>Minimum records per leaf (trees)
                    <input type="number" min={2} max={200} value={defense.min_samples_leaf}
                      onChange={(e) => num(e.target.value, 2, 200, (v) => setDefense({ ...defense, min_samples_leaf: v }))} /></label>)}
                {defense.kind === "complexity_reduction" && (
                  <span className="row gap">
                    <label>Max depth<input type="number" min={1} max={32} value={defense.max_depth}
                      onChange={(e) => num(e.target.value, 1, 32, (v) => setDefense({ ...defense, max_depth: v }))} /></label>
                    <label>Trees<input type="number" min={1} max={100} value={defense.n_estimators}
                      onChange={(e) => num(e.target.value, 1, 100, (v) => setDefense({ ...defense, n_estimators: v }))} /></label>
                  </span>)}
                {defense.kind === "output_restriction" && defense.policy.kind === "quantized_probability" && (
                  <label>Buckets<input type="number" min={2} max={1000} value={defense.policy.bins}
                    onChange={(e) => num(e.target.value, 2, 1000, (v) => setDefense({ kind: "output_restriction", policy: { kind: "quantized_probability", bins: v } }))} /></label>)}
              </fieldset>
              <div className="row gap wrap mt">
                <button className="btn good" disabled={!!busy || !attack || attack.status !== "completed"}
                  onClick={() => run("recalibrated attack", () => api.membershipDefend(attack!.experiment_id, defense),
                    (r) => setIds((s) => ({ ...s, defenses: { ...s.defenses, [defenseKey(defense)]: r.experiment_id } })))}>
                  Run adaptive attack
                </button>
              </div>
              {!attack && <RequiresStep base="ml04" step="attack" what="a membership-inference attack" />}
              <div className="callout mt">
                <strong>Output restriction is not differential privacy.</strong> Every defense here is evaluated by a NEW attack
                whose reference and shadow models are retrained with the defended recipe and see outputs through the same policy,
                on the same candidates as {attack ? <ExpLink id={attack.experiment_id} /> : "the attack"}. A deterministic release
                has no privacy bound (ε = ∞); label-only output still leaks membership through correctness.
              </div>
            </Panel>
            <Panel title="Defended result" subtitle={latest ? <ExpLink id={latest.experiment_id} /> : "not run for this defense"}>
              {latest ? <><UtilityTiles rec={latest} /><MembershipSummary rec={latest} /></> : <p className="muted">No recalibrated attack against {defenseLabel(defense)} yet.</p>}
            </Panel>
          </div>
          <Panel title="Defense comparison and privacy–utility trade-off" subtitle="Undefended vs every defense run in this session (same candidates).">
            <DefenseComparison rows={[...(attack ? [attack] : []), ...defended]} />
          </Panel>
          {latest?.status === "completed" && (
            <div className="grid-2">
              <Panel title={`ROC curves — ${defenseLabel(latest.defense_config)}`}><RocCurves rec={latest} /></Panel>
              <Panel title="Confusion matrices"><ConfusionMatrices rec={latest} /></Panel>
            </div>
          )}
        </>
      )}

      {tab === "evidence" && (
        <Panel title="Session evidence" subtitle="JSON (mlbreak.membership/v1), Markdown report and CSV (one row per candidate) for every experiment.">
          <EvidenceTable disabled={!!busy} headline={{ label: "Membership acc. (threshold)", key: "threshold_membership_accuracy" }}
            rows={[["baseline", baseline], ["attack (membership inference)", attack],
              ...defended.map((r) => [`adaptive · ${defenseLabel(r.defense_config)}`, r] as [string, MembershipRecord])]} />
          <p className="muted small mt">Records are evaluator evidence and include the membership ground truth. There is no
            non-adaptive replay in ML04: a defense changes the model or its outputs, so each one is evaluated with a recalibrated
            attack (linked to its parent).</p>
        </Panel>
      )}
    </div>
  );
}

function BaselineResults({ rec }: { rec: MembershipRecord }) {
  return (
    <>
      <UtilityTiles rec={rec} />
      <div className="tiles">
        <Tile label="No-signal baseline" hint="always guessing the larger group"><RatioView r={mratio(rec, "no_signal_accuracy")} /></Tile>
        <Tile label="Evaluation candidates"><span className="num">{rec.candidate_selection.n_members} members · {rec.candidate_selection.n_non_members} non-members</span></Tile>
      </div>
      <p className="muted small">A large gap means the model behaves differently on its training records — the signal membership
        inference exploits. It is not itself a membership result: run the attack to measure leakage.</p>
      <div className="small muted">Experiment <ExpLink id={rec.experiment_id} /> · <StatusBadge status={rec.status} /></div>
    </>
  );
}
