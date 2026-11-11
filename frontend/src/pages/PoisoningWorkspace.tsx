import { Fragment, useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { Link, NavLink, useParams } from "react-router-dom";
import { api, evidenceLinks, type Ctx } from "../api";
import { LabSwitcher } from "../components/LabSwitcher";
import { PoisonPlot } from "../components/PoisonPlot";
import { Badge, ErrorBox, EvalBadge, ExpLink, Panel, RatioView, StatusBadge, Tile, VulnerableBanner } from "../components/ui";
import type { BoundaryResponse, KnnDefense, LabelFlipConfig, ModelKey, PoisoningRecord, Ratio, ReplayResult,
  VerifyReport } from "../types";

type Tab = "baseline" | "attack" | "defend" | "evidence";
const TABS: { id: Tab; label: string }[] = [
  { id: "baseline", label: "1 · Clean training" },
  { id: "attack", label: "2 · Poison & retrain" },
  { id: "defend", label: "3 · Defend (replay)" },
  { id: "evidence", label: "4 · Evidence & replay" },
];

interface Session { baseline?: string; attack?: string; defend?: string }
const SESSION_KEY = "mlbreak.ml02.session";
const DEFAULT_POISON: LabelFlipConfig = { strategy: "targeted", fraction: 0.1, target_class: 1, selection: "boundary" };
const DEFAULT_DEFENSE: KnnDefense = { kind: "knn_label_sanitization", k: 10, agreement_threshold: 0.5, action: "relabel" };

function loadSession(): { ctx: Ctx; ids: Session } {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (raw) return JSON.parse(raw);
  } catch { /* storage unavailable */ }
  return { ctx: { model_id: "mlp-tanh-16x16", seed: 7 }, ids: {} };
}

/** Decision-boundary grid of one model of a stored ML02 experiment (server retrains it and checks its hash). */
export function useExperimentBoundary(id: string | undefined, model: ModelKey, enabled = true) {
  const [b, setB] = useState<BoundaryResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    setB(null); setErr(null);
    if (id && enabled) {
      api.experimentBoundary(id, model).then((r) => { if (live) setB(r); })
        .catch((e) => { if (live) setErr(String(e.message ?? e)); });
    }
    return () => { live = false; };
  }, [id, model, enabled]);
  return { boundary: b, error: err };
}

export function PoisoningWorkspace() {
  const { tab = "baseline" } = useParams<{ tab: Tab }>();
  const initial = useMemo(loadSession, []);
  const [ctx, setCtx] = useState<Ctx>(initial.ctx);
  const [ids, setIds] = useState<Session>(initial.ids);
  const [records, setRecords] = useState<Record<string, PoisoningRecord>>({});
  const [poison, setPoison] = useState<LabelFlipConfig>(DEFAULT_POISON);
  const [defense, setDefense] = useState<KnnDefense>(DEFAULT_DEFENSE);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [replays, setReplays] = useState<Record<string, ReplayResult>>({});
  const [verifies, setVerifies] = useState<Record<string, VerifyReport>>({});

  useEffect(() => {
    try { localStorage.setItem(SESSION_KEY, JSON.stringify({ ctx, ids })); } catch { /* ignore */ }
  }, [ctx, ids]);

  useEffect(() => {
    Object.values(ids).forEach((id) => {
      if (id && !records[id]) {
        api.anyExperiment(id).then((r) => {
          if ((r as PoisoningRecord).family === "poisoning") setRecords((m) => ({ ...m, [id]: r as PoisoningRecord }));
        }).catch(() => {});
      }
    });
  }, [ids]); // eslint-disable-line react-hooks/exhaustive-deps

  const run = useCallback(async (key: keyof Session, what: string, fn: () => Promise<PoisoningRecord>) => {
    setBusy(what); setError(null);
    try {
      const rec = await fn();
      setRecords((m) => ({ ...m, [rec.experiment_id]: rec }));
      setIds((s) => (key === "attack" ? { ...s, attack: rec.experiment_id, defend: undefined } : { ...s, [key]: rec.experiment_id }));
      if (rec.status !== "completed") setError(`Experiment ${rec.experiment_id} failed: ${rec.error}`);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(null);
    }
  }, []);

  const changeCtx = (c: Ctx) => { setCtx(c); setIds({}); setRecords({}); setReplays({}); setVerifies({}); };
  const rec = (k: keyof Session) => (ids[k] ? records[ids[k]!] : undefined);
  const baseline = rec("baseline"), attack = rec("attack"), defend = rec("defend");
  const domain = (attack ?? baseline)?.dataset.domain as number[][] | undefined;

  const act = async (id: string, kind: "replay" | "verify") => {
    setBusy(`${kind} ${id}`); setError(null);
    try {
      if (kind === "replay") { const r = await api.replay(id); setReplays((m) => ({ ...m, [id]: r })); }
      else { const v = await api.verify(id); setVerifies((m) => ({ ...m, [id]: v })); }
    } catch (e) { setError(String((e as Error).message ?? e)); } finally { setBusy(null); }
  };

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <div className="eyebrow">OWASP ML02:2023 · Data Poisoning Attack</div>
          <h1>Poisoning lab — label flipping</h1>
          <p className="muted">
            Flip the labels of part of the training set, retrain, and measure the damage; then replay the exact
            poisoned set through a label-validation defense. Every chart is built from a stored experiment: boundary
            plots come from models the server retrains from the record and checks against its artifact hash.
          </p>
        </div>
        <div className="ctx-controls">
          <LabSwitcher current="ml02" tab={tab} />
          <label>Model
            <select value={ctx.model_id} onChange={(e) => changeCtx({ ...ctx, model_id: e.target.value })}>
              <option value="mlp-tanh-16x16">MLP tanh 16×16</option>
              <option value="logreg">Logistic regression</option>
            </select>
          </label>
          <label>Seed
            <input type="number" min={0} value={ctx.seed}
              onChange={(e) => { const v = parseInt(e.target.value, 10); if (v >= 0) changeCtx({ ...ctx, seed: v }); }} />
          </label>
        </div>
      </div>
      <VulnerableBanner>
        The undefended training pipeline in this lab deliberately trusts its labels, so label-flipping poisoning succeeds.
        For education and defensive research only, on bundled synthetic data. Local use only.
      </VulnerableBanner>

      <nav className="tabs" aria-label="Lab steps">
        {TABS.map((t) => (
          <NavLink key={t.id} to={`/lab/ml02/${t.id}`} className={({ isActive }) => (isActive ? "tab active" : "tab")}>
            {t.label}
          </NavLink>
        ))}
        <div className="pipeline muted small">
          {(["baseline", "attack", "defend"] as (keyof Session)[]).map((k) => (
            <span key={k} className={ids[k] ? "pipe done" : "pipe"}>{ids[k] ? "✓" : "○"} {k}</span>
          ))}
        </div>
      </nav>

      <ErrorBox error={error} />
      {busy && <div className="busy" role="status"><span className="spinner" /> Running: {busy}…</div>}

      {tab === "baseline" && <BaselineTab ctx={ctx} baseline={baseline} busy={!!busy} domain={domain}
        onRun={() => run("baseline", "clean training", () => api.poisonBaseline(ctx))} />}

      {tab === "attack" && (
        <>
          <div className="grid-2">
            <Panel title="Poisoning configuration">
              <PoisonForm cfg={poison} onChange={setPoison} />
              <div className="row gap wrap mt">
                <button className="btn danger" disabled={!!busy}
                  onClick={() => run("attack", "label flipping + retraining", () => api.poisonAttack(ctx, poison))}>
                  Poison training set & retrain
                </button>
              </div>
            </Panel>
            <Panel title="Model performance: clean vs poisoned"
              subtitle={attack ? <ExpLink id={attack.experiment_id} /> : "run a poisoning experiment"}>
              {attack ? <MetricsCompare rec={attack} /> : <p className="muted">No poisoning experiment yet.</p>}
            </Panel>
          </div>
          {attack && attack.status === "completed" && <AttackPlots rec={attack} domain={domain} />}
          {attack?.attacker_capabilities && (
            <Panel title="Attacker capabilities & selection">
              <KV obj={attack.attacker_capabilities} />
              {attack.poisoning?.selection_rule && (
                <p className="muted small mt">Selection: {attack.poisoning.selection_rule}. Flipped {attack.poisoning.n_flips}{" "}
                  labels (budget {attack.poisoning.budget}, verified server-side).</p>
              )}
            </Panel>
          )}
        </>
      )}

      {tab === "defend" && (
        <>
          <div className="grid-2">
            <Panel title="Defense: kNN label sanitisation">
              <DefenseForm cfg={defense} onChange={setDefense} />
              <div className="row gap wrap mt">
                <button className="btn good" disabled={!!busy || !attack || attack.status !== "completed"}
                  onClick={() => run("defend", "defensive replay", () => api.poisonDefend(attack!.experiment_id, defense))}>
                  Replay poisoned set through the defense
                </button>
              </div>
              {!attack && <p className="muted small mt">Run a poisoning experiment in the{" "}
                <Link to="/lab/ml02/attack">Poison</Link> step first.</p>}
              <div className="callout mt">
                <strong>Replay ≠ robustness.</strong> The defense sees the <em>exact</em> poisoned labels of{" "}
                {attack ? <ExpLink id={attack.experiment_id} /> : "the poisoning experiment"}, and the server checks that the
                regenerated poisoned model reproduces its artifact hash. The poisoner did not know about the defense: try
                the <em>boundary</em> selection, whose flips sit where neighbours already agree with them.
              </div>
            </Panel>
            <Panel title="Defense results" subtitle={defend ? <ExpLink id={defend.experiment_id} /> : "not run"}>
              {defend ? <DefenseResults rec={defend} /> : <p className="muted">No defensive replay yet.</p>}
            </Panel>
          </div>
          {defend && defend.status === "completed" && <DefendPlots rec={defend} domain={domain} />}
          {defend && <Panel title="Clean vs poisoned vs defended"><MetricsCompare rec={defend} /></Panel>}
        </>
      )}

      {tab === "evidence" && (
        <Panel title="Session evidence" subtitle="JSON (mlbreak.poisoning/v1), Markdown and CSV (training + test rows).">
          <div className="table-wrap"><table className="table">
            <thead><tr><th>Step</th><th>Experiment</th><th>Type</th><th>Status</th><th>Attack success</th><th>Export</th>
              <th>Integrity</th><th>Exact replay</th></tr></thead>
            <tbody>
              {(["baseline", "attack", "defend"] as (keyof Session)[]).map((k) => {
                const r = rec(k);
                if (!r) return <tr key={k}><td>{k}</td><td colSpan={7} className="muted">not run</td></tr>;
                const l = evidenceLinks(r.experiment_id), rp = replays[r.experiment_id], v = verifies[r.experiment_id];
                return (
                  <tr key={k}>
                    <td>{k}</td><td><ExpLink id={r.experiment_id} /></td><td><EvalBadge evaluation={r.evaluation} /></td>
                    <td><StatusBadge status={r.status} /></td><td><RatioView r={r.metrics.attack_success_rate} /></td>
                    <td className="row gap"><a href={l.json} download>JSON</a><a href={l.md} download>MD</a><a href={l.csv} download>CSV</a></td>
                    <td><button className="btn small" disabled={!!busy} onClick={() => act(r.experiment_id, "verify")}>Verify</button>{" "}
                      {v && (v.ok ? <Badge tone="good">consistent</Badge> : <Badge tone="critical">failed</Badge>)}</td>
                    <td><button className="btn small" disabled={!!busy || r.status !== "completed"}
                      onClick={() => act(r.experiment_id, "replay")}>Replay</button>{" "}
                      {rp && (rp.identical ? <Badge tone="good">identical</Badge> : <Badge tone="critical">diverged</Badge>)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table></div>
          <p className="muted small mt">Verify recomputes the SHA-256 fingerprints and re-derives every metric from the
            stored rows. It proves internal consistency, <strong>not authenticity</strong>: anyone who can rewrite the file can
            recompute the hashes. Replay re-executes the experiment, and is refused if verification fails.</p>
          {Object.values(replays).map((rp) => (
            <details key={rp.original_experiment_id} className="mt">
              <summary>Replay checks for {rp.original_experiment_id} → {rp.replay_experiment_id}</summary>
              <table className="table small"><tbody>{rp.checks.map((c) => (
                <tr key={c.check}><td>{c.check}</td><td className="mono hash">{String(c.original)}</td>
                  <td>{c.match ? <Badge tone="good">match</Badge> : <Badge tone="critical">differs</Badge>}</td></tr>
              ))}</tbody></table>
            </details>
          ))}
        </Panel>
      )}
    </div>
  );
}

function BaselineTab({ ctx, baseline, busy, domain, onRun }: { ctx: Ctx; baseline?: PoisoningRecord; busy: boolean;
  domain?: number[][]; onRun: () => void }) {
  const { boundary, error } = useExperimentBoundary(baseline?.experiment_id, "clean", baseline?.status === "completed");
  return (
    <div className="grid-2">
      <Panel title="Original dataset & clean decision boundary"
        subtitle="Training points coloured by their TRUE label; boundary of the model trained on clean labels.">
        {baseline ? (
          <>
            <PoisonPlot boundary={boundary} rows={baseline.training_samples} labelMode="true" highlight="none"
              domain={domain} title="Clean training" />
            <ErrorBox error={error} />
          </>
        ) : <p className="muted">Run the clean training baseline to plot the dataset and the clean boundary.</p>}
      </Panel>
      <div className="stack">
        <Panel title="Clean training baseline" actions={
          <button className="btn primary" disabled={busy} onClick={onRun}>Train on clean labels</button>}>
          {baseline ? (
            <>
              <div className="tiles">
                <Tile label="Test accuracy"><RatioView r={baseline.metrics.test_accuracy} emphasize /></Tile>
                <Tile label="Train accuracy"><RatioView r={baseline.metrics.train_accuracy} /></Tile>
                <Tile label="Recall class 0"><RatioView r={baseline.metrics.recall_class_0 as Ratio} /></Tile>
                <Tile label="Recall class 1"><RatioView r={baseline.metrics.recall_class_1 as Ratio} /></Tile>
              </div>
              <div className="small muted">Experiment <ExpLink id={baseline.experiment_id} /> · <StatusBadge status={baseline.status} /></div>
            </>
          ) : <p className="muted">Train {ctx.model_id} (seed {ctx.seed}) on the clean labels and evaluate it on the clean test
            split. This is the reference every poisoning experiment is compared with.</p>}
        </Panel>
        {baseline && (
          <Panel title="Provenance">
            <dl className="kv">
              <dt>Estimator</dt><dd>{baseline.model.estimator}</dd>
              <dt>Clean model SHA-256</dt><dd className="mono hash">{baseline.model.artifact_sha256}</dd>
              <dt>Split</dt><dd className="mono">{baseline.dataset.split_id}</dd>
              <dt>Train split SHA-256</dt><dd className="mono hash">{baseline.dataset.train_sha256}</dd>
            </dl>
          </Panel>
        )}
      </div>
    </div>
  );
}

export function AttackPlots({ rec, domain }: { rec: PoisoningRecord; domain?: number[][] }) {
  const clean = useExperimentBoundary(rec.experiment_id, "clean");
  const pois = useExperimentBoundary(rec.experiment_id, "poisoned");
  return (
    <div className="grid-2">
      <Panel title="Clean decision boundary" subtitle="Model trained on the TRUE labels.">
        <PoisonPlot boundary={clean.boundary} rows={rec.training_samples} labelMode="true" highlight="poisoned"
          domain={domain} height={380} title={`clean-${rec.experiment_id}`} />
      </Panel>
      <Panel title="Poisoned decision boundary"
        subtitle="Model retrained on the flipped labels (points coloured by the label the learner saw). Dashed = clean boundary; ✕ = test samples broken by the poisoning.">
        <PoisonPlot boundary={pois.boundary} compare={clean.boundary} rows={rec.training_samples} labelMode="observed"
          highlight="poisoned" testRows={rec.test_samples} brokenKey="pred_poisoned" domain={domain} height={380}
          title={`poisoned-${rec.experiment_id}`} />
        <ErrorBox error={pois.error ?? clean.error} />
      </Panel>
    </div>
  );
}

export function DefendPlots({ rec, domain }: { rec: PoisoningRecord; domain?: number[][] }) {
  const clean = useExperimentBoundary(rec.experiment_id, "clean");
  const pois = useExperimentBoundary(rec.experiment_id, "poisoned");
  const def = useExperimentBoundary(rec.experiment_id, "defended");
  return (
    <div className="grid-2">
      <Panel title="Suspicious-label detection"
        subtitle="Poisoned training set over the poisoned boundary. ◇ caught · ✕ missed · △ clean label flagged (false positive).">
        <PoisonPlot boundary={pois.boundary} rows={rec.training_samples} labelMode="observed" highlight="detection"
          domain={domain} height={380} title={`detect-${rec.experiment_id}`} />
      </Panel>
      <Panel title="Defended decision boundary"
        subtitle="Model retrained on the sanitised labels. Dashed = clean boundary; ✕ = test samples still broken.">
        <PoisonPlot boundary={def.boundary} compare={clean.boundary} rows={rec.training_samples} labelMode="sanitized"
          highlight="none" testRows={rec.test_samples} brokenKey="pred_defended" domain={domain} height={380}
          title={`defended-${rec.experiment_id}`} />
        <ErrorBox error={def.error ?? pois.error ?? clean.error} />
      </Panel>
    </div>
  );
}

export function DefenseResults({ rec }: { rec: PoisoningRecord }) {
  const m = rec.metrics;
  return (
    <>
      <div className="tiles">
        <Tile label="Detection recall" hint={m.detection_recall?.definition}><RatioView r={m.detection_recall} emphasize /></Tile>
        <Tile label="Detection precision" hint={m.detection_precision?.definition}><RatioView r={m.detection_precision} /></Tile>
        <Tile label="False-positive rate" hint={m.false_positive_rate?.definition}><RatioView r={m.false_positive_rate} /></Tile>
        <Tile label="Recovery" hint={m.recovery_rate?.definition}><RatioView r={m.recovery_rate} /></Tile>
      </div>
      <table className="table">
        <thead><tr><th>Effect</th><th>Undefended</th><th>Defended</th></tr></thead>
        <tbody>
          <tr><td>Attack success rate</td><td><RatioView r={m.undefended_attack_success_rate} /></td>
            <td><RatioView r={m.attack_success_rate} emphasize /></td></tr>
          <tr><td>Test accuracy</td><td><RatioView r={m.undefended_test_accuracy} /></td><td><RatioView r={m.test_accuracy} /></td></tr>
          {m.target_class_recall && <tr><td>Target-class recall</td><td><RatioView r={m.undefended_target_class_recall} /></td>
            <td><RatioView r={m.target_class_recall} /></td></tr>}
        </tbody>
      </table>
      <p className="muted small mt">Cost with no attack: the same sanitisation on the CLEAN labels flags{" "}
        <RatioView r={m.clean_data_flag_rate} /> of training labels; resulting test accuracy{" "}
        <RatioView r={m.clean_data_test_accuracy} />.</p>
    </>
  );
}

export function MetricsCompare({ rec }: { rec: PoisoningRecord }) {
  const by = rec.metrics.by_model ?? {};
  const cols = (["clean", "poisoned", "defended", "clean_sanitized"] as ModelKey[]).filter((k) => by[k]);
  const label: Record<ModelKey, string> = { clean: "Clean", poisoned: "Poisoned", defended: "Defended",
    clean_sanitized: "Defense on clean data" };
  const tc = rec.metrics.target_class;
  const rows: [string, (k: ModelKey) => ReactNode][] = [
    ["Test accuracy", (k) => <RatioView r={by[k]!.test_accuracy} emphasize={k !== "clean"} />],
    ["Recall class 0", (k) => <RatioView r={by[k]!.recall_class_0} />],
    ["Recall class 1", (k) => <RatioView r={by[k]!.recall_class_1} />],
    ["Train accuracy (true labels)", (k) => <RatioView r={by[k]!.train_accuracy} />],
    ["Train accuracy (labels given)", (k) => <RatioView r={by[k]!.train_accuracy_observed} />],
  ];
  const m = rec.metrics;
  return (
    <div className="table-wrap">
      <table className="table">
        <thead><tr><th>Metric</th>{cols.map((k) => <th key={k}>{label[k]}</th>)}</tr></thead>
        <tbody>{rows.map(([n, f]) => <tr key={n}><td>{n}</td>{cols.map((k) => <td key={k}>{f(k)}</td>)}</tr>)}</tbody>
      </table>
      {rec.evaluation !== "clean_training" && (
        <dl className="kv mt">
          <dt>Poison rate</dt><dd><RatioView r={m.poison_rate} /></dd>
          {tc != null && <><dt>Target class</dt><dd>class {tc}</dd></>}
          <dt>Attack success rate</dt><dd><RatioView r={m.attack_success_rate} emphasize /></dd>
          {m.collateral_damage_rate && <><dt>Collateral (other class)</dt><dd><RatioView r={m.collateral_damage_rate} /></dd></>}
          {m.test_outcomes && <><dt>Test outcomes (clean → poisoned)</dt>
            <dd className="small">{Object.entries(m.test_outcomes.counts).map(([k, v]) => `${k} ${v}`).join(" · ")}</dd></>}
        </dl>
      )}
      <p className="muted small">Hover a ratio for its exact definition and population.</p>
    </div>
  );
}

function PoisonForm({ cfg, onChange }: { cfg: LabelFlipConfig; onChange: (c: LabelFlipConfig) => void }) {
  const targeted = cfg.strategy === "targeted";
  return (
    <div className="form">
      <label>Strategy
        <select value={cfg.strategy} onChange={(e) => onChange(e.target.value === "random"
          ? { ...cfg, strategy: "random", target_class: null, selection: "random" }
          : { ...cfg, strategy: "targeted", target_class: 1 })}>
          <option value="random">Random label flipping</option>
          <option value="targeted">Targeted (one class)</option>
        </select>
      </label>
      <label>Target class
        <select value={cfg.target_class ?? ""} disabled={!targeted}
          onChange={(e) => onChange({ ...cfg, target_class: parseInt(e.target.value, 10) as 0 | 1 })}>
          {!targeted && <option value="">— (untargeted)</option>}
          <option value={0}>class 0</option><option value={1}>class 1</option>
        </select>
      </label>
      <label className="span-2">Poisoned fraction of the training set = <span className="num">{(100 * cfg.fraction).toFixed(0)}%</span>
        {" "}<span className="muted">({Math.round(cfg.fraction * 350)} of 350 labels)</span>
        <input type="range" min={0.01} max={0.5} step={0.01} value={cfg.fraction}
          onChange={(e) => onChange({ ...cfg, fraction: parseFloat(e.target.value) })} />
      </label>
      <label className="span-2">Which target-class samples
        <select value={cfg.selection} disabled={!targeted}
          onChange={(e) => onChange({ ...cfg, selection: e.target.value as "random" | "boundary" })}>
          <option value="random">Random among the target class (knows labels only)</option>
          <option value="boundary">Nearest the clean boundary (knows the clean model's margins)</option>
        </select>
      </label>
      <p className="span-2 muted small">The attacker controls only training LABELS, before training. Features, test data,
        the learner and its hyper-parameters are untouched. Selection is seeded and reproducible.</p>
    </div>
  );
}

function DefenseForm({ cfg, onChange }: { cfg: KnnDefense; onChange: (c: KnnDefense) => void }) {
  return (
    <div className="form">
      <label>Neighbours k = <span className="num">{cfg.k}</span>
        <input type="range" min={3} max={25} step={1} value={cfg.k}
          onChange={(e) => onChange({ ...cfg, k: parseInt(e.target.value, 10) })} />
      </label>
      <label>Agreement threshold = <span className="num">{cfg.agreement_threshold.toFixed(2)}</span>
        <input type="range" min={0.1} max={0.9} step={0.05} value={cfg.agreement_threshold}
          onChange={(e) => onChange({ ...cfg, agreement_threshold: parseFloat(e.target.value) })} />
      </label>
      <label className="span-2">Action on suspicious labels
        <select value={cfg.action} onChange={(e) => onChange({ ...cfg, action: e.target.value as "relabel" | "remove" })}>
          <option value="relabel">Relabel to the neighbourhood majority</option>
          <option value="remove">Remove from the training set</option>
        </select>
      </label>
      <p className="span-2 muted small">A label is suspicious when fewer than {Math.round(cfg.agreement_threshold * 100)}% of its{" "}
        {cfg.k} nearest training neighbours share it (after Paudice et al., 2018).</p>
    </div>
  );
}

function KV({ obj }: { obj: Record<string, unknown> }) {
  return <dl className="kv">{Object.entries(obj).map(([k, v]) => (
    <Fragment key={k}><dt>{k.replace(/_/g, " ")}</dt><dd>{String(v)}</dd></Fragment>))}</dl>;
}
