import { Fragment, useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { Link, NavLink, useParams } from "react-router-dom";
import { api, evidenceLinks, type Ctx } from "../api";
import { BoundaryPlot, CLASS_COLORS } from "../components/BoundaryPlot";
import { LabSwitcher, ScenarioSwitcher } from "../components/LabSwitcher";
import { Badge, ErrorBox, EvalBadge, ExpLink, Panel, RatioView, StatusBadge, Tile, VulnerableBanner } from "../components/ui";
import type { AttackConfig, BoundaryResponse, DatasetResponse, Defense, ExperimentRecord, ReplayResult, SampleResult } from "../types";

type Tab = "baseline" | "attack" | "defend" | "evidence";
const TABS: { id: Tab; label: string }[] = [
  { id: "baseline", label: "1 · Baseline" },
  { id: "attack", label: "2 · Attack" },
  { id: "defend", label: "3 · Defend" },
  { id: "evidence", label: "4 · Evidence & replay" },
];

interface Session {
  baseline?: string; single?: string; batch?: string; replay?: string; adaptive?: string;
}
const SESSION_KEY = "mlbreak.ml01.session";

const DEFAULT_ATTACK: AttackConfig = {
  attack: "pgd", norm: "linf", epsilon: 0.3, steps: 40, step_size: null, confidence: 0,
  random_start: false, query_budget: 500,
};
const DEFAULT_DEFENSES: Record<Defense["kind"], Defense> = {
  adversarial_training: { kind: "adversarial_training", epsilon: 0.2, norm: "linf", steps: 20, rounds: 3 },
  confidence_rejection: { kind: "confidence_rejection", threshold: 0.8 },
};

function loadSession(): { ctx: Ctx; ids: Session } {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (raw) return JSON.parse(raw);
  } catch { /* storage unavailable */ }
  return { ctx: { model_id: "mlp-tanh-16x16", seed: 7 }, ids: {} };
}

function saveSession(s: { ctx: Ctx; ids: Session }) {
  try { localStorage.setItem(SESSION_KEY, JSON.stringify(s)); } catch { /* ignore */ }
}

function ClassDot({ y }: { y: number | null | undefined }) {
  if (y == null) return <span className="muted">—</span>;
  if (y === -1) return <Badge tone="warning">abstain</Badge>;
  return <span><span className="dot" style={{ background: CLASS_COLORS[y] }} /> class {y}</span>;
}

export function Workspace() {
  const { tab = "baseline" } = useParams<{ tab: Tab }>();
  const initial = useMemo(loadSession, []);
  const [ctx, setCtx] = useState<Ctx>(initial.ctx);
  const [ids, setIds] = useState<Session>(initial.ids);
  const [records, setRecords] = useState<Record<string, ExperimentRecord>>({});
  const [dataset, setDataset] = useState<DatasetResponse | null>(null);
  const [boundary, setBoundary] = useState<BoundaryResponse | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [attack, setAttack] = useState<AttackConfig>(DEFAULT_ATTACK);
  const [defense, setDefense] = useState<Defense>(DEFAULT_DEFENSES.adversarial_training);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [replays, setReplays] = useState<Record<string, ReplayResult>>({});

  useEffect(() => saveSession({ ctx, ids }), [ctx, ids]);

  // Dataset + undefended boundary for the current model/seed.
  useEffect(() => {
    setDataset(null); setBoundary(null); setError(null);
    Promise.all([api.dataset(ctx), api.boundary(ctx, null)])
      .then(([d, b]) => {
        setDataset(d); setBoundary(b);
        setSelected((cur) => {
          if (cur != null && cur < d.test.X.length) return cur;
          // Default selection: the least confident correctly-classified test sample.
          let best = 0, bestConf = 2;
          d.test.p1.forEach((p, i) => {
            const conf = Math.max(p, 1 - p);
            if (d.test.pred[i] === d.test.y[i] && conf < bestConf && conf > 0.6) { best = i; bestConf = conf; }
          });
          return best;
        });
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, [ctx]);

  // Reload session records stored in localStorage.
  useEffect(() => {
    Object.values(ids).forEach((id) => {
      if (id && !records[id]) api.experiment(id).then((r) => setRecords((m) => ({ ...m, [id]: r }))).catch(() => {});
    });
  }, [ids]); // eslint-disable-line react-hooks/exhaustive-deps


  const run = useCallback(async (key: keyof Session, what: string, fn: () => Promise<ExperimentRecord>) => {
    setBusy(what); setError(null);
    try {
      const rec = await fn();
      setRecords((m) => ({ ...m, [rec.experiment_id]: rec }));
      setIds((s) => ({ ...s, [key]: rec.experiment_id }));
      if (rec.status !== "completed") setError(`Experiment ${rec.experiment_id} failed: ${rec.error}`);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(null);
    }
  }, []);

  const changeCtx = (c: Ctx) => { setCtx(c); setIds({}); setRecords({}); setReplays({}); setSelected(null); };

  const rec = (k: keyof Session) => (ids[k] ? records[ids[k]!] : undefined);
  const baseline = rec("baseline"), single = rec("single"), batch = rec("batch");
  const replay = rec("replay"), adaptive = rec("adaptive");

  const selectedSample: SampleResult | undefined =
    single?.samples.find((s) => s.test_index === selected) ?? batch?.samples.find((s) => s.test_index === selected);

  const doReplay = async (id: string) => {
    setBusy(`replay ${id}`); setError(null);
    try { const r = await api.replay(id); setReplays((m) => ({ ...m, [id]: r })); }
    catch (e) { setError(String((e as Error).message ?? e)); }
    finally { setBusy(null); }
  };

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <div className="eyebrow">OWASP ML01:2023 · Input Manipulation Attack</div>
          <h1>Evasion lab — two-moons classifier</h1>
          <p className="muted">
            Perturb a correctly classified input within an ε-ball so the deployed system returns the wrong
            class. Every number below comes from an executed experiment with stored evidence.
          </p>
        </div>
        <div className="ctx-controls">
          <LabSwitcher current="ml01" tab={tab} path="ml01" />
          <ScenarioSwitcher current="ml01" tab={tab} />
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
      <VulnerableBanner />

      <nav className="tabs" aria-label="Lab steps">
        {TABS.map((t) => (
          <NavLink key={t.id} to={`/lab/ml01/${t.id}`} className={({ isActive }) => (isActive ? "tab active" : "tab")}>
            {t.label}
          </NavLink>
        ))}
        <div className="pipeline muted small">
          {(["baseline", "batch", "replay", "adaptive"] as (keyof Session)[]).map((k) => (
            <span key={k} className={ids[k] ? "pipe done" : "pipe"}>{ids[k] ? "✓" : "○"} {k}</span>
          ))}
        </div>
      </nav>

      <ErrorBox error={error} />
      {busy && <div className="busy" role="status"><span className="spinner" /> Running: {busy}…</div>}

      {tab === "baseline" && (
        <div className="grid-2">
          <Panel title="Dataset & decision boundary" subtitle="Click a test point to select it. Heatmap = model P(class 1); white line = decision boundary.">
            <BoundaryPlot boundary={boundary} dataset={dataset} selected={selected} onSelect={setSelected}
              title="Undefended decision boundary" />
          </Panel>
          <div className="stack">
            <Panel title="Baseline evaluation" actions={
              <button className="btn primary" disabled={!!busy}
                onClick={() => run("baseline", "baseline evaluation", () => api.baseline(ctx))}>
                Run baseline
              </button>}>
              {baseline ? (
                <>
                  <div className="tiles">
                    <Tile label="Clean test accuracy"><RatioView r={baseline.metrics.clean_accuracy} emphasize /></Tile>
                    <Tile label="Train accuracy"><RatioView r={baseline.metrics.train_accuracy} /></Tile>
                  </div>
                  <div className="small muted">Experiment <ExpLink id={baseline.experiment_id} /> · <StatusBadge status={baseline.status} /></div>
                </>
              ) : <p className="muted">Evaluate the unmodified model on the clean test split. Results are stored as evidence.</p>}
            </Panel>
            <Panel title="Model & data provenance">
              {dataset ? (
                <dl className="kv">
                  <dt>Estimator</dt><dd>{dataset.model.estimator}</dd>
                  <dt>Artifact SHA-256</dt><dd className="mono hash">{dataset.model.artifact_sha256}</dd>
                  <dt>Optimizer iterations</dt><dd>{dataset.model.n_iter}</dd>
                  <dt>Split</dt><dd className="mono">{dataset.dataset.split_id}</dd>
                  <dt>Test split SHA-256</dt><dd className="mono hash">{String(dataset.dataset.test_sha256)}</dd>
                  <dt>Train / test</dt><dd>{String(dataset.dataset.n_train)} / {dataset.dataset.n_test}</dd>
                </dl>
              ) : <p className="muted">Loading…</p>}
            </Panel>
            <SampleInspector dataset={dataset} selected={selected} />
          </div>
        </div>
      )}

      {tab === "attack" && (
        <div className="grid-2">
          <Panel title="Original vs adversarial input"
            subtitle="Dotted box/circle = the ε-ball around the selected sample. ✕ = evaded, ○ = attack failed.">
            <BoundaryPlot boundary={boundary} dataset={dataset} selected={selected} onSelect={setSelected}
              epsilon={attack.epsilon} norm={attack.norm} samples={(batch ?? single)?.samples}
              showTrain={false} title="Attack view" />
          </Panel>
          <div className="stack">
            <Panel title="Attack configuration">
              <AttackForm cfg={attack} onChange={setAttack} />
              <div className="row gap wrap mt">
                <button className="btn danger" disabled={!!busy || selected == null}
                  onClick={() => run("single", `attack on sample #${selected}`, () => api.attack(ctx, attack, [selected!]))}>
                  Attack sample #{selected ?? "—"}
                </button>
                <button className="btn danger outline" disabled={!!busy}
                  onClick={() => run("batch", "attack on full test split", () => api.attack(ctx, attack, null))}>
                  Attack full test split
                </button>
              </div>
            </Panel>
            <Panel title={`Selected sample #${selected ?? "—"}`}>
              {selectedSample?.x_adversarial ? (
                <table className="table">
                  <thead><tr><th></th><th>Original</th><th>Adversarial</th></tr></thead>
                  <tbody>
                    <tr><td>Input (x1, x2)</td>
                      <td className="mono">({selectedSample.x_original.map((v) => v.toFixed(4)).join(", ")})</td>
                      <td className="mono">({selectedSample.x_adversarial.map((v) => v.toFixed(4)).join(", ")})</td></tr>
                    <tr><td>Prediction</td><td><ClassDot y={selectedSample.pred_original} /></td>
                      <td><ClassDot y={selectedSample.pred_adversarial} /></td></tr>
                    <tr><td>P(class 1)</td><td className="num">{selectedSample.p1_original.toFixed(4)}</td>
                      <td className="num">{selectedSample.p1_adversarial?.toFixed(4)}</td></tr>
                    <tr><td>Confidence</td><td className="num">{selectedSample.confidence_original.toFixed(4)}</td>
                      <td className="num">{selectedSample.confidence_adversarial?.toFixed(4)}</td></tr>
                    <tr><td>‖δ‖∞ / ‖δ‖₂</td><td colSpan={2} className="num">
                      {selectedSample.perturbation_linf?.toFixed(4)} / {selectedSample.perturbation_l2?.toFixed(4)}</td></tr>
                    <tr><td>True label</td><td colSpan={2}><ClassDot y={selectedSample.y_true} /></td></tr>
                    <tr><td>Outcome</td><td colSpan={2}>{selectedSample.attack_success
                      ? <Badge tone="critical">evaded — prediction changed</Badge>
                      : <Badge tone="good">attack failed within budget</Badge>}</td></tr>
                  </tbody>
                </table>
              ) : selectedSample && !selectedSample.clean_correct ? (
                <p className="muted">This sample is already misclassified on its clean input, so it is not attacked.</p>
              ) : <p className="muted">Run an attack on this sample (or the full split) to see the comparison.</p>}
            </Panel>
            <Panel title="Baseline vs attack">
              <MetricsCompare cols={[
                ["Baseline", baseline], ["Attack (selected)", single], ["Attack (test split)", batch]]} />
              {(batch ?? single)?.attacker_capabilities && (
                <CapabilityBox caps={(batch ?? single)!.attacker_capabilities!} />
              )}
            </Panel>
          </div>
        </div>
      )}

      {tab === "defend" && (
        <>
          <div className="grid-2">
            <Panel title="Defense configuration">
              <DefenseForm cfg={defense} onChange={setDefense} />
              <div className="row gap wrap mt">
                <button className="btn good" disabled={!!busy || !batch || batch.status !== "completed"}
                  onClick={() => run("replay", "defense replay", () => api.defend(batch!.experiment_id, defense))}>
                  Evaluate defense (replay perturbations)
                </button>
                <button className="btn warning" disabled={!!busy}
                  onClick={() => run("adaptive", "adaptive attack", () =>
                    api.attack(ctx, attack, null, defense, batch?.experiment_id ?? null))}>
                  Adaptive attack (new experiment)
                </button>
              </div>
              {!batch && <p className="muted small mt">Replay needs a test-split attack: run one in the <Link to="/lab/ml01/attack">Attack</Link> step first.</p>}
              <div className="callout mt">
                <strong>Replay ≠ robustness.</strong> <em>Replay</em> re-submits the exact perturbations from{" "}
                {batch ? <ExpLink id={batch.experiment_id} /> : "the attack experiment"} — crafted without knowledge of the
                defense. The <em>adaptive attack</em> is a separate experiment: the same attack procedure ({attack.attack},{" "}
                {attack.norm}, ε={attack.epsilon}) regenerated against the defended system by an attacker who knows the defense
                {defense.kind === "confidence_rejection" ? " (it targets confidence above τ)" : " (it uses the defended model's gradients)"}.
              </div>
            </Panel>
            <Panel title="Defense comparison">
              <MetricsCompare cols={[["Undefended attack", batch], ["Defended · replay", replay], ["Defended · adaptive", adaptive]]} />
              {replay?.metrics.notes && <ul className="notes">{replay.metrics.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
            </Panel>
          </div>
          <div className="grid-3">
            <Panel title="Undefended" subtitle={batch ? <>perturbations from <ExpLink id={batch.experiment_id} /></> : "no attack yet"}>
              <BoundaryPlot boundary={boundary} dataset={dataset} samples={batch?.samples} showTrain={false} height={360} title="Undefended" />
            </Panel>
            <Panel title="Defended · replay" subtitle={replay ? <ExpLink id={replay.experiment_id} /> : "not run"}>
              <DefendedPlot ctx={ctx} defense={replay?.defense_config ?? defense} dataset={dataset}
                samples={replay?.samples} title="Defended replay" />
            </Panel>
            <Panel title="Defended · adaptive" subtitle={adaptive ? <ExpLink id={adaptive.experiment_id} /> : "not run"}>
              <DefendedPlot ctx={ctx} defense={adaptive?.defense_config ?? defense} dataset={dataset}
                samples={adaptive?.samples} title="Defended adaptive" />
            </Panel>
          </div>
          <p className="muted small">Plots of stored experiments use the defense that experiment ran with; empty plots
            preview the defense currently configured.</p>
        </>
      )}

      {tab === "evidence" && (
        <Panel title="Session evidence" subtitle="Each experiment stores machine-readable JSON, a Markdown report and per-sample CSV.">
          <table className="table">
            <thead><tr><th>Step</th><th>Experiment</th><th>Type</th><th>Status</th><th>Attack success</th><th>Export</th><th>Exact replay</th></tr></thead>
            <tbody>
              {(["baseline", "single", "batch", "replay", "adaptive"] as (keyof Session)[]).map((k) => {
                const r = rec(k);
                if (!r) return <tr key={k}><td>{k}</td><td colSpan={6} className="muted">not run</td></tr>;
                const links = evidenceLinks(r.experiment_id);
                const rp = replays[r.experiment_id];
                return (
                  <tr key={k}>
                    <td>{k}</td><td><ExpLink id={r.experiment_id} /></td><td><EvalBadge evaluation={r.evaluation} /></td>
                    <td><StatusBadge status={r.status} /></td>
                    <td><RatioView r={r.metrics.attack_success_rate} /></td>
                    <td className="row gap"><a href={links.json} download>JSON</a><a href={links.md} download>MD</a><a href={links.csv} download>CSV</a></td>
                    <td>
                      <button className="btn small" disabled={!!busy || r.status !== "completed"} onClick={() => doReplay(r.experiment_id)}>Replay</button>{" "}
                      {rp && (rp.identical ? <Badge tone="good">identical</Badge> : <Badge tone="critical">diverged</Badge>)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {Object.values(replays).map((rp) => (
            <details key={rp.original_experiment_id} className="mt">
              <summary>Replay checks for {rp.original_experiment_id} → {rp.replay_experiment_id}</summary>
              <table className="table small">
                <thead><tr><th>Check</th><th>Original</th><th>Replay</th><th></th></tr></thead>
                <tbody>{rp.checks.map((c) => (
                  <tr key={c.check}><td>{c.check}</td><td className="mono hash">{String(c.original)}</td>
                    <td className="mono hash">{String(c.replay)}</td>
                    <td>{c.match ? <Badge tone="good">match</Badge> : <Badge tone="critical">differs</Badge>}</td></tr>
                ))}</tbody>
              </table>
            </details>
          ))}
        </Panel>
      )}
    </div>
  );
}

function DefendedPlot({ ctx, defense, dataset, samples, title }: {
  ctx: Ctx; defense: Defense; dataset: DatasetResponse | null; samples?: SampleResult[]; title: string;
}) {
  const [b, setB] = useState<BoundaryResponse | null>(null);
  const key = JSON.stringify([ctx, defense]);
  useEffect(() => {
    let live = true;
    setB(null);
    api.boundary(ctx, defense).then((r) => { if (live) setB(r); }).catch(() => {});
    return () => { live = false; };
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps
  return <BoundaryPlot boundary={b} dataset={dataset} samples={samples} showTrain={false} height={360} title={title} />;
}

function SampleInspector({ dataset, selected }: { dataset: DatasetResponse | null; selected: number | null }) {
  if (!dataset || selected == null) return null;
  const x = dataset.test.X[selected], p1 = dataset.test.p1[selected];
  return (
    <Panel title={`Selected test sample #${selected}`}>
      <dl className="kv">
        <dt>Input (x1, x2)</dt><dd className="mono">({x[0].toFixed(4)}, {x[1].toFixed(4)})</dd>
        <dt>True label</dt><dd><ClassDot y={dataset.test.y[selected]} /></dd>
        <dt>Prediction</dt><dd><ClassDot y={dataset.test.pred[selected]} /></dd>
        <dt>P(class 1)</dt><dd className="num">{p1.toFixed(4)}</dd>
        <dt>Confidence</dt><dd className="num">{Math.max(p1, 1 - p1).toFixed(4)}</dd>
        <dt>Clean outcome</dt><dd>{dataset.test.pred[selected] === dataset.test.y[selected]
          ? <Badge tone="good">correct</Badge> : <Badge tone="critical">misclassified</Badge>}</dd>
      </dl>
    </Panel>
  );
}

function AttackForm({ cfg, onChange }: { cfg: AttackConfig; onChange: (c: AttackConfig) => void }) {
  const set = (p: Partial<AttackConfig>) => onChange({ ...cfg, ...p });
  const bb = cfg.attack === "random_search";
  return (
    <div className="form">
      <label>Technique
        <select value={cfg.attack} onChange={(e) => {
          const attack = e.target.value as AttackConfig["attack"];
          set({ attack, norm: attack === "fgsm" ? "linf" : cfg.norm });
        }}>
          <option value="pgd">PGD — white-box, iterative (Madry et al. 2018)</option>
          <option value="fgsm">FGSM — white-box, one step (Goodfellow et al. 2015)</option>
          <option value="random_search">Random search — black-box, labels only</option>
        </select>
      </label>
      <label>Norm
        <select value={cfg.norm} disabled={cfg.attack === "fgsm"} onChange={(e) => set({ norm: e.target.value as "linf" | "l2" })}>
          <option value="linf">L∞</option><option value="l2">L2</option>
        </select>
      </label>
      <label className="span-2">Budget ε = <span className="num">{cfg.epsilon.toFixed(2)}</span>
        <input type="range" min={0.01} max={1} step={0.01} value={cfg.epsilon}
          onChange={(e) => set({ epsilon: parseFloat(e.target.value) })} />
      </label>
      {!bb && cfg.attack === "pgd" && (
        <label>Steps
          <input type="number" min={1} max={200} value={cfg.steps}
            onChange={(e) => set({ steps: Math.min(200, Math.max(1, parseInt(e.target.value, 10) || 1)) })} />
        </label>
      )}
      {!bb && (
        <label>Confidence κ (logit margin)
          <input type="number" min={0} max={10} step={0.5} value={cfg.confidence}
            onChange={(e) => set({ confidence: Math.min(10, Math.max(0, parseFloat(e.target.value) || 0)) })} />
        </label>
      )}
      {cfg.attack === "pgd" && (
        <label className="check"><input type="checkbox" checked={cfg.random_start}
          onChange={(e) => set({ random_start: e.target.checked })} /> Random start (seeded)</label>
      )}
      {bb && (
        <label>Query budget / sample
          <input type="number" min={1} max={2000} value={cfg.query_budget}
            onChange={(e) => set({ query_budget: Math.min(2000, Math.max(1, parseInt(e.target.value, 10) || 1)) })} />
        </label>
      )}
      <p className="span-2 muted small">
        {bb ? "Black-box: the attacker sees only the label returned by the deployed system and pays one query per input."
          : "White-box: the attacker knows the model parameters and uses exact input gradients."}
      </p>
    </div>
  );
}

function DefenseForm({ cfg, onChange }: { cfg: Defense; onChange: (d: Defense) => void }) {
  return (
    <div className="form">
      <label className="span-2">Control
        <select value={cfg.kind} onChange={(e) => onChange(DEFAULT_DEFENSES[e.target.value as Defense["kind"]])}>
          <option value="adversarial_training">Adversarial training (model-level)</option>
          <option value="confidence_rejection">Confidence rejection (system-level)</option>
        </select>
      </label>
      {cfg.kind === "adversarial_training" ? (
        <>
          <label>Training ε = <span className="num">{cfg.epsilon.toFixed(2)}</span>
            <input type="range" min={0.05} max={0.6} step={0.05} value={cfg.epsilon}
              onChange={(e) => onChange({ ...cfg, epsilon: parseFloat(e.target.value) })} />
          </label>
          <label>Rounds
            <input type="number" min={1} max={5} value={cfg.rounds}
              onChange={(e) => onChange({ ...cfg, rounds: Math.min(5, Math.max(1, parseInt(e.target.value, 10) || 1)) })} />
          </label>
          <p className="span-2 muted small">Retrains on clean + PGD adversarial copies of the training set, regenerated
            against the current model each round.</p>
        </>
      ) : (
        <>
          <label className="span-2">Threshold τ = <span className="num">{cfg.threshold.toFixed(2)}</span>
            <input type="range" min={0.55} max={0.99} step={0.01} value={cfg.threshold}
              onChange={(e) => onChange({ ...cfg, threshold: parseFloat(e.target.value) })} />
          </label>
          <p className="span-2 muted small">The system abstains when max(P) &lt; τ. Abstention is safe but costs coverage
            on clean inputs.</p>
        </>
      )}
    </div>
  );
}

function MetricsCompare({ cols }: { cols: [string, ExperimentRecord | undefined][] }) {
  const rows: [string, (r: ExperimentRecord) => ReactNode][] = [
    ["Experiment", (r) => <ExpLink id={r.experiment_id} />],
    ["Clean accuracy (test split)", (r) => <RatioView r={r.metrics.clean_accuracy} />],
    ["Coverage", (r) => <RatioView r={r.metrics.coverage} />],
    ["Adversarial accuracy", (r) => <RatioView r={r.metrics.adversarial_accuracy} />],
    ["Robust accuracy", (r) => <RatioView r={r.metrics.robust_accuracy} />],
    ["Attack success rate", (r) => <RatioView r={r.metrics.attack_success_rate} emphasize />],
    ["Adversarial abstention", (r) => <RatioView r={r.metrics.abstention_rate_adversarial} />],
    ["Mean ‖δ‖∞ (successful)", (r) => {
      const s = r.metrics.perturbation?.successful;
      return s?.n ? <span className="num">{s.linf_mean!.toFixed(4)}</span> : <span className="muted">—</span>;
    }],
    ["Max ‖δ‖/ε (verified ≤ 1)", (r) => r.metrics.perturbation
      ? <span className="num">{r.metrics.perturbation.max_budget_ratio.toFixed(4)}</span> : <span className="muted">—</span>],
    ["Queries (total)", (r) => r.metrics.queries ? <span className="num">{r.metrics.queries.total.toLocaleString()}</span> : <span className="muted">—</span>],
    ["Status", (r) => <StatusBadge status={r.status} />],
  ];
  return (
    <div className="table-wrap">
      <table className="table">
        <thead><tr><th>Metric</th>{cols.map(([l]) => <th key={l}>{l}</th>)}</tr></thead>
        <tbody>
          {rows.map(([name, f]) => (
            <tr key={name}><td>{name}</td>{cols.map(([l, r]) => <td key={l}>{r ? f(r) : <span className="muted">not run</span>}</td>)}</tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">Hover a ratio for its exact definition. Success rate = evaded / attacked inputs that
        were decided correctly when clean.</p>
    </div>
  );
}

function CapabilityBox({ caps }: { caps: Record<string, unknown> }) {
  return (
    <dl className="kv mt">
      {Object.entries(caps).map(([k, v]) => (
        <Fragment key={k}><dt>{k.replace(/_/g, " ")}</dt><dd>{String(v)}</dd></Fragment>
      ))}
    </dl>
  );
}

