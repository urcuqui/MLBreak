import { Fragment, useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { Link, NavLink, useParams } from "react-router-dom";
import { api, evidenceLinks } from "../api";
import { LabSwitcher, ScenarioSwitcher } from "../components/LabSwitcher";
import { SpamBar, TextDiff } from "../components/TextDiff";
import { Badge, ErrorBox, EvalBadge, ExpLink, Panel, RatioView, StatusBadge, Tile, VulnerableBanner } from "../components/ui";
import type { ClassifyResult, ReplayResult, TextAttackConfig, TextCharset, TextDefense, TextMessage, TextModelId, TextOp,
  TextOutcome, TextRecord, TextSample, VerifyReport } from "../types";

type Tab = "baseline" | "attack" | "defend" | "evidence";
const TABS: { id: Tab; label: string }[] = [
  { id: "baseline", label: "1 · Classifier" },
  { id: "attack", label: "2 · Attack" },
  { id: "defend", label: "3 · Defend" },
  { id: "evidence", label: "4 · Evidence & replay" },
];
interface Session { baseline?: string; single?: string; batch?: string; replay?: string; adaptive?: string }
interface Ctx { model_id: TextModelId; seed: number }
const SESSION_KEY = "mlbreak.ml01spam.session";

export const DEFAULT_TEXT_ATTACK: TextAttackConfig = {
  attack: "deepwordbug_greedy", goal: "spam_to_ham", min_target_confidence: 0.5,
  operations: ["swap", "substitute", "delete", "insert"], charsets: ["ascii_letters"], query_budget: 200,
  max_edit_distance: 10, max_char_ratio: 0.15, max_modified_word_ratio: 0.5, max_edits_per_word: 1,
  candidates_per_word: 8, protect_numbers: true, protect_stopwords: true, min_word_length: 2,
};
const DEFENSES: Record<TextDefense["kind"], TextDefense> = {
  text_normalization: { kind: "text_normalization", unicode: true, leetspeak: true, spell_snap: true, spell_min_count: 2 },
  text_adversarial_training: { kind: "text_adversarial_training", max_train_samples: 300, query_budget: 100,
    candidates_per_word: 6, max_edit_distance: 10, max_char_ratio: 0.15 },
};
const OPS: TextOp[] = ["swap", "substitute", "delete", "insert"];
const CHARSETS: { id: TextCharset; label: string }[] = [
  { id: "ascii_letters", label: "ASCII letters" }, { id: "confusables", label: "confusable homoglyphs (substitute)" },
  { id: "invisible", label: "invisible zero-width (insert)" },
];
const OUTCOME: Record<TextOutcome, [Parameters<typeof Badge>[0]["tone"], string]> = {
  evaded: ["critical", "evaded within constraints"], failed: ["good", "attack failed"],
  budget_exhausted: ["warning", "query budget exhausted"], skipped_misclassified: ["neutral", "skipped: clean misclassification"],
  not_perturbed: ["neutral", "not perturbed"],
};

function loadSession(): { ctx: Ctx; ids: Session } {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (raw) return JSON.parse(raw);
  } catch { /* storage unavailable */ }
  return { ctx: { model_id: "tfidf-word-logreg", seed: 7 }, ids: {} };
}

export function OutcomeBadge({ o }: { o: TextOutcome }) {
  const [tone, label] = OUTCOME[o];
  return <Badge tone={tone}>{label}</Badge>;
}

/** Everything about one attacked message. */
export function SampleView({ s, cfg }: { s: TextSample; cfg: TextAttackConfig | null }) {
  const label = (y: number | null) => (y == null ? "—" : y === 1 ? "spam" : "ham");
  return (
    <div className="stack">
      <div className="row gap wrap"><OutcomeBadge o={s.outcome} /><span className="muted small">test message #{s.test_index} · true label {label(s.y_true)}</span></div>
      <TextDiff original={s.text_original} adversarial={s.text_adversarial} />
      <div className="table-wrap"><table className="table">
        <thead><tr><th></th><th>Original</th><th>Adversarial</th></tr></thead>
        <tbody>
          <tr><td>Prediction</td><td>{label(s.pred_original)}</td><td>{label(s.pred_adversarial)}</td></tr>
          <tr><td>Class probabilities</td><td><SpamBar p={s.p_spam_original} /></td>
            <td>{s.p_spam_adversarial == null ? "—" : <SpamBar p={s.p_spam_adversarial} />}</td></tr>
        </tbody>
      </table></div>
      <dl className="kv">
        <dt>Queries</dt><dd className="num">{s.queries ?? "—"}{cfg && s.queries != null && s.queries > 0 ? ` / ${cfg.query_budget} budget` : s.queries === 0 ? " (replayed: no new queries)" : ""}</dd>
        <dt>Edit distance</dt><dd className="num">{s.edit_distance ?? "—"}{cfg && s.edit_distance != null ? ` / ≤ ${cfg.max_edit_distance} · ${(100 * (s.char_ratio ?? 0)).toFixed(1)}% of characters (≤ ${(100 * cfg.max_char_ratio).toFixed(0)}%)` : ""}</dd>
        <dt>Words modified</dt><dd className="num">{s.words_modified ?? "—"}{s.words_modifiable != null ? ` of ${s.words_modifiable} modifiable` : ""}</dd>
        {s.edits && s.edits.length > 0 && <><dt>Edit script</dt><dd className="small mono">{s.edits.map((e, k) => (
          <div key={k}>{e.op}: {JSON.stringify(e.word_before)} → {JSON.stringify(e.word_after)}</div>))}</dd></>}
      </dl>
      {s.constraints && (
        <ul className="checklist" aria-label="constraint validation">
          {Object.entries(s.constraints).map(([k, ok]) => (
            <li key={k}>{ok ? <Badge tone="good">{k.replace(/_/g, " ")}</Badge> : <Badge tone="critical">{k.replace(/_/g, " ")}</Badge>}</li>))}
        </ul>
      )}
    </div>
  );
}

export function TextCompare({ cols }: { cols: [string, TextRecord | undefined][] }) {
  const rows: [string, (r: TextRecord) => ReactNode][] = [
    ["Experiment", (r) => <ExpLink id={r.experiment_id} />],
    ["Clean accuracy (test split)", (r) => <RatioView r={r.metrics.clean_accuracy} />],
    ["Spam recall (clean)", (r) => <RatioView r={r.metrics.spam_recall} />],
    ["Ham false-positive rate (clean)", (r) => <RatioView r={r.metrics.ham_false_positive_rate} />],
    ["Attack success rate", (r) => <RatioView r={r.metrics.attack_success_rate} emphasize />],
    ["Accuracy under attack", (r) => <RatioView r={r.metrics.accuracy_under_attack} />],
    ["Budget exhausted", (r) => <RatioView r={r.metrics.budget_exhaustion_rate} />],
    ["Constraint violations", (r) => <RatioView r={r.metrics.constraint_violation_rate} />],
    ["Mean queries (attacked)", (r) => <span className="num">{r.metrics.effort?.mean_queries_attacked?.toFixed(1) ?? "—"}</span>],
    ["Mean edit distance (evaded)", (r) => <span className="num">{r.metrics.effort?.mean_edit_distance_evaded?.toFixed(2) ?? "—"}</span>],
    ["Outcomes", (r) => <span className="small">{r.metrics.outcomes ? Object.entries(r.metrics.outcomes.counts).filter(([, v]) => v)
      .map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`).join(" · ") : "—"}</span>],
    ["Status", (r) => <StatusBadge status={r.status} />],
  ];
  return (
    <div className="table-wrap">
      <table className="table">
        <thead><tr><th>Metric</th>{cols.map(([l]) => <th key={l}>{l}</th>)}</tr></thead>
        <tbody>{rows.map(([n, f]) => (
          <tr key={n}><td>{n}</td>{cols.map(([l, r]) => <td key={l}>{r ? f(r) : <span className="muted">not run</span>}</td>)}</tr>))}</tbody>
      </table>
      <p className="muted small">Success = target class reached AND every text-preservation constraint satisfied. Hover a ratio for
        its definition; denominators are the attacked messages (clean decision correct).</p>
    </div>
  );
}

export function SpamWorkspace() {
  const { tab = "baseline" } = useParams<{ tab: Tab }>();
  const initial = useMemo(loadSession, []);
  const [ctx, setCtx] = useState<Ctx>(initial.ctx);
  const [ids, setIds] = useState<Session>(initial.ids);
  const [records, setRecords] = useState<Record<string, TextRecord>>({});
  const [messages, setMessages] = useState<TextMessage[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [attack, setAttack] = useState<TextAttackConfig>(DEFAULT_TEXT_ATTACK);
  const [batchSize, setBatchSize] = useState(100);
  const [defense, setDefense] = useState<TextDefense>(DEFENSES.text_normalization);
  const [effort, setEffort] = useState({ max_edits_per_word: 2, query_budget: 400 });
  const [freeText, setFreeText] = useState("WINNER!! You have won a £1000 cash prize. Call 09061701461 now to claim.");
  const [classified, setClassified] = useState<ClassifyResult | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [replays, setReplays] = useState<Record<string, ReplayResult>>({});
  const [verifies, setVerifies] = useState<Record<string, VerifyReport>>({});

  useEffect(() => { try { localStorage.setItem(SESSION_KEY, JSON.stringify({ ctx, ids })); } catch { /* ignore */ } }, [ctx, ids]);
  useEffect(() => {
    Object.values(ids).forEach((id) => {
      if (id && !records[id]) api.anyExperiment(id).then((r) => {
        if ((r as TextRecord).family === "text_evasion") setRecords((m) => ({ ...m, [id]: r as TextRecord }));
      }).catch(() => {});
    });
  }, [ids]); // eslint-disable-line react-hooks/exhaustive-deps
  const source = attack.goal === "spam_to_ham" ? 1 : 0;
  useEffect(() => {
    api.textMessages(ctx.model_id, ctx.seed, source as 0 | 1, 150).then((r) => {
      setMessages(r.messages);
      setSelected((cur) => (cur != null && r.messages.some((m) => m.test_index === cur) ? cur
        : r.messages.find((m) => m.pred === source && Math.abs(m.p_spam - 0.5) < 0.45)?.test_index ?? r.messages[0]?.test_index ?? null));
    }).catch((e) => setError(String(e.message ?? e)));
  }, [ctx, source]);

  const run = useCallback(async (key: keyof Session, what: string, fn: () => Promise<TextRecord>) => {
    setBusy(what); setError(null);
    try {
      const rec = await fn();
      setRecords((m) => ({ ...m, [rec.experiment_id]: rec }));
      setIds((s) => (key === "batch" ? { ...s, batch: rec.experiment_id, replay: undefined, adaptive: undefined } : { ...s, [key]: rec.experiment_id }));
      if (rec.status !== "completed") setError(`Experiment ${rec.experiment_id} failed: ${rec.error}`);
    } catch (e) { setError(String((e as Error).message ?? e)); } finally { setBusy(null); }
  }, []);
  const act = async (id: string, kind: "replay" | "verify") => {
    setBusy(`${kind} ${id}`); setError(null);
    try {
      if (kind === "replay") { const r = await api.replay(id); setReplays((m) => ({ ...m, [id]: r })); }
      else { const v = await api.verify(id); setVerifies((m) => ({ ...m, [id]: v })); }
    } catch (e) { setError(String((e as Error).message ?? e)); } finally { setBusy(null); }
  };
  const classify = async () => {
    setBusy("classify"); setError(null);
    try { setClassified(await api.textClassify(ctx.model_id, ctx.seed, freeText)); }
    catch (e) { setError(String((e as Error).message ?? e)); } finally { setBusy(null); }
  };
  const changeCtx = (c: Ctx) => { setCtx(c); setIds({}); setRecords({}); setReplays({}); setVerifies({}); setClassified(null); };

  const rec = (k: keyof Session) => (ids[k] ? records[ids[k]!] : undefined);
  const baseline = rec("baseline"), single = rec("single"), batch = rec("batch"), replay = rec("replay"), adaptive = rec("adaptive");
  const sampleIn = (r?: TextRecord) => r?.samples.find((s) => s.test_index === selected);
  const shown = sampleIn(single) ?? sampleIn(batch);
  const selMsg = messages.find((m) => m.test_index === selected);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <div className="eyebrow">OWASP ML01:2023 · Input Manipulation Attack · scenario 2</div>
          <h1>Spam-filter evasion — character-level attacks</h1>
          <p className="muted">A TF-IDF spam filter trained on the bundled SMS Spam Collection (CC BY 4.0) is attacked with a
            DeepWordBug-style black-box search that swaps, substitutes, deletes or inserts characters under explicit
            text-preservation constraints. Every number below comes from an executed experiment with stored evidence.</p>
        </div>
        <div className="ctx-controls">
          <LabSwitcher current="ml01" tab={tab} path="ml01/spam" />
          <ScenarioSwitcher current="ml01/spam" tab={tab} />
          <label>Model
            <select value={ctx.model_id} onChange={(e) => changeCtx({ ...ctx, model_id: e.target.value as TextModelId })}>
              <option value="tfidf-word-logreg">TF-IDF + logistic regression</option>
              <option value="tfidf-word-nb">TF-IDF + naive Bayes</option>
            </select>
          </label>
          <label>Seed
            <input type="number" min={0} value={ctx.seed}
              onChange={(e) => { const v = parseInt(e.target.value, 10); if (v >= 0) changeCtx({ ...ctx, seed: v }); }} />
          </label>
        </div>
      </div>
      <VulnerableBanner>The undefended spam filter in this scenario is deliberately exposed to character-level evasion, on
        bundled public data, for education and defensive research. Local use only.</VulnerableBanner>

      <nav className="tabs" aria-label="Lab steps">
        {TABS.map((t) => <NavLink key={t.id} to={`/lab/ml01/spam/${t.id}`} className={({ isActive }) => (isActive ? "tab active" : "tab")}>{t.label}</NavLink>)}
        <div className="pipeline muted small">
          {(["baseline", "batch", "replay", "adaptive"] as (keyof Session)[]).map((k) => (
            <span key={k} className={ids[k] ? "pipe done" : "pipe"}>{ids[k] ? "✓" : "○"} {k}</span>))}
        </div>
      </nav>
      <ErrorBox error={error} />
      {busy && <div className="busy" role="status"><span className="spinner" /> Running: {busy}…</div>}

      {tab === "baseline" && (
        <div className="grid-2">
          <div className="stack">
            <Panel title="Bundled classifier" actions={<button className="btn primary" disabled={!!busy}
              onClick={() => run("baseline", "clean evaluation", () => api.textBaseline(ctx.model_id, ctx.seed))}>Load & evaluate</button>}>
              {baseline ? (
                <>
                  <div className="tiles">
                    <Tile label="Clean accuracy"><RatioView r={baseline.metrics.clean_accuracy} emphasize /></Tile>
                    <Tile label="Spam recall"><RatioView r={baseline.metrics.spam_recall} /></Tile>
                    <Tile label="Spam precision"><RatioView r={baseline.metrics.spam_precision} /></Tile>
                    <Tile label="Ham false-positive rate"><RatioView r={baseline.metrics.ham_false_positive_rate} /></Tile>
                  </div>
                  <div className="small muted">Experiment <ExpLink id={baseline.experiment_id} /> · <StatusBadge status={baseline.status} /></div>
                </>
              ) : <p className="muted">Train the approved configuration on the training split (in-process, no pickle) and evaluate it on
                the held-out test split. Results are stored as evidence.</p>}
            </Panel>
            {baseline && (
              <Panel title="Provenance">
                <dl className="kv">
                  <dt>Dataset</dt><dd>{String(baseline.dataset.source)} · {String(baseline.dataset.license)} · DOI {String(baseline.dataset.doi)}</dd>
                  <dt>Messages</dt><dd>{String(baseline.dataset.n_raw_messages)} raw, {String(baseline.dataset.n_duplicates_removed)} duplicates removed before the split</dd>
                  <dt>Train / test</dt><dd>{String(baseline.dataset.n_train)} / {String(baseline.dataset.n_test)} ({String(baseline.dataset.n_test_spam)} test spam)</dd>
                  <dt>Split</dt><dd className="mono small">{baseline.dataset.split_id}</dd>
                  <dt>Estimator</dt><dd>{baseline.model.estimator}</dd>
                  <dt>TF-IDF</dt><dd className="mono small">{JSON.stringify(baseline.model.tfidf)}</dd>
                  <dt>Model SHA-256</dt><dd className="mono hash">{baseline.model.artifact_sha256}</dd>
                </dl>
              </Panel>
            )}
          </div>
          <Panel title="Classify a message" subtitle="Live prediction from the bundled classifier (not stored as evidence).">
            <textarea aria-label="message to classify" value={freeText} maxLength={2000} onChange={(e) => setFreeText(e.target.value)} />
            <div className="row gap mt"><button className="btn" disabled={!!busy || !freeText.trim()} onClick={classify}>Classify</button></div>
            {classified && (
              <div className="mt stack">
                <div className="row gap">{classified.prediction === 1 ? <Badge tone="warning">spam</Badge> : <Badge tone="info">ham</Badge>}</div>
                <SpamBar p={classified.p_spam} />
              </div>
            )}
          </Panel>
        </div>
      )}

      {tab === "attack" && (
        <>
          <div className="grid-2">
            <Panel title={`Held-out ${source ? "spam" : "ham"} messages`} subtitle="Click a message to select it. P(spam) is the undefended filter's.">
              <div className="table-wrap msg-list"><table className="table small">
                <thead><tr><th>#</th><th>Message</th><th>P(spam)</th></tr></thead>
                <tbody>{messages.map((m) => (
                  <tr key={m.test_index} className={m.test_index === selected ? "row-active" : ""} onClick={() => setSelected(m.test_index)}>
                    <td className="num">{m.test_index}</td><td>{m.text.length > 110 ? m.text.slice(0, 110) + "…" : m.text}</td>
                    <td className="num">{m.p_spam.toFixed(3)}{m.pred !== m.y_true && <> <Badge tone="neutral">misclassified</Badge></>}</td>
                  </tr>))}</tbody>
              </table></div>
            </Panel>
            <Panel title="Attack configuration" subtitle="DeepWordBug-style greedy search (black-box, scores only).">
              <TextAttackForm cfg={attack} onChange={setAttack} />
              <div className="row gap wrap mt">
                <button className="btn danger" disabled={!!busy || selected == null}
                  onClick={() => run("single", `attack on message #${selected}`, () => api.textAttack(ctx.model_id, ctx.seed, attack, [selected!], 1))}>
                  Attack message #{selected ?? "—"}</button>
                <button className="btn danger outline" disabled={!!busy}
                  onClick={() => run("batch", `attack on ${batchSize} messages`, () => api.textAttack(ctx.model_id, ctx.seed, attack, null, batchSize))}>
                  Attack {batchSize} messages</button>
                <label className="small muted">batch <input type="number" min={1} max={300} value={batchSize} style={{ width: 80 }}
                  onChange={(e) => setBatchSize(Math.min(300, Math.max(1, parseInt(e.target.value, 10) || 1)))} /></label>
              </div>
            </Panel>
          </div>
          <div className="grid-2">
            <Panel title={`Message #${selected ?? "—"}`} subtitle={shown ? <>from <ExpLink id={(sampleIn(single) ? single : batch)!.experiment_id} /></> : "not attacked yet"}>
              {shown ? <SampleView s={shown} cfg={(sampleIn(single) ? single : batch)!.attack_config} />
                : selMsg ? <><div className="msg">{selMsg.text}</div><div className="mt"><SpamBar p={selMsg.p_spam} /></div></>
                : <p className="muted">Select a message.</p>}
            </Panel>
            <Panel title="Attack outcome" subtitle={batch ? <>batch <ExpLink id={batch.experiment_id} /></> : "run a batch attack to measure success"}>
              <TextCompare cols={[["Single message", single], ["Batch", batch]]} />
              {batch && <SampleTable rec={batch} onPick={setSelected} selected={selected} />}
            </Panel>
          </div>
        </>
      )}

      {tab === "defend" && (
        <>
          <div className="grid-2">
            <Panel title="Defensive control">
              <TextDefenseForm cfg={defense} onChange={setDefense} />
              <div className="form mt">
                <label>Adaptive effort: edits per word
                  <input type="number" min={1} max={3} value={effort.max_edits_per_word}
                    onChange={(e) => setEffort({ ...effort, max_edits_per_word: Math.min(3, Math.max(1, parseInt(e.target.value, 10) || 1)) })} /></label>
                <label>Adaptive effort: query budget
                  <input type="number" min={1} max={1000} value={effort.query_budget}
                    onChange={(e) => setEffort({ ...effort, query_budget: Math.min(1000, Math.max(1, parseInt(e.target.value, 10) || 1)) })} /></label>
              </div>
              <div className="row gap wrap mt">
                <button className="btn good" disabled={!!busy || !batch || batch.status !== "completed"}
                  onClick={() => run("replay", "defense replay", () => api.textDefend(batch!.experiment_id, defense))}>Replay adversarial messages</button>
                <button className="btn warning" disabled={!!busy || !batch}
                  onClick={() => run("adaptive", "adaptive attack", () => api.textAttack(ctx.model_id, ctx.seed,
                    { ...(batch!.attack_config as TextAttackConfig), ...effort }, batch!.samples.map((s) => s.test_index), 0, defense, batch!.experiment_id))}>
                  Adaptive attack (new experiment)</button>
              </div>
              {!batch && <p className="muted small mt">Run a batch attack in the <Link to="/lab/ml01/spam/attack">Attack</Link> step first.</p>}
              <div className="callout mt"><strong>Replay ≠ robustness.</strong> <em>Replay</em> re-submits the exact adversarial messages of{" "}
                {batch ? <ExpLink id={batch.experiment_id} /> : "the attack"}. The <em>adaptive attack</em> is a new search that queries the
                defended pipeline (normalisation is inside its loop), on the same messages and under the same text-preservation
                constraints; only its effort (edits per word, query budget) may increase.</div>
            </Panel>
            <Panel title="Defense comparison">
              <TextCompare cols={[["Undefended attack", batch], ["Defended · replay", replay], ["Defended · adaptive", adaptive]]} />
            </Panel>
          </div>
          {selected != null && (batch || replay || adaptive) && (
            <div className="grid-3">
              {([["Undefended", batch], ["Defended · replay", replay], ["Defended · adaptive", adaptive]] as [string, TextRecord | undefined][]).map(([t, r]) => {
                const s = sampleIn(r);
                return (
                  <Panel key={t} title={`${t} · message #${selected}`} subtitle={r ? <ExpLink id={r.experiment_id} /> : "not run"}>
                    {s ? <SampleView s={s} cfg={r!.attack_config} /> : <p className="muted">{r ? "message not in this experiment" : "not run"}</p>}
                  </Panel>);
              })}
            </div>
          )}
        </>
      )}

      {tab === "evidence" && (
        <Panel title="Session evidence" subtitle="JSON (mlbreak.text-evasion/v1), Markdown report and per-message CSV.">
          <div className="table-wrap"><table className="table">
            <thead><tr><th>Step</th><th>Experiment</th><th>Type</th><th>Status</th><th>Attack success</th><th>Export</th><th>Integrity</th><th>Exact replay</th></tr></thead>
            <tbody>{(["baseline", "single", "batch", "replay", "adaptive"] as (keyof Session)[]).map((k) => {
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
                  <td><button className="btn small" disabled={!!busy || r.status !== "completed"} onClick={() => act(r.experiment_id, "replay")}>Replay</button>{" "}
                    {rp && (rp.identical ? <Badge tone="good">identical</Badge> : <Badge tone="critical">diverged</Badge>)}</td>
                </tr>);
            })}</tbody>
          </table></div>
          <p className="muted small mt">Verify re-derives every metric and re-validates every adversarial message's constraints from the
            stored text and edit script. It proves internal consistency, <strong>not authenticity</strong>. Replay re-executes the
            experiment and is refused if verification fails.</p>
        </Panel>
      )}
    </div>
  );
}

function SampleTable({ rec, onPick, selected }: { rec: TextRecord; onPick: (i: number) => void; selected: number | null }) {
  return (
    <div className="table-wrap msg-list mt"><table className="table small">
      <thead><tr><th>#</th><th>Outcome</th><th>P(spam) clean → adv</th><th>Queries</th><th>Edits</th></tr></thead>
      <tbody>{rec.samples.map((s) => (
        <tr key={s.test_index} className={s.test_index === selected ? "row-active" : ""} onClick={() => onPick(s.test_index)}>
          <td className="num">{s.test_index}</td><td><OutcomeBadge o={s.outcome} /></td>
          <td className="num">{s.p_spam_original.toFixed(3)} → {s.p_spam_adversarial?.toFixed(3) ?? "—"}</td>
          <td className="num">{s.queries ?? "—"}</td><td className="num">{s.edit_distance ?? "—"}</td>
        </tr>))}</tbody>
    </table></div>
  );
}

function TextAttackForm({ cfg, onChange }: { cfg: TextAttackConfig; onChange: (c: TextAttackConfig) => void }) {
  const set = (p: Partial<TextAttackConfig>) => onChange({ ...cfg, ...p });
  const toggle = <T,>(list: T[], v: T) => (list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);
  const num = (v: string, lo: number, hi: number, int = true) => Math.min(hi, Math.max(lo, (int ? parseInt(v, 10) : parseFloat(v)) || lo));
  return (
    <div className="form">
      <label>Goal
        <select value={cfg.goal} onChange={(e) => set({ goal: e.target.value as TextAttackConfig["goal"] })}>
          <option value="spam_to_ham">evade the filter (spam → ham)</option>
          <option value="ham_to_spam">false positive (ham → spam)</option>
        </select></label>
      <label>Target confidence ≥ <span className="num">{cfg.min_target_confidence.toFixed(2)}</span>
        <input type="range" min={0.5} max={0.95} step={0.05} value={cfg.min_target_confidence}
          onChange={(e) => set({ min_target_confidence: parseFloat(e.target.value) })} /></label>
      <div className="span-2"><div className="muted small">Permitted operations</div><div className="checks">
        {OPS.map((o) => <label key={o}><input type="checkbox" checked={cfg.operations.includes(o)} disabled={cfg.operations.length === 1 && cfg.operations.includes(o)}
          onChange={() => set({ operations: toggle(cfg.operations, o) })} />{o}</label>)}</div></div>
      <div className="span-2"><div className="muted small">Permitted characters</div><div className="checks">
        {CHARSETS.map((c) => <label key={c.id}><input type="checkbox" checked={cfg.charsets.includes(c.id)} disabled={cfg.charsets.length === 1 && cfg.charsets.includes(c.id)}
          onChange={() => set({ charsets: toggle(cfg.charsets, c.id) })} />{c.label}</label>)}</div></div>
      <label>Query budget / message<input type="number" min={1} max={1000} value={cfg.query_budget} onChange={(e) => set({ query_budget: num(e.target.value, 1, 1000) })} /></label>
      <label>Max edit distance<input type="number" min={1} max={50} value={cfg.max_edit_distance} onChange={(e) => set({ max_edit_distance: num(e.target.value, 1, 50) })} /></label>
      <label>Max altered characters = <span className="num">{(100 * cfg.max_char_ratio).toFixed(0)}%</span>
        <input type="range" min={0.01} max={0.5} step={0.01} value={cfg.max_char_ratio} onChange={(e) => set({ max_char_ratio: parseFloat(e.target.value) })} /></label>
      <label>Max modified words = <span className="num">{(100 * cfg.max_modified_word_ratio).toFixed(0)}%</span>
        <input type="range" min={0.05} max={1} step={0.05} value={cfg.max_modified_word_ratio} onChange={(e) => set({ max_modified_word_ratio: parseFloat(e.target.value) })} /></label>
      <label>Edits per word<input type="number" min={1} max={3} value={cfg.max_edits_per_word} onChange={(e) => set({ max_edits_per_word: num(e.target.value, 1, 3) })} /></label>
      <label>Candidates per word<input type="number" min={1} max={32} value={cfg.candidates_per_word} onChange={(e) => set({ candidates_per_word: num(e.target.value, 1, 32) })} /></label>
      <div className="span-2 checks">
        <label><input type="checkbox" checked={cfg.protect_numbers} onChange={(e) => set({ protect_numbers: e.target.checked })} />protect numbers, URLs, e-mails</label>
        <label><input type="checkbox" checked={cfg.protect_stopwords} onChange={(e) => set({ protect_stopwords: e.target.checked })} />protect function words</label>
      </div>
      <p className="span-2 muted small">A label change only counts as an evasion if every constraint holds; the server re-validates each
        adversarial message from its edit script. Turning number protection off lets the attack corrupt phone numbers, which
        defeats the spam's purpose and roughly doubles the apparent success.</p>
    </div>
  );
}

function TextDefenseForm({ cfg, onChange }: { cfg: TextDefense; onChange: (d: TextDefense) => void }) {
  return (
    <div className="form">
      <label className="span-2">Control
        <select value={cfg.kind} onChange={(e) => onChange(DEFENSES[e.target.value as TextDefense["kind"]])}>
          <option value="text_normalization">Text normalisation (pre-processing)</option>
          <option value="text_adversarial_training">Adversarial data augmentation (retraining)</option>
        </select></label>
      {cfg.kind === "text_normalization" ? (
        <div className="span-2 checks">
          {(["unicode", "leetspeak", "spell_snap"] as const).map((k) => (
            <label key={k}><input type="checkbox" checked={cfg[k]} onChange={(e) => onChange({ ...cfg, [k]: e.target.checked })} />
              {{ unicode: "Unicode (NFKC, strip zero-width, map homoglyphs)", leetspeak: "leetspeak (fr3e → free)",
                spell_snap: "spell-snap unknown words to the training vocabulary" }[k]}</label>))}
        </div>
      ) : (
        <>
          <label>Training spam attacked<input type="number" min={10} max={600} value={cfg.max_train_samples}
            onChange={(e) => onChange({ ...cfg, max_train_samples: Math.min(600, Math.max(10, parseInt(e.target.value, 10) || 10)) })} /></label>
          <label>Queries per training message<input type="number" min={10} max={500} value={cfg.query_budget}
            onChange={(e) => onChange({ ...cfg, query_budget: Math.min(500, Math.max(10, parseInt(e.target.value, 10) || 10)) })} /></label>
          <p className="span-2 muted small">Adversarial examples are generated from TRAINING spam only and added with label spam; the
            held-out test messages are never used.</p>
        </>
      )}
    </div>
  );
}

export function TextDetailBody({ rec }: { rec: TextRecord }) {
  const [pick, setPick] = useState<number | null>(rec.samples.find((s) => s.outcome === "evaded")?.test_index ?? rec.samples[0]?.test_index ?? null);
  const s = rec.samples.find((x) => x.test_index === pick);
  return (
    <>
      {rec.mode !== "baseline" && (
        <div className="grid-2">
          <Panel title={`Message #${pick ?? "—"}`}>{s ? <SampleView s={s} cfg={rec.attack_config} /> : <p className="muted">No messages.</p>}</Panel>
          <Panel title="Messages"><SampleTable rec={rec} onPick={setPick} selected={pick} /></Panel>
        </div>
      )}
      <div className="grid-2">
        <Panel title="Metrics"><TextCompare cols={[["This experiment", rec]]} /></Panel>
        <div className="stack">
          {rec.metrics.comparison && (
            <Panel title="Undefended vs defended">
              <table className="table"><thead><tr><th>Metric</th><th>Undefended</th><th>Defended</th></tr></thead>
                <tbody>{Object.entries(rec.metrics.comparison).map(([k, v]) => (
                  <tr key={k}><td>{k.replace(/_/g, " ")}</td><td><RatioView r={v.undefended ?? undefined} /></td><td><RatioView r={v.defended ?? undefined} /></td></tr>))}</tbody>
              </table>
            </Panel>)}
          {rec.attacker_capabilities && (
            <Panel title="Attacker capabilities"><dl className="kv">{Object.entries(rec.attacker_capabilities).map(([k, v]) => (
              <Fragment key={k}><dt>{k.replace(/_/g, " ")}</dt><dd>{String(v)}</dd></Fragment>))}</dl></Panel>)}
          <Panel title="Configuration & provenance"><dl className="kv">
            <dt>Model</dt><dd>{rec.model.model_id} · {rec.model.estimator}</dd>
            <dt>Model SHA-256</dt><dd className="mono hash">{rec.model.artifact_sha256}</dd>
            <dt>Deployed system SHA-256</dt><dd className="mono hash">{rec.target_system.model_artifact_sha256}</dd>
            <dt>TF-IDF</dt><dd className="mono small">{JSON.stringify(rec.model.tfidf)}</dd>
            <dt>Split</dt><dd className="mono small">{rec.dataset.split_id}</dd>
            <dt>Attack</dt><dd className="mono small">{rec.attack_config ? JSON.stringify(rec.attack_config) : "—"}</dd>
            <dt>Defense</dt><dd className="mono small">{rec.defense_config ? JSON.stringify(rec.defense_config) : "none"}</dd>
          </dl></Panel>
        </div>
      </div>
    </>
  );
}
