import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { ErrorBox, EvalBadge, ExpLink, Panel, StatusBadge, pct } from "../components/ui";
import type { ExperimentSummary } from "../types";

export function Experiments() {
  const [rows, setRows] = useState<ExperimentSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("all");
  useEffect(() => { api.experiments().then(setRows).catch((e) => setError(String(e.message ?? e))); }, []);
  const shown = useMemo(() => (rows ?? []).filter((r) => filter === "all" || r.evaluation === filter), [rows, filter]);
  return (
    <div className="page">
      <div className="page-head"><div>
        <div className="eyebrow">History</div><h1>Experiments</h1>
        <p className="muted">Every baseline, attack, defense replay, adaptive attack and exact replay that ran on this machine.</p>
      </div></div>
      <ErrorBox error={error} />
      <Panel title={`${shown.length} experiment${shown.length === 1 ? "" : "s"}`} actions={
        <select value={filter} onChange={(e) => setFilter(e.target.value)} aria-label="Filter by type">
          <option value="all">All types</option><option value="clean">Baseline</option><option value="attack">Attack</option>
          <option value="replay">Defense replay</option><option value="adaptive_attack">Adaptive attack</option>
          <option value="clean_training">ML02 clean training</option><option value="poisoning">ML02 poisoning</option>
          <option value="poisoning_replay">ML02 defense replay</option>
        </select>}>
        <div className="table-wrap"><table className="table">
          <thead><tr><th>Experiment</th><th>Created (UTC)</th><th>Type</th><th>Status</th><th>Model</th><th>Attack</th>
            <th>Defense</th><th title="Accuracy on clean test inputs of the evaluated model (for ML02: the poisoned/defended model)">Clean-input acc.</th><th>Attack success</th><th>Parent / replay of</th></tr></thead>
          <tbody>{shown.map((e) => (
            <tr key={e.experiment_id}>
              <td><ExpLink id={e.experiment_id} /></td><td className="mono small">{e.created_at.slice(0, 19).replace("T", " ")}</td>
              <td><EvalBadge evaluation={e.evaluation} /></td><td><StatusBadge status={e.status} /></td>
              <td className="mono small">{e.model_id}</td>
              <td className="small">{!e.attack ? "—" : e.family === "text_evasion"
                ? `${e.attack} · ≤ ${e.max_edit_distance} edits · ${e.query_budget} queries`
                : e.family === "poisoning"
                ? `${e.attack} · ${Math.round((e.poison_fraction ?? 0) * 100)}%${e.target_class != null ? ` · class ${e.target_class}` : ""}`
                : `${e.attack} · ${e.norm} · ε=${e.epsilon}`}</td>
              <td className="small">{e.defense ?? "—"}</td>
              <td className="num">{pct(e.clean_accuracy)}</td>
              <td className="num">{e.attack_success_rate == null ? "—" : `${pct(e.attack_success_rate)} (${e.attack_success})`}</td>
              <td className="small">{e.replay_of ? <>replay of <ExpLink id={e.replay_of} /></> : <ExpLink id={e.parent_experiment_id} />}</td>
            </tr>))}</tbody>
        </table></div>
        {rows && rows.length === 0 && <p className="muted">No experiments stored yet.</p>}
      </Panel>
    </div>
  );
}
