import { useEffect, useState } from "react";
import { api, evidenceLinks } from "../api";
import { ErrorBox, EvalBadge, ExpLink, Panel, StatusBadge } from "../components/ui";
import type { ExperimentSummary } from "../types";

export function Evidence() {
  const [rows, setRows] = useState<ExperimentSummary[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [md, setMd] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api.experiments().then((r) => { setRows(r); if (r[0]) setSel(r[0].experiment_id); }).catch((e) => setError(String(e.message ?? e))); }, []);
  useEffect(() => { if (sel) api.report(sel).then(setMd).catch((e) => setError(String(e.message ?? e))); }, [sel]);
  return (
    <div className="page">
      <div className="page-head"><div>
        <div className="eyebrow">Evidence</div><h1>Evidence & Reports</h1>
        <p className="muted">Machine-readable JSON (schema <code>mlbreak.experiment/v1</code>), human-readable Markdown and per-sample CSV for every experiment.</p>
      </div></div>
      <ErrorBox error={error} />
      <div className="grid-2">
        <Panel title="Experiments">
          <div className="table-wrap scroll-y"><table className="table">
            <thead><tr><th>Experiment</th><th>Type</th><th>Status</th><th>Download</th></tr></thead>
            <tbody>{rows.map((r) => {
              const l = evidenceLinks(r.experiment_id);
              return (
                <tr key={r.experiment_id} className={sel === r.experiment_id ? "row-active" : ""}>
                  <td><button className="linklike mono" onClick={() => setSel(r.experiment_id)}>{r.experiment_id}</button></td>
                  <td><EvalBadge evaluation={r.evaluation} /></td><td><StatusBadge status={r.status} /></td>
                  <td className="row gap"><a href={l.json} download>JSON</a><a href={l.md} download>MD</a><a href={l.csv} download>CSV</a></td>
                </tr>);
            })}</tbody>
          </table></div>
          {rows.length === 0 && <p className="muted">No evidence yet.</p>}
        </Panel>
        <Panel title="Report preview" subtitle={sel ? <ExpLink id={sel} /> : undefined}>
          <pre className="report">{md || "Select an experiment."}</pre>
        </Panel>
      </div>
    </div>
  );
}
