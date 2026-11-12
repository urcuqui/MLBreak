import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { Badge, ErrorBox } from "../components/ui";
import type { OwaspCategory } from "../types";

const SCENARIO_LINKS: Record<string, { to: string; label: string }> = {
  "ml01-moons-evasion": { to: "/lab/ml01/baseline", label: "2D Classifier Evasion" },
  "ml01-spam-filter-evasion": { to: "/lab/ml01/spam/baseline", label: "Spam Filter Evasion" },
  "ml02-moons-label-flip": { to: "/lab/ml02/baseline", label: "Label-flipping poisoning" },
};

export function Owasp() {
  const [data, setData] = useState<{ taxonomy: string; version: string; url: string; categories: OwaspCategory[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api.owasp().then(setData).catch((e) => setError(String(e.message ?? e))); }, []);
  return (
    <div className="page">
      <div className="page-head"><div>
        <div className="eyebrow">Taxonomy</div>
        <h1>OWASP ML Top 10</h1>
        <p className="muted">
          {data ? <>{data.taxonomy} ({data.version}) — <a href={data.url} target="_blank" rel="noreferrer noopener">official project page</a>. </> : null}
          Only categories marked <em>implemented</em> have a working, tested laboratory. The rest are metadata only —
          no placeholder attacks.
        </p>
      </div></div>
      <ErrorBox error={error} />
      <div className="cards">
        {data?.categories.map((c) => (
          <article key={c.id} className={`card ${c.lab_status === "implemented" ? "card-live" : "card-off"}`}>
            <div className="row between">
              <span className="mono strong">{c.id}</span>
              {c.lab_status === "implemented" ? <Badge tone="good">lab implemented</Badge> : <Badge tone="neutral">not implemented</Badge>}
            </div>
            <h3>{c.name}</h3>
            <p className="muted small">{c.summary}</p>
            {c.scenario_ids.length > 0 && (
              <ul className="scenario-list">
                {c.scenario_ids.map((sid) => SCENARIO_LINKS[sid] && (
                  <li key={sid}><Link to={SCENARIO_LINKS[sid].to}>{SCENARIO_LINKS[sid].label}</Link>{" "}
                    <span className="muted small mono">{sid}</span></li>))}
              </ul>
            )}
            <div className="row between mt">
              <a href={c.reference_url} target="_blank" rel="noreferrer noopener" className="small">OWASP reference ↗</a>
              {c.lab_status === "implemented" && (
                <Link className="btn small primary" to={`/lab/${c.code.toLowerCase()}/baseline`}>Open lab</Link>)}
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
