import type {
  AttackConfig, BoundaryResponse, Dashboard, DatasetResponse, Defense, ExperimentRecord,
  ExperimentSummary, KnnDefense, LabelFlipConfig, ModelKey, OwaspCategory, PoisoningCase, PoisoningRecord,
  RegressionReport, ReplayResult, SecurityCase, VerifyReport,
} from "./types";

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const body = await res.json();
      msg = typeof body.detail === "string" ? body.detail
        : Array.isArray(body.detail) ? body.detail.map((d: { loc: string[]; msg: string }) =>
            `${d.loc.slice(1).join(".")}: ${d.msg}`).join("; ")
        : JSON.stringify(body);
    } catch { /* keep statusText */ }
    throw new ApiError(res.status, msg);
  }
  return res.json() as Promise<T>;
}

// Boundary grids can retrain models server-side and share the experiment concurrency gate (HTTP 429 when
// busy). Pages request several at once, so they are serialised client-side and retried with backoff on 429.
let boundaryQueue: Promise<unknown> = Promise.resolve();
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function serialBoundary<T>(path: string): Promise<T> {
  const run = async (): Promise<T> => {
    for (let attempt = 0; ; attempt++) {
      try {
        return await req<T>(path);
      } catch (e) {
        if (!(e instanceof ApiError) || e.status !== 429 || attempt >= 6) throw e;
        await sleep(250 * (attempt + 1));
      }
    }
  };
  const next = boundaryQueue.then(run, run);
  boundaryQueue = next.catch(() => undefined);
  return next;
}

const post = <T>(path: string, body: unknown) => req<T>(path, { method: "POST", body: JSON.stringify(body) });

export interface Ctx { model_id: string; seed: number }

export const api = {
  dashboard: () => req<Dashboard>("/api/dashboard"),
  owasp: () => req<{ taxonomy: string; version: string; url: string; categories: OwaspCategory[] }>("/api/owasp"),
  dataset: (c: Ctx) => req<DatasetResponse>(
    `/api/scenarios/ml01-moons-evasion/dataset?model_id=${encodeURIComponent(c.model_id)}&seed=${c.seed}`),
  boundary: (c: Ctx, defense: Defense | null, resolution = 90) =>
    post<BoundaryResponse>("/api/boundary", { ...c, defense, resolution }),
  baseline: (c: Ctx, defense: Defense | null = null) =>
    post<ExperimentRecord>("/api/experiments/baseline", { ...c, defense }),
  attack: (c: Ctx, attack: AttackConfig, indices: number[] | null,
           target_defense: Defense | null = null, parent_experiment_id: string | null = null) =>
    post<ExperimentRecord>("/api/experiments/attack", {
      ...c, attack, samples: indices ? { indices } : {}, target_defense, parent_experiment_id,
    }),
  defend: (parent_experiment_id: string, defense: Defense) =>
    post<ExperimentRecord>("/api/experiments/defend", { parent_experiment_id, defense }),
  experiments: () => req<ExperimentSummary[]>("/api/experiments"),
  experiment: (id: string) => req<ExperimentRecord>(`/api/experiments/${encodeURIComponent(id)}`),
  replay: (id: string) => post<ReplayResult>(`/api/experiments/${encodeURIComponent(id)}/replay`, {}),
  report: async (id: string) => {
    const r = await fetch(`/api/experiments/${encodeURIComponent(id)}/report.md`);
    if (!r.ok) throw new ApiError(r.status, r.statusText);
    return r.text();
  },
  verify: (id: string) => req<VerifyReport>(`/api/experiments/${encodeURIComponent(id)}/verify`),
  anyExperiment: (id: string) => req<ExperimentRecord | PoisoningRecord>(`/api/experiments/${encodeURIComponent(id)}`),
  // ML02 data poisoning
  poisonBaseline: (c: Ctx) => post<PoisoningRecord>("/api/experiments/poisoning/baseline", c),
  poisonAttack: (c: Ctx, poison: LabelFlipConfig) =>
    post<PoisoningRecord>("/api/experiments/poisoning/attack", { ...c, poison }),
  poisonDefend: (parent_experiment_id: string, defense: KnnDefense) =>
    post<PoisoningRecord>("/api/experiments/poisoning/defend", { parent_experiment_id, defense }),
  experimentBoundary: (id: string, model: ModelKey, resolution = 80) =>
    serialBoundary<BoundaryResponse>(
      `/api/experiments/${encodeURIComponent(id)}/boundary?model=${model}&resolution=${resolution}`),
  cases: () => req<(SecurityCase | PoisoningCase)[]>("/api/regression/cases"),
  runRegression: () => post<RegressionReport>("/api/regression/run", {}),
  latestRegression: () => req<RegressionReport | null>("/api/regression/latest"),
};

export const evidenceLinks = (id: string) => ({
  json: `/api/experiments/${id}/evidence.json`,
  md: `/api/experiments/${id}/report.md`,
  csv: `/api/experiments/${id}/samples.csv`,
});
