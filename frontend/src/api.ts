import type {
  AttackConfig, BoundaryResponse, Dashboard, DatasetResponse, Defense, ExperimentRecord,
  ExperimentSummary, KnnDefense, LabelFlipConfig, ModelKey, OwaspCategory, PoisoningCase, PoisoningRecord,
  RegressionReport, ReplayResult, SecurityCase, VerifyReport, ClassifyResult, TextAttackConfig, TextDefense,
  TextMessage, TextModelId, TextRecord,
} from "./types";

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

// HTTP 429 means the server's concurrency gate refused the request BEFORE doing any work or writing any
// evidence (see EngineBusy in src/mlbreak/experiments.py), so retrying is side-effect free. Pages fire several
// requests at once (and a cold model fit can hold a slot for a few seconds), so 429s are retried with backoff.
const RETRY_429 = [300, 600, 1000, 1500, 2500, 4000];

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res = await fetch(path, { ...init, headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) } });
  for (const wait of RETRY_429) {
    if (res.status !== 429) break;
    await new Promise((r) => setTimeout(r, wait));
    res = await fetch(path, { ...init, headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) } });
  }
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
  anyExperiment: (id: string) => req<ExperimentRecord | PoisoningRecord | TextRecord>(`/api/experiments/${encodeURIComponent(id)}`),
  // ML02 data poisoning
  poisonBaseline: (c: Ctx) => post<PoisoningRecord>("/api/experiments/poisoning/baseline", c),
  poisonAttack: (c: Ctx, poison: LabelFlipConfig) =>
    post<PoisoningRecord>("/api/experiments/poisoning/attack", { ...c, poison }),
  poisonDefend: (parent_experiment_id: string, defense: KnnDefense) =>
    post<PoisoningRecord>("/api/experiments/poisoning/defend", { parent_experiment_id, defense }),
  experimentBoundary: (id: string, model: ModelKey, resolution = 80) =>
    serialBoundary<BoundaryResponse>(
      `/api/experiments/${encodeURIComponent(id)}/boundary?model=${model}&resolution=${resolution}`),
  // ML01 spam-filter evasion
  textMessages: (model_id: TextModelId, seed: number, label: 0 | 1 | null, limit = 150) =>
    req<{ dataset: Record<string, unknown>; model: Record<string, unknown>; messages: TextMessage[] }>(
      `/api/text/messages?model_id=${model_id}&seed=${seed}${label == null ? "" : `&label=${label}`}&limit=${limit}`),
  textClassify: (model_id: TextModelId, seed: number, text: string, defense: TextDefense | null = null) =>
    post<ClassifyResult>("/api/text/classify", { model_id, seed, text, defense }),
  textBaseline: (model_id: TextModelId, seed: number, defense: TextDefense | null = null) =>
    post<TextRecord>("/api/experiments/text/baseline", { model_id, seed, defense }),
  textAttack: (model_id: TextModelId, seed: number, attack: TextAttackConfig, indices: number[] | null, maxSamples: number,
               target_defense: TextDefense | null = null, parent_experiment_id: string | null = null) =>
    post<TextRecord>("/api/experiments/text/attack", {
      model_id, seed, attack, samples: indices ? { indices } : { max_samples: maxSamples }, target_defense,
      parent_experiment_id }),
  textDefend: (parent_experiment_id: string, defense: TextDefense) =>
    post<TextRecord>("/api/experiments/text/defend", { parent_experiment_id, defense }),
  cases: () => req<(SecurityCase | PoisoningCase)[]>("/api/regression/cases"),
  runRegression: () => post<RegressionReport>("/api/regression/run", {}),
  latestRegression: () => req<RegressionReport | null>("/api/regression/latest"),
};

export const evidenceLinks = (id: string) => ({
  json: `/api/experiments/${id}/evidence.json`,
  md: `/api/experiments/${id}/report.md`,
  csv: `/api/experiments/${id}/samples.csv`,
});
