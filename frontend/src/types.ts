export interface Ratio { numerator: number; denominator: number; value: number | null; definition: string; population?: string }

export type Outcome = "evaded" | "abstained_adversarial" | "robust" | "clean_misclassified" | "clean_abstained" | "not_perturbed";

export interface PerturbationStats {
  n: number; linf_mean?: number; linf_max?: number; linf_median?: number;
  l2_mean?: number; l2_max?: number; l2_median?: number;
}

export interface Metrics {
  clean_accuracy?: Ratio; clean_accuracy_selected?: Ratio; coverage?: Ratio; selective_accuracy?: Ratio;
  train_accuracy?: Ratio; adversarial_accuracy?: Ratio; attack_success_rate?: Ratio;
  abstained_on_adversarial?: Ratio;
  robust_accuracy?: Ratio; abstention_rate_adversarial?: Ratio; attack_resisted_rate?: Ratio;
  abstention_rate_clean?: Ratio; abstention_rate_clean_selected?: Ratio;
  adversarial_coverage?: Ratio; adversarial_selective_accuracy?: Ratio;
  outcomes?: { population: string; n_selected: number; attack_eligible: number; counts: Record<Outcome, number>; definition: string };
  perturbation?: { norm: string; epsilon: number; max_budget_ratio: number; attacked: PerturbationStats; successful: PerturbationStats };
  queries?: { total: number; mean_per_attacked_sample: number; kind: string };
  comparison?: Record<string, { undefended: Ratio; defended: Ratio }>;
  notes?: string[];
}

export interface SampleResult {
  test_index: number; y_true: number; x_original: number[]; pred_original: number;
  p1_original: number; confidence_original: number; x_adversarial: number[] | null;
  pred_adversarial: number | null; p1_adversarial: number | null; confidence_adversarial: number | null;
  perturbation_linf: number | null; perturbation_l2: number | null; clean_correct: boolean;
  attack_success: boolean | null; queries: number | null; outcome?: Outcome | null;
}

export interface AttackConfig {
  attack: "fgsm" | "pgd" | "random_search"; norm: "linf" | "l2"; epsilon: number; steps: number;
  step_size?: number | null; confidence: number; random_start: boolean; query_budget: number;
}

export type Defense =
  | { kind: "adversarial_training"; epsilon: number; norm: "linf" | "l2"; steps: number; rounds: number }
  | { kind: "confidence_rejection"; threshold: number };

export interface ExperimentRecord {
  experiment_id: string; created_at: string; mode: "baseline" | "attack" | "defend";
  evaluation: "clean" | "attack" | "adaptive_attack" | "replay"; status: "completed" | "failed";
  error: string | null; intentionally_vulnerable: boolean; scenario_id: string;
  owasp: { id: string; name: string; taxonomy: string; taxonomy_version: string; reference_url: string };
  attacker_capabilities: Record<string, unknown> | null;
  model: { model_id: string; artifact_sha256: string; estimator: string; hyperparameters: Record<string, unknown>; training: string };
  target_system: { defense: Defense | null; model_artifact_sha256: string; decision_policy: string };
  dataset: Record<string, unknown> & { dataset_id: string; split_id: string; test_sha256: string; train_sha256: string };
  seed: number; attack_config: AttackConfig | null; defense_config: Defense | null;
  parent_experiment_id: string | null; replay_of: string | null; request: Record<string, unknown>;
  schema_version?: string; lineage?: Record<string, unknown> | null;
  metrics: Metrics; samples: SampleResult[]; reproducibility: Record<string, unknown>; duration_ms: number;
}

export interface ExperimentSummary {
  experiment_id: string; created_at: string; mode: string; evaluation: string; status: string;
  scenario_id: string; owasp_id: string; model_id: string; attack: string | null; epsilon: number | null;
  norm: string | null; defense: string | null; n_samples: number; clean_accuracy: number | null;
  attack_success_rate: number | null; attack_success: string | null; parent_experiment_id: string | null;
  replay_of: string | null; error: string | null; family?: "evasion" | "poisoning";
  poison_fraction?: number | null; target_class?: number | null;
}

export interface OwaspCategory {
  id: string; code: string; name: string; summary: string; reference_url: string;
  lab_status: "implemented" | "not_implemented"; scenario_ids: string[];
}

export interface DatasetResponse {
  dataset: Record<string, unknown> & { domain: number[][]; n_test: number; split_id: string };
  model: { model_id: string; artifact_sha256: string; estimator: string; n_iter: number };
  train: { X: number[][]; y: number[] };
  test: { X: number[][]; y: number[]; pred: number[]; p1: number[] };
}

export interface BoundaryResponse {
  x: number[]; y: number[]; p1: number[][]; abstain: number[][];
  target_system: { defense: Defense | KnnDefense | null; model_artifact_sha256: string; decision_policy: string };
}

export interface ReplayResult {
  original_experiment_id: string; replay_experiment_id: string; identical: boolean;
  original_integrity?: { ok: boolean; guarantee?: string };
  checks: { check: string; original: unknown; replay: unknown; match: boolean }[];
}

export interface Invariant {
  id: string; kind: string; posture: string; metric: string; op: string; value: number; description: string;
  observed?: number | null; passed?: boolean; experiment_id?: string | null;
}

export interface SecurityCase {
  id: string; title: string; owasp_id: string; attack: AttackConfig; defense: Defense;
  defended_evaluation: "replay" | "adaptive"; invariants: Invariant[]; rationale: string;
}

export interface RegressionReport {
  generated_at: string; total: number; passed: number; failed: number; ok: boolean;
  cases: { id: string; title: string; passed: boolean; defended_evaluation: string;
    undefended_experiment_id: string; defended_experiment_id: string | null; invariants: Invariant[] }[];
}

export interface Dashboard {
  experiments_total: number; experiments_by_evaluation: Record<string, number>; experiments_failed: number;
  recent: ExperimentSummary[]; owasp_implemented: number; owasp_total: number; regression_cases: number;
  regression_latest: { generated_at: string; total: number; passed: number; failed: number; ok: boolean } | null;
}

// ---------------------------------------------------------------------------
// ML02 data poisoning
// ---------------------------------------------------------------------------
export interface LabelFlipConfig {
  strategy: "random" | "targeted"; fraction: number; target_class: 0 | 1 | null; selection: "random" | "boundary";
}

export interface KnnDefense {
  kind: "knn_label_sanitization"; k: number; agreement_threshold: number; action: "relabel" | "remove";
}

export interface TrainRow {
  train_index: number; x: number[]; y_true: number; y_observed: number; poisoned: boolean;
  pred_clean: number; pred_poisoned: number | null; pred_defended: number | null;
  knn_agreement: number | null; flagged: boolean | null; y_sanitized: number | null; removed: boolean | null;
  detection: "true_positive" | "false_positive" | "false_negative" | "true_negative" | null;
  flagged_on_clean: boolean | null;
}

export interface TestRow {
  test_index: number; x: number[]; y_true: number; pred_clean: number; p1_clean: number;
  pred_poisoned: number | null; p1_poisoned: number | null; pred_defended: number | null; p1_defended: number | null;
  pred_clean_sanitized: number | null;
  outcome: "kept" | "broken" | "fixed" | "still_wrong" | null;
  defended_outcome: "kept" | "broken" | "fixed" | "still_wrong" | null;
}

export type ModelKey = "clean" | "poisoned" | "defended" | "clean_sanitized";

export interface PoisoningRecord {
  schema_version: string; family: "poisoning"; experiment_id: string; created_at: string;
  mode: "baseline" | "attack" | "defend"; evaluation: "clean_training" | "poisoning" | "poisoning_replay";
  status: "completed" | "failed"; error: string | null; intentionally_vulnerable: boolean; scenario_id: string;
  owasp: { id: string; name: string; taxonomy: string; taxonomy_version: string; reference_url: string };
  attacker_capabilities: Record<string, unknown> | null;
  model: { model_id: string; artifact_sha256: string; estimator: string };
  models: Partial<Record<ModelKey, { artifact_sha256: string; training: string } | null>>;
  target_system: { model_artifact_sha256: string | null; defense: KnnDefense | null };
  dataset: Record<string, unknown> & { dataset_id: string; split_id: string; train_sha256: string };
  seed: number; poison_config: LabelFlipConfig | null; defense_config: KnnDefense | null;
  poisoning: (Record<string, unknown> & { selection_rule?: string; poisoned_indices?: number[]; n_flips?: number;
    budget?: number }) | null;
  parent_experiment_id: string | null; replay_of: string | null; lineage: Record<string, unknown> | null;
  request: Record<string, unknown>; metrics: PoisoningMetrics;
  training_samples: TrainRow[]; test_samples: TestRow[]; reproducibility: Record<string, unknown>; duration_ms: number;
}

export interface ModelBlock {
  test_accuracy: Ratio; recall_class_0: Ratio; recall_class_1: Ratio; train_accuracy?: Ratio;
  train_accuracy_observed?: Ratio;
}

export interface PoisoningMetrics {
  [key: string]: unknown;
  by_model?: Partial<Record<ModelKey, ModelBlock>>;
  test_accuracy?: Ratio; clean_test_accuracy?: Ratio; undefended_test_accuracy?: Ratio;
  train_accuracy?: Ratio; train_accuracy_observed?: Ratio; target_class?: number;
  target_class_recall?: Ratio; clean_target_class_recall?: Ratio; undefended_target_class_recall?: Ratio;
  poison_rate?: Ratio; attack_success_rate?: Ratio; undefended_attack_success_rate?: Ratio;
  collateral_damage_rate?: Ratio; test_accuracy_delta?: { value: number | null; definition: string };
  test_outcomes?: { counts: Record<string, number>; reference: string; compared: string };
  defended_test_outcomes?: { counts: Record<string, number>; reference: string; compared: string };
  flagged?: Ratio; detection_recall?: Ratio; detection_precision?: Ratio; false_positive_rate?: Ratio;
  labels_repaired?: Ratio; labels_corrupted?: Ratio; removed?: Ratio; recovery_rate?: Ratio;
  clean_data_test_accuracy?: Ratio; clean_data_flag_rate?: Ratio;
  detection_counts?: Record<string, number>;
}

export interface PoisoningCase {
  id: string; family: "poisoning"; title: string; owasp_id: string; poison: LabelFlipConfig; defense: KnnDefense;
  defended_evaluation: "replay"; invariants: Invariant[]; rationale: string;
}

export interface VerifyReport {
  experiment_id: string; ok: boolean; guarantee: string;
  checks: { check: string; status: "pass" | "fail" | "skipped"; detail: string }[];
}
