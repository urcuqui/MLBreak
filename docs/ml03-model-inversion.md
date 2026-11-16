# ML03 · Private Attribute Inversion

Laboratory for **OWASP ML03:2023 Model Inversion Attack**: how the outputs of a traditional ML model can reveal a
sensitive attribute of a specific record, how much of that is really caused by the model, and what output
restrictions change. Synthetic data only.

Code: [`src/mlbreak/inversion/`](../src/mlbreak/inversion/) · UI: **OWASP ML03 → Model Inversion → Private
Attribute Inversion** (`/lab/ml03/baseline`) · CLI: `mlbreak invert …`, `mlbreak demo --lab ml03` · Evidence:
`mlbreak.inversion/v1` · Regression: MLB-REG-014 … 018.

## 1. OWASP mapping

| OWASP ML03:2023 element | MLBreak |
|---|---|
| Model outputs are used to infer sensitive information about inputs | A risk score computed on a confidential attribute is used to reconstruct that attribute (§4) |
| Attacker has API access to the model | Bounded black-box prediction service, counted and budgeted queries (§2) |
| Impact: confidentiality / privacy of individuals | Reconstruction accuracy **and** the advantage over an auxiliary-only attacker (§5) |
| Prevention: limit model output information | Quantised probability and label-only output policies, each evaluated with a new adaptive attack (§6) |
| Prevention: access control, rate limiting | Per-target query budget (attack parameter); operational rate limiting is not modelled |
| Prevention: differential privacy, regularisation | **Not implemented** — no formally specified mechanism is validated here (§10) |

**What this lab is not.** *Membership inference* (ML04) asks whether a record was in the training set; here every
target is a held-out record that was never trained on, so success does not depend on membership. *Model
extraction* (ML05) copies the model; here the attacker never learns or estimates the model's parameters — it only
compares the service's answers for two crafted records with one observed answer. Reconstruction of whole training
records (Fredrikson et al., 2015, face reconstruction) is out of scope.

## 2. Threat model

**System under test.** A fictional *account-risk scoring service*. For each account holder it receives four public
features and one confidential attribute and returns a risk output. A consumer (e.g. a decision letter, a partner
API) sees that output.

**Sensitive information.** `confidential_support_programme` ∈ {0, 1}: whether the (fictional) person is enrolled in
a confidential hardship-support programme. It is an **input** of the model and is distinct from the **prediction
target** `elevated_payment_risk` ∈ {0, 1}.

| The attacker KNOWS | The attacker does NOT know |
|---|---|
| the target's four public features, exactly | the target's private attribute |
| the service's released output for the target's real record (it is "observed", e.g. a leaked decision letter — not a query) | the target's label |
| that the attribute is binary, {0, 1} | the model's weights, intercept or type |
| an auxiliary population sample: public features + attribute of 600 *other* people, no labels, no victim outputs | the private training records, or which records were trained on |
| | the exact training set |

**Capabilities.** Black-box access to the same prediction service with crafted records (public features + a
candidate attribute), at most `query_budget` queries per target (default 2, max 16), counted by the oracle and
refused beyond the budget. The attacker cannot change the model, the output policy or the decision threshold.

**Assumptions behind every inference** (stated in each record's `attacker_capabilities`):

1. The service is **deterministic**: the same record always gets the same output. (True here; predictions are
   computed without BLAS so one row and a batch give bit-identical results.)
2. The attacker's public features are **exactly** the values the service used for the target.
3. The attribute domain is **exhaustive** ({0, 1}), so excluding one value implies the other.
4. The observed output was produced under the **same output policy** as the attacker's queries.

Under 1–4 the real target's own candidate always reproduces the observed output. That is what makes an
*elimination* inference valid, and it is also why full-probability reconstruction is near-certain here (§7):
these are strong, favourable-to-the-attacker assumptions, stated rather than hidden.

## 3. System under test

**Data** (`synthetic-account-risk-v1`, [data.py](../src/mlbreak/inversion/data.py)), seeded with
`default_rng([seed, 3003])`, seed bounded to 0 … 100 000:

* public features x ∈ R⁴ ~ N(0, I);
* private attribute P(s = 1 | x) = σ(−0.2 + 0.9·x₀ − 0.7·x₂) — partly predictable from public data;
* label P(y = 1 | x, s) = σ(−0.9 + 0.5·x₀ + 0.8·x₁ − 0.6·x₃ + **1.8·s**) — the attribute genuinely matters.

Three disjoint splits: **train** 1 200 (victim's private records: x, s, y), **auxiliary** 600 (attacker's
population sample: x, s, no y), **evaluation** 400 (held-out targets). Each split has a SHA-256 in every record.
No real personal, medical or financial record is used or imitated.

**Victim** (`logreg`): scikit-learn `LogisticRegression(C=1, lbfgs, max_iter=1000, random_state=seed)` on
[x, s] → y, trained in-process (no pickle). Artifact hash over the parameters and configuration. Seed 7: attribute
weight **1.64**; held-out decision accuracy 301/400 = 75.25 % vs 297/400 = 74.25 % for a reference model trained
without the attribute; log-loss 0.496 vs 0.549. The attribute is relevant to the prediction, which is the
precondition for leakage.

**Negative-control victim** (`logreg-public-only`): same service interface — it still receives the attribute in
every record — but the model was trained on the public features only (attribute weight exactly 0). Its output
carries no attribute information by construction.

**Attacker's auxiliary model**: logistic regression x → s fitted on the auxiliary split. It never sees a victim
output.

## 4. Attack methodology

Candidate matching ([attacks.py](../src/mlbreak/inversion/attacks.py)), after the known-output inversion of
Fredrikson et al. (2014). For each selected evaluation record:

1. take its public features (attacker knowledge);
2. the evaluation engine obtains the service's released output **for the real record** and hands the attacker an
   `AttackerView` = (public features, observed output, attacker's auxiliary prior P(s=1|x));
3. order candidates: the one the auxiliary prior prefers first (0 on an exact 0.5), then the other;
4. query `(public features, s = c)` in that order, while budget remains, until only one candidate is consistent;
5. a candidate is **consistent** if its response is within `tolerance` (default 1e-9) of the observed output; an
   unqueried candidate cannot be excluded and stays consistent;
6. decide: one consistent queried candidate → `unique_match`; one consistent unqueried candidate (all others
   excluded) → `elimination`; several → **ambiguous**, resolved by `tie_policy`: `auxiliary_prior` (the
   prior's choice — never better than the baseline on that target) or `abstain` (no inference, counted as not
   correct); none → `no_consistent_candidate` (impossible under §2 assumptions; treated as ambiguous);
7. the evaluation engine compares the inference with ground truth.

**Ground-truth isolation.** The attack code receives only an `AttackerView` and a counting `query` closure. The
module imports nothing that holds data, models or the engine (enforced by a test); a test permutes the hidden
attributes held by the evaluator and checks that every inference is unchanged. The attacker-facing endpoints
(`POST /api/inversion/targets`, `POST /api/inversion/query`) never return the attribute, the label or the
pre-policy score, and the query endpoint evaluates exactly the record it is given (no lookup by target id). Ground
truth appears only in the stored evidence, in an evaluation-only block.

**Determinism.** No randomness in the attack; target selection `order: "random"` uses `default_rng([seed, 3103])`.

## 5. Baseline comparison and metrics

The **auxiliary-only attacker** guesses the attribute from the public features with its auxiliary model, with no
victim output and no query, on the **same targets**. High inversion accuracy alone is never treated as leakage.

Metrics ([metrics.py](../src/mlbreak/inversion/metrics.py)), each ratio with numerator, denominator, population and
definition; an empty denominator gives `null`, never 0. ML03 does **not** reuse ML01's "attack success": a
reconstruction is *correct* when the inferred value equals the true attribute, and abstention is never correct.

| Metric | Definition |
|---|---|
| `attribute_inference_accuracy` | correct reconstructions / targets |
| `baseline_inference_accuracy` | correct auxiliary-only guesses / targets |
| `inference_advantage` | (correct by inversion − correct by baseline) / targets — signed |
| `leakage_p_value` | exact one-sided McNemar on paired per-target correctness, inversion vs baseline; **leakage claimed** only if p < 0.05 and inversion wins more discordant pairs |
| `recall_private_0/1`, `baseline_recall_private_0/1` | per-attribute recall |
| `ambiguity_rate`, `unambiguous_accuracy`, `abstention_rate` | targets with > 1 consistent candidate; accuracy on the rest; abstentions |
| `queries` (mean, max, total), `query_budget_utilisation`, `budget_exhaustion_rate` | queries sent (the observed output is not a query); targets left ambiguous because the budget ran out first |
| `victim_prediction_accuracy` | service decision 1[P ≥ 0.5] on the full score vs true label (whole evaluation split) |
| `consumer_decision_accuracy`, `released_brier_score`, `released_ranking_auc`, `score_distortion` | utility of the RELEASED output to a consumer |
| `undefended_attribute_inference_accuracy`, `defense_accuracy_reduction`, `defense_reduction_p_value` | adaptive runs only: the same targets against full probability, paired reduction and its one-sided McNemar test |

## 6. Defensive controls (output policies)

| Policy | Releases |
|---|---|
| `full_probability` | P(high risk), float64 — the undefended service |
| `quantized_probability` | midpoint of the equal-width bucket containing P (`bins` ∈ 2 … 1000) |
| `label_only` | 1[P ≥ 0.5] |

The policy is applied **identically** to the observed target output and to every candidate query (a service
cannot tell them apart). A policy is evaluated by a **new adaptive inversion** (`POST
/api/experiments/inversion/defend`) with the parent attack's model, seed, targets and attack configuration;
replaying the parent's observations would be meaningless because the policy changes the observation itself. The
service always decides on the full score, so decision accuracy is unaffected; a consumer that thresholds the
released value is measured separately.

**Output restriction is not a formal privacy guarantee.** It removes the information this attack uses; it bounds
nothing. Differential privacy is deliberately not implemented: a correct mechanism needs a specified neighbouring
relation, sensitivity analysis and privacy accounting that this lab does not validate.

## 7. Results and privacy–utility analysis (seed 7, 200 targets, default attack)

Auxiliary-only attacker: **138/200 = 69.0 %** correct (recall 80/109 for s=0, 58/91 for s=1).

| Policy | Inversion correct | Advantage | Leakage (p) | Ambiguous | Reduction vs full (p) | Consumer decisions | Brier ↓ | Ranking AUC | Score distortion |
|---|---|---|---|---|---|---|---|---|---|
| full probability | 200/200 | +62 (+31.0 %) | claimed (2.2e-19) | 0 | — | 301/400 | 0.1647 | 0.839 | 0 |
| quantised, 10 buckets | 200/200 | +62 | claimed (2.2e-19) | 2 | 0 (p = 1) | 301/400 | 0.1659 | 0.835 | 0.024 |
| quantised, 4 buckets | 193/200 | +55 (+27.5 %) | claimed (2.8e-17) | 20 | 7 (p = 0.0078) | 301/400 | 0.1684 | 0.820 | 0.060 |
| label only | 168/200 | +30 (+15.0 %) | claimed (9.3e-10) | 104 | 32 (p = 2.3e-10) | 301/400 | 0.2475 | 0.753 | 0.255 |
| negative control (attribute ignored), full prob. | 138/200 | 0 | **not claimed** (p = 1) | 200 | — | 297/400 | 0.1830 | 0.798 | 0 |

Mean queries per target: 1.69 (full), 1.85 (label only); with `query_budget = 1, tie_policy = abstain` the attack
is correct only by elimination: 62/200, 138 abstentions.

Reading the table:

* **Full probability leaks the attribute exactly** under the §2 assumptions; 62 of the 200 correct
  reconstructions are attributable to the model (the other 138 the auxiliary attacker also gets).
* **Quantisation helps only when the bucket width approaches the attribute's effect on the score.** 10 buckets
  change nothing; 4 buckets remove 7 reconstructions at a small cost in score quality. Over seeds 1–20 the 4-bucket
  reduction was positive on 20/20 seeds but significant (p < 0.05) on only 7/20.
* **Label-only output limits but does not eliminate leakage**: whenever the attribute flips the decision, the label
  reveals it. Over seeds 1–20 the advantage stayed between +14.0 and +23.5 points (p ≤ 3.7e-9) while the
  reduction versus full probability was significant on every seed. Its utility cost is in calibration and ranking
  (Brier 0.165 → 0.248, AUC 0.839 → 0.753), not in decisions.
* **The negative control** gives identical accuracy to the auxiliary attacker, zero advantage and no claim — high
  raw accuracy (69 %) with no model-induced leakage.

## 8. Reproducibility and evidence

* Records `mlbreak.inversion/v1`: per target, an **attacker trace** (public features, observed output, every query
  with its response and distance, consistent set, rule, inference, budget flags) and an **evaluation-only** block
  (true attribute and label); per evaluation record, a service row (full score, released value, decisions).
* Integrity verification ([integrity.py](../src/mlbreak/inversion/integrity.py)) re-derives every decision from
  its stored queries with the attack's own `decide`, checks budgets, query order, distances and that every
  released value is one the policy can produce, cross-checks each observed output against the service row of the
  same record, recomputes correctness and every metric, and verifies lineage to the parent attack. A forged
  decision fails even if all hashes are recomputed (tested).
* Exact replay re-executes the request and compares split, model and auxiliary-model hashes, inferred values,
  metrics and results fingerprints.
* Reports have the shared seven sections (see [evidence.md](evidence.md)).

## 9. Regression cases

Thresholds were derived from the seed-7 runs and a 20-seed sweep (seeds 1–20, 200 targets, default attack); each
invariant's description records the observed value.

| Case | Claim | Key invariants (observed at seed 7) |
|---|---|---|
| MLB-REG-014 | inversion gains measurable information over the auxiliary-only baseline | advantage ≥ 0.20 (0.31; sweep 0.27–0.39), leakage p ≤ 0.001 (2.2e-19), baseline ≥ 0.55 (0.69) |
| MLB-REG-015 | quantisation changes effectiveness without being assumed secure | paired reduction > 0 (0.035), defended advantage ≥ 0.20 (0.275), leakage p ≤ 0.001, consumer decisions ≥ 0.74 (0.7525) — significance deliberately NOT asserted |
| MLB-REG-016 | label-only limits score information but does not eliminate leakage | paired McNemar ≤ 0.001 (2.3e-10), ambiguity ≥ 0.30 (0.52), advantage ≥ 0.10 (0.15; sweep 0.14–0.235), leakage p ≤ 0.001 |
| MLB-REG-017 | negative control: no spurious leakage claim | advantage == 0, leakage p ≥ 0.05 (1.0), ambiguity == 1.0 |
| MLB-REG-018 | budget and evidence consistent and reproducible | 1-query budget: utilisation ≤ 1, exhaustion ≥ 0.5 (0.69), abstentions not counted (accuracy ≤ 0.40, observed 0.31), both records verify and the attack replays bit-identically |

## 10. Limitations

* **Strong attacker assumptions**: exact public features and a deterministic service. With noisy auxiliary
  knowledge, exact matching would fail and a probabilistic (MAP) inversion would be needed; not implemented.
* **Binary attribute, linear model, synthetic data**: results do not transfer to multi-valued or continuous
  attributes, non-linear models or real populations. The generator makes the attribute strongly predictive by
  design.
* **One attack**: candidate matching with an auxiliary prior. Its accuracy against a policy is a lower bound for
  that policy.
* **No formal privacy mechanism**: output policies are not differential privacy; no rate limiting across targets,
  no anomaly detection of crafted queries.
* **Held-out targets only**: no training-record reconstruction, no membership inference (ML04).
* **Scores are LR probabilities** with their usual calibration; utility metrics are on 400 synthetic records.
* Single seed per regression case; margins and seed ranges are documented above.

## 11. Responsible use

The lab demonstrates a privacy risk on fictional records to teach measurement and defense. Do not point the attack
at real services or real people's data. The vulnerable service, its data and the attack all run locally; nothing is
downloaded at runtime. See [responsible-use.md](responsible-use.md).

## Running it

```bash
make demo                                                        # all labs, including ML03
PYTHONPATH=src .venv/bin/python -m mlbreak demo --lab ml03       # baseline → inversion → 4 buckets → label-only → negative control → replay
PYTHONPATH=src .venv/bin/python -m mlbreak invert attack --queries 2
PYTHONPATH=src .venv/bin/python -m mlbreak invert defend <attack-id> --policy label_only
```
