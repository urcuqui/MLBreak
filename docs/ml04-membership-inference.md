# ML04 · Training Set Membership Inference

> **Intentionally vulnerable laboratory.** Synthetic data only — no real person, account or transaction. For
> education and defensive research; local use only. See [responsible use](#14-responsible-use).

MLBreak's ML04 lab measures whether a black-box attacker can tell, from a classifier's released predictions, if a
given record was in its **training set**. The attacks run for real, are calibrated only on the attacker's own data,
are evaluated on untouched members and non-members, and are recorded as verifiable, replayable evidence
(`mlbreak.membership/v1`).

Every number in this document was produced by the code in this repository: the seed-7 experiments of
`python -m mlbreak demo --lab ml04`, and two 20-seed studies (`mlbreak member study --seeds 0-19`, with and without
`--label-unknown`).

## 1. OWASP ML04 mapping

| OWASP ML04:2023 element | MLBreak |
|---|---|
| Attacker queries a model with a record and infers from the output whether it was in the training data | one black-box query per candidate record; the attacker decides member / non-member |
| Overfitting makes models behave differently on training records | intentionally overfitting random forest (training accuracy 1.0, held-out 0.725 at seed 7) vs a lower-capacity logistic regression |
| Shadow-model attacks | `n` shadow models trained on the attacker's own auxiliary population with known splits; gradient-boosting meta-classifier |
| Threshold / loss attacks | calibrated threshold on true-label confidence, −NLL or maximum confidence |
| Prevention: regularisation, simpler models, restricting outputs | leaf-size regularisation / L2 strength, complexity reduction (depth, trees), quantised or label-only output — each evaluated by a **recalibrated** attack |
| Reference | <https://owasp.org/www-project-machine-learning-security-top-10/docs/ML04_2023-Membership_Inference_Attack> |

How it differs from the neighbouring categories:

| | Question the attacker answers | Lab |
|---|---|---|
| **ML03** Model inversion | *What is this record's hidden attribute?* (the record need not be a training member) | `ml03-private-attribute-inversion` |
| **ML04** Membership inference | *Was this exact record used to train the model?* | `ml04-training-membership-inference` |
| **ML05** Model theft | *Can I reproduce the model's function or parameters?* | not implemented |

## 2. Threat model

**Asset.** Confidentiality of training-set membership — that a fictional customer's record was used to train a
transaction-risk classifier.

**Trust boundary.** Prediction service → output consumer. The attacker sees the service's released output for the
records it submits. It does not see the training set, training indices, the fitted model, or any evaluator evidence.

**The attacker knows:**

* the 8 feature values of each candidate record;
* each candidate's true label — **only when `label_known = true`** (default). With `label_known = false` the
  threshold attack must use `max_confidence` and the shadow meta-classifier sees only the released output;
* the victim's **training recipe** (model family and hyperparameters, including any training-time defense) and the
  training-set size (600). This is the standard shadow-model assumption (Shokri et al., 2017) — Kerckhoffs'
  principle applied to the training procedure. A defense that relies on the recipe being secret is not evaluated;
* two **auxiliary populations** drawn from the same distribution as the victim's data but sharing no record with it:
  `shadow` (2400 records) and `calibration` (1200 records), with labels;
* the output policy of the service (it sees its outputs).

**The attacker does not know** whether any candidate is a member, the training indices, the model parameters, or
anything in the evaluator's evidence. It can query the victim **once per candidate** (counted, budgeted); shadow and
reference models are trained locally and **never query the victim**.

**Goal.** Decide membership for each candidate better than the no-signal baseline, at operating points chosen before
looking at the evaluation population.

**Ground truth.** Only the evaluation engine (`src/mlbreak/membership/engine.py`) knows which candidate is a member.
The attack functions receive an `AttackerKnowledge` (candidate features, optional labels, auxiliary populations,
recipe, policy) and a counting query function; the evaluator attaches `MembershipTruth` after the attack returns.
Candidate order is shuffled so position does not reveal the group. This is checked executably: re-running an attack
with the evaluation ground truth **permuted** must give bit-identical attacker outputs (`attacker_independence`,
MLB-REG-020/021).

**Role separation in the API.** Every ML04 endpoint is **evaluator-facing** (tag `ml04`): records contain the
membership ground truth. There is no attacker-facing ML04 endpoint; with `MLBREAK_EVALUATOR_TOKEN` set, all of them
require the evaluator token (tested).

## 3. Synthetic data and populations

Dataset `synthetic-transaction-risk-v1`: `x ~ N(0, I₈)`, `P(y = 1 | x) = σ(−0.2 + x·w + 0.7·x₂·x₃)` with
`w = [1.0, −0.8, 0.6, 0, 0.5, 0, −0.4, 0]`. Labels are noisy by construction, so a model that fits its training labels
perfectly has memorised individual records — the controlled overfitting that makes membership observable. Generated
with `numpy.random.default_rng([seed, 4004])`; the linear part is an explicit column sum (no BLAS), so the data are
identical in every numerical environment.

| Population | Size | Role | Who knows it |
|---|---|---|---|
| `victim_train` | 600 | victim's training records = **members** | evaluator only |
| `victim_test` | 600 | held-out victim utility and generalisation gap (never attacked) | evaluator only |
| `shadow` | 2400 | attacker's shadow models (known in/out splits of 600/600) | attacker |
| `calibration` | 1200 | attacker's reference models for threshold calibration (600/600 splits) | attacker |
| `evaluation_non_members` | 600 | non-member pool of the final evaluation | evaluator only |
| `decoy` | 600 | training set of the negative-control victim only | evaluator only |

**Disjointness is verified, not assumed:** every record's float64 feature vector is hashed (SHA-256) and the record
stores the number of duplicate records and the pairwise overlap of every pair of populations (all 0;
`population_disjointness` = 1 in every record, MLB-REG-020).

**Final evaluation candidates** (`CandidateSelection`): `n_members` records of `victim_train` and `n_non_members` of
`evaluation_non_members` (default 250 + 250, balanced). Unequal counts set a different evaluation prevalence;
`operational_prevalences` (default 50%, 10%, 1%) are additional prevalences for which precision and false discovery
rate are **projected** from the measured TPR and FPR.

## 4. Victim models

All victims are scikit-learn estimators trained in-process (no pickle/joblib), hashed over every fitted parameter, and
served through the same `MembershipService` interface (batch of records in, one released value per record out).

| `model_id` | Recipe | Trained on | Train / held-out accuracy (seed 7) |
|---|---|---|---|
| `random-forest-overfit` | 100 fully grown trees, `min_samples_leaf = 1`, `max_features = sqrt` | `victim_train` | 1.000 / 0.725 (gap 0.275) |
| `logreg-l2` | logistic regression, L2, C = 1.0 | `victim_train` | 0.727 / 0.740 (gap −0.013) |
| `random-forest-decoy` | same as the overfit forest | `decoy` — **negative control** | 1.000 / 0.702 |

We do not assume any model leaks; leakage is measured. The logistic regression is a lower-capacity comparison, not a
guaranteed non-leaking control (see §9).

## 5. Attacks

Both attacks run in every attack experiment, on the same candidates, with the same victim queries and the same
calibration data.

### A. Threshold attack

Signal per candidate, oriented so that higher = more member-like:

* `true_label_confidence` — released probability of the true label (label known);
* `negative_log_likelihood` — `log p_y` = −NLL (label known). For a single released probability this is a **monotone
  transform** of the true-label confidence: identical ranking, ROC and decisions (verified at seed 7: identical AUC
  0.896 and advantage 0.616); it is offered because both appear in the literature, not as a different attack;
* `max_confidence` — `max(p, 1 − p)` (label unknown). Under label-only output it is constant: no signal.

Thresholds: **global**, or **class-conditional** (per true label, or per released decision when the label is unknown)
when each class has at least `min_class_calibration` members **and** non-members in the calibration data; otherwise
that class falls back to the global threshold and the record says why (`class_modes`).

### B. Shadow-model attack

1. Train `n_shadow_models` (default 8, max 16) models with the victim's recipe, each on 600 random records of the
   attacker's `shadow` population, holding out 600 others. Their membership labels are known by construction and are
   never the victim's.
2. Pass each shadow model's outputs through the service's output policy; features: released probability, and when the
   label is known, the label, true-label confidence, its log and the maximum confidence.
3. Train a `GradientBoostingClassifier` (100 trees, depth 2) meta-classifier on all shadow examples (9 600 by default).
4. Calibrate its operating thresholds on the reference models (§6), exactly like the threshold attack.
5. Score the victim's released outputs for the candidates.

The record stores every shadow model's artifact hash and training-row hash, the meta-classifier's hash and the hash of
its training examples (`shadow` block).

## 6. Calibration / evaluation separation

Thresholds are **never** chosen on the final evaluation population. Both attacks are calibrated on **reference
models**: `n_reference_models` (default 2) models trained by the attacker with the victim's recipe on 600 records of
its `calibration` population, each with 600 held-out records — 2 400 calibration records with attacker-known
membership. The calibration scores, membership flags and class keys are stored, and integrity verification
**re-derives every threshold from them**.

Two operating points:

* **balanced** — the threshold maximising Youden's J = TPR − FPR on calibration data (ties → the highest threshold).
  Accuracy, balanced accuracy, precision, recall, advantage and the confusion matrix are reported here. If J ≤ 0 for
  every threshold the point is `undefined` and every candidate is called a non-member.
* **low_fpr** — the lowest threshold whose **calibration** FPR ≤ `target_fpr` (default 1%). Its status is:
  * `calibrated` — at least 10 calibration non-members may be false positives at that target (≥ 1 000 non-members for
    1%: the default 2 reference models give 1 200);
  * `approximate` — fewer than 10 (e.g. one reference model, 600 non-members → 6): coarse FPR resolution;
  * `undefined` — no threshold reaches the target FPR while calling any calibration member a member (e.g. quantised or
    label-only scores are too coarse). **TPR at 1% FPR is then reported as undefined, never as 0 or as a guess.**

The evaluation FPR actually obtained at the low-FPR threshold is reported next to the TPR with a 95% Wilson interval;
if the interval's lower bound exceeds the target the record flags `exceeds_target` (the calibrated threshold did not
transfer). Seed 7: threshold attack TPR 45/250 = 18.0% at an evaluation FPR of 1/250 = 0.4%.

The ROC curve of the evaluation population is reported for description only; the operating points marked on it in the
UI are the calibrated ones.

## 7. Leakage metrics

All metrics are derived from the per-candidate evidence and re-derived on verification; every ratio carries its
numerator, denominator, population and definition.

| Metric | Definition |
|---|---|
| `<attack>_roc_auc` | P(member score > non-member score), ties ½ — threshold-free; 0.5 = no signal |
| `<attack>_leakage_p_value` | one-sided Mann–Whitney test of AUC > 0.5 (normal approximation, tie and continuity corrections) |
| `leakage_claimed` | 1 iff some attack has p < 0.05 / 2 (Bonferroni over the two attacks) **and** AUC > 0.5 |
| `<attack>_membership_accuracy`, `_balanced_accuracy` | at the balanced operating point |
| `<attack>_membership_precision` / `_recall` / `_false_positive_rate` | TP/(TP+FP), TP/P, FP/N at the balanced point |
| `<attack>_membership_advantage` | TPR − FPR (Yeom et al.; 0 = the no-signal baseline) |
| `no_signal_accuracy`, `<attack>_accuracy_over_no_signal` | majority-group accuracy, and the attack's margin over it |
| `<attack>_tpr_at_low_fpr`, `_fpr_at_low_fpr` | at the calibrated low-FPR threshold, with Wilson intervals and status |
| `<attack>_operating_point_p_value` | one-sided Fisher exact test at the balanced point |
| `<attack>_operational_precision` | projected precision / FDR at each declared prevalence π: TPR·π / (TPR·π + FPR·(1−π)) |
| `<attack>_confusion`, `<attack>_roc_curve` | TP/FP/TN/FN; (FPR, TPR) points of the evaluation population |
| `<attack>_calibration` | calibration AUC and calibration TPR/FPR of both operating points (**calibration population**) |
| `train_accuracy`, `test_accuracy`, `generalization_gap`, `test_roc_auc`, `consumer_decision_accuracy`, `released_brier_score` | victim utility |
| `queries`, `query_budget_utilisation` | one victim query per candidate, against `query_budget` |
| `<attack>_defense_reduction_p_value`, `_correctness_reduction`, `_undefended_membership_accuracy` | defend only: paired over the same candidates (one-sided exact McNemar) |

Precision depends on prevalence: at seed 7 the threshold attack's precision is 75.2% (230/306) at a 50% prevalence,
a projected 25.2% at 10% and 3.0% at 1% — most records called "member" would be false discoveries when members are
rare.

## 8. Defenses, privacy–utility trade-off and where defenses fail

Each defense is evaluated by a **recalibrated** attack: the attacker retrains its reference and shadow models with the
defended recipe and observes them through the defended output policy, then attacks the same 500 candidates.

**Seed 7** (`mlbreak demo --lab ml04`):

| Configuration | Held-out acc. | Gap | Threshold AUC | Shadow AUC | Advantage (thr) | TPR @ 1% FPR (thr) | Leakage claimed | Paired reduction p (thr) |
|---|---|---|---|---|---|---|---|---|
| Undefended forest | 0.725 | 0.275 | 0.896 | 0.895 | +0.616 | 18.0% | yes | — |
| Regularisation (leaf ≥ 25) | 0.703 | 0.085 | 0.568 | 0.539 | +0.048 | 3.2% | yes (threshold, p = 0.0045) | 1.8e-43 |
| Complexity reduction (depth 4, 30 trees) | 0.678 | 0.153 | 0.597 | 0.606 | +0.128 | 2.8% | yes | 1.9e-37 |
| Quantised output (4 buckets) | 0.725 | 0.275 | 0.817 | 0.826 | +0.608 | undefined | yes | 0.34 |
| Label-only output | 0.725 | 0.275 | 0.640 | 0.661 | +0.280 | undefined | yes | 3.5e-15 |
| Logistic regression (comparison) | 0.740 | −0.013 | 0.511 | 0.468 | +0.024 | 2.4% | no | — |

**20 seeds (0–19), label known** — min / median / max:

| Configuration | Threshold AUC | Shadow AUC | Utility drop vs undefended | Significant reduction (thr / shadow) | Seeds with leakage still claimed |
|---|---|---|---|---|---|
| Undefended forest | 0.850 / 0.902 / 0.928 | 0.850 / 0.904 / 0.931 | — | — | 20/20 |
| Regularisation | 0.525 / 0.570 / 0.617 | 0.530 / 0.557 / 0.604 | −0.028 / −0.003 / +0.022 | 20 / 20 | 17/20 |
| Complexity reduction | 0.550 / 0.602 / 0.630 | 0.553 / 0.598 / 0.632 | −0.028 / +0.012 / +0.047 | 20 / 20 | 20/20 |
| Quantised (4 buckets) | 0.800 / 0.847 / 0.862 | 0.805 / 0.859 / 0.879 | 0 | **0 / 0** | 20/20 |
| Label-only | 0.628 / 0.643 / 0.666 | 0.619 / 0.659 / 0.698 | 0 | 20 / 20 | 20/20 |

(Negative utility drop = the defended model was more accurate on held-out data.)

**20 seeds, label unknown** (`max_confidence`): undefended AUC 0.835–0.912; regularisation 0.504–0.605 (leakage still
claimed on 8/20 seeds); complexity reduction 0.524–0.602 (13/20); quantised 0.766–0.836 (20/20); **label-only: AUC
exactly 0.500 for the threshold attack on every seed and no claim on any seed** — without the label and without a
score the attacker has nothing to threshold.

What this shows:

* **Regularisation and complexity reduction reduce the measured leakage substantially, at little utility cost**
  (AUC drop 0.26–0.38 on every seed, utility within ±5 points) because they shrink the generalisation gap the attacks
  exploit. **They do not remove it**: with the label known, leakage is still significant on 17/20 (regularisation) and
  20/20 (complexity) seeds. MLB-REG-022 pins both the reduction and the residual signal.
* **Defense failure — quantisation.** Four buckets barely change the ranking of member vs non-member confidences; the
  paired reduction is never significant (0/20 seeds) and the low-FPR operating point becomes undefined (the scores are
  too coarse to calibrate a 1% FPR), which reduces the attacker's *precision at low FPR* but not its accuracy.
* **Defense failure — label-only against a label-aware attacker.** Hiding the score removes the confidence signal but
  not *correctness*: an overfit model is right on (almost) every member and wrong on ~28% of non-members, so an
  attacker who knows the label still gains an advantage of 0.256–0.332 (MLB-REG-025). Label-only output only defeats
  the label-*unaware* attacker.

### Why output restriction is not differential privacy

Quantisation and label-only output are **deterministic** functions of a model that is itself a deterministic function
of the training set. Two neighbouring training sets can therefore produce different released outputs with probability
1, so the privacy loss of releasing them is unbounded — **ε = ∞**. Output restriction removes information that some
attacks use; it does not bound what any attack can learn. This output restriction is **not differential privacy**, and
neither are regularisation or complexity reduction. No DP training mechanism (e.g. DP-SGD) is implemented in this lab.

## 9. Negative controls

* **Decoy victim** (`random-forest-decoy`): the same overfitting recipe trained on the disjoint `decoy` population, so
  **no** evaluation candidate is in its training set (`in_training = false` for every candidate). Same candidates,
  same attacks. It is as overfit as the real victim (gap 0.298 at seed 7), so a null result is due to non-membership,
  not to a weaker model. Seed 7: threshold AUC 0.498, shadow AUC 0.505, no claim (MLB-REG-023).
  Over 20 seeds: AUC 0.451–0.544 (both attacks) and **0/20** claims with the label known; with the label unknown AUC
  0.438–0.572 and **2/20** claims. With α = 0.05 (Bonferroni-split across two attacks), at most about 1 false claim in
  20 seeds is expected under the null; 2 of 20 is within chance (P ≈ 0.26), and it is reported here rather than hidden. A single-seed
  leakage claim is evidence, not proof.
* **Logistic regression**: its training records ARE members, but its capacity is low (gap ≈ 0). AUC 0.438–0.557 over
  20 seeds (both attacks, both studies); leakage claimed on 2/20 seeds in each study. We report it as "no material leakage measured by these
  attacks", not as "no leakage".
* The **no-signal baseline** (always guessing the larger group; advantage 0) is reported in every record.

## 10. Evidence and reproducibility

Record family `mlbreak.membership/v1` (new; no existing schema version changed). Each record stores:

* the victim configuration, recipe and artifact hash, its training population and that population's hash;
* the dataset generator, population sizes and fingerprints, and the row-level disjointness check;
* attacker capabilities, the attack configuration, the output policy and the defense (with lineage to the parent
  attack: parent hashes, shared dataset and candidates, the defense parameters applied);
* calibration provenance: reference-model hashes and training-row hashes, the calibration population hash, and per
  attack the calibration scores, attacker-known membership, class keys, operating points and calibration AUC;
* shadow provenance: shadow-model hashes, training and held-out row hashes, meta-classifier hash, features and the
  hash of its training examples;
* per candidate: an opaque `candidate_id`, the **attacker observation** (features, label if granted, released output,
  query index), both attacks' **scores and decisions** at both operating points, and — evaluator-only — the
  **ground truth** (designated group, real training membership, source population and index); in defend records also
  the parent's correctness per attack;
* per-record victim utility rows (train and held-out, full and released output);
* metrics with definitions and denominators; numerical environment; `config_sha256`, `metrics_sha256`,
  `results_sha256`, `record_sha256`; replay and verify commands.

`mlbreak check <id>` / `GET /api/experiments/<id>/verify` re-derive, without re-running any model: every threshold
score from the stored released output, every operating point from the stored calibration data, every decision, the
query order and budget, policy-valid outputs, ground-truth consistency (group ⇔ population; no training member for the
decoy) and every metric. Shadow meta-classifier scores cannot be re-derived without re-training; they are covered by
the results fingerprint and by **exact replay** (`mlbreak replay <id>`, refused if verification fails; compares
dataset, victim, calibration, shadow, attacker-output, metrics and results fingerprints). Integrity is not
authenticity (see [reproducibility](reproducibility.md#integrity-is-not-authenticity)).

The **multi-seed study** (`mlbreak member study`, `POST /api/membership/study`) stores per-seed rows and a summary
re-derived on verification (`MLB-MIS-<12 hex>` under `<data dir>/membership_studies/`), and can be replayed.

## 11. Security regression cases

| Case | What it pins (seed 7, recalibrated attacks; ranges are the 20-seed study) |
|---|---|
| **MLB-REG-019** | overfit victim leaks: gap ≥ 0.2 (0.275), threshold and shadow AUC ≥ 0.8 (0.896 / 0.895), advantage ≥ 0.5 (0.616), leakage claimed, TPR ≥ 10% at the calibrated 1% FPR threshold (18%) |
| **MLB-REG-020** | threshold calibration without evaluation labels: attacker independence (permuted ground truth ⇒ identical outputs; populations disjoint), disjointness = 1, evaluation FPR ≤ 3% at the 1%-calibrated threshold (0.4%), balanced point transfers (accuracy ≥ 0.75, 0.808) |
| **MLB-REG-021** | shadow attack learns from auxiliary membership only (label unknown): attacker independence, shadow AUC ≥ 0.8 (0.880; 0.836–0.912), p ≤ 0.001 |
| **MLB-REG-022** | leaf-size regularisation: defended AUC ≤ 0.65 for both attacks (0.568 / 0.539), paired reduction p ≤ 0.001, gap ≤ 0.15 (0.085), held-out accuracy ≥ 0.67 (0.703), **and** a residual-signal canary AUC ≥ 0.52 |
| **MLB-REG-023** | negative control: no leakage claim, AUC ≤ 0.6 for both attacks, advantage ≤ 0.15, while the decoy still overfits (gap ≥ 0.2) |
| **MLB-REG-024** | evidence integrity + lineage + bit-identical replay; query budget enforced (a budget one below the candidate count is refused and the failed record verifies); ≤ 1 query per candidate; complexity reduction significantly reduces the shadow attack with held-out accuracy ≥ 0.65 |
| **MLB-REG-025** | documented defense failure: label-only output reduces but does not stop a label-aware attacker (advantage ≥ 0.2, 0.280; leakage still claimed) while consumer decisions are unchanged |

Thresholds were set from the seed-7 values and the 20-seed ranges, with margins inside the observed range; the
ML01–ML03 cases (MLB-REG-001…018) are unchanged.

## 12. Running it

```bash
PYTHONPATH=src .venv/bin/python -m mlbreak demo --lab ml04            # baseline → attack → 4 defenses → controls → independence → replay
PYTHONPATH=src .venv/bin/python -m mlbreak member attack              # undefended forest, label known
PYTHONPATH=src .venv/bin/python -m mlbreak member attack --label-unknown --signal max_confidence
PYTHONPATH=src .venv/bin/python -m mlbreak member defend MLB-EXP-… --defense regularization --min-samples-leaf 25
PYTHONPATH=src .venv/bin/python -m mlbreak member study --seeds 0-19  # ≈ 2 minutes
PYTHONPATH=src .venv/bin/python -m mlbreak member independence MLB-EXP-…
```

API (all evaluator-facing, tag `ml04`): `POST /api/experiments/membership/{baseline,attack,defend}`,
`GET /api/membership/independence/{experiment_id}`, `POST /api/membership/study`,
`GET /api/membership/study/{report_id}` and `/verify`. UI: **OWASP ML04 → Training Set Membership Inference**
(`/lab/ml04/baseline`).

## 13. Known limitations

* **Two attacks, both lower bounds.** Stronger per-example attacks (e.g. likelihood-ratio attacks with many reference
  models per candidate) can reach much higher TPR at low FPR. A defense that holds here may not hold against them.
* **Strong attacker assumptions.** Recipe, training-set size and same-distribution auxiliary data are known; a real
  attacker may have less. Stale-calibration (non-adaptive) attacks are not evaluated.
* **One distribution, one size.** 8 Gaussian features, 600 training records, binary labels with a fixed noise level;
  leakage depends strongly on all of these. The overfit forest is deliberately extreme.
* **TPR at 1% FPR rests on 1 200 calibration non-members and 250 evaluation non-members.** The evaluation FPR at the
  calibrated threshold is a 250-sample estimate (Wilson interval reported); the 20-seed range of TPR at 1% FPR is
  4.0–23.2%.
* **Single-seed regression thresholds.** The cases pin seed 7; the multi-seed study shows where seeds differ (e.g. the
  TPR bound in MLB-REG-019 is seed-specific).
* **No formal privacy.** No differential-privacy mechanism, no bound on what an attacker can learn; regularisation
  and output restriction are mitigations only.
* **Evidence size.** An attack record is ≈ 0.9 MB of indented JSON (calibration scores and per-candidate evidence are
  stored so that thresholds and decisions can be re-derived).
* **Synthetic data only.** No conclusion transfers to a real dataset or model without measuring it there.

## 14. Responsible use

Membership inference reveals that a person's data was used to train a model — which can itself be sensitive (e.g. a
model trained only on patients with a condition). This lab exists to measure and reduce that risk in systems you own
or are authorised to test. Use it with synthetic or properly authorised data, keep the server local (it binds to
127.0.0.1 by default), and do not point the techniques at third-party services. Results are measurements on synthetic
data, not guarantees about any real system. See [responsible use](responsible-use.md).
