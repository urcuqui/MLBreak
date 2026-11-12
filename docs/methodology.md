# Experimental methodology

## Scenario

* **Dataset** `moons-v1`: `sklearn.datasets.make_moons(n_samples=500, noise=0.25, random_state=seed)`,
  stratified 70/30 split with `random_state=seed`. Feature domain `x1 ∈ [-2.5, 3.5]`, `x2 ∈ [-2.0, 2.5]`.
* **Models**: `mlp-tanh-16x16` (MLPClassifier, tanh, lbfgs, α=0.5) and `logreg`. Both are trained
  with `random_state=seed`. The MLP's L2 regularisation was chosen so the model is accurate
  (96.7% at seed 7) *and* not trivially over-confident. An over-confident model would make the
  rejection defense meaningless.

## Modes

| Mode / evaluation | What runs | Target |
|---|---|---|
| `baseline` / `clean` | Clean test split | undefended (or a defended system, if requested) |
| `attack` / `attack` | New attack | undefended |
| `defend` / `replay` | Parent attack's **stored** adversarial inputs, unchanged | defended |
| `attack` / `adaptive_attack` | **New** attack by an attacker who knows the defense | defended |

Replay and adaptive evaluation are always separate experiments, linked by `parent_experiment_id`.
Exact replay (`mlbreak replay`) re-executes any experiment to check reproducibility. It is not a
robustness evaluation.

## Attacks

* **FGSM** (Goodfellow, Shlens & Szegedy, 2015): `x' = Π(x − ε·sign(∇ₓ m(x)))`, L∞ only.
* **PGD** (Madry et al., 2018): iterated signed (L∞) or normalised (L2) steps of size
  `α = 2.5ε/steps` by default, projected onto the ε-ball ∩ domain. Optional seeded random start.
  The objective is the true-class margin `m(x) = s·logit(x)`. The attack stops early per sample once
  `m(x) < −κ` and returns the iterate with the lowest margin.
* **Adaptive PGD against rejection**: κ is raised to `logit(τ) + 10⁻³`, so the adversarial point is
  classified wrongly *with* confidence ≥ τ.
* **Random search** (decision-based black-box): `query_budget` points are sampled uniformly from the
  ε-ball (seeded per sample), clipped to the domain, sorted by perturbation norm and queried in
  batches of 50. The first point that changes the label (and is not an abstention) is returned. This
  is the minimum-norm success among the drawn candidates. Otherwise the attack reports failure.

Gradients are analytic (manual back-propagation over the fitted MLP parameters). Tests check them
against central finite differences and check the forward pass against `predict_proba`.

## Defenses

* **Adversarial training**: for `rounds` rounds, generate PGD examples (training ε, steps) on the
  training set against the current model, then retrain from scratch on clean + all adversarial
  copies.
* **Confidence rejection**: abstain if `max(p, 1−p) < τ`.

## Metrics

All metrics are computed in [`src/mlbreak/metrics.py`](../src/mlbreak/metrics.py) from the per-sample
evidence. Each ratio is stored with its `numerator`, `denominator`, `population` and `definition`. An empty
denominator gives `value: null` (undefined), never 0.

### Outcome partition (schema v2)

Every **selected** sample of an attack, adaptive attack or replay gets exactly one `outcome`. The counts
are stored in `metrics.outcomes` and always sum to `n_selected`.

| Outcome | Clean decision | Adversarial decision | Meaning |
|---|---|---|---|
| `evaded` | correct | other class (not abstain) | attack success |
| `abstained_adversarial` | correct | **abstain** | blocked by abstaining, **not** a correct prediction |
| `robust` | correct | true class | attack resisted |
| `clean_misclassified` | wrong class | — (not attacked) | |
| `clean_abstained` | abstain | — (not attacked) | |
| `not_perturbed` | correct | — (replay only: parent did not perturb it) | decision under evaluation = clean decision |

*Attack-eligible* = `evaded + abstained_adversarial + robust`: samples decided correctly on the clean
input for which an adversarial input was evaluated. For a replay, eligibility needs **both** a
perturbation from the parent **and** a correct clean decision by the defended system.

### Definitions

| Metric | Numerator | Denominator (population) |
|---|---|---|
| `clean_accuracy` | correct clean decisions | **all test-split samples** (abstentions are not correct) |
| `abstention_rate_clean` | clean inputs abstained on | all test-split samples |
| `coverage`* | non-abstained clean decisions | all test-split samples |
| `selective_accuracy`* | correct clean decisions | non-abstained clean decisions |
| `clean_accuracy_selected` | `evaded + abstained_adversarial + robust + not_perturbed` | selected samples |
| `adversarial_accuracy` | `robust + not_perturbed` | selected samples |
| `robust_accuracy` | `robust` | selected samples |
| `attack_success_rate` | `evaded` | attack-eligible samples |
| `abstention_rate_adversarial` | `abstained_adversarial` | attack-eligible samples |
| `attack_resisted_rate` | `robust` | attack-eligible samples |
| `abstention_rate_clean_selected` | `clean_abstained` | selected samples |
| `adversarial_coverage`* | selected − `clean_abstained` − `abstained_adversarial` | selected samples |
| `adversarial_selective_accuracy`* | `robust + not_perturbed` | selected samples with a non-abstained decision under the evaluation |
| `abstained_on_adversarial`* | alias of `abstention_rate_adversarial` (kept for API compatibility) | |

\* only reported for systems that can abstain (confidence rejection). For an argmax system these are
trivially 1 (coverage) or equal to their non-selective counterparts.

Invariants checked on every record (`metrics_arithmetic`): `attack_success_rate + abstention_rate_adversarial
+ attack_resisted_rate = 1` over the eligible population; `robust ≤ adversarial ≤ clean (selected)`.

**Adversarial vs robust accuracy.** *Adversarial accuracy* asks "how often is the system right under this
evaluation?". A sample with no adversarial input keeps its clean decision. *Robust accuracy* only counts
samples that were actually tested with an adversarial input **and** stayed correct. They coincide for a
fresh attack (every correctly classified sample is attacked) and differ in a replay when the parent left a
sample unperturbed.

**What changed in schema v2.** v1 computed `adversarial_accuracy = (clean-correct − successes) / selected`.
Because abstentions are not successes, every adversarial abstention was counted as a correct prediction. On
the reference rejection replay (τ = 0.8), that reported 86.7% instead of the correct 35.3%: 77 abstentions
were counted as correct. Attack success rate and clean metrics are unchanged. For systems that cannot
abstain, adversarial accuracy is also unchanged.

### Other measurements

* `perturbation`: L∞ and L2 statistics over attacked and over successful (`evaded`) samples, plus the
  largest observed ‖δ‖/ε, which is verified to be ≤ 1.
* `queries`: label queries (black-box) or model evaluations (white-box). Each sample uses at most
  `query_budget` (black-box) or `1 + 2·steps` (white-box).
* `comparison` (replay only): the undefended side is **recomputed from the parent's per-sample evidence**
  with the same definitions as the defended side. It is never copied from the parent's stored metrics.

## Lineage

A `parent_experiment_id` is accepted only for **defend/replay** and **adaptive attacks**. The parent must:

1. exist, have completed, and be a **non-adaptive attack against the undefended model**
   (`mode = attack`, `evaluation = attack`). Baselines, adaptive attacks, replays and failed runs are rejected;
2. pass the evidence integrity check (`mlbreak check`), so a tampered parent is rejected;
3. share scenario, model, seed, dataset split hashes and base-model artifact hash with the child;
4. for adaptive attacks, also use the same **norm and ε** and the **same selected test samples**. The
   attack type may differ (e.g. a black-box adaptive attack), and this is recorded as `lineage.attack_differs`.

The validated provenance is stored in `lineage`: relation, parent hashes (config, results, record), shared
hashes and the list of checks passed. It is covered by the child's `record_sha256`. A non-adaptive attack
cannot have a parent. Incompatible requests return HTTP 422 with the reason.

## Security regression

`security_cases/MLB-REG-*.json` declare invariants of three kinds:

* `security`: the defense must hold.
* `utility`: the defense must not destroy clean performance.
* `attack_canary`: the attack, or a documented limitation, must still be observable. This guards
  against "the defense passes because the attack silently broke".

Each invariant applies to one **posture**: `undefended` (the attack), `defended` (replay or adaptive
attack against the defense) or `paired`. A `paired` invariant compares the two on the **same test
samples** (`src/mlbreak/stats.py`):

| Statistic | Definition (paired population: samples eligible for the attack in both evaluations) |
|---|---|
| `absolute_reduction` | success rate undefended − success rate defended |
| `relative_reduction` | absolute_reduction / success rate undefended (undefined if that is 0) |
| `mcnemar_p_value` | exact one-sided McNemar test of "the defense reduces success": P(X ≤ c), X ~ Bin(b + c, ½), with b = successes only without the defense and c = only with it |

A defense's claim ("it reduces attack success") is comparative, so its primary invariants are paired.
A single bound on the defended posture cannot tell a working defense from a no-op one when the undefended
rate is already low, and it breaks when numerical noise moves one number across a fixed line. That is what
happened to MLB-REG-002 ([regression-analysis-mlb-reg-002.md](regression-analysis-mlb-reg-002.md)).
Significance uses α = 0.05, fixed in advance. Effect-size thresholds and residual bounds are set from
measurements across numerical environments, with the observed ranges recorded in each invariant's
description.

`MLB-REG-002` requires adversarial training to reduce adaptive PGD success significantly
(McNemar p ≤ 0.05), by ≥ 5 pp and ≥ 15% relative, with residual success ≤ 35% for its pinned seed.
`MLB-REG-004` encodes a known limitation as a canary: confidence rejection is bypassed by an
adaptive attacker.

ML02 cases (`owasp_id: ML02:2023`) use the same invariant kinds. *undefended* is the poisoning experiment,
and *defended* is its defensive replay:

| Case | Pins |
|---|---|
| MLB-REG-005 | kNN sanitisation detects ≥ 80% of random flips and restores accuracy, at low clean-data cost |
| MLB-REG-006 | sparse targeted flips are detected and targeted success is reduced |
| MLB-REG-007 | known limitation: boundary-targeted flips are invisible to kNN (detection ≤ 20%, success stays ≥ 15%) |
| MLB-REG-008 | known limitation: dense targeted flips make sanitisation relabel clean rows and do not restore the class |

ML01 spam-filter cases (`family: text_evasion`) use the same invariant kinds; *undefended* is the attack on the
undefended filter and *defended* its replay or a new adaptive attack:

| Case | Pins |
|---|---|
| MLB-REG-009 | normalisation significantly reduces a stronger-effort adaptive attacker (paired McNemar, ≥ 5 pp), with utility kept |
| MLB-REG-010 | normalisation blocks replayed adversarial messages (and why replay alone overstates it) |
| MLB-REG-011 | Unicode normalisation neutralises invisible-character insertion, adaptively |
| MLB-REG-012 | adversarial data augmentation significantly reduces a stronger-effort adaptive attacker |
| MLB-REG-013 | known limitation: for naive Bayes, normalisation does not significantly reduce an adaptive attacker |

Every text case also asserts `constraint_violation_rate ≤ 0`: no recorded evasion violates a declared constraint.

The ML02 methodology (threat model, metrics, evaluation assumptions) is in
[ml02-data-poisoning.md](ml02-data-poisoning.md).
