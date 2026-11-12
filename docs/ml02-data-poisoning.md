# ML02:2023 — Data Poisoning lab (label flipping)

Scenario `ml02-moons-label-flip`. Code: `src/mlbreak/poisoning/`. Regression cases: `MLB-REG-005` to
`MLB-REG-008`. Evidence schema: `mlbreak.poisoning/v1`.

> **Intentionally vulnerable.** The undefended training pipeline trusts its labels. The lab uses only the
> bundled synthetic `moons-v1` dataset and is for education and defensive research.

## 1. System under test

The victim trains a classifier (`mlp-tanh-16x16` or `logreg`, the same approved configurations as ML01) on
the 350-sample training split of `moons-v1`, then deploys it on the 150-sample test split. The **clean
model** is trained on the true labels. It is bit-identical to the ML01 base model for the same model and
seed, and every ML02 record states its artifact hash.

## 2. Threat model

| | |
|---|---|
| **Asset** | Integrity of the training labels, and therefore of the trained model's decisions |
| **Trust boundary** | Label source → training pipeline (annotation vendor, crowdsourcing, a writable feedback store) |
| **Attacker can** | Change the **labels** of at most `round(fraction × 350)` training samples **before training** (`fraction` ≤ 0.5) |
| **Attacker cannot** | Modify features, add or remove samples, touch test data, the learning algorithm, its hyper-parameters, the seed, or the trained model |
| **Timing** | Before training. The victim always **retrains** on the poisoned set; no step reuses the clean model's weights |

### Attacker knowledge per strategy

| Strategy | Selection | Knowledge required | Goal |
|---|---|---|---|
| `random` | uniform over all training rows | none (blind label noise) | lower overall accuracy (availability) |
| `targeted` | uniform over rows of `target_class` | training labels | make the model miss `target_class` (targeted integrity) |
| `targeted` + `boundary` | the `target_class` rows with the smallest clean-model margin \|logit\| | training labels **and** the clean model's margins | same, with fewer flips |

For `boundary`, the attacker is given the clean model itself: a **perfect-knowledge surrogate**. That is an
upper bound on what a real attacker with a surrogate trained on similar data would achieve.

The budget is enforced, not assumed. `enforce_poison_budget` fails the experiment (`PoisonBudgetViolation`)
in three cases: the poisoned label vector differs from the clean one outside the selected rows, the flip
count exceeds the budget, or a targeted attack touches another class. The integrity check re-verifies this
from the stored rows (`poisoned_indices`, `poison_budget`).

### Defender

**kNN label sanitisation** (after Paudice et al., 2018, *Label Sanitization against Label Flipping
Poisoning Attacks*). For each training row, take its `k` nearest training neighbours in feature space,
Euclidean, excluding itself, with ties broken by index. If the fraction that share its label is below
`agreement_threshold`, the label is **suspicious**. The defense then either **relabels** it to the
neighbourhood majority (a tie keeps the original label) or **removes** it. The model is retrained on the
result. The defender does not know which rows were poisoned and does not see test data.

## 3. Evaluations

| Mode / evaluation | What runs |
|---|---|
| `baseline` / `clean_training` | Train on clean labels; evaluate on the clean test split |
| `attack` / `poisoning` | Select rows (seeded), flip labels, verify the budget, **retrain**, evaluate the poisoned model next to the clean model |
| `defend` / `poisoning_replay` | **Defensive replay**: load an integrity-verified poisoning record and take its *exact* observed label vector. Retrain the poisoned model and require its artifact hash to match the parent's. Then sanitise, retrain, evaluate. The same sanitisation is also applied to the **clean** labels, to measure the defense's cost when there is no attack |

Exact replay (`mlbreak replay <id>`) re-executes any of these and compares the model artifact hashes, label
vector hashes, poisoned indices, metrics and per-row results. It is refused if the stored record fails
integrity verification.

### Evaluation assumptions

* Test inputs and test labels are **clean**. Poisoning acts only through training.
* Every compared model is trained with the **same** features, configuration and seed. Only the labels
  differ, so any difference in behaviour is caused by the labels.
* The defense is evaluated against a **non-adaptive** poisoner: the poisoned set was chosen without knowledge
  of the defense. MLB-REG-007 and MLB-REG-008 show that informed or dense poisoning defeats it. An attacker
  that explicitly optimises against kNN sanitisation is **not** implemented (see Limitations).
* Results are for one dataset and one seed by default. Effect sizes on 150 test samples are a handful of
  samples, so every rate is reported with its numerator and denominator.

## 4. Metrics

All metrics are re-derived from the per-row evidence by `mlbreak check`. Every ratio carries
`numerator`, `denominator`, `population` and `definition`. An empty denominator is `value: null`
(undefined), never 0.

| Metric | Definition (population) |
|---|---|
| `test_accuracy` | correct predictions / test samples, for the model under evaluation (all test samples) |
| `clean_test_accuracy` | the same, for the clean model |
| `train_accuracy` | predictions equal to the **true** label / training samples |
| `train_accuracy_observed` | predictions equal to the label the learner was **given** / rows it was trained on. Shows how far the model fitted the manipulated labels |
| `recall_class_0`, `recall_class_1` | per-class recall on the test split |
| `target_class_recall` | recall of `target_class` (targeted attacks only) |
| `poison_rate` | flipped labels / training samples |
| `attack_success_rate` | test samples the **clean** model classifies correctly and the evaluated model misclassifies / test samples the clean model classifies correctly. Population: the whole test split (random) or the `target_class` test samples (targeted) |
| `collateral_damage_rate` | the same, on the *other* class (targeted only) |
| `test_outcomes` | partition of test samples, clean → evaluated model: `kept`, `broken`, `fixed`, `still_wrong` |
| `detection_recall` | flagged poisoned rows / poisoned rows |
| `detection_precision` | flagged poisoned rows / flagged rows |
| `false_positive_rate` | flagged clean rows / clean rows |
| `labels_repaired` / `labels_corrupted` | (relabel) poisoned rows whose sanitised label is the true label / poisoned rows; clean rows whose sanitised label is wrong / clean rows |
| `removed` | (remove) removed rows / training rows |
| `recovery_rate` | test samples broken by the poisoning that the defended model gets right / test samples broken by the poisoning |
| `clean_data_test_accuracy`, `clean_data_flag_rate` | the defense applied to the clean labels: resulting test accuracy and flagged fraction (the defense's cost with no attack) |

## 5. Reference results

MLP, seed 7, k = 10, threshold 0.5, relabel. Clean model: test accuracy 96.7% (145/150), class-1 recall
94.7% (71/75). Produced by `mlbreak demo --lab ml02` and the regression suite.

| Poisoning | Undefended ASR | kNN detection recall | Defended ASR | Reading |
|---|---|---|---|---|
| random 20% | 3.4% (5/145) | 87.1% (61/70) | 1.4% (2/145) | The regularised MLP tolerates random noise; sanitisation repairs most of it |
| targeted class 1, random rows, 10% | 5.6% (4/71) | 88.6% (31/35) | 2.8% (2/71) | Sparse targeted flips are still local outliers |
| targeted class 1, boundary rows, 10% | **23.9%** (17/71) | **0% (0/35)** | 22.5% (16/71) | Informed flips sit where neighbours already agree with them: the defense is blind |
| targeted class 1, random rows, 20% | **47.9%** (34/71) | 40.0% (28/70) | **52.1%** (37/71) | Dense flips shift neighbourhood majorities; sanitisation relabels clean class-1 rows (15.4% false positives) and makes it **worse** |

The point of the lab is the last two rows. A defense that looks good against blind label noise can do
nothing, or harm, against a modestly informed poisoner.

### Statistical support for each claim

Paired comparison of undefended vs defended on the same test samples (exact one-sided McNemar). These
values are identical in all 7 numerical environments tested: ML02 outcomes are prediction-level, and the
predictions agree even where the model hashes differ.

| Case | Broken test samples (undefended → defended) | b | c | p | What the evidence supports |
|---|---|---|---|---|---|
| MLB-REG-005 random 20% | 5 → 2 (of 145) | 3 | 0 | 0.125 | Detection (87.1%) and label repair. The test-time recovery is **not significant**: the poisoning barely hurt the model |
| MLB-REG-006 targeted, sparse | 4 → 2 (of 71) | 2 | 0 | 0.25 | Detection (88.6%) and label repair. The test-time reduction is **not significant** |
| MLB-REG-007 targeted, boundary | 17 → 16 (of 71) | 1 | 0 | 0.5 | Sanitisation is bypassed: no significant reduction, 0% detection |
| MLB-REG-008 targeted, dense | 34 → 37 (of 71) | 7 | 10 | 0.83 | Sanitisation **backfires**: success rises by 4.2 pp |

The regression cases encode exactly these claims. 005 and 006 assert label repair (`labels_repaired ≥ 0.80`),
detection and "not made worse" (paired `absolute_reduction ≥ 0`). Their test-time bounds are kept, but
described as absolute bounds rather than evidence of recovery. 007 and 008 assert the bypass and the
backfire with paired canaries.

## 6. Limitations

* **Two features, 350 training rows.** Neighbourhoods are meaningful in 2-D; kNN sanitisation degrades in
  high dimensions, and real label noise is not uniform.
* **Label flipping only.** No feature or clean-label poisoning, backdoors/triggers, or optimisation-based
  poisoning (e.g. bilevel or gradient-based attacks). Those belong to future ML02 scenarios.
* **No adaptive poisoner.** The boundary strategy defeats kNN sanitisation as a side effect, not by design.
  No attack is optimised against the deployed defense, so the defended results are an upper bound on
  defense effectiveness, not a robustness guarantee.
* **Perfect-knowledge surrogate** for `boundary` selection, which overstates a realistic attacker.
* **Small effect sizes** for random and sparse targeted flips (2–5 test samples). Their test-time effect is
  not statistically significant on 150 test samples (see the table above). Several absolute thresholds have
  a 1–2 sample margin. They were identical across the 7 numerical environments tested, but a dependency
  upgrade could move them.
* **One defense family.** No loss-based filtering, influence functions, robust losses, or provenance controls
  on the label source, which is the strongest mitigation in practice.
* **No poisoning detection at inference time**, and no monitoring of the label pipeline over time (that
  overlaps with ML08 Model Skewing).
* **Integrity, not authenticity**, as for all MLBreak evidence (see `reproducibility.md`).
