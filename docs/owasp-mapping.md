# OWASP mapping

Taxonomy: **OWASP Machine Learning Security Top 10, 2023 edition**
(<https://owasp.org/www-project-machine-learning-security-top-10/>). The registry lives in
`src/mlbreak/owasp.py`. Check the OWASP project page for the current publication status of the list.

| ID | Name | MLBreak status | Scenario |
|---|---|---|---|
| ML01:2023 | Input Manipulation Attack | **Implemented** (2 scenarios) | `ml01-moons-evasion` (2-D numerical evasion), `ml01-spam-filter-evasion` (character-level text evasion) |
| ML02:2023 | Data Poisoning Attack | **Implemented** | `ml02-moons-label-flip` |
| ML03:2023 | Model Inversion Attack | Not implemented | — |
| ML04:2023 | Membership Inference Attack | Not implemented | — |
| ML05:2023 | Model Theft | Not implemented | — |
| ML06:2023 | AI Supply Chain Attacks | Not implemented | — |
| ML07:2023 | Transfer Learning Attack | Not implemented | — |
| ML08:2023 | Model Skewing | Not implemented | — |
| ML09:2023 | Output Integrity Attack | Not implemented | — |
| ML10:2023 | Model Poisoning | Not implemented | — |

A category is marked implemented only when it has a registered scenario, passing acceptance tests and
at least one passing security regression case. `tests/test_regression.py` enforces this.

## ML02 coverage

| OWASP ML02 element | MLBreak |
|---|---|
| Attacker manipulates training data so the model behaves as intended | Label flipping of a bounded fraction of the training set, then retraining |
| Untargeted (availability) vs targeted poisoning | `random` vs `targeted` (one class; `random` or `boundary` row selection) |
| Prevention: data validation and verification of training labels | kNN label sanitisation (relabel or remove suspicious labels), evaluated by defensive replay |
| Prevention: separating and monitoring training data sources | Not modelled (out of scope; see [ml02-data-poisoning.md](ml02-data-poisoning.md#6-limitations)) |

Details: [ml02-data-poisoning.md](ml02-data-poisoning.md).

## ML01 coverage

| OWASP ML01 element | MLBreak |
|---|---|
| Attacker manipulates input data at inference time | Bounded test-time perturbation of input features |
| Adversarial examples crafted to cause misclassification | FGSM, PGD (white-box), random search (black-box) |
| Prevention: adversarial training | `adversarial_training` defense, evaluated by replay and adaptive attack |
| Prevention: robust models / detection of anomalous inputs | `confidence_rejection` (a simple abstention control; shown to be bypassable adaptively) |
| Input validation | Feature-domain constraint enforced for all inputs |

ML01 has **two working scenarios**:

| Scenario | Input | Attacks | Defenses | Details |
|---|---|---|---|---|
| `ml01-moons-evasion` | 2 numerical features | FGSM, PGD (white-box), random search (black-box) | adversarial training, confidence rejection | [methodology.md](methodology.md) |
| `ml01-spam-filter-evasion` | SMS text, TF-IDF | DeepWordBug-style character edits (black-box, score-based), optional homoglyph / zero-width charsets | text normalisation, adversarial data augmentation | [ml01-spam-filter-evasion.md](ml01-spam-filter-evasion.md) |

For the spam scenario, the OWASP ML01 elements map as follows: the attacker manipulates *message text* at
inference time; adversarial examples are character-level edits under explicit text-preservation constraints;
"input validation" is text normalisation (Unicode, leetspeak, spell-snap); "adversarial training" is data
augmentation with adversarial training spam.

Each category entry has the same metadata contract (`id`, `code`, `name`, `summary`,
`reference_url`, `taxonomy_version`, `lab_status`, `scenario_ids`). A category moves to
`implemented` only together with a working, tested scenario.
