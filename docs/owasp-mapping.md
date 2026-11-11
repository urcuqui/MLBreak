# OWASP mapping

Taxonomy: **OWASP Machine Learning Security Top 10, 2023 edition**
(<https://owasp.org/www-project-machine-learning-security-top-10/>). The registry lives in
`src/mlbreak/owasp.py`. Check the OWASP project page for the current publication status of the list.

| ID | Name | MLBreak status | Scenario |
|---|---|---|---|
| ML01:2023 | Input Manipulation Attack | **Implemented** | `ml01-moons-evasion` |
| ML02:2023 | Data Poisoning Attack | Not implemented | — |
| ML03:2023 | Model Inversion Attack | Not implemented | — |
| ML04:2023 | Membership Inference Attack | Not implemented | — |
| ML05:2023 | Model Theft | Not implemented | — |
| ML06:2023 | AI Supply Chain Attacks | Not implemented | — |
| ML07:2023 | Transfer Learning Attack | Not implemented | — |
| ML08:2023 | Model Skewing | Not implemented | — |
| ML09:2023 | Output Integrity Attack | Not implemented | — |
| ML10:2023 | Model Poisoning | Not implemented | — |

## ML01 coverage

| OWASP ML01 element | MLBreak |
|---|---|
| Attacker manipulates input data at inference time | Bounded test-time perturbation of input features |
| Adversarial examples crafted to cause misclassification | FGSM, PGD (white-box), random search (black-box) |
| Prevention: adversarial training | `adversarial_training` defense, evaluated by replay and adaptive attack |
| Prevention: robust models / detection of anomalous inputs | `confidence_rejection` (a simple abstention control; shown to be bypassable adaptively) |
| Input validation | Feature-domain constraint enforced for all inputs |

Each category entry has the same metadata contract (`id`, `code`, `name`, `summary`,
`reference_url`, `taxonomy_version`, `lab_status`, `scenario_ids`). A category moves to
`implemented` only together with a working, tested scenario.
