# Threat model — ML01:2023 evasion lab

The ML02:2023 data-poisoning threat model (label-source trust boundary, label-flipping capabilities) is in
[ml02-data-poisoning.md](ml02-data-poisoning.md#2-threat-model).

## System under test

A *deployed system* that receives a 2-feature input and returns a decision:

* **Undefended (intentionally vulnerable):** `argmax` of an MLP / logistic-regression classifier.
* **Defended:** an adversarially trained model, or the original model with a confidence-rejection
  policy that may return *abstain* (`-1`).

## Assets

| Asset | Why it matters |
|---|---|
| Integrity of decisions | The attacker's goal is a wrong decision on an input whose clean version is decided correctly. |
| Clean-input utility | A defense that refuses or misclassifies legitimate inputs has a real cost. |
| Evidence integrity | Conclusions must be traceable to the exact model, data, configuration and outputs. |

## Trust boundary

**Input → model.** Test-time inputs are attacker-controlled within constraints. Training data,
model parameters, the decision policy and the evaluation harness are trusted. Attacks on those
belong to other OWASP categories (ML02, ML05, ML06, ML10) and are out of scope here.

## Attacker capabilities

| | White-box (FGSM, PGD) | Black-box (random search) |
|---|---|---|
| Knowledge | Architecture, parameters and exact input gradients | None about internals |
| Interaction | Unlimited local computation, counted as model evaluations | Hard-label queries to the deployed system, fixed budget per sample |
| Observes | Logits/probabilities | Returned label or abstention only |
| Can modify | The two input features of a test-time sample | Same |
| Constraint | ‖δ‖ₚ ≤ ε (p ∈ {∞, 2}); input stays in the declared feature domain | Same |
| Knows the defense | Only in **adaptive** experiments (`target_defense` set) | Queries the deployed (possibly defended) system directly |

The constraints are **enforced, not assumed**: every adversarial input passes through
`enforce_constraints`. Any violation fails the experiment (`status: failed`, `BudgetViolation`).

## Attacker goal

Untargeted evasion: `decision(x + δ) ∉ {y, abstain}` for a sample with `decision(x) = y`.

## Out of scope

Physical-world perturbations, semantic constraints beyond the box domain, targeted attacks,
poisoning, model extraction, and denial of service through abstention (an attacker *forcing*
abstentions is not counted as success here; see limitations).

## Platform threat model (the lab itself)

The lab is intentionally vulnerable in its *ML behaviour*, not its *software*. Controls: localhost
binding, no uploads, no deserialisation of untrusted artifacts, no code-execution surfaces, strict
request schemas, resource limits, ID validation against path traversal, and a CSP. See
`tests/test_security_controls.py` and `tests/test_api.py`.
