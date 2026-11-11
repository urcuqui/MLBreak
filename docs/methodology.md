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

## Metrics (all with explicit numerator/denominator)

* `clean_accuracy`: correct decisions / all test samples. Abstentions count as not correct.
* `coverage`, `selective_accuracy`: for rejection systems.
* `attack_success_rate`: (attacked samples with a correct clean decision whose adversarial decision
  is a *different class, not abstain*) / (attacked samples with a correct clean decision). For replay,
  the denominator is restricted to samples the parent actually perturbed **and** the defended system
  decides correctly on the clean input.
* `adversarial_accuracy`: selected samples still decided correctly after the attack / selected samples.
* `perturbation`: L∞ and L2 statistics over attacked and over successful samples, plus the largest
  observed ‖δ‖/ε, which is verified to be ≤ 1.
* `queries`: label queries (black-box) or model evaluations (white-box).

## Security regression

`security_cases/MLB-REG-*.json` declare invariants of three kinds:

* `security`: the defense must hold.
* `utility`: the defense must not destroy clean performance.
* `attack_canary`: the attack, or a documented limitation, must still be observable. This guards
  against "the defense passes because the attack silently broke".

`MLB-REG-004` encodes a known limitation as a canary: confidence rejection is bypassed by an
adaptive attacker.
