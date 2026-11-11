# Limitations of the first laboratory

* **Two features.** The 2-D setting makes boundaries visible, but it is far easier than
  high-dimensional inputs. In high dimensions small perturbations are much more effective, and
  geometric intuitions from this lab do not transfer directly.
* **ε is not semantics.** An Lp ball over synthetic features is not a model of "imperceptible" or
  "plausible" input changes. Real systems need domain-specific constraints.
* **Attack strength.** PGD has no restarts by default and stops early at the first boundary
  crossing. Its success rate is a *lower bound* on what a stronger attacker achieves. Increase
  steps, enable random start or raise κ to test further. Random search is a simple baseline, not a
  state-of-the-art decision-based attack (e.g. HopSkipJump).
* **Early stopping biases replay.** Perturbations that barely cross the boundary are low-confidence,
  so replaying them against confidence rejection looks perfect (0%). The adaptive attack and
  `MLB-REG-004` exist to show why that number must not be read as robustness.
* **Adversarial training** here retrains a full-batch L-BFGS learner on a fixed augmented set. It
  approximates, but is not, Madry-style min-max SGD training. It reduces adaptive success, but does
  not eliminate it.
* **Abstention is counted as a non-success.** In settings where forcing abstentions is harmful
  (availability), it would count as an attack outcome too.
* **No certified robustness.** None of the defenses come with guarantees.
* **IBM ART** was evaluated and not adopted. For a 2-feature scikit-learn MLP, analytic gradients
  and ~150 lines of attack code keep the lab transparent and fully tested, with no heavy dependency.
  ART is the right choice when the lab grows to image models or stronger attacks.
* **Reproducibility is platform-scoped**: see [reproducibility.md](reproducibility.md).
* **Timeouts are cooperative.** A running scikit-learn `fit` cannot be interrupted, so an experiment
  can exceed its wall-clock limit by up to one model fit before it is marked as failed. A hard limit
  would need process isolation, which the lab does not use (no subprocesses, no model serialisation).
* **Integrity is tamper-evident, not tamper-proof.** `record_sha256` detects edits of stored evidence,
  but it is not a signature. Someone who can rewrite the file can also recompute every hash. The lab is
  local-first and has no keys to sign with.
* **Strict lineage.** Adaptive attacks must reuse the parent's ε, norm and test samples. Comparing
  defenses across different budgets or populations needs separate, unlinked experiments.
