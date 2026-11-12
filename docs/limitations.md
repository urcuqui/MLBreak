# Limitations

This page covers the ML01 2-D evasion lab and the platform. Scenario-specific limitations:
[ML01 spam-filter evasion](ml01-spam-filter-evasion.md#7-limitations) and
[ML02 data poisoning](ml02-data-poisoning.md#6-limitations).

## ML01 evasion lab and platform


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
* **Reproducibility is environment-scoped.** Identical package versions still give different trained models on
  different BLAS libraries and CPU kernels. Bit-identical replay holds only within one numerical environment
  (`numerical_environment_sha256`). The adaptive-attack result against adversarial training varies by up to
  6.7 pp across environments (25.0–31.7% for MLB-REG-002), and 25.5–43.4% across seeds. See
  [regression-analysis-mlb-reg-002.md](regression-analysis-mlb-reg-002.md).
* **Timeouts are cooperative, not preemptive.** The deadline is checked between uninterruptible segments.
  An experiment can overshoot by the longest segment that started before the deadline (typically one
  scikit-learn fit), plus the short unchecked tail that builds and writes the evidence. That tail
  means a run can be stored as `completed` with `duration_ms` slightly above the limit. The thread keeps
  its concurrency slot while it overshoots, and the regression-suite limit is only checked between
  cases. A hard limit would need process isolation, which the lab does not use (no subprocesses, no
  model serialisation). Details: [reproducibility.md](reproducibility.md#execution-limits-and-timeouts).
* **Integrity, not authenticity.** `record_sha256` and the other fingerprints are unkeyed SHA-256 digests
  stored in the record itself. They detect corruption and inconsistent edits, and replays refuse any
  record that fails them. But anyone who can rewrite an evidence file can recompute every digest, and
  nothing proves who produced a record. Authenticated evidence would need an HMAC or signature with an
  externally held key, or an append-only external log. See
  [reproducibility.md](reproducibility.md#integrity-is-not-authenticity).
* **Strict lineage.** Adaptive attacks must reuse the parent's ε, norm and test samples. Comparing
  defenses across different budgets or populations needs separate, unlinked experiments.
