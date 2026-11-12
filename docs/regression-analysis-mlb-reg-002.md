# MLB-REG-002 regression: root-cause analysis

**CI run:** <https://github.com/urcuqui/MLBreak/actions/runs/43106288300> (commit `b9bae84`)

```text
FAILED tests/test_regression.py::test_security_regression_suite_passes
MLB-REG-002 / adaptive-success-reduced: observed 0.31724137931034485, expected <= 0.30
```

## Summary

The defense did not regress. **The invariant measured one machine's floating-point rounding.**

The same code with the same package versions (NumPy 2.5.4, scikit-learn 1.9.1, SciPy 1.18.1) trains a
different MLP depending on the **BLAS kernel**: Apple Accelerate on macOS, or OpenBLAS with a CPU-specific
kernel on Linux. Adversarial training (three rounds of PGD-on-train followed by a refit) amplifies that
difference, and the adaptive attack is computed from the resulting model's gradients. Across seven
numerical environments, the adaptive attack success against the defended model ranges from **25.0% to
31.7%**. The invariant `≤ 30%` had been calibrated in one of them (25.5%) and fails in three.

In **every** environment, adversarial training still reduces adaptive success **significantly on the same
samples**: one-sided exact McNemar p ≤ 0.0038, absolute reduction ≥ 8.3 pp, relative reduction ≥ 20.7%.

## Reproduction

CI's backend job installed unpinned dependencies (`pip install -e ".[dev]"`) on `ubuntu-latest` (x86_64).
The CI log is not downloadable without authentication, so the run was reproduced in Docker (`python:3.12-slim`,
`--platform linux/amd64`) with the same versions and `OPENBLAS_CORETYPE` forced to the kernels OpenBLAS
selects on GitHub's x86_64 CPUs:

| Environment (NumPy 2.5.4, scikit-learn 1.9.1, SciPy 1.18.1 unless noted) | BLAS | Base model SHA-256 | L-BFGS iters (base / defended) | Undefended ASR | Adaptive ASR (defended) |
|---|---|---|---|---|---|
| macOS arm64 | Accelerate | `59aa4a5b…` | 312 / 1144 | 57/145 = 39.3% | 37/145 = **25.5%** |
| Linux x86_64, OpenBLAS **Haswell** (= Zen) | scipy-openblas 0.3.34 | `dfb387bb…` | 303 / 1143 | 58/145 = 40.0% | 46/145 = **31.7%** ← **exactly the CI value** |
| Linux x86_64, OpenBLAS SandyBridge | scipy-openblas | `38b51653…` | 273 / 956 | 58/145 = 40.0% | 45/145 = **31.0%** |
| Linux x86_64, OpenBLAS Nehalem | scipy-openblas | `bad8678a…` | 290 / 1266 | 58/145 = 40.0% | 44/145 = **30.3%** |
| Linux x86_64, OpenBLAS Prescott | scipy-openblas | `6080c855…` | 312 / 1188 | 58/145 = 40.0% | 36/144 = **25.0%** |
| Linux arm64, OpenBLAS neoversen1 | scipy-openblas | `942ef14f…` | 304 / 820 | 59/145 = 40.7% | 41/144 = **28.5%** |
| Linux x86_64, Python 3.11, NumPy 2.4.6, SciPy 1.17.1, Haswell | scipy-openblas | (= py3.12 Haswell) | — | 58/145 = 40.0% | 46/145 = **31.7%** |

The **Haswell** row reproduces CI to the last digit (0.31724137931034485 = 46/145).

Ruled out:

* **Thread count**: 1, 4 and 12 OpenBLAS/OpenMP threads give the identical artifact hash.
* **Package versions**: CI resolved exactly the versions installed locally.
* **Metric denominators**: the eligible population is 145 (or 144) in every run, built the same way
  (clean-correct samples that were attacked); the outcome partition sums to 150 everywhere.

## Cause classification (as requested)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| 1. Numerical variation between dependency versions | **Partly. It's the numerical environment, not the versions** | Same versions, different BLAS kernel → different trained parameters |
| 2. Changes in model-training behaviour | No | Same code and configuration; no change in training between the last green run and the failure |
| 3. Initialisation / solver convergence | **Yes (mechanism)** | Same `random_state`; L-BFGS takes 273–312 iterations and lands on different parameters |
| 4. Adaptive PGD evaluation | **Yes (amplifier)** | Undefended ASR varies by ±1 sample; replay by ±4; the adaptive result against adversarial training by 10 |
| 5. Incorrect metric denominators | No | Identical populations and partition sums in every environment |
| 6. Unrealistic regression threshold | **Yes** | `≤ 30%` set from one environment with a 4.5 pp margin, inside the 6.7 pp spread |
| 7. Genuine loss of defensive effectiveness | **No** | Significant paired reduction in all 7 environments (table below) |

## Is the defense still effective? (paired, same samples)

For each environment: *b* = samples evaded only without the defense, *c* = only with it; exact one-sided
McNemar test of "the defense reduces success".

| Environment | paired n | undefended → defended | b | c | absolute | relative | p |
|---|---|---|---|---|---|---|---|
| macOS / Accelerate | 144 | 56 → 36 | 25 | 5 | +13.9 pp | 35.7% | 1.6e-4 |
| Linux x86_64 / Haswell (CI) | 145 | 58 → 46 | 15 | 3 | +8.3 pp | 20.7% | 3.8e-3 |
| Linux x86_64 / SandyBridge | 145 | 58 → 45 | 15 | 2 | +9.0 pp | 22.4% | 1.2e-3 |
| Linux x86_64 / Nehalem | 145 | 58 → 44 | 16 | 2 | +9.7 pp | 24.1% | 6.6e-4 |
| Linux x86_64 / Prescott | 144 | 57 → 36 | 26 | 5 | +14.6 pp | 36.8% | 9.6e-5 |
| Linux arm64 / neoversen1 | 144 | 58 → 41 | 21 | 4 | +11.8 pp | 29.3% | 4.6e-4 |
| Linux x86_64 py3.11 / Haswell | 145 | 58 → 46 | 15 | 3 | +8.3 pp | 20.7% | 3.8e-3 |

### Beyond seed 7

Seeds 0–9 with the MLB-REG-002 configuration (macOS/Accelerate and Linux/OpenBLAS-Haswell, 20 runs):

* the defense reduces adaptive success in **20/20** runs (absolute 2.0–19.1 pp; mean 10.6 pp macOS, 8.2 pp Linux);
* the reduction is significant at α = 0.05 in **18/20** runs (not significant: seed 8 on macOS, seed 0 on Linux);
* residual adaptive success ranges **25.5–43.4%**.

So "residual ≤ 30%" was never a property of adversarial training here. It held only for one seed on one
machine.

## The fix

**MLB-REG-002 now states the claim the defense actually makes: it reduces success compared with no defense.**

| Invariant | Kind | Posture | Criterion | Observed range (7 envs) | Rationale |
|---|---|---|---|---|---|
| attack-still-effective | canary | undefended | ASR ≥ 0.30 | 0.393–0.407 | unchanged |
| adaptive-reduction-significant | security | paired | McNemar p ≤ 0.05 | 9.6e-5 – 3.8e-3 | conventional α fixed in advance, not fitted to data |
| adaptive-absolute-reduction | security | paired | ≥ 0.05 | 0.083–0.146 | minimum meaningful effect (≈ 7 of 145 samples) |
| adaptive-relative-reduction | security | paired | ≥ 0.15 | 0.207–0.368 | at least ~⅐ of successes removed |
| adaptive-residual-bounded | security | defended | ASR ≤ 0.35 | 0.250–0.317 | worst observed + ≥ 4.5-sample margin; documented as specific to seed 7 |

**Why this is not a weaker test.**

* The old invariant could not detect a defense that does nothing: if undefended success were ≤ 30%, a no-op
  defense passed. The paired invariants fail it: absolute reduction 0, p = 1.0
  (`tests/test_defense_effect.py::test_single_bound_cannot_tell_a_useless_defense_from_a_useful_one`).
* Neutralising adversarial training (returning the undefended model) fails all three paired invariants
  while the canary passes. A much weaker adversarial training (1 round, ε = 0.01, 1 step) also fails
  (`test_mlb_reg_002_fails_when_adversarial_training_is_neutralised`, `..._too_weak`).
* A seed sweep (`test_adversarial_training_reduces_adaptive_success_across_seeds`) requires a reduction in
  every seed 0–9, a mean absolute reduction ≥ 5 pp and significance in ≥ 7 of 10 seeds. That catches a
  defense that only works for the pinned seed.
* The residual bound moved from 0.30 to 0.35 only after the measurements above. The comparative invariants
  carry the security claim; the bound documents how much success remains.

## Other cases, audited the same way

The same cross-environment and paired analysis was run for every case (`MLB-REG-001` … `008`).

* **ML01 replay and rejection** (001, 003, 004) are stable (±1 sample) and strongly significant.
  MLB-REG-001 gained a paired significance invariant (observed p ≤ 1.6e-8).
* **ML02** metrics are **identical in all 7 environments** even though model hashes differ: its outcomes
  are prediction-level, and the poisoned and defended models agree on every test prediction.
* Two ML02 claims were **not statistically supported**. MLB-REG-005 "accuracy restored" rests on 5 → 2 broken
  samples (p = 0.125), and MLB-REG-006 "targeted success reduced" on 4 → 2 (p = 0.25). Neither was removed:
  both are kept as the absolute bounds they actually are, re-described honestly, and joined by what *is*
  measurable: `labels_repaired ≥ 0.80` (observed 87.1% / 88.6%) and a paired "not made worse" check.
* The limitation cases gained paired canaries. MLB-REG-007 shows no significant reduction (p = 0.5) and
  MLB-REG-008 shows the defense *increases* success (−4.2 pp).

## Making CI deterministic

* `requirements/ci-py3.11.txt` and `ci-py3.12.txt`: hash-pinned locks; CI installs them with
  `--require-hashes`, then the project with `--no-deps`. `pyproject.toml` keeps its ranges for library users.
* `OPENBLAS_CORETYPE=Haswell` in CI: GitHub's x86_64 pool mixes AMD and Intel CPUs (AVX-512 Intel would pick
  the `SkylakeX` kernel), so without it the result could change between runs of the same commit.
* Every record stores `dependencies`, `numerical_backend` (NumPy BLAS build, runtime kernels,
  `OPENBLAS_CORETYPE`) and `numerical_environment_sha256`. A replay in a different numerical environment
  reports the difference instead of implying a reproducibility failure.
* The security gate now runs even when pytest fails, so both signals are visible.

## Results before / after

| | Before (b9bae84) | After |
|---|---|---|
| macOS / Accelerate | 8/8 | 8/8 |
| CI-equivalent (Linux x86_64, Haswell) | **7/8 — MLB-REG-002 fails** | 8/8 |
| Linux x86_64 SandyBridge / Nehalem | **7/8** / **7/8** | 8/8 / 8/8 |
| Linux x86_64 Prescott, Linux arm64 (neoversen1) | 8/8, 8/8 | 8/8, 8/8 |
| Linux arm64, generic armv8 kernel | not tested | 8/8 |

After the fix, the **full pytest suite** also passes in every environment above, including the
locked CI installs (`--require-hashes`) for Python 3.11 and 3.12.
| Python 3.11 (NumPy 2.4.6), Haswell | not tested | 8/8 |
