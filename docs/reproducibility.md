# Reproducibility

Every source of randomness derives from the experiment `seed`:

| Component | Seeding |
|---|---|
| Dataset generation and split | `make_moons(random_state=seed)`, `train_test_split(random_state=seed)` |
| Model initialisation | `random_state=seed` |
| PGD random start / random search | `numpy.random.default_rng([seed, test_index])`, so each sample is independent of selection order |
| Adversarial training | `default_rng([seed, round, i])` |

Each evidence record stores the full validated request, dataset and split hashes, the model artifact
SHA-256 (over its parameter arrays) and the deployed-system artifact hash. It also stores
Python/NumPy/scikit-learn/pydantic/platform versions and the BLAS thread pools, and these fingerprints:

| Field | Covers |
|---|---|
| `config_sha256` | the validated request |
| `samples_sha256` | per-sample results, excluding the derived `outcome`, so it is comparable across schema versions |
| `metrics_sha256` | the metrics block |
| `results_sha256` | samples (with `outcome`) + metrics; same formula as v1 for v1 records |
| `record_sha256` | the whole record except itself (v2): any edit, including timestamps and lineage, is detected |

## Which fields are reproducible

`reproducibility.fields` in every record lists this explicitly.

* **Bit-for-bit on replay** (same platform, CPU architecture, BLAS build and library versions): the
  request/`config_sha256`, dataset split hashes, model and target-system artifact hashes, every
  per-sample value (adversarial inputs, predictions, probabilities, queries, outcome), the metrics,
  `samples_sha256`, `metrics_sha256`, `results_sha256` and the parent hashes in `lineage`.
* **Runtime-dependent**: `experiment_id`, `created_at`, `duration_ms`, `replay_of`, the environment
  block, and therefore `record_sha256`, which deliberately covers these fields to make the stored
  file tamper-evident. A replay never compares `record_sha256`. Whether an experiment completes or
  times out also depends on runtime, because limits are wall-clock.

## Evidence integrity

```bash
mlbreak check MLB-EXP-xxxxxxxxxxxx        # or GET /api/experiments/<id>/verify
```

This recomputes every fingerprint and **re-derives the metrics from the per-sample results**, without
re-running the experiment. It also checks the outcome partition arithmetic and that each sample is
internally consistent (an adversarial result requires a correct clean decision). Parents of
defend/adaptive experiments must pass this check before they are used. The demo verifies every record
it writes.

## Replaying

```bash
mlbreak replay MLB-EXP-xxxxxxxxxxxx      # or the "Replay" button in the UI
```

The replay creates a new experiment (`replay_of` → original). It compares status, `config_sha256`, model
and target-system artifact hashes, train/test split hashes, `samples_sha256`, `metrics_sha256` and
`results_sha256`, and it exits non-zero if anything differs. Environment differences (library versions,
platform) are reported separately as `environment_differences`. For a **v1** original, metrics and
results hashes are not comparable (the metric definitions changed), so only per-sample results are
compared and a note explains why.

## Execution limits and timeouts

Every experiment runs inside a concurrency gate (`max_concurrent_experiments`, otherwise HTTP 429) and
against a wall-clock deadline (`max_experiment_seconds`). The deadline covers base-model training,
**adversarial training** (checked before each round, at every PGD step on the training set and after
each fit), attack loops, replay and evaluation. A timed-out experiment is stored with
`status: failed` and no results. An interrupted adversarial training is **never cached**. Decision-
boundary grids and dataset views, which can also train models, share the same gate and deadline (HTTP
504 on timeout). Only one security regression suite runs at a time, and a suite-level limit
(`max_regression_seconds`) fails any case not started in time instead of passing it.

The deadline is cooperative. A single scikit-learn `fit` cannot be interrupted, so the worst-case
overshoot is one fit. That is bounded by the model's `max_iter` (MLP: 5000 L-BFGS iterations) and the fixed
dataset size (adversarial training: at most 350 × (1 + 5 rounds) samples).

## Scope of "identical"

Results are bit-identical on the same platform with the same library versions. Floating-point
differences across CPU architectures, BLAS builds or scikit-learn versions can change the trained
parameters in the last bits, and therefore the hashes. They can occasionally also change a
borderline sample's outcome. When the artifact hash differs, `defend` refuses to replay stored
perturbations against a regenerated model. Regression thresholds leave margin for this.
