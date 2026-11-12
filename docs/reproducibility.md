# Reproducibility

Every source of randomness derives from the experiment `seed`:

| Component | Seeding |
|---|---|
| Dataset generation and split | `make_moons(random_state=seed)`, `train_test_split(random_state=seed)` |
| Model initialisation | `random_state=seed` |
| PGD random start / random search | `numpy.random.default_rng([seed, test_index])`, so each sample is independent of selection order |
| Adversarial training | `default_rng([seed, round, i])` |
| ML02 poisoned-row selection | `default_rng([seed, 202302])` (random / targeted-random); stable margin sort for `boundary` |
| ML02 kNN sanitisation | deterministic: Euclidean distances, ties broken by index (no randomness) |
| ML01 spam corpus split | SHA-256-verified bundled file, exact de-duplication, `train_test_split(stratify, random_state=seed)` |
| ML01 spam attack candidates | `default_rng([seed, 1001, test_index])` per message; augmentation `default_rng([seed, 4242, train_index])` |

ML02 records (`mlbreak.poisoning/v1`) use the same fingerprint scheme. `results_sha256` covers the training
rows, test rows and metrics. The `poisoning` block also stores `clean_labels_sha256`,
`observed_labels_sha256`, `sanitized_labels_sha256` and the exact `poisoned_indices`.

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
internally consistent (an adversarial result requires a correct clean decision). ML02 poisoning records
are verified the same way: their metrics are re-derived from the per-sample label, flag and prediction
vectors.

The check is **enforced**, not advisory:

* **Replay** (`mlbreak replay`, `POST /api/experiments/<id>/replay`) verifies the stored record first.
  If any check fails, the replay is refused before anything executes: HTTP **409** with the full
  verification report, or CLI exit code **3**. Re-executing the request of an edited record would
  otherwise run an arbitrary request and compare it against fabricated results.
* **Lineage**: the parent of a defend/adaptive experiment (ML01) or of a defensive replay (ML02) must
  pass the check before it is used.

### Integrity is not authenticity

The fingerprints are **unkeyed SHA-256 digests stored inside the record they protect**. What that does and
does not give you:

| Threat | Detected? |
|---|---|
| Accidental corruption, truncation, a bad merge | ✅ yes |
| Editing one field (a metric, a prediction, the request, a timestamp) without updating the digests | ✅ yes: `results_sha256`, `metrics_derivation`, `config_sha256` or `record_sha256` fails |
| Editing results *and* recomputing the derived metrics and `results_sha256` | ✅ yes: `record_sha256` still covers the whole record |
| Rewriting a record **and recomputing every digest** (including `record_sha256`) | ❌ **no** |
| Proving *who* produced a record, or *when* | ❌ **no** |

Anyone who can write `var/experiments/*.json` can produce a record that passes `mlbreak check`. Every
verification report says so in its `guarantee` field.
`tests/test_integrity.py::test_fully_reforged_record_passes_integrity_but_not_replay` pins this
limitation. For a forged record, *replay* still exposes the fabrication, because re-execution produces
the genuine results and the comparison fails. Replay is only as trustworthy as the code and environment
that run it, though.

**Authenticated evidence** would need at least one of:

* a keyed MAC (HMAC-SHA-256) or a digital signature (e.g. Ed25519) over `record_sha256`, with the key held
  outside the evidence directory (an OS keychain, an HSM, a CI secret);
* an append-only, externally held log of record digests (a transparency log or a signed git history);
* storage and transport controls (read-only evidence directories, signed artifact upload in CI).

MLBreak is local-first and has no key management, so it provides **integrity only**. Treat evidence from
an untrusted source as unverified until you replay it in an environment you control.

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

Every experiment (ML01 and ML02) runs inside a concurrency gate (`max_concurrent_experiments`,
otherwise HTTP 429) and against a wall-clock deadline (`max_experiment_seconds`, 60 s by default).
Decision-boundary grids and dataset views, which can also train models, share the same gate and
deadline (HTTP 504 on timeout). A timed-out experiment is stored with `status: failed` and no results.
An adversarial training interrupted by the deadline is **never cached**.

### The deadline is cooperative, not preemptive

Python threads cannot be killed safely, and MLBreak deliberately uses no subprocesses (no model
serialisation). So the deadline is a **checkpoint**: the code calls `_check_deadline` between
uninterruptible segments, and work only stops at the next checkpoint after the deadline. Checkpoints sit:

* after the (cached) base-model fit and after building the deployed system;
* adversarial training: before each round, at every PGD step over the training set, and after each fit;
* attacks: every PGD/FGSM step, and before each sample in black-box random search;
* ML02: after each fit (clean, poisoned, sanitised, and the sanitiser applied to clean data) and after the
  kNN label check;
* ML01 attack and replay: after evaluating the adversarial inputs, before the record is built (ML01 baseline
  has no checkpoint after its single clean evaluation).

What that means in practice:

* **Overshoot is bounded by the longest segment that starts before the deadline**, not by zero. The
  dominant segment is one scikit-learn `fit`, bounded by `max_iter` (MLP: 5000 L-BFGS iterations) and the
  fixed dataset size (at most 350 × (1 + 5 rounds) samples for adversarial training). Other segments are
  much shorter: one PGD step, one random-search sample (≤ 2000 queries), one kNN query over 350 points.
* **The unchecked tail after the last checkpoint is not covered.** Building metrics, hashing and writing
  the evidence files run after the final check. An experiment can therefore be stored as `completed` with
  `duration_ms` slightly above the limit. This is expected, not a bug.
* **The thread keeps its slot while it overshoots.** The concurrency gate is released only when the
  experiment returns, and the HTTP client waits for it.
* **A cold base-model fit that crosses the deadline is still cached.** It is a complete fit, not a partial
  one, so later requests reuse it. The experiment that triggered it still fails.
* **Regression suite limit** (`max_regression_seconds`): checked only *between* cases. A case already
  running is not interrupted, so the suite can overshoot by up to one case (two or three experiments,
  each with its own deadline). Cases not started in time are reported as **failed**, never as passed.
* **Timing is not reproducible.** Whether a run times out depends on the machine and its load, so
  `status` on a timeout is a runtime field and a replay may differ there.

A hard limit would need process isolation (a worker process killed on timeout) or OS limits
(`ulimit -t`, cgroups). `compose.yaml` caps CPU, memory and PIDs for the whole container, which bounds
resource use but not the duration of an individual experiment.

## Numerical environment and dependency strategy

**The same package versions do not guarantee the same numbers.** NumPy's BLAS differs by platform (Apple
Accelerate on macOS, OpenBLAS on Linux), and OpenBLAS picks a CPU-specific kernel at runtime (Haswell,
Zen, SkylakeX, Neoverse, …). Each choice changes floating-point rounding. The MLP's L-BFGS fit then
converges to different parameters, so the **model artifact hashes differ between environments even with
identical versions**. Most metrics move by at most a sample or two; ML01 adaptive attacks against adversarial
training move by up to ~10 samples. See [regression-analysis-mlb-reg-002.md](regression-analysis-mlb-reg-002.md).

What MLBreak does about it:

| Layer | Mechanism |
|---|---|
| Library users | `pyproject.toml` keeps compatible ranges (`numpy>=1.26`, …). Results are scientifically equivalent but not bit-identical across stacks |
| CI and Docker | Hash-pinned locks: `requirements/ci-py3.11.txt`, `requirements/ci-py3.12.txt` (with dev tools), `requirements/runtime-py3.12.txt` (Docker). Installed with `pip install --require-hashes -r …` then `pip install --no-deps -e .` |
| CPU kernel | CI sets `OPENBLAS_CORETYPE=Haswell`, because GitHub's x86_64 runner pool mixes AMD and Intel (AVX-512) CPUs |
| Evidence | Every record stores `dependencies` (exact versions), `numerical_backend` (NumPy BLAS build, runtime kernels, `OPENBLAS_CORETYPE`) and `numerical_environment_sha256` |
| Replay | Compares environments first. If `numerical_environment_sha256` differs, the replay reports it in `environment_differences` and explains why results are not bit-identical |
| Regression | Security claims use **paired comparisons** (same samples, undefended vs defended) and margins measured across environments, not single-environment numbers |

Print your environment with `mlbreak env`.

### Supported and tested environments

| Environment | Python | NumPy / SciPy / scikit-learn | BLAS (kernel) | Status |
|---|---|---|---|---|
| **CI reference** (Linux x86_64) | 3.12 | 2.5.4 / 1.18.1 / 1.9.1 (locked) | OpenBLAS (Haswell, pinned) | tests + gate + demo pass |
| **CI reference** (Linux x86_64) | 3.11 | 2.4.6 / 1.17.1 / 1.9.1 (locked) | OpenBLAS (Haswell, pinned) | tests + gate + demo pass |
| Developer macOS arm64 | 3.12 | 2.5.4 / 1.18.1 / 1.9.1 | Accelerate | tests + gate pass |
| Linux x86_64, other kernels | 3.12 | locked | OpenBLAS SandyBridge / Nehalem / Prescott | tests + gate pass |
| Linux arm64 | 3.12 | locked | OpenBLAS (neoversen1; generic armv8) | tests + gate pass |

Python 3.11 resolves to NumPy 2.4.x because NumPy 2.5 dropped 3.11. On the Haswell kernel, the two CI
stacks produced identical regression results.

Regenerate the locks after an intentional dependency upgrade (commands at the top of each file), then run
`pytest` and `mlbreak verify` in CI before merging. A change of numerical stack is a change of the
measurement instrument, so treat it like a code change.

## Scope of "identical"

Results are bit-identical **only within the same numerical environment** (`numerical_environment_sha256`):
same platform and CPU architecture, BLAS build and kernel, and library versions. MLBreak does **not**
claim bit-identical results across environments. Across them, model artifact hashes differ, and a
borderline sample's outcome can change. Consequences:

* `defend` (ML01) and defensive replay (ML02) refuse to reuse a parent whose model artifact cannot be
  regenerated bit-for-bit in the current environment, instead of silently comparing different models.
* ML02 boundary plots retrain the model and require its recorded artifact hash. A record created in another
  numerical environment therefore cannot be plotted on this one; re-run the experiment locally.
* Regression thresholds are set from measurements across the environments above, and comparative claims are
  tested with paired statistics.
