# Reproducibility

Every source of randomness derives from the experiment `seed`:

| Component | Seeding |
|---|---|
| Dataset generation and split | `make_moons(random_state=seed)`, `train_test_split(random_state=seed)` |
| Model initialisation | `random_state=seed` |
| PGD random start / random search | `numpy.random.default_rng([seed, test_index])`, so each sample is independent of selection order |
| Adversarial training | `default_rng([seed, round, i])` |

Each evidence record stores the full validated request, dataset and split hashes, the model artifact
SHA-256 (over its parameter arrays), the deployed-system artifact hash, `config_sha256`,
`results_sha256` (over all per-sample results and metrics), and Python/NumPy/scikit-learn/platform
versions.

## Replaying

```bash
mlbreak replay MLB-EXP-xxxxxxxxxxxx      # or the "Replay" button in the UI
```

The replay creates a new experiment (`replay_of` → original) and compares status, model and
target-system artifact hashes, train/test split hashes and `results_sha256`. It exits non-zero if
anything differs.

## Scope of "identical"

Results are bit-identical on the same platform with the same library versions. Floating-point
differences across CPU architectures, BLAS builds or scikit-learn versions can change the trained
parameters in the last bits, and therefore the hashes. They can occasionally also change a
borderline sample's outcome. When the artifact hash differs, `defend` refuses to replay stored
perturbations against a regenerated model. Regression thresholds leave margin for this.
