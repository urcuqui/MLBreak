"""Label-flipping poisoning (OWASP ML02:2023) and kNN label sanitisation.

Threat model:

* The attacker controls the LABELS of a bounded fraction of the training set before training (e.g. a
  compromised annotation pipeline, crowdsourced labels, or a writable feedback store). They do not
  modify features, test data, the learning algorithm, its hyper-parameters or the trained model.
* The budget is ``round(fraction × n_train)`` flipped labels, checked by :func:`enforce_poison_budget`.
  A violation fails the experiment instead of being silently clipped.

Everything here is a pure, deterministic function of its inputs and the seed.
"""

from __future__ import annotations

import numpy as np

from .schemas import KnnLabelSanitization, LabelFlipConfig

# Domain separation for the poisoning RNG, so it never shares a stream with ML01 (seed, test_index).
_POISON_STREAM = 2023_02


class PoisonBudgetViolation(RuntimeError):
    """The poisoned label vector differs from the clean labels in more (or other) places than allowed."""


class InvalidPoisonConfig(ValueError):
    """A well-formed config that cannot be applied to this training set."""


def n_flips(cfg: LabelFlipConfig, n_train: int) -> int:
    return int(round(cfg.fraction * n_train))


def select_poison_indices(cfg: LabelFlipConfig, y: np.ndarray, seed: int,
                          clean_margin: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    """Indices (sorted) of the training samples whose labels are flipped, and the rule that chose them.

    ``clean_margin`` (|logit| of the clean model on the training set) is required only for
    ``targeted`` + ``boundary``.
    """
    y = np.asarray(y)
    k = n_flips(cfg, len(y))
    if k < 1:
        raise InvalidPoisonConfig(f"fraction {cfg.fraction} of {len(y)} training samples flips no labels")
    rng = np.random.default_rng([seed, _POISON_STREAM])
    if cfg.strategy == "random":
        idx = rng.choice(len(y), size=k, replace=False)
        rule = f"uniform random sample of {k}/{len(y)} training indices (rng=[seed, {_POISON_STREAM}])"
    else:
        pool = np.flatnonzero(y == cfg.target_class)
        if k > len(pool):
            raise InvalidPoisonConfig(f"budget of {k} flips exceeds the {len(pool)} training samples of "
                                      f"target class {cfg.target_class}")
        if cfg.selection == "random":
            idx = rng.choice(pool, size=k, replace=False)
            rule = (f"uniform random sample of {k}/{len(pool)} class-{cfg.target_class} training indices "
                    f"(rng=[seed, {_POISON_STREAM}])")
        else:
            if clean_margin is None:
                raise ValueError("boundary selection needs the clean model's margins")
            order = np.argsort(np.asarray(clean_margin)[pool], kind="stable")  # ties → lower index first
            idx = pool[order[:k]]
            rule = (f"the {k}/{len(pool)} class-{cfg.target_class} training samples with the smallest "
                    "clean-model margin |logit| (stable sort, ties by index)")
    return np.sort(idx.astype(np.int64)), rule


def flip_labels(y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    y_obs = np.array(y, dtype=np.int64, copy=True)
    y_obs[idx] = 1 - y_obs[idx]
    return y_obs


def enforce_poison_budget(y_true: np.ndarray, y_obs: np.ndarray, idx: np.ndarray, cfg: LabelFlipConfig) -> None:
    """The ONLY differences between clean and observed labels are the selected flips, within budget, and
    (targeted) only on the target class."""
    changed = np.flatnonzero(np.asarray(y_true) != np.asarray(y_obs))
    if not np.array_equal(changed, np.sort(idx)):
        raise PoisonBudgetViolation("observed labels differ from the clean labels outside the selected indices")
    if len(changed) > n_flips(cfg, len(y_true)):
        raise PoisonBudgetViolation(f"{len(changed)} flips exceed the budget of {n_flips(cfg, len(y_true))}")
    if cfg.strategy == "targeted" and np.any(np.asarray(y_true)[changed] != cfg.target_class):
        raise PoisonBudgetViolation("a targeted attack flipped a sample outside the target class")


# --------------------------------------------------------------------------
# Defense: kNN label sanitisation
# --------------------------------------------------------------------------
def knn_neighbours(X: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest neighbours of every row (Euclidean, self excluded, ties by index)."""
    X = np.asarray(X, dtype=float)
    d = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1, kind="stable")[:, :k]


def knn_label_check(X: np.ndarray, y_obs: np.ndarray, cfg: KnnLabelSanitization):
    """Return (agreement, flagged, majority) for every training sample."""
    nb = knn_neighbours(X, cfg.k)
    y_obs = np.asarray(y_obs)
    agreement = (y_obs[nb] == y_obs[:, None]).mean(axis=1)
    flagged = agreement < cfg.agreement_threshold
    ones = y_obs[nb].sum(axis=1)
    # Majority label of the neighbourhood; a tie keeps the sample's own label.
    majority = np.where(2 * ones > cfg.k, 1, np.where(2 * ones < cfg.k, 0, y_obs))
    return agreement, flagged, majority.astype(np.int64)


def sanitize(y_obs: np.ndarray, flagged: np.ndarray, majority: np.ndarray, action: str):
    """Return (keep_mask, y_sanitized) for the defended training set."""
    y_obs = np.asarray(y_obs, dtype=np.int64)
    if action == "remove":
        return ~flagged, y_obs.copy()
    return np.ones(len(y_obs), dtype=bool), np.where(flagged, majority, y_obs).astype(np.int64)
