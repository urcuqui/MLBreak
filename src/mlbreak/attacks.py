"""Test-time evasion attacks (OWASP ML01:2023 — Input Manipulation).

Threat model per attack:

* **FGSM / PGD — white-box.** The attacker knows the model architecture and
  parameters and can compute exact input gradients of the logit. They can
  modify the two input features of a test-time sample within an Lp ball of
  radius ε, and the result must stay inside the dataset's feature domain.
* **Random search — black-box, decision-based.** The attacker can only submit
  inputs to the *deployed system* and observe the returned label (or an
  abstention). They have a fixed query budget and the same ε / domain
  constraints. No gradients, scores or parameters are available.

Every attack result passes through :func:`enforce_constraints`. A violation
raises :class:`BudgetViolation` and the experiment is recorded as failed — it
is never silently clipped after the fact.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .schemas import AttackConfig

ABSTAIN = -1
_BATCH = 50  # black-box queries are issued in batches of this size
_TOL = 1e-9


class BudgetViolation(RuntimeError):
    """An adversarial example exceeded ε or left the feature domain."""


class ExperimentTimeout(RuntimeError):
    """The experiment exceeded its wall-clock limit."""


class QueryBudgetExceeded(RuntimeError):
    """A black-box attacker tried to exceed its query budget."""


@dataclass
class AttackOutput:
    x_adv: np.ndarray
    reached_goal: np.ndarray  # attacker-side belief that the goal was met
    queries: np.ndarray  # model evaluations used per sample
    effective_confidence: float


def lp_norm(delta: np.ndarray, norm: str) -> np.ndarray:
    delta = np.atleast_2d(delta)
    if norm == "linf":
        return np.max(np.abs(delta), axis=1)
    if norm == "l2":
        return np.sqrt(np.sum(delta**2, axis=1))
    raise ValueError(norm)


def project(x: np.ndarray, x0: np.ndarray, eps: float, norm: str,
            lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Project onto {x : ||x - x0||_p <= eps} ∩ domain box.

    Because x0 lies inside the box, the box clip moves every coordinate towards
    x0 and therefore never increases the perturbation norm.
    """
    d = x - x0
    if norm == "linf":
        d = np.clip(d, -eps, eps)
    else:
        n = lp_norm(d, "l2")
        d = d * np.minimum(1.0, eps / np.maximum(n, 1e-12))[:, None]
    return np.clip(x0 + d, lo, hi)


def enforce_constraints(x0: np.ndarray, x_adv: np.ndarray, eps: float, norm: str,
                        lo: np.ndarray, hi: np.ndarray) -> float:
    """Raise :class:`BudgetViolation` unless every sample respects ε and the domain.

    Returns the largest observed ||δ||_p / ε ratio (≤ 1).
    """
    if x0.shape != x_adv.shape or not np.all(np.isfinite(x_adv)):
        raise BudgetViolation("adversarial inputs have the wrong shape or are not finite")
    norms = lp_norm(x_adv - x0, norm)
    worst = float(np.max(norms)) if len(norms) else 0.0
    if worst > eps + _TOL:
        raise BudgetViolation(f"perturbation {worst:.6g} exceeds {norm} budget ε={eps}")
    if np.any(x_adv < lo - _TOL) or np.any(x_adv > hi + _TOL):
        raise BudgetViolation("adversarial input left the valid feature domain")
    return worst / eps


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise ExperimentTimeout("experiment exceeded its wall-clock limit")


def _ball_sample(rng: np.random.Generator, n: int, dim: int, eps: float, norm: str) -> np.ndarray:
    if norm == "linf":
        return rng.uniform(-eps, eps, size=(n, dim))
    direction = rng.normal(size=(n, dim))
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    radius = eps * rng.uniform(size=(n, 1)) ** (1.0 / dim)
    return direction * radius


def gradient_attack(model, X0: np.ndarray, y: np.ndarray, cfg: AttackConfig,
                    lo: np.ndarray, hi: np.ndarray, rngs: list[np.random.Generator],
                    deadline: float | None = None, min_confidence: float = 0.0) -> AttackOutput:
    """FGSM (one signed step of size ε) or PGD (iterated, projected steps).

    The attacker minimises the true-class margin ``m(x) = s·logit(x)`` with
    ``s = +1`` for class 1 and ``-1`` for class 0, and stops early once
    ``m(x) < -κ``. ``κ`` is ``max(cfg.confidence, min_confidence)``;
    ``min_confidence`` lets an *adaptive* attacker clear a rejection threshold.
    For PGD the iterate with the lowest margin is returned (best-iterate PGD).
    """
    X0 = np.asarray(X0, dtype=float)
    s = np.where(y == 1, 1.0, -1.0)
    kappa = max(cfg.confidence, min_confidence)
    eps, norm, alpha = cfg.epsilon, cfg.norm, cfg.effective_step_size
    queries = np.zeros(len(X0), dtype=np.int64)

    x = X0.copy()
    if cfg.random_start:
        noise = np.stack([_ball_sample(r, 1, X0.shape[1], eps, norm)[0] for r in rngs])
        x = project(X0 + noise, X0, eps, norm, lo, hi)

    margin = s * model.logit(x)
    queries += 1
    best_x, best_m = x.copy(), margin.copy()
    done = margin < -kappa

    for _ in range(cfg.steps):
        _check_deadline(deadline)
        active = np.flatnonzero(~done)
        if active.size == 0:
            break
        g = s[active, None] * model.grad_logit(x[active])
        if norm == "linf":
            step = -alpha * np.sign(g)
        else:
            step = -alpha * g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        x[active] = project(x[active] + step, X0[active], eps, norm, lo, hi)
        m = s[active] * model.logit(x[active])
        queries[active] += 2  # one gradient + one forward evaluation
        better = m < best_m[active]
        best_x[active[better]] = x[active[better]]
        best_m[active[better]] = m[better]
        done[active] |= m < -kappa

    x_out = x if cfg.attack == "fgsm" else best_x
    reached = s * model.logit(x_out) < -kappa
    return AttackOutput(x_out, reached, queries, kappa)


def random_search_attack(oracle: Callable[[np.ndarray], np.ndarray], X0: np.ndarray,
                         y: np.ndarray, cfg: AttackConfig, lo: np.ndarray, hi: np.ndarray,
                         rngs: list[np.random.Generator],
                         deadline: float | None = None) -> AttackOutput:
    """Decision-based black-box random search.

    For each sample, draw ``query_budget`` candidates uniformly from the ε-ball,
    clip them to the domain, sort by perturbation norm and query the deployed
    system in batches, smallest perturbations first. The first candidate whose
    returned label differs from the true label (and is not an abstention) is
    the minimum-norm success among the drawn candidates. If none is found the
    original input is returned and the attempt counts as a failure.
    """
    X0 = np.asarray(X0, dtype=float)
    x_adv = X0.copy()
    reached = np.zeros(len(X0), dtype=bool)
    queries = np.zeros(len(X0), dtype=np.int64)
    for i, rng in enumerate(rngs):
        _check_deadline(deadline)
        cand = X0[i] + _ball_sample(rng, cfg.query_budget, X0.shape[1], cfg.epsilon, cfg.norm)
        cand = np.clip(cand, lo, hi)
        cand = cand[np.argsort(lp_norm(cand - X0[i], cfg.norm), kind="stable")]
        for start in range(0, len(cand), _BATCH):
            batch = cand[start:start + _BATCH]
            if queries[i] + len(batch) > cfg.query_budget:
                raise QueryBudgetExceeded("random search attempted to exceed its query budget")
            labels = oracle(batch)
            queries[i] += len(batch)
            hit = np.flatnonzero((labels != y[i]) & (labels != ABSTAIN))
            if hit.size:
                x_adv[i] = batch[hit[0]]
                reached[i] = True
                break
    return AttackOutput(x_adv, reached, queries, 0.0)
