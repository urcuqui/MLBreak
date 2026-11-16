"""Private-attribute inversion attacks. Attacker-side code ONLY.

Nothing in this module can reach ground truth, model parameters or private training records: the attack receives
an :class:`AttackerView` (public features, the observed output, the candidate domain and the attacker's own
auxiliary prior) and a ``query`` function (see :func:`~.defenses.counting_oracle`). The evaluation engine compares
the inference with ground truth afterwards.

Candidate matching (after the "known-output" inversion of Fredrikson et al., 2014):

1. order candidates deterministically — the one the auxiliary prior prefers first (value 0 first on an exact
   0.5 prior), then the rest in ascending order;
2. query ``(public features, s = c)`` for each candidate in that order while budget remains, and keep the
   candidates whose response is within ``tolerance`` of the observed output; an unqueried candidate cannot be
   excluded and stays consistent;
3. stop as soon as exactly one candidate is consistent.

Inference rule (:func:`decide`, also used by the integrity check to re-derive every stored decision):

* one consistent candidate that was queried → ``unique_match``; one that was never queried (all others were
  excluded) → ``elimination``. Both rely on the service being DETERMINISTIC and the attacker knowing every
  other input exactly: the real target's own candidate then reproduces the observed output;
* several consistent candidates → AMBIGUOUS; ``tie_policy`` decides: ``auxiliary_prior`` returns the prior's
  choice among them, ``abstain`` returns no inference;
* no consistent candidate (impossible for a deterministic service with exact public features) →
  ``no_consistent_candidate``, treated as ambiguous over the whole domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .schemas import AttackerTrace, CandidateQuery, InversionAttackConfig

DOMAIN: tuple[int, ...] = (0, 1)  # possible values of the binary private attribute (attacker knowledge)


@dataclass(frozen=True)
class AttackerView:
    """What the adversary knows about ONE target record. Deliberately has no field for the private attribute,
    the target label, the model or any training record."""

    public_features: tuple[float, ...]
    observed_output: float
    auxiliary_p1: float
    candidate_domain: tuple[int, ...] = DOMAIN


def prior_choice(p1: float, among: Sequence[int] = DOMAIN) -> int:
    """The auxiliary prior's choice restricted to ``among`` (binary: 1 iff P(s=1) > 0.5; ties → 0)."""
    preferred = 1 if p1 > 0.5 else 0
    return preferred if preferred in among else min(among)


def candidate_order(view: AttackerView) -> list[int]:
    first = prior_choice(view.auxiliary_p1, view.candidate_domain)
    return [first] + sorted(c for c in view.candidate_domain if c != first)


def decide(observed: float, responses: dict[int, float], domain: Sequence[int], p1: float, tie_policy: str,
           tolerance: float) -> tuple[list[int], bool, str, int | None]:
    """→ (consistent candidates, ambiguous, rule, inferred value or None)."""
    consistent = [c for c in domain if c not in responses or abs(responses[c] - observed) <= tolerance]
    if len(consistent) == 1:
        c = consistent[0]
        return consistent, False, ("unique_match" if c in responses else "elimination"), c
    pool, rule = (consistent, None) if consistent else (list(domain), "no_consistent_candidate")
    if tie_policy == "abstain":
        return consistent, True, rule or "tie_abstain", None
    return consistent, True, rule or "tie_auxiliary_prior", prior_choice(p1, pool)


def candidate_matching(view: AttackerView, query: Callable[[tuple[float, ...], int], float],
                       cfg: InversionAttackConfig) -> AttackerTrace:
    responses: dict[int, float] = {}
    queries: list[CandidateQuery] = []
    for c in candidate_order(view):
        consistent, _, _, _ = decide(view.observed_output, responses, view.candidate_domain, view.auxiliary_p1,
                                     cfg.tie_policy, cfg.tolerance)
        if len(consistent) == 1 or len(queries) >= cfg.query_budget:
            break
        r = query(view.public_features, c)
        responses[c] = r
        queries.append(CandidateQuery(query_index=len(queries) + 1, candidate=c, response=r,
                                      distance=abs(r - view.observed_output)))
    consistent, ambiguous, rule, inferred = decide(view.observed_output, responses, view.candidate_domain,
                                                   view.auxiliary_p1, cfg.tie_policy, cfg.tolerance)
    separation = abs(responses[1] - responses[0]) if 0 in responses and 1 in responses else None
    return AttackerTrace(
        public_features=list(view.public_features), observed_output=view.observed_output,
        candidate_domain=list(view.candidate_domain), auxiliary_p1=view.auxiliary_p1, queries=queries,
        n_queries=len(queries), query_budget=cfg.query_budget,
        budget_exhausted=len(responses) < len(view.candidate_domain) and len(queries) >= cfg.query_budget
        and ambiguous,
        consistent_candidates=consistent, ambiguous=ambiguous, rule=rule, inferred=inferred, separation=separation)


def auxiliary_guess(p1: np.ndarray) -> np.ndarray:
    """The auxiliary-only attacker: no victim output, no query — the prior's choice for every target."""
    return np.array([prior_choice(float(p)) for p in p1], dtype=np.int64)


# --------------------------------------------------------------------------------------------------------------
# Degraded-knowledge attacker (sensitivity studies only; the reference experiment uses candidate_matching above)
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PartialView:
    """Attacker knowledge when public features are incomplete or imprecise.

    ``public_features`` holds what the attacker BELIEVES each feature to be (possibly noisy), ``None`` where the
    feature is unknown. ``imputations`` are complete public-feature rows from the attacker's own auxiliary
    population, used to fill the unknown features. No ground truth is present."""

    public_features: tuple[float | None, ...]
    observed_output: float
    auxiliary_p1: float
    imputations: tuple[tuple[float, ...], ...] = ()
    candidate_domain: tuple[int, ...] = DOMAIN


@dataclass(frozen=True)
class NearestResult:
    inferred: int | None
    ambiguous: bool
    rule: str
    n_queries: int
    scores: dict[int, float]


def nearest_matching(view: PartialView, query: Callable[[tuple[float, ...], int], float], budget: int,
                     tie_policy: str, margin: float) -> NearestResult:
    """Nearest-response matching with marginalisation over unknown features.

    Completions: the believed features with unknown entries filled from each imputation row (one completion if
    nothing is unknown). For each completion in turn, query every candidate (prior-preferred first) while budget
    remains. Score(c) = mean |response − observed| over c's queries; the lower score wins if it beats the other by
    more than ``margin``; otherwise — or if some candidate was never queried — the target is ambiguous and
    ``tie_policy`` applies. Exact matching would fail here because the attacker's records differ from the target's.
    """
    missing = [j for j, v in enumerate(view.public_features) if v is None]
    rows = view.imputations if missing else ((),)
    completions = []
    for row in rows:
        comp = [v for v in view.public_features]
        for j in missing:
            comp[j] = row[j]
        completions.append(tuple(float(v) for v in comp))
    dist: dict[int, list[float]] = {c: [] for c in view.candidate_domain}
    order = candidate_order(AttackerView((), view.observed_output, view.auxiliary_p1, view.candidate_domain))
    n = 0
    for comp in completions:
        for c in order:
            if n >= budget:
                break
            dist[c].append(abs(query(comp, c) - view.observed_output))
            n += 1
    scores = {c: sum(d) / len(d) for c, d in dist.items() if d}
    if len(scores) == len(view.candidate_domain):
        ranked = sorted(scores, key=lambda c: (scores[c], c))
        if scores[ranked[1]] - scores[ranked[0]] > margin:
            return NearestResult(ranked[0], False, "nearest_response", n, scores)
    if tie_policy == "abstain":
        return NearestResult(None, True, "tie_abstain", n, scores)
    return NearestResult(prior_choice(view.auxiliary_p1), True, "tie_auxiliary_prior", n, scores)
